"""Bounded validation-selected unit interventions, with explicit control limits."""
import itertools
import numpy as np
import torch
from .common import metrics
from .models import predict


def subset(split, maximum, seed):
    ids = np.sort(np.random.default_rng(seed).choice(len(split["y"]), min(maximum, len(split["y"])), replace=False))
    return {k: v[ids] for k, v in split.items()}


def binary_overrides(model, device):
    # Same >=0.5 convention as the model's STE forward, including exact zeros.
    inputs, heads, neurons = model.get_mask_probabilities()
    return {"input_mask_override": (inputs >= .5).float().to(device),
            "head_mask_overrides": {i: (p >= .5).float().to(device) for i, p in enumerate(heads)},
            "neuron_mask_overrides": {i: (p >= .5).float().to(device) for i, p in enumerate(neurons)}}


def clone_masks(masks):
    return {k: {i: t.clone() for i, t in v.items()} if isinstance(v, dict) else v.clone() for k, v in masks.items()}


def nse(p, y):
    return metrics(p, y)["nse"]


def interpret(model, splits, meta, cfg):
    val = subset(splits["val"], cfg["analysis_val_samples"], cfg["analysis_seed"])
    test = subset(splits["test"], cfg["analysis_test_samples"], cfg["analysis_seed"]+1)
    base = binary_overrides(model, cfg["device"])
    pv, pt = predict(model, val, cfg, base), predict(model, test, cfg, base)
    bv, bt = nse(pv, val["y"]), nse(pt, test["y"])
    if bv is None or bt is None:
        raise ValueError("Constant target in bounded analysis subset; increase analysis sample sizes")
    # Input removals set standardized values to zero (training mean), not raw zero.
    inputs = []
    val_losses = {}
    for i, name in enumerate(meta["feature_names"]):
        mask = clone_masks(base)
        mask["input_mask_override"][i] = 0
        a, b = predict(model, val, cfg, mask), predict(model, test, cfg, mask)
        val_losses[i] = np.mean((a-val["y"])**2)
        inputs.append({"feature": name, "active": bool(base["input_mask_override"][i]),
                       "val_delta_nse": bv-nse(a, val["y"]), "test_delta_nse": bt-nse(b, test["y"])})
    pairs = []
    base_mse = np.mean((pv-val["y"])**2)
    active_inputs = [i for i in range(len(inputs)) if inputs[i]["active"]]
    for i, j in itertools.combinations(active_inputs, 2):
        mask = clone_masks(base)
        mask["input_mask_override"][[i,j]] = 0
        loss = np.mean((predict(model, val, cfg, mask)-val["y"])**2)
        pairs.append((float(loss-val_losses[i]-val_losses[j]+base_mse), i, j))
    pairs.sort(key=lambda x: (-abs(x[0]), x[1], x[2]))
    interactions = []
    test_loss = np.mean((pt-test["y"])**2)
    for score, i, j in pairs[:3]:
        mask = clone_masks(base)
        mask["input_mask_override"][[i,j]] = 0
        pair_loss = np.mean((predict(model, test, cfg, mask)-test["y"])**2)
        singles = []
        for z in (i,j):
            mask = clone_masks(base)
            mask["input_mask_override"][z] = 0
            singles.append(np.mean((predict(model, test, cfg, mask)-test["y"])**2))
        interactions.append({"features": [meta["feature_names"][i],meta["feature_names"][j]],
                             "val_interaction_mse": score, "test_interaction_mse": float(pair_loss-sum(singles)+test_loss)})
    pools, rankings, coverage = {}, {}, []
    for key in ("head_mask_overrides", "neuron_mask_overrides"):
        for layer, flags in base[key].items():
            module = model.layers[layer]
            if key.startswith("neuron"):
                incoming, outgoing = module.mlp.fc1.weight, module.mlp.fc2.weight.T
                strengths = (incoming.norm(dim=1)*outgoing.norm(dim=1)).detach()
                degrees = [(int((incoming[i]!=0).sum()), int((outgoing[i]!=0).sum())) for i in range(len(flags))]
            else:
                att = module.attn
                strengths, degrees = [], []
                for h in range(len(flags)):
                    ids = torch.arange(h*att.d_head, (h+1)*att.d_head, device=flags.device)
                    rows = torch.cat([ids, ids+att.d_model, ids+2*att.d_model])
                    a, b = att.qkv.weight[rows], att.out_proj.weight[:, ids]
                    strengths.append(float((a.norm()*b.norm()).detach()))
                    degrees.append((int((a!=0).sum()), int((b!=0).sum())))
            pool = [i for i in range(len(flags)) if flags[i] > 0 and min(degrees[i]) > 0]
            group = (key, layer)
            pools[group] = (pool, degrees)
            cap = cfg["max_neuron_candidates_per_layer"] if key.startswith("neuron") else 0
            candidates = sorted(pool, key=lambda i: (-float(strengths[i]), i))
            if cap > 0:
                candidates = candidates[:cap]
            ranking = []
            for i in candidates:
                masks = clone_masks(base)
                masks[key][layer][i] = 0
                delta = bv-nse(predict(model, val, cfg, masks), val["y"])
                ranking.append((delta, i))
            rankings[group] = sorted(ranking, key=lambda v: (-v[0], v[1]))
            coverage.append({"kind": key, "layer": layer, "trained_active": int(flags.sum()),
                             "with_nonzero_weight_paths": len(pool), "validation_ranked": len(candidates)})
    core = clone_masks(base)
    selected = {}
    for group, ranking in rankings.items():
        key, layer = group
        k = max(1, int(np.ceil(len(ranking)*cfg["core_fraction"]))) if ranking else 0
        selected[group] = [i for _, i in ranking[:k]]
        core[key][layer].zero_()
        if k:
            core[key][layer][selected[group]] = 1
    removed = clone_masks(base)
    for (key, layer), ids in selected.items():
        removed[key][layer][ids] = 0
    pc, pr = predict(model, test, cfg, core), predict(model, test, cfg, removed)
    sc, sr = nse(pc, test["y"]), nse(pr, test["y"])
    rng = np.random.default_rng(cfg["seed"]+cfg["analysis_seed"])
    random_rows = []
    for repeat in range(cfg["random_controls"]):
        masks = clone_masks(base)
        deviations, overlaps, count = [], 0, 0
        for (key, layer), chosen in selected.items():
            pool, degrees = pools[(key,layer)]
            available = list(pool)
            masks[key][layer].zero_()
            # Match layer/type/count exactly and weight degrees approximately.
            # Randomized order and nearest-degree ties keep this a structural
            # control. Report mismatch and overlap; do not claim exact graph matching.
            for i in rng.permutation(chosen):
                distances = [abs(degrees[z][0]-degrees[i][0])+abs(degrees[z][1]-degrees[i][1]) for z in available]
                best = min(distances)
                choices = [z for z,d in zip(available, distances) if d == best]
                pick = int(rng.choice(choices))
                available.remove(pick)
                masks[key][layer][pick] = 1
                deviations.append(best)
                overlaps += int(pick in chosen)
                count += 1
        random_rows.append({"repeat": repeat, "nse": nse(predict(model, test, cfg, masks), test["y"]),
                            "mean_degree_mismatch": float(np.mean(deviations)) if deviations else None,
                            "core_overlap_fraction": overlaps/count if count else 1.})
    random_scores = [r["nse"] for r in random_rows if r["nse"] is not None]
    # Strictly better avoids credit for identical controls. This is a descriptive
    # predeclared criterion, not a p-value or a proof of universal circuit validity.
    win = float(np.mean([sc > score for score in random_scores])) if sc is not None and random_scores else 0.
    nonempty = all(len(ids) > 0 for ids in selected.values())
    passed = bool(nonempty and sc is not None and sr is not None and bv is not None and bt is not None
                  and bt-sc <= cfg["core_max_delta_nse"] and bt-sr >= cfg["remove_min_delta_nse"]
                  and win >= cfg["random_min_win_fraction"])
    return {"val_dates": val["dates"], "test_dates": test["dates"], "inputs": inputs,
            "interactions": interactions, "candidate_coverage": coverage,
            "core_units": {f"{k}:{l}": ids for (k,l),ids in selected.items()},
            "circuit": {"full_nse": bt, "core_nse": sc, "removed_nse": sr,
                        "core_delta_nse": bt-sc if sc is not None else None,
                        "remove_delta_nse": bt-sr if sr is not None else None,
                        "random_win_fraction": win, "pass": passed}, "random_controls": random_rows,
            "limits": "Unit-mask interventions retain input gates, biases, residuals, embedding and readout. Core is validation-selected from bounded candidates; not an isolated closed graph or globally minimal circuit."}


def probes(model, splits, meta, cfg):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    train = subset(splits["train"], cfg["probe_train_samples"], cfg["analysis_seed"]+2)
    test = subset(splits["test"], cfg["analysis_test_samples"], cfg["analysis_seed"]+1)
    rain = next((i for i,n in enumerate(meta["feature_names"]) if n.lower()=="p" or any(t in n.lower() for t in ("prcp", "precip", "rain"))), None)
    if rain is None:
        return {"status": "rain_feature_missing", "results": []}
    _, htrain = predict(model, train, cfg, hidden=True)
    _, htest = predict(model, test, cfg, hidden=True)
    def targets(split):
        p = split["x"][:,:,rain]*meta["x_std"][rain]+meta["x_mean"][rain]
        weights = cfg["api_decay"] ** np.arange(p.shape[1]-1, -1, -1)
        return {"CUM": p.sum(axis=1), "API": p @ weights,
                "CHG": (p[:,-1]-p[:,-2])/(p[:,-2]+cfg["chg_epsilon_mm"])}
    ty, vy = targets(train), targets(test)
    rows = []
    for name in ty:
        for representation, a, b in (("trained_hidden", htrain, htest), ("input_history", train["x"].reshape(len(htrain),-1), test["x"].reshape(len(htest),-1))):
            try:
                fit = Ridge(alpha=cfg["probe_ridge"]).fit(a, ty[name])
                r2 = float(r2_score(vy[name], fit.predict(b))) if np.var(vy[name]) > 0 else None
                rows.append({"target": name, "representation": representation, "r2": r2,
                             "n_train": len(a), "n_test": len(b), "status": "ok" if r2 is not None else "constant_target"})
            except (ValueError, np.linalg.LinAlgError) as e:
                rows.append({"target": name, "representation": representation, "r2": None,
                             "n_train": len(a), "n_test": len(b), "status": "failed", "reason": str(e)})
    return {"status": "computed", "results": rows,
            "formulas": {"CUM": "sum(P_window_mm)", "API": f"sum({cfg['api_decay']}^(L-1-j)*P_j)",
                         "CHG": f"(P_last-P_previous)/(P_previous+{cfg['chg_epsilon_mm']})"},
            "limits": "Unfiltered test R2; negative values and failures retained. Input-derived scores demonstrate decodability, not independent physical states."}
