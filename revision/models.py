import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.flood_transformer.model import ExplainableSparseTransformer
from src.flood_transformer.sparsity import update_sparsity


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Unsupported nondeterministic GPU operations fail instead of silently
    # weakening reproducibility. Record CUDA/PyTorch versions in each run.
    torch.use_deterministic_algorithms(True)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y[:, -1]).squeeze(-1)


def transformer(dim, cfg):
    return ExplainableSparseTransformer(dim, cfg["d_model"], cfg["n_heads"], cfg["num_layers"],
                                        cfg["d_ff"], cfg["dropout"], enable_learnable_masks=False)


def phase_switch(model, phase, activation_ratio=0.):
    masks = phase == "mask"
    model.enable_learnable_masks = masks
    for layer in model.layers:
        layer.attn.enable_learnable_masks = masks
        layer.mlp.enable_learnable_masks = masks
        layer.mlp.enable_activation_sparsity = phase != "dense"
        layer.mlp.activation_sparsity_ratio = activation_ratio
    for name, param in model.named_parameters():
        if "mask_logits" in name:
            param.requires_grad_(masks)


def loader(split, cfg, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(split["x"]), torch.from_numpy(split["y"])),
                      batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], generator=torch.Generator().manual_seed(cfg["seed"]))


@torch.no_grad()
def predict(model, split, cfg, overrides=None, hidden=False):
    model.eval()
    preds, states = [], []
    for x, _ in loader(split, cfg):
        x = x.to(cfg["device"])
        if isinstance(model, LSTM):
            output = model(x)
        else:
            result = model(x, return_intermediates=hidden, **(overrides or {}))
            output = result.pred
            if hidden:
                states.append(result.hidden_states[-1][:, -1].cpu().numpy())
        preds.append(output.cpu().numpy())
    pred = np.concatenate(preds)
    return (pred, np.concatenate(states)) if hidden else pred


def fit(model, splits, cfg, kind="steh", trial=None):
    model.to(cfg["device"])
    train = loader(splits["train"], cfg, shuffle=True)
    phases = [("dense", cfg["epochs_dense"]), ("sparse", cfg["epochs_sparse"]),
              ("mask", cfg["epochs_mask"])] if kind == "steh" else [("lstm", cfg["epochs_lstm"])]
    log = []
    for phase, epochs in phases:
        if epochs < 1:
            raise ValueError("Every requested training phase needs >=1 epoch")
        if kind == "steh":
            phase_switch(model, phase, 0. if phase == "dense" else cfg["target_activation_sparsity"])
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=(trial or {}).get("learning_rate", cfg["learning_rate"]),
                                 weight_decay=cfg["weight_decay"])
        best_loss, best = float("inf"), None
        for epoch in range(epochs):
            model.train()
            squared, count = 0., 0
            for batch, (x, y) in enumerate(train):
                x, y = x.to(cfg["device"]), y.to(cfg["device"])
                sparse = kind == "steh" and phase != "dense"
                if sparse:
                    step = epoch * len(train) + batch + 1
                    ratio = update_sparsity(step, epochs * len(train), cfg["target_weight_sparsity"]) if phase == "sparse" else cfg["target_weight_sparsity"]
                    act = update_sparsity(step, epochs * len(train), cfg["target_activation_sparsity"]) if phase == "sparse" else cfg["target_activation_sparsity"]
                    phase_switch(model, phase, act)
                    model.apply_global_weight_sparsity(ratio, "neuronwise", cfg["minimum_alive_per_neuron"])
                optim.zero_grad(set_to_none=True)
                pred = model(x) if kind == "lstm" else model(x, return_intermediates=False).pred
                mse = ((pred-y)**2).mean()
                loss = mse
                if phase == "mask":
                    internal = sum(torch.sigmoid(layer.attn.head_mask_logits).sum() + torch.sigmoid(layer.mlp.neuron_mask_logits).sum() for layer in model.layers)
                    loss = loss + cfg["lambda_mask_l1"]*internal + cfg["lambda_input_mask_l1"]*torch.sigmoid(model.input_mask_logits).sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optim.step()
                # Projection after update prevents last-step regrowth in checkpoints.
                # Support can change between steps; this is dynamic sparsity, not
                # permanently frozen edges.
                if sparse:
                    model.apply_global_weight_sparsity(ratio, "neuronwise", cfg["minimum_alive_per_neuron"])
                squared += float(mse.detach()) * len(y)
                count += len(y)
            # Every Stage-II candidate is evaluated at the FINAL sparsity budget.
            # Its saved weights and activation flags therefore agree on restore.
            evaluated = model
            if kind == "steh" and phase == "sparse":
                evaluated = copy.deepcopy(model)
                phase_switch(evaluated, phase, cfg["target_activation_sparsity"])
                evaluated.apply_global_weight_sparsity(cfg["target_weight_sparsity"], "neuronwise", cfg["minimum_alive_per_neuron"])
            val_loss = float(np.mean((predict(evaluated, splits["val"], cfg)-splits["val"]["y"])**2))
            log.append({"phase": phase, "epoch": epoch+1, "train_mse": squared/count, "val_mse": val_loss})
            print(f"  {phase} {epoch+1}/{epochs} val_mse={val_loss:.6g}", flush=True)
            if np.isfinite(val_loss) and val_loss < best_loss:
                best_loss, best = val_loss, copy.deepcopy(evaluated.state_dict())
        if best is None:
            raise RuntimeError("No finite validation checkpoint")
        model.load_state_dict(best)
        if kind == "steh" and phase == "sparse":
            phase_switch(model, phase, cfg["target_activation_sparsity"])
    return model, log
