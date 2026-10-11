import hashlib
import json
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def metrics(pred, target):
    p, y = np.asarray(pred, float), np.asarray(target, float)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = p[valid], y[valid]
    if not len(y):
        return {"n": 0, "nse": None, "kge": None, "rmse": None, "bias": None}
    var = np.sum((y - y.mean()) ** 2)
    nse = 1 - np.sum((p - y) ** 2) / var if var > 0 else np.nan
    r = np.corrcoef(p, y)[0, 1] if np.std(p) > 0 and np.std(y) > 0 else np.nan
    kge = 1 - np.sqrt((r - 1)**2 + (np.std(p) / np.std(y) - 1)**2 + (p.mean() / y.mean() - 1)**2) if np.std(y) > 0 and abs(y.mean()) > 1e-12 else np.nan
    return clean({"n": len(y), "nse": nse, "kge": kge, "rmse": np.sqrt(np.mean((p-y)**2)),
                  "mae": np.mean(abs(p-y)), "bias": (p-y).mean()})


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = successes / total
    center = (p + z*z/(2*total)) / (1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [max(0., center-half), min(1., center+half)]


def finite_population_interval(successes, sample_n, population_n, confidence=.95):
    """Equal-tailed exact interval by inverting the hypergeometric distribution.

    Appropriate for a prespecified simple random sample without replacement.
    Each unit must have one complete, fixed pass/fail outcome.
    """
    from scipy.stats import hypergeom
    if not 0 <= successes <= sample_n <= population_n or sample_n < 1:
        raise ValueError("Invalid finite-population counts")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    tail = (1-confidence)/2
    low, high = successes, population_n-sample_n+successes
    # First possible K for which observing >=successes is not rejected.
    while low < high:
        mid = (low+high)//2
        if hypergeom.sf(successes-1,population_n,mid,sample_n) >= tail:
            high = mid
        else:
            low = mid+1
    lower = low
    low, high = successes, population_n-sample_n+successes
    # Last possible K for which observing <=successes is not rejected.
    while low < high:
        mid = (low+high+1)//2
        if hypergeom.cdf(successes,population_n,mid,sample_n) >= tail:
            low = mid
        else:
            high = mid-1
    return {"confidence": confidence, "success_count_bounds": [lower,low],
            "success_fraction_bounds": [lower/population_n,low/population_n]}
