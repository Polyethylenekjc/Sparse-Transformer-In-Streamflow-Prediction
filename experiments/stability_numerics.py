"""Read-only robust functional comparisons of stored representations."""
import numpy as np


def centered(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite sample-by-feature matrix')
    # Translation before averaging makes exactly constant stored columns zero,
    # including columns whose FP32 sample mean accumulates rounding error.
    shifted=values-values[:1]
    return shifted-shifted.mean(axis=0)


def linear_cka_fp64(a,b):
    if len(a)!=len(b):raise ValueError('Samples must align')
    x,y=centered(a),centered(b)
    denominator=np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro')
    return float(np.linalg.norm(x.T@y,'fro')**2/denominator) if denominator else None


def correlation_fp64(a,b):
    x,y=centered(np.asarray(a).reshape(-1,1)),centered(np.asarray(b).reshape(-1,1))
    if len(x)!=len(y):raise ValueError('Samples must align')
    denominator=np.linalg.norm(x)*np.linalg.norm(y)
    return float((x.T@y).item()/denominator) if denominator else None
