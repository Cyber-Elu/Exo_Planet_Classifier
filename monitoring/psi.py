import numpy as np
import pandas as pd

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = expected.replace([np.inf,-np.inf], np.nan).dropna().astype(float)
    a = actual.replace([np.inf,-np.inf], np.nan).dropna().astype(float)
    quantiles = np.linspace(0, 1, bins+1)
    cuts = np.unique(np.quantile(e, quantiles))
    e_counts, _ = np.histogram(e, bins=cuts)
    a_counts, _ = np.histogram(a, bins=cuts)
    e_perc = np.maximum(e_counts / max(e_counts.sum(), 1), 1e-6)
    a_perc = np.maximum(a_counts / max(a_counts.sum(), 1), 1e-6)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))
