import pandas as pd
from src.ml_exoplanete import build_labels, CANON

def test_labeling_heuristic():
    df = pd.DataFrame({
        CANON["mass"]: [5, 50],
        CANON["radius"]: [1.5, 5.0],
        "density_rel_earth": [6.0, 0.8],
        "g_rel_earth": [2.2, 1.1],
        CANON["sma"]: [1.0, 3.0],
        CANON["name"]: ["A","B"]
    })
    out = build_labels(df.copy())
    assert set(out["pl_type"]) == {"Tellurique", "Gazeuse"}
    assert out.loc[out[CANON["name"]]=="A","pl_type"].iloc[0] == "Tellurique"
