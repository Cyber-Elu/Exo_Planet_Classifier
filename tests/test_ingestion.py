import pandas as pd
from src.recup_data import load_raw, build_features

def test_ingestion_filters():
    df = pd.DataFrame({
        "pl_name":["ok","neg"],
        "pl_rade":[1.0,-1.0],
        "pl_bmasse":[1.0,2.0],
        "pl_orbsmax":[1.0,1.0]
    })
    out = load_raw(df)
    assert out.shape[0] == 1
    feat = build_features(out)
    assert {"density_rel_earth","g_rel_earth"}.issubset(feat.columns)
