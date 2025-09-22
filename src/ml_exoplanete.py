import os, sqlite3
from typing import List
import numpy as np, pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from src.config import CFG

RANDOM_STATE = CFG.RANDOM_STATE
MODEL_PATH = "models/exoplanet_clf.joblib"
DB_PATH = "exoplanetes.db"
TABLE_SRC = "exoplanetes"
TABLE_DST = "planet_classifications"

CANON = { "name":"pl_name", "radius":"pl_rade", "mass":"pl_bmasse", "sma":"pl_orbsmax" }
FEATURES: List[str] = [CANON["mass"], CANON["radius"], "density_rel_earth", "g_rel_earth", CANON["sma"]]

def load_data_from_db() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_SRC}", c)

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias = {"pl_masse":CANON["mass"], "pl_radj":CANON["radius"], "pl_dens":"density_rel_earth"}
    for src,dst in alias.items():
        if src in df.columns and dst not in df.columns: df=df.rename(columns={src:dst})
    for c in [CANON["mass"], CANON["radius"], CANON["sma"]]:
        if c not in df.columns: raise KeyError(f"Colonne manquante: {c}")
        df[c]=pd.to_numeric(df[c], errors="coerce")
    if "density_rel_earth" not in df.columns:
        df["density_rel_earth"]=df[CANON["mass"]]/(df[CANON["radius"]]**3)
    if "g_rel_earth" not in df.columns:
        df["g_rel_earth"]=df[CANON["mass"]]/(df[CANON["radius"]]**2)
    df=df.replace([np.inf,-np.inf], np.nan).dropna(subset=FEATURES)
    df=df[(df[CANON["mass"]]>0)&(df[CANON["radius"]]>0)&(df[CANON["sma"]]>0)]
    return df

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    cond=(df[CANON["mass"]]<10)&(df[CANON["radius"]]<2)&(df["density_rel_earth"]>5)
    df["pl_type"]=np.where(cond,"Tellurique","Gazeuse")
    return df

def make_pipeline()->Pipeline:
    rf=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced")
    return Pipeline([("clf", rf)])

def train_model(df: pd.DataFrame)->Pipeline:
    X, y = df[FEATURES], df["pl_type"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE,stratify=y)
    pipe=make_pipeline()
    grid={"clf__n_estimators":[300,600], "clf__max_depth":[None,10,20], "clf__min_samples_leaf":[1,2,4]}
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs=GridSearchCV(pipe, grid, cv=cv, n_jobs=1, scoring="f1_macro", verbose=1)
    gs.fit(Xtr,ytr)
    print("Best params:", gs.best_params_)
    ypred=gs.predict(Xte)
    print(classification_report(yte, ypred))
    print("Confusion matrix:\n", confusion_matrix(yte, ypred))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(gs.best_estimator_, MODEL_PATH)
    return gs.best_estimator_

def predict_and_save_results(model:Pipeline, df:pd.DataFrame)->None:
    preds=model.predict(df[FEATURES])
    out=pd.DataFrame({"nom":df[CANON["name"]], "type_predit":preds,
                      "masse":df[CANON["mass"]], "rayon":df[CANON["radius"]],
                      "densite":df["density_rel_earth"], "distance_etoile":df[CANON["sma"]]})
    with sqlite3.connect(DB_PATH) as conn:
        cur=conn.cursor()
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE_DST} (
            nom TEXT, type_predit TEXT, masse REAL, rayon REAL, densite REAL, distance_etoile REAL)""")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_DST}_nom ON {TABLE_DST}(nom)")
        cur.executemany(
            f"INSERT INTO {TABLE_DST} (nom,type_predit,masse,rayon,densite,distance_etoile) VALUES (?,?,?,?,?,?)",
            out.itertuples(index=False, name=None)
        )
        conn.commit()

if __name__=="__main__":
    df=load_data_from_db()
    df=harmonize_columns(df)
    df=build_labels(df)
    model=train_model(df)
    predict_and_save_results(model, df)
