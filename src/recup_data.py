import sqlite3
from typing import Optional
import pandas as pd
import requests
from io import StringIO
from requests.adapters import HTTPAdapter, Retry

REQUIRED_COLS = ["pl_rade", "pl_bmasse", "pl_orbsmax"]
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TAP_QUERY = """
SELECT pl_name, pl_rade, pl_bmasse, pl_orbsmax
FROM ps
WHERE pl_rade IS NOT NULL AND pl_bmasse IS NOT NULL AND pl_orbsmax IS NOT NULL
"""

def fetch_data_with_tap(query: str = TAP_QUERY) -> pd.DataFrame:
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    r = sess.get(TAP_URL, params={"query": query, "format": "csv"}, timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), low_memory=False)

def load_raw(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLS)
    df = df[(df["pl_rade"]>0) & (df["pl_bmasse"]>0) & (df["pl_orbsmax"]>0)]
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["density_rel_earth"] = df["pl_bmasse"]/(df["pl_rade"]**3)
    df["g_rel_earth"] = df["pl_bmasse"]/(df["pl_rade"]**2)
    return df[["pl_name"]+REQUIRED_COLS+["density_rel_earth","g_rel_earth"]]

def create_database():
    with sqlite3.connect("exoplanetes.db") as conn:
        cur = conn.cursor()
        cur.execute('DROP TABLE IF EXISTS exoplanetes')
        cur.execute("""
            CREATE TABLE exoplanetes (
                pl_name TEXT,
                pl_rade REAL NOT NULL,
                pl_bmasse REAL NOT NULL,
                pl_orbsmax REAL NOT NULL,
                density_rel_earth REAL,
                g_rel_earth REAL
            )
        """)
        cur.execute('CREATE INDEX IF NOT EXISTS idx_exo_plname ON exoplanetes(pl_name)')
        conn.commit()

def insert_data_into_database(data: pd.DataFrame):
    data = data.where(pd.notna(data), None)
    with sqlite3.connect("exoplanetes.db") as conn:
        cur = conn.cursor()
        sql = """INSERT INTO exoplanetes
                 (pl_name, pl_rade, pl_bmasse, pl_orbsmax, density_rel_earth, g_rel_earth)
                 VALUES (?, ?, ?, ?, ?, ?)"""
        cur.executemany(sql, data[["pl_name","pl_rade","pl_bmasse","pl_orbsmax","density_rel_earth","g_rel_earth"]].itertuples(index=False, name=None))
        conn.commit()

if __name__ == "__main__":
    df = fetch_data_with_tap()
    df = load_raw(df)
    df = build_features(df)
    create_database()
    insert_data_into_database(df)
