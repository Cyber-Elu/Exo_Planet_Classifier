from flask import Flask, request, jsonify, render_template
import sqlite3, joblib, os, time
from typing import Optional
from pydantic import BaseModel, ValidationError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.config import CFG

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

# Prometheus metrics
PRED_COUNT = Counter("pred_requests_total", "Total prediction requests")
PRED_LATENCY = Histogram("pred_latency_seconds", "Prediction latency")

# Load model
MODEL_PATH = CFG.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Modèle introuvable: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

class Features(BaseModel):
    pl_rade: float
    pl_bmasse: float
    pl_orbsmax: float
    density_rel_earth: Optional[float] = None
    g_rel_earth: Optional[float] = None

def get_db_connection():
    conn = sqlite3.connect(CFG.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/healthz")
def healthz():
    try:
        with get_db_connection() as c:
            c.execute("SELECT 1")
        return jsonify({"status":"ok"}), 200
    except Exception as e:
        return jsonify({"status":"error","detail":str(e)}), 500

@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.post("/predict")
def predict():
    t0=time.time()
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error":"Invalid or missing JSON"}), 400
    try:
        feat = Features(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    if feat.pl_rade <= 0 or feat.pl_bmasse <= 0 or feat.pl_orbsmax <= 0:
        return jsonify({"error": "All physical features must be positive"}), 400

    density = feat.density_rel_earth or (feat.pl_bmasse / (feat.pl_rade ** 3))
    g_rel   = feat.g_rel_earth      or (feat.pl_bmasse / (feat.pl_rade ** 2))
    X = [[feat.pl_bmasse, feat.pl_rade, density, g_rel, feat.pl_orbsmax]]  # même ordre que FEATURES

    y = model.predict(X)[0]
    proba_fn = getattr(model, "predict_proba", None)
    proba = proba_fn(X)[0].tolist() if callable(proba_fn) else None

    PRED_COUNT.inc()
    PRED_LATENCY.observe(time.time()-t0)
    return jsonify({"label": str(y), "proba": proba}), 200

@app.get("/")
def index():
    with get_db_connection() as conn:
        planets = conn.execute("SELECT * FROM planet_classifications").fetchall()
    return render_template("index.html", planets=planets)

@app.get("/planet/<planet_name>")
def planet(planet_name: str):
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM planet_classifications WHERE nom = ?", (planet_name,)).fetchone()
    if row is None:
        return "Planète non trouvée", 404
    return render_template("planet.html", planet=row)

if __name__ == "__main__":
    app.run(host=CFG.API_HOST, port=CFG.API_PORT, debug=True)
