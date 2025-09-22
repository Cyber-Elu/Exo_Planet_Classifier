# src/web.py — recherche, filtres et ajout de planète

import os
import time
import sqlite3
from typing import Optional

from flask import Flask, request, jsonify, render_template, redirect, url_for
from pydantic import BaseModel, ValidationError, field_validator
from jinja2 import FileSystemLoader, ChoiceLoader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import joblib

from src.config import CFG

# --- Répertoires (absolus) ----------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")  # facultatif

# --- Application Flask ---------------------------------------------------------
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# Loader Jinja + auto-reload en dev
app.jinja_loader = ChoiceLoader([FileSystemLoader(TEMPLATES_DIR, followlinks=True)])
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.cache = {}

# JSON + garde-fous
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 Mo

print("Jinja search path:", getattr(app.jinja_loader, "searchpath", None))
print("Templates dir exists?", os.path.isdir(TEMPLATES_DIR), "->", TEMPLATES_DIR)

# --- Prometheus metrics --------------------------------------------------------
PRED_COUNT = Counter("pred_requests_total", "Total prediction requests")
PRED_LATENCY = Histogram("pred_latency_seconds", "Prediction latency")

# --- Modèle -------------------------------------------------------------------
MODEL_PATH = CFG.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Modèle introuvable: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# --- Schémas d'entrée ----------------------------------------------------------
class Features(BaseModel):
    pl_rade: float
    pl_bmasse: float
    pl_orbsmax: float
    density_rel_earth: Optional[float] = None
    g_rel_earth: Optional[float] = None

    @field_validator("pl_rade", "pl_bmasse", "pl_orbsmax")
    @classmethod
    def positive(cls, v):
        if v is None or v <= 0:
            raise ValueError("Valeur physique attendue strictement > 0")
        return float(v)

class NewPlanet(BaseModel):
    pl_name: str
    pl_rade: float
    pl_bmasse: float
    pl_orbsmax: float

    @field_validator("pl_name")
    @classmethod
    def name_ok(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Le nom ne peut pas être vide")
        if len(v) > 100:
            raise ValueError("Nom trop long (max 100)")
        return v

    @field_validator("pl_rade", "pl_bmasse", "pl_orbsmax")
    @classmethod
    def positive(cls, v):
        if v is None or v <= 0:
            raise ValueError("Valeur physique attendue strictement > 0")
        return float(v)

# --- Accès DB -----------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(CFG.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_tables():
    """Crée les tables attendues si elles n'existent pas (robustesse)."""
    with get_db_connection() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS exoplanetes (
              pl_name TEXT,
              pl_rade REAL NOT NULL,
              pl_bmasse REAL NOT NULL,
              pl_orbsmax REAL NOT NULL,
              density_rel_earth REAL,
              g_rel_earth REAL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_exo_plname ON exoplanetes(pl_name)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS planet_classifications (
              nom TEXT,
              type_predit TEXT,
              masse REAL,
              rayon REAL,
              densite REAL,
              distance_etoile REAL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_planet_cls_nom ON planet_classifications(nom)")
        c.commit()

ensure_tables()

# --- Utilitaires --------------------------------------------------------------
def derive_features(mass, radius):
    """Retourne (density_rel_earth, g_rel_earth)"""
    density = mass / (radius ** 3)
    g_rel = mass / (radius ** 2)
    return density, g_rel

def predict_label(mass, radius, sma):
    density, g_rel = derive_features(mass, radius)
    X = [[mass, radius, density, g_rel, sma]]  # respecter l'ordre d'entraînement
    y = model.predict(X)[0]
    proba_fn = getattr(model, "predict_proba", None)
    proba = proba_fn(X)[0].tolist() if callable(proba_fn) else None
    return y, proba, density, g_rel

# --- Endpoints techniques -----------------------------------------------------
@app.get("/healthz")
def healthz():
    try:
        with get_db_connection() as c:
            c.execute("SELECT 1")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.get("/debug/paths")
def debug_paths():
    import glob
    return {
        "cwd": os.getcwd(),
        "BASE_DIR": BASE_DIR,
        "TEMPLATES_DIR": TEMPLATES_DIR,
        "templates_present": [
            os.path.basename(p) for p in glob.glob(os.path.join(TEMPLATES_DIR, "*.html"))
        ],
    }

# --- Endpoints fonctionnels (UI) ---------------------------------------------
@app.get("/")
def index():
    """
    Page liste avec :
      - recherche par nom : param ?q=...
      - filtre par type :  param ?type=Tellurique|Gazeuse
    """
    q = (request.args.get("q") or "").strip()
    type_filter = (request.args.get("type") or "").strip()

    sql = "SELECT * FROM planet_classifications"
    params = []
    clauses = []

    if q:
        clauses.append("LOWER(nom) LIKE ?")
        params.append(f"%{q.lower()}%")

    if type_filter in ("Tellurique", "Gazeuse"):
        clauses.append("type_predit = ?")
        params.append(type_filter)

    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    sql += " ORDER BY nom COLLATE NOCASE"

    with get_db_connection() as conn:
        planets = conn.execute(sql, params).fetchall()

    return render_template("index.html", planets=planets, q=q, type_filter=type_filter)


@app.get("/planet/<planet_name>")
def planet(planet_name: str):
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM planet_classifications WHERE nom = ?",
            (planet_name,),
        ).fetchone()
    if row is None:
        return "Planète non trouvée", 404
    return render_template("planet.html", planet=row)


@app.get("/add")
def add_form():
    """Formulaire d'ajout d'une planète."""
    return render_template("add.html", error=None, form={"pl_name":"", "pl_rade":"", "pl_bmasse":"", "pl_orbsmax":""})


@app.post("/add")
def add_submit():
    """
    Traite le formulaire : valide, prédit, insère dans exoplanetes + planet_classifications,
    puis redirige vers la fiche /planet/<nom>.
    """
    data = {
        "pl_name": request.form.get("pl_name", "").strip(),
        "pl_rade": request.form.get("pl_rade", "").strip(),
        "pl_bmasse": request.form.get("pl_bmasse", "").strip(),
        "pl_orbsmax": request.form.get("pl_orbsmax", "").strip(),
    }

    # Validation simple via Pydantic
    try:
        parsed = NewPlanet(
            pl_name=data["pl_name"],
            pl_rade=float(data["pl_rade"]),
            pl_bmasse=float(data["pl_bmasse"]),
            pl_orbsmax=float(data["pl_orbsmax"]),
        )
    except Exception as e:
        # On réaffiche le formulaire avec le message d'erreur
        return render_template("add.html", error=str(e), form=data), 400

    # Prédiction
    label, proba, density, g_rel = predict_label(parsed.pl_bmasse, parsed.pl_rade, parsed.pl_orbsmax)

    # Insertion DB (UPSERT simple via delete+insert pour rester compatible SQLite)
    with get_db_connection() as conn:
        cur = conn.cursor()
        # On garde aussi une trace dans exoplanetes (avec features dérivées) pour cohérence
        cur.execute("DELETE FROM exoplanetes WHERE pl_name = ?", (parsed.pl_name,))
        cur.execute(
            """INSERT INTO exoplanetes
               (pl_name, pl_rade, pl_bmasse, pl_orbsmax, density_rel_earth, g_rel_earth)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (parsed.pl_name, parsed.pl_rade, parsed.pl_bmasse, parsed.pl_orbsmax, density, g_rel),
        )
        cur.execute("DELETE FROM planet_classifications WHERE nom = ?", (parsed.pl_name,))
        cur.execute(
            """INSERT INTO planet_classifications
               (nom, type_predit, masse, rayon, densite, distance_etoile)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (parsed.pl_name, label, parsed.pl_bmasse, parsed.pl_rade, density, parsed.pl_orbsmax),
        )
        conn.commit()

    # Redirection vers la fiche planète
    return redirect(url_for("planet", planet_name=parsed.pl_name))


# --- Endpoint JSON /predict déjà existant -------------------------------------
@app.post("/predict")
def predict():
    t0 = time.time()
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON"}), 400
    try:
        feat = Features(**data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400

    density = feat.density_rel_earth or (feat.pl_bmasse / (feat.pl_rade ** 3))
    g_rel = feat.g_rel_earth or (feat.pl_bmasse / (feat.pl_rade ** 2))
    X = [[feat.pl_bmasse, feat.pl_rade, density, g_rel, feat.pl_orbsmax]]

    y = model.predict(X)[0]
    proba_fn = getattr(model, "predict_proba", None)
    proba = proba_fn(X)[0].tolist() if callable(proba_fn) else None

    PRED_COUNT.inc()
    PRED_LATENCY.observe(time.time() - t0)
    return jsonify({"label": str(y), "proba": proba}), 200


# --- Lancement ----------------------------------------------------------------
if __name__ == "__main__":
    # Lancer depuis la racine:  python -m src.web
    app.run(host=CFG.API_HOST, port=CFG.API_PORT, debug=True)
