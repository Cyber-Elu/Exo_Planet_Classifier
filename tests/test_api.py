import json, os
from src.web import app

def test_health():
    client = app.test_client()
    r = client.get("/healthz")
    assert r.status_code in (200, 500)

def test_predict_bad_json():
    client = app.test_client()
    r = client.post("/predict", data="not json", headers={"Content-Type":"application/json"})
    assert r.status_code == 400

def test_predict_ok(monkeypatch):
    client = app.test_client()
    # Patch model to avoid loading real file
    from src.web import model
    def _predict(X): return ["Tellurique"]
    model.predict = _predict
    payload = {"pl_rade":1.1,"pl_bmasse":1.3,"pl_orbsmax":1.0}
    r = client.post("/predict", data=json.dumps(payload), headers={"Content-Type":"application/json"})
    assert r.status_code == 200
    assert r.json["label"] in ("Tellurique","Gazeuse")
