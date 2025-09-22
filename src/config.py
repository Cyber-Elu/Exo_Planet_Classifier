import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/exoplanet_clf.joblib")
    DB_PATH: str = os.getenv("DB_PATH", "exoplanetes.db")
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "5000"))
    RANDOM_STATE: int = 42

CFG = Settings()
