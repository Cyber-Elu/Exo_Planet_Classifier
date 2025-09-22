DROP TABLE IF EXISTS exoplanetes;
CREATE TABLE exoplanetes (
  pl_name TEXT,
  pl_rade REAL NOT NULL,
  pl_bmasse REAL NOT NULL,
  pl_orbsmax REAL NOT NULL,
  density_rel_earth REAL,
  g_rel_earth REAL
);
CREATE INDEX IF NOT EXISTS idx_exo_plname ON exoplanetes(pl_name);

CREATE TABLE IF NOT EXISTS planet_classifications (
  nom TEXT,
  type_predit TEXT,
  masse REAL,
  rayon REAL,
  densite REAL,
  distance_etoile REAL
);
CREATE INDEX IF NOT EXISTS idx_planet_cls_nom ON planet_classifications(nom);
