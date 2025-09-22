# Model Card – Exo Planet Classifier

- **Modèle**: RandomForestClassifier (sklearn)
- **Version**: v1 (YYYY-MM-DD, commit SHA)
- **Features**: pl_bmasse, pl_rade, density_rel_earth, g_rel_earth, pl_orbsmax
- **Cible**: pl_type (Tellurique/Gazeuse) — heuristique de labellisation (masse<10, rayon<2, densité>5)
- **Métriques**: F1 macro (CV 5 folds), rapport par classe, matrice de confusion
- **Données d'entraînement**: NASA Exoplanet Archive (table `ps`) — champs sélectionnés
- **Limites connues**: Labellisation heuristique (non supervisée “vérité terrain”), biais possibles selon détection
- **Utilisation prévue**: tri rapide des planètes candidates; non destiné à de la recherche scientifique sans validation
- **Sécurité**: Entrées bornées >0; validation Pydantic; taille JSON limitée
