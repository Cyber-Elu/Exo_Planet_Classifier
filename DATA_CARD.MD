# Data Card – NASA Exoplanet Archive (ps)

- **Source**: https://exoplanetarchive.ipac.caltech.edu
- **Colonnes utilisées**: pl_name, pl_rade (R⊕), pl_bmasse (M⊕), pl_orbsmax (UA)
- **Nettoyage**: suppression NA/<=0, coercition numérique, dérivées density_rel_earth & g_rel_earth
- **Licence / usage**: données publiques; citer la source
- **Rafraîchissement**: script `recup_data.py` — extraction à la demande
- **Risques/biais**: couverture d’observation non uniforme; incertitudes de mesure; detection bias
