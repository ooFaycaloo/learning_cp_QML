# Hackathon Finance – Gold (XAU) – Starter Kit

Ce repo est une base **pragmatique** pour:
- charger le dataset fourni (`dataset_train.xlsx`, feuille **Gold**),
- faire du feature engineering (rendements, volatilité, momentum, indicateurs),
- définir la cible à **20 jours** avec un **score ∈ [-1, +1]**,
- entraîner une baseline ML robuste + validation time-series,
- intégrer une brique QML via **MerLin** (option hybride).

## Structure
- `notebooks/` : exécution pas à pas (EDA → features → modèles → MerLin → résultats)
- `src/` : fonctions réutilisables (chargement, features, splits, modèles, métriques)
- `outputs/` : artefacts (csv prédictions, métriques)
- `logs/` : logs d’exécution

## Démarrage
```bash
pip install -r requirements.txt
jupyter lab
```

## Notes importantes (Finance)
- On travaille sur **les rendements**, pas les prix bruts.
- Aucune fuite d’info: à la date *t*, features calculées avec des infos ≤ *t*.
- Validation **out-of-sample** via split temporel (walk-forward / TimeSeriesSplit).
