# Architecture

```text
 SATELLITE                       WEATHER                          SOIL
 16-day NDVI composites          daily rain, Tmax/Tmin,           pH, N, OC, sand/silt/clay
 (simulator.py)                  solar, RH                        (simulator.py, yearly samples)
        |                        (nasa_power.py REAL  |  simulator.py synthetic)        |
        |                                  |                                            |
        v                                  v                                            |
 ndvi_phase_means()              weather_features()     <-- season.py: crop-calendar   |
 early/mid/peak/late             rain, flowering rain,      windows (wheat Nov-Apr,     |
        |                        dry spell, GDD, heat days  rice Jun-Oct)               |
        +----------------------------------+--------------------------------------------+
                                           |
                      build_dataset.py: 1 row per district x crop x season (440 rows)
                                           |
                      features.py: NDVI slope / senescence / AUC, temp range,
                                   rainfall x soil-N, crop flag  -> 26 features
                                           |
              +----------------------------+-----------------------------+
              |                                                          |
        EARLY FUSION                                               LATE FUSION
   one table -> Ridge / RF / XGBoost /                  LightGBM per modality -> out-of-fold
   LightGBM / MLP (models.py)                           predictions -> non-negative linear
              |                                          meta-learner (fusion.py)
              +----------------------------+-----------------------------+
                                           |
                 validation.py: leave-one-year-out (11 folds), leave-districts-out (8 folds),
                                modality ablation (7 combinations), RMSE / MAE / R2 / MAPE
                                           |
                 train.py: deploy best explainable model (LightGBM), SHAP importance,
                           80% intervals from LOYO residuals, district presets, figures
                                           |
                           models/model_bundle.joblib + reports/*.csv/json/png
                                           |
                 api/main.py (FastAPI)  <---  cropyield/predict.py (predict + TreeSHAP)
                     /api/predict  /api/metrics  /api/meta  /api/dataset  /api/district ...
                                           |
                 api/static  <- React dashboard (web/): Overview, Explore, Results, Predict, Method
```

## Design decisions
| Decision | Reason |
|---|---|
| Same aggregation code for synthetic and real weather | swapping in NASA POWER changes data, not the pipeline |
| Pipelines (imputer/scaler inside) | preprocessing is learned per fold, so there is no leakage from the test year |
| LOYO as the primary metric | matches operational forecasting of an unseen season |
| Deploy the best *explainable* model | every forecast in the UI must come with an exact breakdown |
| Built-in TreeSHAP (`pred_contrib`) | exact, fast, and no extra heavy dependency |
| Empirical intervals from LOYO residuals | honest uncertainty with no distributional assumption |
| Prebuilt UI served by FastAPI | one process and one port; no Node.js needed to run |
