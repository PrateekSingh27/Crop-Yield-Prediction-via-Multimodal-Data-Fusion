# Crop Yield Prediction via Multimodal Data Fusion: Project Report

**Team:** Snehaal Mukherjee (1/23/SET/BCS/350), Krishna Bhatia (1/23/SET/BCS/358), Prateek Singh (1/23/SET/BCS/362), Aryaman (1/23/SET/BCS/364)
**Programme:** B.Tech CSE (AI & ML), Manav Rachna International Institute of Research and Studies, Faridabad

---

## Abstract
We predict district-level wheat and rice yield in India by fusing three modalities: satellite vegetation indices (NDVI),
seasonal weather, and soil properties. Daily and composite data are aligned to crop-stage windows and engineered into 26
features, then combined by early fusion (one joint model) and late fusion (stacked per-modality models). Five learners are
compared under leave-one-year-out (LOYO) validation, which mirrors forecasting an unseen season. Early-fusion LightGBM
performs best (RMSE 0.372 t/ha, MAPE 9.7 %, R² 0.822). An ablation study shows every single modality is weaker on its own
(R² 0.55–0.70) than the fused model, and SHAP analysis attributes 43 % of the explanation to satellite, 34 % to weather and
20 % to soil features. The system is delivered as a FastAPI service with a React dashboard that returns a forecast, an 80 %
interval and a per-feature explanation.

## 1. Introduction and motivation
Accurate pre-harvest yield estimates support food-security planning, procurement, crop insurance and targeted extension
services. Single-source models leave blind spots:
* **Satellite-only** models see canopy condition but not sub-surface constraints or the cause of stress.
* **Weather-only** models miss spatial differences in management and soil.
* **Soil-only** models miss the season-to-season variation driven by rainfall and heat.

Fusing the three reduces these blind spots and makes predictions more robust.

## 2. Objectives
1. Build an aligned district × crop × season dataset with satellite, weather and soil modalities.
2. Engineer phenology, stress and cross-modal interaction features.
3. Compare early and late fusion strategies.
4. Compare Ridge, Random Forest, XGBoost, LightGBM and a neural network.
5. Evaluate with leave-one-year-out validation using RMSE, MAE, R² and MAPE; add a spatial (leave-districts-out) check.
6. Quantify each modality's value through ablation and SHAP.
7. Deploy the model behind an API with an interactive, explainable dashboard.

## 3. Data
### 3.1 Study design
* 32 districts in 11 states (Punjab, Haryana, Rajasthan, Uttar Pradesh, Madhya Pradesh, Maharashtra, Andhra Pradesh,
  Karnataka, Tamil Nadu, Bihar, Chhattisgarh).
* Crops: wheat (rabi, sown ~10 Nov, 150-day season) and rice (kharif, transplanted ~20 Jun, 125-day season).
* Seasons 2014–2024 → 40 district-crop pairs × 11 seasons = **440 rows**.

### 3.2 Sources
| Modality | Production source | This build |
|---|---|---|
| Satellite | Sentinel-2 / MODIS NDVI (Google Earth Engine) | simulated 16-day composites |
| Weather | NASA POWER daily API | **real NASA POWER** (one command) or synthetic generator |
| Soil | SoilGrids / Soil Health Card | simulated district profiles + yearly sampling noise |
| Yield | data.gov.in / ICRISAT district statistics | simulated with crop-response model |

### 3.3 Why simulation, and how it is made realistic
Official district yields, raster processing and soil cards require accounts and extensive GIS work beyond a semester
project. Instead of a simple formula (which would make a linear model look best and teach nothing), the simulator encodes
established agronomic behaviour:

* **Daily weather:** seasonal temperature cycle by state, monsoon rainfall with wet-day probability and gamma amounts,
  year-to-year monsoon strength following approximate IMD all-India departures (e.g. 2014 −12 %, 2015 −14 %, 2019 +10 %), and
  a 2022 spring temperature anomaly representing the March heatwave that reduced North-Indian wheat.
* **Water response:** irrigation fills part of the deficit; soil texture changes water holding; deficits hurt sharply,
  floods mildly.
* **Heat response:** exponential penalty per day above 33 °C (wheat) / 35 °C (rice) during flowering and grain filling.
* **Nitrogen:** saturating Mitscherlich curve. **pH:** quadratic optimum near 6.8. **Organic carbon, texture, radiation:**
  mild effects.
* **Hidden management:** an irrigation/variety/practice index per district that the model never sees, the main source of
  irreducible error in real yield data.
* **NDVI:** a double-logistic canopy curve whose height follows vigour (management × water × nitrogen) and whose decline
  starts early under heat and drought; monsoon clouds contaminate rice composites.

Because daily weather passes through the same aggregation code (`season.py`) whether it is synthetic or real, switching to
NASA POWER requires no change to the model pipeline.

## 4. Feature engineering
| Group | Features |
|---|---|
| Satellite | phase-mean NDVI (early 0–40, mid 40–80, peak 80–110, late 110–150 days after sowing), green-up slope, senescence drop, seasonal NDVI integral (AUC) |
| Weather | seasonal rainfall, weekly-rainfall CV, flowering rainfall, longest dry spell, mean Tmax/Tmin, diurnal range, solar radiation, humidity, GDD (base 5 °C wheat / 10 °C rice), heat-stress days |
| Soil | pH, available N, organic carbon, sand, silt, clay |
| Cross-modal | rainfall × soil N |
| Context | crop indicator |

Full definitions and units: `docs/DATA_DICTIONARY.md`.

## 5. Fusion strategies
* **Early fusion:** all features concatenated into one row per district-season; a single model learns interactions
  across modalities (e.g. high heat matters more when the canopy is already stressed).
* **Late fusion:** one LightGBM per modality; a non-negative linear meta-learner combines their predictions. The meta-learner is
  trained on out-of-fold base predictions (inner GroupKFold by year) to prevent leakage.
* **Intermediate / deep fusion** (CNN encoder for imagery + LSTM for daily weather) is left as future work: it needs far
  more labelled seasons than 440.

## 6. Models
All models are scikit-learn pipelines (median imputation; scaling where needed) fitted inside each validation fold.

| Model | Configuration |
|---|---|
| Ridge Regression | standardised inputs, α chosen by internal CV (RidgeCV) |
| Random Forest | 400 trees, min leaf 2, 50 % features per split |
| XGBoost | 600 trees, depth 4, learning rate 0.03, row/column subsampling, L2 = 2 |
| LightGBM | 600 trees, 15 leaves, learning rate 0.03, min 10 samples/leaf, bagging + feature fraction |
| MLP | 64-32 ReLU, L2 = 0.01, early stopping |

## 7. Validation methodology
* **Leave-one-year-out (primary):** for each of the 11 seasons, train on the other 10 and predict the held-out season.
  Random row splits would put districts from the same season (sharing the same monsoon) in both train and test and inflate scores.
* **Leave-districts-out (secondary):** 8-fold GroupKFold over districts tests generalisation to unseen places.
* **Metrics:** RMSE, MAE (t/ha), R², MAPE (%).

## 8. Results
### 8.1 Model comparison (LOYO)
| Model | Fusion | RMSE | MAE | R² | MAPE % |
|---|---|---|---|---|---|
| **LightGBM** | Early | **0.372** | **0.290** | **0.822** | **9.7** |
| XGBoost | Early | 0.374 | 0.291 | 0.820 | 9.7 |
| Random Forest | Early | 0.378 | 0.291 | 0.817 | 9.9 |
| Ridge Regression | Early | 0.388 | 0.303 | 0.806 | 10.5 |
| Late fusion (stacked) | Late | 0.412 | 0.326 | 0.781 | 11.1 |
| MLP | Early | 0.419 | 0.332 | 0.774 | 11.4 |

Gradient boosting beats the linear baseline because the crop responds non-linearly (heat thresholds, saturating N).
Early fusion beats late fusion because separate per-modality models cannot learn cross-modal interactions. The MLP trails on
440 rows, consistent with the expectation that deep models are data-hungry.

### 8.2 Modality ablation (LightGBM, LOYO R²)
Soil 0.55, Weather 0.64, Satellite 0.70; Satellite+Weather 0.77, Weather+Soil 0.79, Satellite+Soil 0.79; **All three 0.82**.
Every added modality improves accuracy, which supports the central hypothesis of the project.

### 8.3 Explainability
Top SHAP features: seasonal NDVI integral, heat-stress days, mid-season NDVI, GDD, sand, organic carbon, peak NDVI, soil pH.
Modality share of mean |SHAP|: satellite 43 %, weather 34 %, soil 20 %, cross-modal 3 %. The late-fusion meta-learner weights
(satellite 0.52, weather 0.30, soil 0.32) independently agree that satellite carries the most signal.

### 8.4 Error analysis
* Per crop: wheat RMSE 0.34 (R² 0.88), rice RMSE 0.40 (R² 0.72). Rice is harder because monsoon clouds contaminate NDVI.
* Per season: highest error in 2022 (wheat heatwave), a year unlike the training seasons.
* Spatial check: on unseen districts RMSE rises to ~0.41–0.42 and LightGBM no longer beats Ridge. Part of what the trees
  learn is each district's fixed soil signature, so forecasting new regions needs more districts in training.

### 8.5 Uncertainty
80 % prediction intervals are taken from the 10th–90th percentile of LOYO residuals per crop (about −0.43/+0.41 t/ha for wheat,
−0.46/+0.54 t/ha for rice).

## 9. System implementation
* **Backend:** FastAPI (`api/main.py`) with Pydantic validation (ranges, Tmin < Tmax, texture sums to 100 %), Swagger docs at `/docs`.
* **Inference:** `cropyield/predict.py` fills missing inputs from district presets, predicts, and returns exact TreeSHAP
  contributions that sum to the prediction, grouped by modality, with range warnings for extrapolation.
* **Frontend:** React + Vite + Tailwind + Recharts, served by FastAPI from `api/static`: NDVI season stripes, data explorer with
  district NDVI curves, evaluation dashboard, live what-if predictor with scenarios (heatwave, rainfall failure, low N).
* **Testing:** 11 pytest tests: dataset integrity, reproducibility and no side effects on saved data, LOYO split integrity, aggregation maths, NASA parser
  with missing values, API additivity, monotonic heat response, input validation.

## 10. Limitations
1. Yield, NDVI and soil are simulated; scores describe the simulated world and demonstrate the methodology.
2. District aggregation hides field-level variation.
3. Fixed crop calendars; real sowing dates vary by year and district.
4. SHAP explains model associations, not causal effects.

## 11. Conclusion and future scope
Fusing satellite, weather and soil data gives clearly better forecasts than any single source, and an explainable gradient
boosting model under strict temporal validation reaches about 10 % error. Next steps: replace simulated labels with
data.gov.in / ICRISAT district yields, extract real Sentinel-2 NDVI with Google Earth Engine, use SoilGrids rasters, model
year-specific sowing dates, and try CNN-LSTM intermediate fusion once more labelled seasons are available.

## References
1. NASA POWER Project, Prediction Of Worldwide Energy Resources, https://power.larc.nasa.gov
2. Copernicus Sentinel-2 mission, https://dataspace.copernicus.eu
3. ISRIC SoilGrids, https://soilgrids.org
4. Open Government Data Platform India, https://data.gov.in
5. Ke et al. (2017), LightGBM: A Highly Efficient Gradient Boosting Decision Tree, NeurIPS.
6. Chen & Guestrin (2016), XGBoost: A Scalable Tree Boosting System, KDD.
7. Lundberg & Lee (2017), A Unified Approach to Interpreting Model Predictions (SHAP), NeurIPS.
