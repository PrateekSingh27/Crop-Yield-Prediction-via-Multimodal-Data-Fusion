# Viva guide: likely questions and answers

## The 60-second pitch
"We predict district-level wheat and rice yield by fusing satellite NDVI, weather and soil data. Each season is aligned to
crop stages, turned into 26 features, and modelled with gradient boosting. We test it the way it would be used: train on ten
seasons, forecast the eleventh. LightGBM gets 0.37 t/ha RMSE, about 10 % error. Any single data source alone is clearly worse
(R² 0.55–0.70 vs 0.82 fused). Every prediction comes with a SHAP breakdown showing how much satellite, weather and soil pushed it."

## Demo order (3–4 minutes)
1. **Overview:** point at the 2022 wheat row in the NDVI stripes: the crop browns early (March heatwave) and its yield bar is shortest.
2. **Model results:** LightGBM is deployed, early fusion beats late fusion, ablation shows fusion > any single source.
3. **Predict:** Karnal wheat, then click "Heatwave at grain filling": the yield drops and heat-stress days turn red in "Why this number".
4. `/docs`: show the API is a real backend with validation (try soil pH = 14 → 422 error).

## Questions

**Q: Is the data real?**
Weather can be real: `python -m cropyield.nasa_power` downloads NASA POWER daily data for all 32 districts. NDVI, soil and yield
are simulated with crop-science response curves, because official district yields, satellite rasters and Soil Health Card data
need accounts and heavy GIS work. The simulator is non-linear and has a hidden management effect, so the scores are realistic,
not perfect. We state this openly in the dashboard and report.

**Q: Why not just use a formula to generate data?**
The original version did, and Linear Regression won, which contradicts the proposal. A realistic simulator has thresholds (heat),
saturation (nitrogen) and optima (pH), so non-linear models win for the right reason.

**Q: What is NDVI?**
(NIR − Red) / (NIR + Red). Healthy green leaves reflect near-infrared strongly and absorb red, so higher NDVI means a denser,
healthier canopy. Range −1 to 1; crops are typically 0.2 (bare) to 0.85 (dense).

**Q: Why leave-one-year-out instead of a random 80/20 split?**
All districts in a season share the same monsoon and heat. A random split puts the same season in train and test, so the model
"knows" that year's weather and the score is inflated. LOYO forecasts a season the model has never seen, exactly like real use.
(The notebook shows both: random 5-fold looks slightly better, RMSE 0.369 vs 0.372. The gap is small on our data because simulated
season-wide shocks are moderate; with real droughts that hit every district at once it is much larger.)

**Q: Early vs late fusion: which is better, and why?**
Early fusion (0.372 RMSE) beat late fusion (0.412). Late fusion trains one model per modality, so it cannot learn interactions
such as "heat matters more when the canopy is already weak". Late fusion is useful when a modality is often missing.

**Q: Why LightGBM?**
It is the most accurate under LOYO, handles mixed-scale tabular features and missing values, trains in seconds, and gives exact
SHAP values. XGBoost is nearly identical (0.374); Random Forest is close; Ridge is 4 % worse; the MLP is worst because 440 rows are too few for a neural net.

**Q: What does the ablation show?**
Soil alone R² 0.55, weather 0.64, satellite 0.70, any pair 0.77–0.79, all three 0.82. Each source adds information the others lack: the core claim of the project.

**Q: What is SHAP?**
A game-theory method that splits a prediction into additive contributions: prediction = base value + Σ contributions. Our API
returns them for every forecast (and a test checks they add up exactly).

**Q: Which features matter most?**
Seasonal NDVI integral, heat-stress days, mid-season NDVI, GDD, sand, organic carbon. By modality: satellite 43 %, weather 34 %, soil 20 %.

**Q: Why is rice harder to predict than wheat?**
Monsoon clouds contaminate optical satellite images during kharif, so NDVI is noisier (rice R² 0.72 vs wheat 0.88).
Radar (Sentinel-1 SAR) would help because it sees through clouds.

**Q: What happens for a district the model has never seen?**
We tested this with leave-districts-out: RMSE rises to about 0.41 and LightGBM loses its edge over Ridge, because trees partly
memorise district soil signatures. More districts would fix this.

**Q: How do you get the prediction interval?**
From out-of-sample LOYO errors: the 10th and 90th percentile residuals per crop, added to the prediction (80 % interval).

**Q: What are heat-stress days and GDD?**
Heat-stress days: days with Tmax above 33 °C (wheat) / 35 °C (rice) between flowering and grain filling, when heat kills
pollen and shortens grain fill. GDD: accumulated warmth above a base temperature, a proxy for crop development.

**Q: How would you deploy this for real?**
Swap simulated labels for data.gov.in / ICRISAT district yields, pull Sentinel-2 NDVI through Google Earth Engine and SoilGrids
rasters, keep the same pipeline, retrain every season, and serve forecasts mid-season (e.g. at peak NDVI) via the API.

**Q: What is the tech stack?**
Python, pandas, scikit-learn, LightGBM, XGBoost, FastAPI + Pydantic, React + Vite + Tailwind + Recharts, pytest.

**Q: Limitations?**
Simulated labels, district-level aggregation, fixed sowing dates, SHAP is associational not causal, and 440 rows is too few for deep fusion.
