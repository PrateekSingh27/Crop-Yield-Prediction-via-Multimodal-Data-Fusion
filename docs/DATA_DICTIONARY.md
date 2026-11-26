# Data dictionary

`data/crop_yield_multimodal.csv`: one row per district × crop × season (440 rows).
Season year = harvest year (wheat 2020 = sown Nov 2019, harvested Apr 2020).

## Identifiers
| Column | Description |
|---|---|
| district, state | administrative unit |
| crop | Wheat (rabi) or Rice (kharif) |
| year | season (harvest) year, 2014–2024 |
| lat, lon | approximate district centroid used for NASA POWER |
| weather_source | `synthetic` or `NASA POWER (real)` |

## Satellite (NDVI, 0–1)
| Column | Definition |
|---|---|
| ndvi_early | mean of 16-day composites, 0–40 days after sowing (DAS) |
| ndvi_mid | mean, 40–80 DAS |
| ndvi_peak | mean, 80–110 DAS |
| ndvi_late | mean, 110–150 DAS |
| ndvi_growth_slope* | (peak − early) / 2: green-up rate |
| ndvi_senescence* | (peak − late) × 100: canopy decline after peak |
| ndvi_auc* | trapezoidal integral of the four phase means: season-long greenness |

## Weather (crop season)
| Column | Unit | Definition |
|---|---|---|
| rainfall_mm | mm | total season rainfall |
| rainfall_cv | ratio | coefficient of variation of weekly rainfall |
| flowering_rainfall_mm | mm | rain during flowering window (wheat 95–115, rice 70–92 DAS) |
| dry_spell_days | days | longest run of days with < 1 mm rain |
| tmax_mean_c, tmin_mean_c | °C | mean daily max / min temperature |
| temp_range_c* | °C | tmax − tmin |
| solar_radiation_mj | MJ/m²/day | mean daily shortwave radiation |
| rh_mean_pct | % | mean relative humidity |
| gdd | °C·day | Σ (clip(Tmean, base, cap) − base); base/cap 5/30 °C wheat, 10/35 °C rice |
| heat_stress_days | days | days with Tmax > 33 °C (wheat) / 35 °C (rice) from flowering to end of grain fill |

## Soil
| Column | Unit |
|---|---|
| soil_ph | pH |
| soil_n_kg_ha | kg/ha available nitrogen |
| soil_oc_pct | % organic carbon |
| sand_pct, silt_pct, clay_pct | % (sum = 100) |

## Derived / context
| Column | Definition |
|---|---|
| rainfall_x_soil_n* | rainfall_mm × soil_n_kg_ha (cross-modal interaction) |
| crop_is_rice* | 1 for rice, 0 for wheat |

## Target
| Column | Unit |
|---|---|
| yield_tonnes_ha | t/ha (grain; rice expressed as milled-equivalent) |

\* created in `cropyield/features.py`, not stored in the CSV.

## Other files
| File | Content |
|---|---|
| data/daily_weather.csv | daily weather each season row was aggregated from |
| data/ndvi_composites.csv | 16-day NDVI composites (district, crop, year, days_after_sowing, ndvi) |
| data/raw/nasa_power/*.csv | cached NASA POWER downloads (only after running the downloader) |
