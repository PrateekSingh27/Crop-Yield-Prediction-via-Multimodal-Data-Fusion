"""Build the fused district x crop x season dataset.

    python -m cropyield.build_dataset                  # fully synthetic (offline)
    python -m cropyield.build_dataset --weather nasa   # REAL NASA POWER weather

Outputs:
    data/crop_yield_multimodal.csv  one row per district-crop-season (model input)
    data/daily_weather.csv          the daily weather each row was aggregated from
    data/ndvi_composites.csv        16-day NDVI composites per season
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import (DAILY_WEATHER_PATH, DATA_DIR, DATASET_PATH, DISTRICTS,
                     NDVI_SERIES_PATH, RANDOM_SEED, YEARS)
from .season import slice_season
from .simulator import make_district_traits, make_year_noise, simulate_record, synthetic_daily_weather


def build(weather: str = "synthetic", seed: int = RANDOM_SEED, save: bool = True) -> pd.DataFrame:
    """Build the dataset; with save=False nothing is written to disk (used by tests)."""
    if weather not in {"synthetic", "nasa"}:
        raise ValueError("weather must be 'synthetic' or 'nasa'")
    rng = np.random.default_rng(seed)
    traits = make_district_traits(rng)
    year_noise = make_year_noise(rng)

    real_weather = {}
    if weather == "nasa":
        from .nasa_power import load_district
        real_weather = {d.name: load_district(d.name) for d in DISTRICTS}

    records, ndvi_frames, daily_frames = [], [], []
    for year in YEARS:
        for district in DISTRICTS:
            for crop in district.crops:
                if weather == "nasa":
                    daily = slice_season(real_weather[district.name], crop, year)
                    source = "NASA POWER (real)"
                else:
                    daily = synthetic_daily_weather(district, crop, year, traits[district.name],
                                                    year_noise[year], rng)
                    source = "synthetic"
                record, series = simulate_record(district, crop, year, daily, traits[district.name],
                                                 year_noise[year], rng, source)
                records.append(record)
                ndvi_frames.append(series)
                daily_frames.append(daily.assign(district=district.name, crop=crop, year=year))

    dataset = pd.DataFrame(records)
    if not save:
        return dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.round(4).to_csv(DATASET_PATH, index=False)
    pd.concat(ndvi_frames).round(4).to_csv(NDVI_SERIES_PATH, index=False)
    daily_cols = ["district", "crop", "year", "date", "rainfall_mm", "tmax_c", "tmin_c",
                  "solar_radiation_mj", "rh_pct"]
    pd.concat(daily_frames)[daily_cols].round(2).to_csv(DAILY_WEATHER_PATH, index=False)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weather", choices=["synthetic", "nasa"], default="synthetic")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    df = build(args.weather, args.seed)
    print(f"Built {len(df)} district-crop-season rows "
          f"({df.district.nunique()} districts, {df.state.nunique()} states, "
          f"{df.year.min()}-{df.year.max()}, weather={args.weather}) -> {DATASET_PATH}")
    print(df.groupby("crop")["yield_tonnes_ha"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
