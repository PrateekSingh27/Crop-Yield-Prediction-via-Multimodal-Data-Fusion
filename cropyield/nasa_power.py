"""Download REAL daily weather from NASA POWER for every study district.

NASA POWER (https://power.larc.nasa.gov) is free and needs no API key.
Run on a machine with internet:

    python -m cropyield.nasa_power            # downloads + caches 32 districts
    python -m cropyield.build_dataset --weather nasa
    python -m cropyield.train

Files are cached in data/raw/nasa_power/<district>.csv, so re-runs are instant.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import requests

from .config import DISTRICTS, NASA_DIR, YEARS

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = {
    "PRECTOTCORR": "rainfall_mm",          # precipitation, mm/day
    "T2M_MAX": "tmax_c",                   # max temperature at 2 m, deg C
    "T2M_MIN": "tmin_c",                   # min temperature at 2 m, deg C
    "ALLSKY_SFC_SW_DWN": "solar_radiation_mj",  # MJ/m^2/day (AG community)
    "RH2M": "rh_pct",                      # relative humidity at 2 m, %
}
START = f"{YEARS[0] - 1}1101"   # wheat 2014 is sown in Nov 2013
END = f"{YEARS[-1]}1231"


def parse_response(payload: dict) -> pd.DataFrame:
    """Convert NASA POWER JSON into a tidy daily DataFrame (-999 -> NaN)."""
    params = payload["properties"]["parameter"]
    dates = sorted(params["PRECTOTCORR"].keys())
    frame = pd.DataFrame({"date": pd.to_datetime(dates, format="%Y%m%d").date})
    for api_name, column in PARAMETERS.items():
        values = np.array([params[api_name].get(day, np.nan) for day in dates], dtype=float)
        values[values <= -998] = np.nan
        frame[column] = values
    # Short gaps (rare) are linearly interpolated; rain gaps set to 0.
    frame["rainfall_mm"] = frame["rainfall_mm"].fillna(0.0)
    numeric = list(PARAMETERS.values())
    frame[numeric] = frame[numeric].interpolate(limit=5).ffill().bfill()
    return frame


def fetch_point(lat: float, lon: float, start: str = START, end: str = END,
                retries: int = 3, timeout: int = 120) -> pd.DataFrame:
    query = {
        "parameters": ",".join(PARAMETERS), "community": "AG",
        "latitude": lat, "longitude": lon, "start": start, "end": end, "format": "JSON",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(API_URL, params=query, timeout=timeout)
            response.raise_for_status()
            return parse_response(response.json())
        except Exception as error:  # network hiccup / rate limit -> back off and retry
            last_error = error
            time.sleep(3 * attempt)
    raise RuntimeError(f"NASA POWER request failed for ({lat}, {lon}): {last_error}")


def cache_path(district_name: str):
    return NASA_DIR / f"{district_name.lower()}.csv"


def download_all(force: bool = False) -> None:
    NASA_DIR.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(DISTRICTS, start=1):
        path = cache_path(d.name)
        if path.exists() and not force:
            print(f"[{i:2d}/{len(DISTRICTS)}] {d.name:<11} cached")
            continue
        frame = fetch_point(d.lat, d.lon)
        frame.to_csv(path, index=False)
        print(f"[{i:2d}/{len(DISTRICTS)}] {d.name:<11} {len(frame):,} days saved")
        time.sleep(1.0)  # be polite to the public API


def load_district(district_name: str) -> pd.DataFrame:
    path = cache_path(district_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m cropyield.nasa_power")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def all_cached() -> bool:
    return all(cache_path(d.name).exists() for d in DISTRICTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NASA POWER daily weather for all districts")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    download_all(force=parser.parse_args().force)
    print("Done. Next: python -m cropyield.build_dataset --weather nasa")
