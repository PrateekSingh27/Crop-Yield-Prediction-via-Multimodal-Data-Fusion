"""Run with:  python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cropyield.build_dataset import build
from cropyield.config import ALL_FEATURES, DISTRICTS, MODEL_BUNDLE_PATH, TARGET
from cropyield.features import build_features
from cropyield.nasa_power import parse_response
from cropyield.season import weather_features


@pytest.fixture(scope="session")
def dataset():
    return build_features(build("synthetic", save=False))


def test_dataset_shape_and_ranges(dataset):
    expected = sum(len(d.crops) for d in DISTRICTS) * 11
    assert len(dataset) == expected
    assert dataset[ALL_FEATURES + [TARGET]].isna().sum().sum() == 0
    assert dataset[TARGET].between(0.5, 8).all()
    assert dataset[["ndvi_early", "ndvi_mid", "ndvi_peak", "ndvi_late"]].stack().between(0, 1).all()
    texture = dataset[["sand_pct", "silt_pct", "clay_pct"]].sum(axis=1)
    assert np.allclose(texture, 100, atol=0.5)


def test_tests_do_not_touch_saved_data(dataset):
    from cropyield.config import DATASET_PATH
    if DATASET_PATH.exists():
        saved = pd.read_csv(DATASET_PATH)
        assert len(saved) == len(dataset)


def test_dataset_is_reproducible():
    a, b = build("synthetic", seed=7, save=False), build("synthetic", seed=7, save=False)
    pd.testing.assert_frame_equal(a, b)


def test_loyo_never_trains_on_test_year(dataset):
    years = dataset["year"].to_numpy()
    for y in np.unique(years):
        train_years = set(years[years != y])
        assert y not in train_years


def test_weather_aggregation():
    days = pd.DataFrame({"date": pd.date_range("2020-06-20", periods=125).date, "rainfall_mm": 2.0,
                         "tmax_c": 36.0, "tmin_c": 24.0, "solar_radiation_mj": 18.0, "rh_pct": 70.0})
    w = weather_features(days, "Rice", 2020)
    assert w["rainfall_mm"] == pytest.approx(250)
    assert w["heat_stress_days"] > 0                     # 36 > 35 threshold
    assert w["gdd"] == pytest.approx(125 * (30 - 10))   # mean temp 30, base 10


def test_nasa_parser_handles_missing_values():
    payload = {"properties": {"parameter": {
        "PRECTOTCORR": {"20200101": 1.0, "20200102": -999.0},
        "T2M_MAX": {"20200101": 20.0, "20200102": 22.0},
        "T2M_MIN": {"20200101": 8.0, "20200102": -999.0},
        "ALLSKY_SFC_SW_DWN": {"20200101": 15.0, "20200102": 16.0},
        "RH2M": {"20200101": 60.0, "20200102": 62.0}}}}
    frame = parse_response(json.loads(json.dumps(payload)))
    assert frame["rainfall_mm"].tolist() == [1.0, 0.0]
    assert frame["tmin_c"].notna().all()


@pytest.mark.skipif(not MODEL_BUNDLE_PATH.exists(), reason="train the model first")
class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from api.main import app
        with TestClient(app) as c:
            yield c

    def test_health(self, client):
        assert client.get("/api/health").json()["status"] == "ok"

    def test_predict_is_additive_and_sane(self, client):
        r = client.post("/api/predict", json={"crop": "Wheat", "district": "Ludhiana", "inputs": {}})
        assert r.status_code == 200
        body = r.json()
        assert 1.0 < body["predicted_yield_t_ha"] < 7.5
        lo, hi = body["interval_80"]
        assert lo < body["predicted_yield_t_ha"] < hi
        total = body["base_value"] + sum(c["contribution"] for c in body["contributions"])
        assert total == pytest.approx(body["predicted_yield_t_ha"], abs=0.01)

    def test_heat_stress_lowers_wheat_yield(self, client):
        cool = client.post("/api/predict", json={"crop": "Wheat", "district": "Karnal",
                                                 "inputs": {"heat_stress_days": 0}}).json()
        hot = client.post("/api/predict", json={"crop": "Wheat", "district": "Karnal",
                                                "inputs": {"heat_stress_days": 30}}).json()
        assert hot["predicted_yield_t_ha"] < cool["predicted_yield_t_ha"]

    def test_validation_rejects_bad_input(self, client):
        assert client.post("/api/predict", json={"crop": "Wheat", "inputs": {"soil_ph": 14}}).status_code == 422
        assert client.post("/api/predict", json={"crop": "Maize"}).status_code == 422

    def test_metrics_and_overview(self, client):
        m = client.get("/api/metrics").json()
        assert len(m["model_comparison"]) >= 4
        assert client.get("/api/overview").json()["kpis"]["rows"] > 0
        assert client.get("/api/district/Karnal?crop=Rice").status_code == 200
