"""
Pruebas unitarias para modelado predictivo y simulación de escenarios (src/modeling.py).
"""

import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.modeling import predict_scenario, train_energy_model


@pytest.fixture
def training_df():
    """Genera datos sintéticos suficientes para entrenamiento y validación de ML."""
    records = []
    for i in range(120):
        hour = i % 24
        temp = 16.0 + 10.0 * (hour / 24.0)
        occ = 0.8 if 8 <= hour <= 18 else 0.1
        area = 10000.0
        # Relación causal conocida
        kwh = (area * 0.015) + (occ * 150.0) + (temp * 5.0)
        records.append({
            "timestamp": f"2026-01-01 {hour:02d}:00:00",
            "building_id": "BLD-101",
            "building_type": "Oficina",
            "surface_area_sqm": area,
            "temperature_celsius": temp,
            "humidity_percent": 50.0,
            "occupancy_rate": occ,
            "hvac_mode": "Normal" if occ > 0.5 else "Eco",
            TARGET_COLUMN: kwh,
        })
    return pd.DataFrame(records)


def test_train_energy_model_random_forest(training_df):
    """Verifica el flujo completo de entrenamiento con Random Forest."""
    res = train_energy_model(
        training_df,
        model_type="Random Forest",
        test_size=0.25,
        n_estimators=30,
        max_depth=6,
        random_state=42,
    )

    assert res.model_name == "Random Forest"
    assert res.test_metrics.r2 > 0.80  # El modelo debe capturar bien la relación
    assert res.test_metrics.mae > 0.0
    assert res.test_metrics.rmse > 0.0
    assert len(res.y_test_real) == res.sample_size_test
    assert len(res.y_test_pred) == res.sample_size_test
    # El modelo debe superar al baseline (DummyRegressor)
    assert res.test_metrics.mae < res.baseline_metrics.mae
    assert not res.feature_importances.empty


def test_train_energy_model_gradient_boosting(training_df):
    """Verifica el entrenamiento con Gradient Boosting."""
    res = train_energy_model(
        training_df,
        model_type="Gradient Boosting",
        test_size=0.20,
        n_estimators=30,
        random_state=42,
    )

    assert res.model_name == "Gradient Boosting"
    assert res.test_metrics.r2 > 0.70


def test_predict_scenario(training_df):
    """Verifica que el simulador 'What-If' genere predicciones consistentes."""
    res = train_energy_model(
        training_df,
        model_type="Random Forest",
        n_estimators=25,
        random_state=42,
    )

    scenario = {
        "surface_area_sqm": 10000.0,
        "temperature_celsius": 25.0,
        "humidity_percent": 45.0,
        "occupancy_rate": 0.80,
        "hour": 14,
        "is_weekend": 0,
        "building_type": "Oficina",
        "hvac_mode": "Normal",
    }

    pred = predict_scenario(res.pipeline, scenario, cost_per_kwh=0.18, co2_factor=0.40)

    assert pred["predicted_consumption_kwh"] > 0
    assert pred["estimated_cost_usd"] > 0
    assert pred["carbon_emissions_kg"] > 0


def test_train_insufficient_data():
    """Verifica que se lance ValueError si hay datos insuficientes."""
    tiny_df = pd.DataFrame({
        "surface_area_sqm": [100.0, 200.0],
        TARGET_COLUMN: [10.0, 20.0],
    })
    with pytest.raises(ValueError):
        train_energy_model(tiny_df)
