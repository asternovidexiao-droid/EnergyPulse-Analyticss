"""
Script de generación determinista de datos sintéticos de consumo energético.
Crea un dataset realista de 60 días para 5 tipos de edificios con patrones físicos,
estacionales, ocupacionales y meteorológicos coherentes.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Permitir ejecución directa del script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import DATA_DIR, SAMPLE_DATA_FILE


def generate_energy_dataset(
    days: int = 60,
    seed: int = 42,
    output_path: Path = SAMPLE_DATA_FILE,
) -> pd.DataFrame:
    """
    Genera un DataFrame sintético pero altamente realista con datos horarios de energía.

    Args:
        days: Número de días a generar (60 días = 1440 horas por edificio).
        seed: Semilla para reproducibilidad estocástica.
        output_path: Ruta destino donde guardar el archivo CSV.

    Returns:
        pd.DataFrame con los datos generados.
    """
    np.random.seed(seed)

    # Configuración de los 5 edificios
    buildings = [
        {
            "id": "BLD-101",
            "type": "Oficina",
            "area": 12500.0,
            "base_intensity": 0.018,  # kWh/m² base
            "has_solar": True,
            "solar_capacity": 65.0,   # kW peak
        },
        {
            "id": "BLD-102",
            "type": "Hospital",
            "area": 28000.0,
            "base_intensity": 0.035,  # Alto consumo crítico 24/7
            "has_solar": True,
            "solar_capacity": 120.0,
        },
        {
            "id": "BLD-103",
            "type": "Centro Comercial",
            "area": 22000.0,
            "base_intensity": 0.022,
            "has_solar": False,
            "solar_capacity": 0.0,
        },
        {
            "id": "BLD-104",
            "type": "Escuela",
            "area": 15000.0,
            "base_intensity": 0.014,
            "has_solar": True,
            "solar_capacity": 45.0,
        },
        {
            "id": "BLD-105",
            "type": "Residencial",
            "area": 8500.0,
            "base_intensity": 0.012,
            "has_solar": False,
            "solar_capacity": 0.0,
        },
    ]

    start_date = datetime(2026, 1, 1, 0, 0, 0)
    total_hours = days * 24
    records = []

    for b in buildings:
        b_id = b["id"]
        b_type = b["type"]
        area = b["area"]
        base_kwh = area * b["base_intensity"]

        for h in range(total_hours):
            current_time = start_date + timedelta(hours=h)
            hour = current_time.hour
            day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday
            is_weekend = day_of_week >= 5

            # 1. Simulación de temperatura ambiente (°C)
            # Variación diaria: mínimo a las 05:00, máximo a las 15:00
            daily_cycle = np.sin((hour - 9) * np.pi / 12)  # -1 a +1
            # Variación estacional suave de fondo
            trend_cycle = np.sin(h / (24 * 15)) * 3.0
            temp = 20.0 + (daily_cycle * 7.5) + trend_cycle + np.random.normal(0, 1.2)
            temp = float(np.clip(temp, 8.0, 38.0))

            # 2. Simulación de humedad relativa (%) - inversa a la temperatura
            humidity = 70.0 - (temp - 20.0) * 1.8 + np.random.normal(0, 4.0)
            humidity = float(np.clip(humidity, 25.0, 95.0))

            # 3. Simulación de tasa de ocupación (0.0 a 1.0)
            if b_type == "Oficina":
                if is_weekend:
                    occupancy = np.random.uniform(0.02, 0.08)
                else:
                    if 8 <= hour <= 18:
                        occupancy = 0.75 + 0.18 * np.sin((hour - 8) * np.pi / 10) + np.random.normal(0, 0.04)
                    elif 6 <= hour < 8 or 18 < hour <= 21:
                        occupancy = 0.20 + np.random.normal(0, 0.05)
                    else:
                        occupancy = np.random.uniform(0.01, 0.05)
            elif b_type == "Hospital":
                # Consistencia 24/7 con leve aumento en horario de visitas (10-18h)
                visit_boost = 0.15 if 10 <= hour <= 18 else 0.0
                occupancy = 0.72 + visit_boost + np.random.normal(0, 0.03)
            elif b_type == "Centro Comercial":
                if 10 <= hour <= 22:
                    peak_mult = 1.25 if is_weekend else 0.95
                    occupancy = 0.65 * peak_mult + np.random.normal(0, 0.05)
                elif 8 <= hour < 10 or 22 < hour <= 23:
                    occupancy = 0.18 + np.random.normal(0, 0.03)
                else:
                    occupancy = np.random.uniform(0.02, 0.05)
            elif b_type == "Escuela":
                if is_weekend:
                    occupancy = 0.01
                elif 7 <= hour <= 16:
                    occupancy = 0.85 + np.random.normal(0, 0.04)
                elif 16 < hour <= 19:
                    occupancy = 0.15 + np.random.normal(0, 0.03)
                else:
                    occupancy = 0.01
            else:  # Residencial
                if 7 <= hour <= 9 or 19 <= hour <= 23:
                    occupancy = 0.85 + np.random.normal(0, 0.05)
                elif 9 < hour < 19:
                    occupancy = 0.40 if is_weekend else 0.22 + np.random.normal(0, 0.04)
                else:
                    occupancy = 0.90 + np.random.normal(0, 0.03)

            occupancy = float(np.clip(occupancy, 0.0, 1.0))

            # 4. Modo HVAC según ocupación y temperatura
            if occupancy < 0.08:
                hvac_mode = "Off" if temp > 18 and temp < 24 else "Eco"
            elif temp > 30.0 or temp < 12.0:
                hvac_mode = "High" if occupancy > 0.4 else "Normal"
            elif temp > 25.0 or temp < 16.0:
                hvac_mode = "Normal" if occupancy > 0.3 else "Eco"
            else:
                hvac_mode = "Eco" if occupancy < 0.5 else "Normal"

            # 5. Generación Solar PV (kWh)
            solar_pv = 0.0
            if b["has_solar"] and 7 <= hour <= 18:
                # Perfil campana solar centrado a las 12:30
                sun_elevation = np.sin((hour - 6) * np.pi / 13)
                if sun_elevation > 0:
                    cloud_factor = np.random.uniform(0.70, 1.0)
                    solar_pv = b["solar_capacity"] * sun_elevation * cloud_factor
            solar_pv = float(max(0.0, solar_pv))

            # 6. Cálculo de Consumo Energético Total (kWh)
            hvac_factors = {"Off": 0.15, "Eco": 0.65, "Normal": 1.0, "High": 1.45}
            thermal_delta = max(0.0, temp - 22.0) * 1.6 + max(0.0, 16.0 - temp) * 1.3
            thermal_load = (area / 1000.0) * thermal_delta * hvac_factors[hvac_mode]
            occupant_load = area * (occupancy * 0.016)

            # Consumo bruto
            consumption = base_kwh + occupant_load + thermal_load + np.random.normal(0, base_kwh * 0.04)

            # Inyección de anomalías ocasionales (para validación de EDA y filtros)
            if np.random.rand() < 0.003:  # 0.3% de probabilidad de spike o fallo
                spike_type = np.random.choice(["spike", "drop"])
                if spike_type == "spike":
                    consumption *= np.random.uniform(1.8, 2.5)
                else:
                    consumption *= np.random.uniform(0.2, 0.4)

            consumption = float(max(10.0, consumption))

            records.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "building_id": b_id,
                "building_type": b_type,
                "surface_area_sqm": round(area, 1),
                "temperature_celsius": round(temp, 2),
                "humidity_percent": round(humidity, 1),
                "occupancy_rate": round(occupancy, 3),
                "hvac_mode": hvac_mode,
                "solar_pv_kwh": round(solar_pv, 2),
                "energy_consumption_kwh": round(consumption, 2),
            })

    df = pd.DataFrame(records)

    # Crear directorio si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Dataset generado exitosamente: {len(df)} registros guardados en '{output_path}'.")
    return df


if __name__ == "__main__":
    generate_energy_dataset()
