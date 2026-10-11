"""
Configuración centralizada de EnergyPulse Analytics.
Define esquemas de validación, constantes físicas, económicas y rutas del sistema.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_FILE = DATA_DIR / "sample_energy_data.csv"

# Columna Objetivo (Target)
TARGET_COLUMN = "energy_consumption_kwh"

# Columnas obligatorias para análisis y modelado
REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "building_id",
    "building_type",
    "surface_area_sqm",
    "temperature_celsius",
    "humidity_percent",
    "occupancy_rate",
    "hvac_mode",
    TARGET_COLUMN,
]

# Columnas opcionales pero recomendadas
OPTIONAL_COLUMNS: List[str] = [
    "solar_pv_kwh",
]

# Rangos válidos para validación de datos
VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "surface_area_sqm": (10.0, 500000.0),
    "temperature_celsius": (-40.0, 60.0),
    "humidity_percent": (0.0, 100.0),
    "occupancy_rate": (0.0, 1.0),
    "energy_consumption_kwh": (0.0, 100000.0),
    "solar_pv_kwh": (0.0, 50000.0),
}

# Categorías válidas esperadas
VALID_CATEGORIES: Dict[str, List[str]] = {
    "building_type": ["Oficina", "Hospital", "Centro Comercial", "Escuela", "Residencial"],
    "hvac_mode": ["Off", "Eco", "Normal", "High"],
}

# Constantes de negocio y sostenibilidad
DEFAULT_COST_PER_KWH_USD: float = 0.165       # Tarifa promedio comercial ($ / kWh)
DEFAULT_CO2_KG_PER_KWH: float = 0.385         # Factor de emisión de la red eléctrica (kg CO2 / kWh)
PEAK_HOURS: List[int] = [10, 11, 12, 13, 14, 17, 18, 19, 20]  # Horas pico típicas
