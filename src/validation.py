"""
Módulo de validación de datos para EnergyPulse Analytics.
Implementa reglas de consistencia de negocio, integridad de esquemas,
detección de valores nulos, duplicados y anomalías de rango.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.config import REQUIRED_COLUMNS, VALID_RANGES, TARGET_COLUMN


@dataclass
class ValidationReport:
    """Contenedor de resultados del diagnóstico de calidad de datos."""
    is_valid: bool
    total_rows: int = 0
    total_columns: int = 0
    duplicate_rows: int = 0
    null_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Retorna un diccionario resumido para presentación en la interfaz."""
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "duplicate_rows": self.duplicate_rows,
            "total_nulls": sum(self.null_counts.values()),
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
        }


def validate_dataframe(df: Optional[pd.DataFrame]) -> ValidationReport:
    """
    Realiza una validación exhaustiva de un DataFrame contra el esquema esperado.

    Args:
        df: DataFrame a evaluar.

    Returns:
        ValidationReport con el estado, errores y advertencias encontrados.
    """
    report = ValidationReport(is_valid=True)

    # 1. Comprobar si el DataFrame es nulo o está vacío
    if df is None:
        report.is_valid = False
        report.errors.append("El archivo o fuente de datos proporcionada es nula.")
        return report

    if not isinstance(df, pd.DataFrame):
        report.is_valid = False
        report.errors.append(f"El objeto proporcionado no es un DataFrame de Pandas (tipo: {type(df)}).")
        return report

    if df.empty:
        report.is_valid = False
        report.errors.append("El archivo CSV no contiene registros (está vacío).")
        return report

    report.total_rows = len(df)
    report.total_columns = len(df.columns)

    # 2. Verificar presencia de columnas obligatorias
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        report.is_valid = False
        report.errors.append(
            f"Faltan columnas obligatorias para el análisis: {', '.join(missing_cols)}."
        )
        return report

    # 3. Registrar tipos de datos y recuento de nulos
    for col in df.columns:
        report.column_types[col] = str(df[col].dtype)
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            report.null_counts[col] = null_count
            if col == TARGET_COLUMN:
                report.warnings.append(
                    f"La columna objetivo '{TARGET_COLUMN}' tiene {null_count} valores nulos que serán descartados en el modelado."
                )
            else:
                report.warnings.append(
                    f"La columna '{col}' contiene {null_count} valores nulos ({null_count / len(df):.1%})."
                )

    # 4. Duplicados
    dup_count = int(df.duplicated().sum())
    report.duplicate_rows = dup_count
    if dup_count > 0:
        report.warnings.append(f"Se detectaron {dup_count} filas duplicadas en el conjunto de datos.")

    # 5. Validación de columna timestamp
    if "timestamp" in df.columns:
        try:
            converted_dates = pd.to_datetime(df["timestamp"], errors="coerce")
            unparseable_count = int(converted_dates.isna().sum()) - int(df["timestamp"].isna().sum())
            if unparseable_count > 0:
                report.warnings.append(
                    f"Se encontraron {unparseable_count} registros con fechas que no pudieron ser interpretadas en 'timestamp'."
                )
        except Exception as e:
            report.warnings.append(f"No fue posible interpretar la columna 'timestamp' como fechas: {str(e)}")

    # 6. Validación de rangos numéricos y reglas de negocio
    for col, (min_val, max_val) in VALID_RANGES.items():
        if col in df.columns:
            # Intentar coerción numérica segura
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            invalid_numeric = numeric_series.isna().sum() - df[col].isna().sum()
            if invalid_numeric > 0:
                report.warnings.append(
                    f"La columna numérica '{col}' contiene {invalid_numeric} valores no numéricos."
                )

            # Chequeo de límites fuera de rango
            out_of_bounds = ((numeric_series < min_val) | (numeric_series > max_val)).sum()
            if out_of_bounds > 0:
                report.warnings.append(
                    f"La columna '{col}' presenta {out_of_bounds} valores fuera del rango físico esperado [{min_val}, {max_val}]."
                )

    # 7. Regla estricta para la columna objetivo
    if TARGET_COLUMN in df.columns:
        valid_targets = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna()
        if len(valid_targets) < 10:
            report.is_valid = False
            report.errors.append(
                f"La variable objetivo '{TARGET_COLUMN}' tiene menos de 10 registros válidos. Se requieren más datos para análisis."
            )

    return report


def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Aplica una limpieza básica pero segura al DataFrame:
    - Convierte 'timestamp' a datetime si existe.
    - Convierte columnas numéricas conocidas a tipos float.
    - Elimina duplicados si se solicita.
    - Elimina filas con target nulo para garantizar modelado robusto.

    Args:
        df: DataFrame original.
        drop_duplicates: Si es True, remueve filas completamente idénticas.

    Returns:
        pd.DataFrame limpio y normalizado.
    """
    clean_df = df.copy()

    if drop_duplicates:
        clean_df = clean_df.drop_duplicates()

    if "timestamp" in clean_df.columns:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")

    # Coerción de columnas numéricas esperadas
    for col in VALID_RANGES.keys():
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Asegurar que el target no tenga nulos
    if TARGET_COLUMN in clean_df.columns:
        clean_df = clean_df.dropna(subset=[TARGET_COLUMN])

    return clean_df.reset_index(drop=True)
