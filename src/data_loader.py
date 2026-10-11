"""
Módulo de carga y exportación de datos para EnergyPulse Analytics.
Gestiona lectura robusta con detección de codificaciones, manejo de excepciones,
carga de dataset de muestra y exportación a CSV.
"""

import io
from pathlib import Path
from typing import Any, Tuple, Union
import pandas as pd

from src.config import SAMPLE_DATA_FILE
from src.validation import ValidationReport, clean_dataframe, validate_dataframe


def load_csv_data(
    source: Union[str, Path, io.BytesIO, io.StringIO, Any],
    clean: bool = True,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Carga un archivo CSV desde una ruta de archivo o un buffer en memoria (Streamlit upload).
    Gestiona automáticamente codificaciones alternativas (UTF-8, Latin-1, CP1252) y excepciones.

    Args:
        source: Ruta al archivo o buffer binario/texto.
        clean: Si es True, aplica limpieza básica de tipos y remoción de duplicados.

    Returns:
        Tupla de (pd.DataFrame, ValidationReport). En caso de error crítico, el DataFrame
        será vacío y el reporte contendrá el detalle del error.
    """
    empty_df = pd.DataFrame()
    encodings_to_try = ["utf-8", "latin1", "cp1252", "utf-16"]
    df: pd.DataFrame = empty_df
    last_exception: Exception | None = None

    # Si es un buffer de Streamlit (UploadedFile o BytesIO)
    is_buffer = hasattr(source, "read")

    for encoding in encodings_to_try:
        try:
            if is_buffer:
                source.seek(0)
            df = pd.read_csv(source, encoding=encoding)
            last_exception = None
            break
        except UnicodeDecodeError as e:
            last_exception = e
            continue
        except pd.errors.EmptyDataError:
            report = ValidationReport(
                is_valid=False,
                errors=["El archivo CSV seleccionado está vacío o no contiene filas con datos."],
            )
            return empty_df, report
        except pd.errors.ParserError as e:
            report = ValidationReport(
                is_valid=False,
                errors=[f"Error de formato CSV (separador, delimitadores o comillas malformadas): {str(e)}"],
            )
            return empty_df, report
        except FileNotFoundError:
            report = ValidationReport(
                is_valid=False,
                errors=[f"No se encontró el archivo especificado en la ruta: {source}"],
            )
            return empty_df, report
        except Exception as e:
            last_exception = e
            break

    if last_exception is not None or df.empty and not isinstance(last_exception, pd.errors.EmptyDataError):
        report = ValidationReport(
            is_valid=False,
            errors=[f"No fue posible leer el archivo con las codificaciones habituales: {str(last_exception)}"],
        )
        return empty_df, report

    # Validar el esquema
    report = validate_dataframe(df)

    # Limpiar solo si la validación fundamental fue superada
    if report.is_valid and clean:
        df = clean_dataframe(df)

    return df, report


def get_or_create_sample_data() -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Carga el conjunto de datos de ejemplo. Si no existe en el disco,
    lo genera automáticamente utilizando el generador determinista.

    Returns:
        Tupla con el DataFrame del sample y su informe de validación.
    """
    if not SAMPLE_DATA_FILE.exists():
        from scripts.generate_sample_data import generate_energy_dataset
        generate_energy_dataset(days=60, output_path=SAMPLE_DATA_FILE)

    return load_csv_data(SAMPLE_DATA_FILE, clean=True)


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convierte un DataFrame en bytes codificados en UTF-8 con BOM para
    compatibilidad total con Microsoft Excel y herramientas analíticas.

    Args:
        df: DataFrame a serializar.

    Returns:
        Bytes listos para descarga directa.
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
