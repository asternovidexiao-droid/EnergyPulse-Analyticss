# ⚡ EnergyPulse Analytics

> **Plataforma Profesional de Inteligencia Energética, Diagnóstico Térmico y Pronóstico Predictivo de Demanda en Edificios Inteligentes**

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple.svg)
![Pytest](https://img.shields.io/badge/Pytest-Tests%20Passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 1. Descripción del Proyecto

**EnergyPulse Analytics** es una solución integral de analítica de datos e inteligencia artificial diseñada para gestores de infraestructuras, empresas de servicios energéticos (ESCOs) y directores de sostenibilidad. La aplicación permite auditar series temporales de telemetría de contadores inteligentes (*Smart Meters*), evaluar la calidad de los datos, caracterizar la curva de carga horaria, aislar anomalías de consumo y predecir la demanda eléctrica mediante modelos supervisados de Machine Learning.

---

## 2. Problema que Resuelve y Objetivos

### El Reto
El sector de la edificación comercial e industrial consume más del 30% de la energía eléctrica global y representa una porción crítica de las emisiones operacionales de carbono. Las ineficiencias comunes incluyen:
- Consumos nocturnos y de fin de semana no justificados (*cargas fantasma*).
- Desajustes entre los sistemas de climatización (HVAC) y la afluencia real de ocupantes.
- Penalizaciones tarifarias por incurrir en picos de demanda durante horas punta (*Peak Demand*).
- Ausencia de herramientas intuitivas para simular escenarios operativos y presupuestarios (*What-If*).

### Objetivos de la Plataforma
1. **Auditoría e Ingesta Fiable**: Validar automáticamente la coherencia de los datos telemétricos, detectando valores nulos, duplicados y rangos imposibles.
2. **Exploración Multidimensional**: Identificar perfiles de carga, correlaciones bioclimáticas (temperatura/humedad) y desviaciones operativas.
3. **Detección Rigurosa de Anomalías**: Aislación matemática de eventos atípicos mediante el método de Rango Intercuartílico (IQR de Tukey).
4. **Predicción y Mitigación de Picos**: Modelar la demanda con algoritmos de regresión (Random Forest, Gradient Boosting, Ridge) libres de fuga de datos (*data leakage*).
5. **Simulación de Decisiones**: Permitir a los tomadores de decisiones ajustar factores operativos en tiempo real y observar el impacto en costes ($) y emisiones de $CO_2$.

---

## 3. Tecnologías Utilizadas

- **Lenguaje Base**: Python (3.10 a 3.13).
- **Procesamiento de Datos**: Pandas & NumPy para transformaciones vectorizadas y cálculo numérico.
- **Machine Learning**: Scikit-learn (Pipelines, ColumnTransformer, StandardScaler, OneHotEncoder, RandomForestRegressor, HistGradientBoostingRegressor, Ridge, DummyRegressor).
- **Visualización Interactiva**: Plotly Express & Plotly Graph Objects (gráficos responsivos con zoom, tooltips unificados y temas limpios).
- **Interfaz de Usuario**: Streamlit con inyección de CSS modular para diseño moderno con tarjetas KPI y paneles responsivos.
- **Garantía de Calidad y Pruebas**: Pytest con fixtures reproducibles para pruebas unitarias y de integración.

---

## 4. Características Principales

- **Panel de Control Ejecutivo**: KPIs clave en tiempo real (Consumo total en kWh/MWh, Demanda punta máxima, Coste estimado en USD, Emisiones en kg $CO_2$ y Cobertura solar fotovoltaica).
- **Gestión de Datos y Validación**:
  - Carga de archivos CSV con detección automática de codificaciones (UTF-8, Latin-1, CP1252).
  - Conjunto de datos de muestra integrado con 7,200 observaciones horarias para 5 tipologías de edificio.
  - Informe automático de calidad: nulos, duplicados, inconsistencias físicas y tipado.
  - Filtros dinámicos combinados: tipología, instalación, modo HVAC y ventana temporal.
  - Exportación de datos filtrados a CSV con codificación UTF-8 BOM para Excel.
- **Módulo de Análisis Exploratorio (EDA)**:
  - Curvas de carga horaria promedio (00:00 - 23:00 h).
  - Sensibilidad bioclimática: Dispersión temperatura vs demanda con línea de tendencia OLS.
  - Resumen estadístico exhaustivo: media, mediana, desviación típica, Q25, Q75, IQR, asimetría (*skewness*) y curtosis (*kurtosis*).
  - Diagramas de caja (*Box Plots*) por tipología y modo HVAC.
  - Matriz de correlación de Pearson interactiva.
  - Detección de anomalías por IQR con factor de severidad configurable y visualización en serie temporal.
- **Modelado Predictivo de Machine Learning**:
  - Comparativa honesta y transparente frente a un modelo base (*Dummy Regressor* basado en la media).
  - Métricas estandarizadas de evaluación: MAE, RMSE, $R^2$ y MAPE.
  - Gráfico de dispersión Valores Reales vs Predichos con diagonal ideal $y=x$.
  - Gráfico de distribución de errores residuales centrado en cero.
  - Importancia relativa de características (*Feature Importance*).
- **Simulador Interactivo de Escenarios "What-If"**:
  - Ajuste dinámico de superficie ($m^2$), temperatura exterior, humedad, tasa de ocupación, modo HVAC y franja horaria.
  - Estimación instantánea de consumo horario, coste monetario y huella de carbono.
  - Recomendaciones inteligentes de eficiencia energética.

---

## 5. Estructura del Proyecto

```text
Feiber/
├── app.py                      # Aplicación principal interactiva (Streamlit)
├── requirements.txt            # Dependencias del proyecto fijadas
├── README.md                   # Documentación técnica exhaustiva
├── .gitignore                  # Exclusiones de Git (entornos, cachés)
├── data/
│   ├── README.md               # Metadatos y diccionario de variables del dataset
│   └── sample_energy_data.csv  # Dataset sintético determinista (7,200 filas)
├── scripts/
│   └── generate_sample_data.py # Generador de datos sintéticos con modelos físicos
├── src/
│   ├── __init__.py             # Módulo raíz del paquete
│   ├── config.py               # Configuración centralizada, esquemas y constantes
│   ├── data_loader.py          # Carga de archivos, manejo de encodings y exportación
│   ├── validation.py           # Validaciones de esquema, reglas físicas y calidad
│   ├── preprocessing.py        # Ingeniería de variables temporales y pipeline ML
│   ├── analysis.py             # KPIs ejecutivos, estadísticas y detección IQR
│   ├── visualization.py        # Visualizaciones interactivas con Plotly
│   └── modeling.py             # Entrenamiento de modelos, métricas y simulador
└── tests/
    ├── __init__.py             # Paquete de pruebas
    ├── test_data_loader.py     # Pruebas de ingesta y exportación
    ├── test_validation.py      # Pruebas de reglas de validación y limpieza
    ├── test_preprocessing.py   # Pruebas de ingeniería de características
    ├── test_analysis.py        # Pruebas de funciones estadísticas y anomalías
    └── test_modeling.py        # Pruebas de entrenamiento y simulación ML
```

---

## 6. Requisitos Previos

- **Python**: Versión 3.10 o superior (compatible y probado en Python 3.13).
- **Gestor de paquetes**: `pip`.
- **Navegador web moderno**: Chrome, Edge, Firefox o Safari.

---

## 7. Instrucciones de Instalación

### En Windows (PowerShell / CMD)

```powershell
# 1. Clonar o acceder al directorio del proyecto
cd c:\Users\soyas\Documents\Feiber

# 2. (Opcional recomendado) Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt
```

### En macOS / Linux (Bash / Zsh)

```bash
# 1. Acceder al directorio
cd /ruta/al/proyecto/Feiber

# 2. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 8. Instrucciones para Ejecutar la Aplicación

Para iniciar el servidor interactivo de Streamlit, ejecute:

```bash
# Ejecución estándar
streamlit run app.py

# O alternativamente a través del módulo de Python:
python -m streamlit run app.py
```

La consola mostrará las URLs de acceso local y en red:
- **URL Local**: `http://localhost:8501`

Abra dicha dirección en su navegador web para interactuar con la plataforma.

---

## 9. Cómo Utilizar el Dataset de Ejemplo

1. Al abrir la aplicación por primera vez, el dataset de muestra se carga de forma automática.
2. Si en cualquier momento sube un archivo personalizado y desea regresar al conjunto original, haga clic en el botón **"🔄 Cargar Dataset de Muestra"** ubicado en la parte superior de la barra lateral izquierda.
3. El dataset de ejemplo contiene 60 días ininterrumpidos de mediciones horarias (7,200 filas) que cubren 5 instalaciones representativas:
   - **BLD-101 (Torre Nova)**: Oficinas corporativas.
   - **BLD-102 (Hospital San Lucas)**: Hospital 24/7 de alta criticidad.
   - **BLD-103 (Plaza Central)**: Centro comercial.
   - **BLD-104 (Campus Universitario)**: Centro educativo.
   - **BLD-105 (Residencial Mirador)**: Complejo residencial.

---

## 10. Cómo Cargar un CSV Propio y Requisitos de Formato

Para cargar su propia serie temporal:
1. En la barra lateral izquierda, localice el componente **"Subir archivo CSV propio"**.
2. Arrastre su archivo `.csv` o selecciónelo desde el explorador de archivos.
3. La aplicación validará automáticamente la estructura antes de procesar los datos.

### Formato y Columnas Requeridas

El archivo debe tener formato CSV delimitado por comas e incluir las siguientes cabeceras:

| Columna Requerida | Tipo | Ejemplo | Descripción |
| :--- | :--- | :--- | :--- |
| `timestamp` | Datetime / Cadena | `2026-01-15 14:00:00` | Fecha y hora en formato reconocible ISO / estándar. |
| `building_id` | Cadena | `BLD-101` | Identificador de la edificación. |
| `building_type` | Cadena | `Oficina` | Tipología funcional del inmueble. |
| `surface_area_sqm` | Numérico | `12500.0` | Superficie útil climatizada en $m^2$ ($> 0$). |
| `temperature_celsius` | Numérico | `24.5` | Temperatura ambiente exterior en °C. |
| `humidity_percent` | Numérico | `52.0` | Humedad relativa del aire exterior (0 a 100%). |
| `occupancy_rate` | Numérico | `0.85` | Ratio de ocupación estimada (0.0 a 1.0). |
| `hvac_mode` | Cadena | `Normal` | Modo del sistema: `Off`, `Eco`, `Normal`, `High`. |
| `energy_consumption_kwh` | Numérico | `315.4` | **Variable Objetivo**: Consumo eléctrico horario registrado en el contador ($\ge 0$). |

*Columna opcional recomendada*:
- `solar_pv_kwh`: Generación eléctrica solar fotovoltaica in-situ en kWh ($\ge 0$).

---

## 11. Métodos Estadísticos y Modelos de Machine Learning

### Métodos Estadísticos
- **Curva de Carga Horaria**: Estimación del valor esperado $E[Y|t]$ y variabilidad $\sigma(Y|t)$ para cada hora $t \in [0, 23]$.
- **Matriz de Correlación de Pearson**: Coeficiente $r_{xy} = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}$ para medir la fuerza de asociaciones lineales.
- **Detección de Anomalías IQR (Tukey)**:
  $$\text{Límite Inferior} = Q_1 - (k \times IQR), \quad \text{Límite Superior} = Q_3 + (k \times IQR)$$
  donde $k=1.5$ por defecto y $IQR = Q_3 - Q_1$.

### Modelos de Regresión
1. **Modelo Base (Baseline - Dummy Regressor)**:
   Predice constantemente la media muestral de entrenamiento ($\hat{y} = \bar{y}_{train}$). Actúa como cota inferior obligatoria: cualquier modelo predictivo aceptable debe superar sustancialmente al baseline.
2. **Random Forest Regressor**:
   Ensamble de árboles de decisión con promediado de predicciones independientes. Captura relaciones no lineales complejas entre temperatura, ocupación, tipología y hora del día sin sobreajuste severo.
3. **HistGradientBoostingRegressor / Gradient Boosting**:
   Ensamble secuencial basado en optimización por descenso de gradiente. Altamente eficiente en grandes volúmenes de telemetría.
4. **Regresión Lineal y Ridge ($\ell_2$)**:
   Modelos paramétricos interpretables con regularización de Tikhonov para verificar linealidad.

### Prevención de Fuga de Datos (*Data Leakage*)
Todo el preprocesamiento está encapsulado en un `Pipeline` con `ColumnTransformer`:
- `StandardScaler` ajusta su media y varianza únicamente con `X_train`.
- `OneHotEncoder(handle_unknown='ignore')` procesa categorías del conjunto de entrenamiento sin exponer datos de evaluación.

---

## 12. Interpretación de las Métricas de Evaluación

- **MAE (Mean Absolute Error - kWh)**: Promedio de los errores absolutos entre la demanda real y la predicha. Representa el desvío medio esperado en unidades físicas directas.
- **RMSE (Root Mean Squared Error - kWh)**: Raíz cuadrada del error cuadrático medio. Penaliza de forma más severa los errores grandes o picos no anticipados.
- **$R^2$ (Coeficiente de Determinación)**: Proporción de la varianza total de la demanda eléctrica que es explicada por el modelo (1.0 es ajuste perfecto, 0.0 indica que el modelo no aporta más información que la media).
- **MAPE (Mean Absolute Percentage Error - %)**: Error porcentual promedio relativo. Facilita la comunicación con perfiles ejecutivos no técnicos (e.g. "error promedio del 4.2%").

---

## 13. Ejecución de Pruebas Automatizadas

El proyecto incluye una suite completa de pruebas unitarias y de integración implementadas con **Pytest**.

Para ejecutar todas las pruebas con reporte detallado:

```bash
python -m pytest tests/ -v
```

### Cobertura de las Pruebas
- `tests/test_validation.py`: Valida DataFrames vacíos, columnas obligatorias faltantes, detección de nulos, duplicados y normalización de tipos.
- `tests/test_data_loader.py`: Valida lectura desde búferes, archivos vacíos, caracteres malformados, persistencia del dataset de ejemplo y exportación UTF-8.
- `tests/test_preprocessing.py`: Comprueba extracción de componentes temporales (hora, fin de semana, punta), KPIs energéticos y transformaciones en ColumnTransformer.
- `tests/test_analysis.py`: Verifica cálculo de métricas agregadas, tablas estadísticas (IQR, skewness, kurtosis), matrices de correlación y detección de outliers.
- `tests/test_modeling.py`: Verifica entrenamiento de Random Forest y Gradient Boosting, superioridad sobre el baseline, extracción de importancia de características y predicciones del simulador.

---

## 14. Limitaciones Conocidas

1. **Datos Extremos**: Los modelos están entrenados con rangos meteorológicos de entre 8°C y 38°C; ante eventos meteorológicos históricos no observados (e.g. tormentas de nieve a -25°C), las predicciones deben considerarse extrapolaciones teóricas.
2. **Tarifas Planas**: El cálculo de costes actual utiliza una tarifa unitaria promedio configurable; no modela de forma nativa estructuras tarifarias complejas con término de potencia contratada en escalones horarios (e.g. tarifas 6.1TD).
3. **Inercia Térmica Transitoria**: El modelo asume respuesta térmica horaria cuasi-estacionaria sin modelar la ecuación diferencial de segundo orden de almacenamiento térmico en paredes.

---

## 15. Posibles Mejoras Futuras

- Integración de pronósticos meteorológicos en tiempo real vía API externa (Open-Meteo o NOAA).
- Modelos recurrentes o de series temporales avanzadas (Prophet, Temporal Fusion Transformers o LSTM) para predicción multi-paso a 24 y 48 horas vista.
- Optimización automática de consignas de termostato mediante aprendizaje por refuerzo (*Reinforcement Learning*).
- Compatibilidad con protocolos de telemetría IoT industrial en tiempo real como MQTT y Modbus TCP.

---

## 16. Licencia

Este proyecto está distribuido bajo la licencia **MIT**. Consulte el archivo de licencia para más detalles.
