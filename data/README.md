# Diccionario y Metadatos del Dataset: EnergyPulse Analytics

## 1. Origen y Naturaleza de los Datos
- **Tipo de datos**: Sintéticos con modelado físico y conductual determinista.
- **Generador**: Implementado en [`scripts/generate_sample_data.py`](../scripts/generate_sample_data.py) con semilla determinista (`seed=42`).
- **Justificación**: Permite simular con rigor la respuesta térmica de edificios, curvas horarias de ocupación, generación renovable y picos de consumo sin violar la privacidad ni acuerdos de confidencialidad de instalaciones comerciales reales.
- **Granularidad**: Registros horarios continuos a lo largo de 60 días consecutivos (7,200 observaciones en total para 5 instalaciones).

## 2. Descripción de Columnas y Variables

| Nombre de Columna | Tipo de Dato | Unidad | Rango Válido | Descripción Detallada |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | Datetime / Cadena | `YYYY-MM-DD HH:MM:SS` | 2026-01-01 a 2026-03-01 | Fecha y hora local de la medición de telemetría. |
| `building_id` | Categórico / Cadena | N/A | `BLD-101` a `BLD-105` | Identificador único de la instalación física. |
| `building_type` | Categórico / Cadena | N/A | `Oficina`, `Hospital`, `Centro Comercial`, `Escuela`, `Residencial` | Tipología funcional y uso principal del inmueble. |
| `surface_area_sqm` | Numérico (Float) | Metros cuadrados ($m^2$) | 10.0 - 50,000.0 | Área construida climatizada total del edificio. |
| `temperature_celsius` | Numérico (Float) | Grados Celsius (°C) | -40.0 - 60.0 | Temperatura meteorológica exterior ambiente. |
| `humidity_percent` | Numérico (Float) | Porcentaje (%) | 0.0 - 100.0 | Humedad relativa del aire exterior. |
| `occupancy_rate` | Numérico (Float) | Ratio (0.0 a 1.0) | 0.0 - 1.0 | Fracción de ocupación humana respecto al aforo máximo. |
| `hvac_mode` | Categórico / Cadena | N/A | `Off`, `Eco`, `Normal`, `High` | Estado de funcionamiento del sistema de climatización. |
| `solar_pv_kwh` | Numérico (Float) | Kilovatios-hora (kWh) | $\ge 0.0$ | Generación eléctrica in-situ proveniente de paneles solares fotovoltaicos. |
| `energy_consumption_kwh` | Numérico (Float) | Kilovatios-hora (kWh) | $\ge 0.0$ | **Variable Objetivo (Target)**: Demanda eléctrica neta total registrada en el contador durante esa hora. |

## 3. Instalaciones Modeladas

1. **BLD-101 (Torre Nova)**: Edificio de oficinas corporativas (12,500 $m^2$). Curva laboral intensa de lunes a viernes (8h a 18h), consumo base mínimo fines de semana. Cuenta con paneles solares (65 kWp).
2. **BLD-102 (Hospital San Lucas)**: Centro hospitalario de alta complejidad (28,000 $m^2$). Operación ininterrumpida 24/7 con alto consumo en equipos médicos y climatización higiénica. Dispone de energía solar (120 kWp).
3. **BLD-103 (Plaza Central)**: Centro comercial y ocio (22,000 $m^2$). Máxima afluencia tardes y fines de semana.
4. **BLD-104 (Campus Universitario)**: Centro educativo (15,000 $m^2$). Horario lectivo de lunes a viernes (7h a 17h), nula ocupación en fines de semana. Generación solar (45 kWp).
5. **BLD-105 (Residencial Mirador)**: Complejo de viviendas plurifamiliar (8,500 $m^2$). Picos de consumo matutinos (7h a 9h) y nocturnos (19h a 23h).

## 4. Métricas de Impacto Económico y Ambiental Derivadas
- **Coste Económico Estimado ($)**: Calculado a una tarifa comercial promedio de \$0.165 por kWh consumido.
- **Emisiones de $CO_2$ Equivalentes ($kg$)**: Estimadas considerando un factor promedio de la red eléctrica de 0.385 kg $CO_2$ / kWh.
