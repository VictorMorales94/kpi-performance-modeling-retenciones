# KPI Impact Simulation Engine

Motor reproducible para análisis, modelamiento y simulación del impacto de variables operativas sobre KPIs críticos de un call center (contacto efectivo, retención, tiempos, etc.).
Desarrollado con un pipeline completo: preprocesamiento, selección de variables, reducción de colinealidad, entrenamiento con Ridge, validación avanzada (permutación + bootstrap) y simulación de escenarios con incrementos controlados en el target.

Objetivo

Proveer un framework claro y transparente que permita:

Identificar qué variables explican realmente un KPI.

Cuantificar su impacto marginal.

Simular qué cambios operativos son necesarios para mover un KPI en X puntos.

Evitar decisiones basadas en intuición o supuestos no validados.


Arquitectura

kpi-impact-simulation-engine/
│
├── data/                       # Entrada (Excel/CSV)
├── outputs/                    # Resultados: coeficientes, perm_importance, bootstrap, simulaciones
│
├── load_data.py                # Lectura robusta de Excel/CSV
├── preprocess.py               # Limpieza, fechas, detección de unidades, filtros por mes
├── feature_selection.py        # Clustering por correlación, VIF, LassoCV + ElasticNetCV
├── model_training.py           # RidgeCV con validación cruzada repetida
├── model_evaluation.py         # Permutation importance + gráficos de coeficientes
├── bootstrap.py                # Bootstrap de coeficientes (intervalos de confianza)
├── simulate.py                 # Simulación de impacto controlado por puntos porcentuales
├── utils.py                    # Tipos de variables, formateos, utilidades
│
├── main_contacto_efectivo.py   # Pipeline completo para target contacto_efectivo/contacto
└── main_retenido.py            # Pipeline completo para target retenido/contacto_efectivo

Pipeline

1. Preprocesamiento

Normalización de columnas.

Detección automática de fecha.

Detección de porcentajes (0–1 vs 0–100).

Selección automática de numéricos.



2. Reducción de multicolinealidad

Correlation Clustering (umbral 0.85).

VIF iterative dropping.



3. Selección de variables

LassoCV + ElasticNetCV.

Conjunción de features estables.



4. Modelo

RidgeCV con búsqueda de alpha en logspace.

Coeficientes desescalados a unidades originales.

R² con RepeatedKFold 5x5.



5. Evaluación

Permutation Importance (30 repeticiones).

Bootstrap de coeficientes (1000 muestras).

Betas estandarizados.



6. Simulación

Interpretación automática de unidades.

Cálculo de delta_x necesario para mover el target en X puntos.

Filtros de factibilidad según reglas operativas.

Exportación de escenarios a CSV.




Ejemplo de uso

python main_contacto_efectivo.py
python main_retenido.py

Los resultados se generan en la carpeta outputs/.

Requisitos

Python 3.10+

pandas

numpy

scikit-learn

scipy

statsmodels

joblib

matplotlib


Enfoque

Este repositorio está diseñado para analistas que no solo necesitan modelos, sino decisiones accionables: qué mover, cuánto mover y qué variable devuelve el mayor impacto marginal en un KPI crítico.
