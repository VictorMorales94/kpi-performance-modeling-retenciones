"""

Preprocesamiento reproducible y metadata de unidades:

- normaliza columnas

- detecta columna fecha robustamente

- convierte fechas

- escala posibles porcentajes (detecta unidad: proportion 0..1 o percent 0..100)

- selecciona numéricos

- opción para filtrar por mes (YYYY-MM)

 

Funciones clave devuelven (df, metadata) donde metadata es dict: {col: 'proportion'|'percent'|'other'}

"""

import pandas as pd

import numpy as np

import re

from typing import Optional, Tuple

 

POSSIBLE_DATE_COLS = ["fecha_completa","fecha","fecha_gestion","fecha_gest","dia","día","day"]

 

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df.columns = (

        df.columns.astype(str)

                  .str.strip()

                  .str.lower()

                  .str.replace(" ", "_")

                  .str.replace("-", "_")

    )

    return df

 

def detect_date_column(df: pd.DataFrame) -> Optional[str]:

    cols = list(df.columns)

    for c in cols:

        lc = c.lower()

        if lc in POSSIBLE_DATE_COLS:

            return c

    for c in cols:

        if pd.api.types.is_datetime64_any_dtype(df[c]):

            return c

    for c in cols:

        if "fecha" in c.lower() or "date" in c.lower():

            return c

    return None

 

def coerce_date(df: pd.DataFrame, date_col: str, new_col="fecha_completa"):

    df = df.copy()

    df[new_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    return df

 

def _detect_unit_series(series: pd.Series) -> str:

    s = pd.to_numeric(series.dropna(), errors="coerce")

    if s.empty:

        return "other"

    p95 = np.nanpercentile(s, 95)

    if p95 <= 1.1:

        return "proportion"

    if s.max() > 1.1:

        return "percent"

    return "other"

 

def scale_possible_percentages(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:

    """

    Detecta columnas que parecen porcentajes y devuelve metadata con unidad.

    No convierte valores, solo detecta la unidad y, opcionalmente, puede normalizar.

    Retorna (df, metadata)

    """

    df = df.copy()

    metadata = {}

    for c in df.columns:

        lc = c.lower()

        if any(k in lc for k in ["%","porcentaje","tasa","nivel","%_","pct","rate"]):

            try:

                unit = _detect_unit_series(df[c])

                metadata[c] = unit

                # No transformamos; dejamos la unidad original, pero si quieres forzar:

                # if unit == "proportion": df[c] = df[c].astype(float)

                # if unit == "percent": df[c] = df[c].astype(float)

            except Exception:

                metadata[c] = "other"

        else:

            metadata[c] = "other"

    return df, metadata

 

def select_numeric(df: pd.DataFrame) -> pd.DataFrame:

    return df.select_dtypes(include=[np.number]).copy()

 

def filter_month(df: pd.DataFrame, month_str: str, date_col: str="fecha_completa"):

    m = pd.to_datetime(month_str + "-01", errors="coerce")

    if pd.isna(m):

        return df

    if date_col not in df.columns:

        return df

    return df[df[date_col].dt.to_period("M") == m.to_period("M")]
