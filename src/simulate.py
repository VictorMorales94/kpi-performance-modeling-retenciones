"""

Funciones para simular impacto en target según coeficientes y reglas.

Versión robusta: interpreta la unidad de las variables (0-1 ó 0-100) automáticamente.

Devuelve dict con base_target, objetivo, y dataframe 'candidates' con columnas:

['feature','coef','base','delta_x','nuevo','feasible','reason']

"""

 

import pandas as pd

import numpy as np

from typing import Callable, Optional, Tuple

 

def obtener_base(df: pd.DataFrame, columna: str, tipo: str = "mean"):

    """Obtiene base (mean o max) de columna en df. Devuelve np.nan si no existe o está vacía."""

    if columna not in df.columns:

        return np.nan

    s = pd.to_numeric(df[columna], errors="coerce").dropna()

    if s.empty:

        return np.nan

    return s.max() if tipo == "max" else s.mean()

 

def detect_unit(series: pd.Series) -> str:

    """

    Detecta si una serie de porcentaje está en 'proportion' (0..1) o 'percent' (0..100).

    Retorna 'proportion' | 'percent' | 'other'

    """

    s = pd.to_numeric(series.dropna(), errors="coerce")

    if s.empty:

        return "other"

    mx = s.max()

    p95 = np.nanpercentile(s, 95)

    # heurística: si el 95% está <= 1.1 -> proportion

    if p95 <= 1.1:

        return "proportion"

    # si el 5% >= 1 y mx > 1 -> percent

    if mx > 1.1:

        return "percent"

    return "other"

 

def _check_bounds(tipo_var: str, nuevo: float, base_k: float, hist_span: Optional[float]) -> Tuple[bool, str]:

    """Reglas simples para descartar soluciones no realistas."""

    if tipo_var == "porcentaje":

        # si base está en 0..1 asumimos new también en 0..1, si 0..100 asumimos 0..100

        if (base_k <= 1.1 and (nuevo < 0 or nuevo > 1.0)) or (base_k > 1.1 and (nuevo < 0 or nuevo > 100.0)):

            return False, "fuera_de_rango_porcentaje"

    if tipo_var == "tiempo":

        if nuevo < 0:

            return False, "tiempo_negativo"

        if hist_span is not None and nuevo > (hist_span * 3 + base_k):

            return False, "excesivo_sobre_historial"

    if tipo_var == "entero":

        # regla: no más del 50% de cambio relativo salvo casos especiales

        if base_k == 0:

            if abs(nuevo) > 1000:

                return False, "delta_abs_demasiado_grande"

        else:

            if abs((nuevo - base_k) / (base_k + 1e-9)) > 0.5:

                return False, "delta_rel>0.50"

    return True, "ok"

 

def _standardize_increment(incremento_pts: float, tipo_t: str, base_target: float) -> float:

    """

    Devuelve delta_target en la misma unidad que base_target.

    - Si tipo_t == 'porcentaje' y base_target <= 1 (0..1):

        - incremento_pts > 1 -> se interpreta como puntos porcentuales (2 -> 0.02)

        - incremento_pts <= 1 -> se interpreta como proporción (0.02)

    - Si tipo_t == 'porcentaje' y base_target > 1 (0..100):

        - incremento_pts > 1 -> puntos (2 -> 2)

        - incremento_pts <= 1 -> proporción (0.02 -> 2)

    - Si tipo_t != 'porcentaje' -> incremento_pts se interpreta como porcentaje relativo (2 -> 2%)

    """

    if tipo_t == "porcentaje":

        if base_target <= 1.1:

            # base en 0..1

            if incremento_pts > 1:

                return incremento_pts / 100.0

            else:

                return incremento_pts

        else:

            # base en 0..100

            if incremento_pts > 1:

                return incremento_pts  # en puntos porcentuales (ej 2 -> 2)

            else:

                return incremento_pts * 100.0

    else:

        # no porcentaje: interpretamos incremento_pts como porcentaje relativo

        return base_target * (incremento_pts / 100.0)

 

def apply_simulation(

    coef_series: pd.Series,

    df_mes: pd.DataFrame,

    target: str,

    incremento_pts: float,

    tipo_variable_func: Callable[[str], str],

    obtener_base_func: Optional[Callable[[pd.DataFrame, str], float]] = None,

    base_type: str = "mean",

    save_path: Optional[str] = None

):

    """

    coef_series: pd.Series index=feature names, values=coef

    df_mes: DataFrame con datos históricos (puede ser el df original o df_mes)

    target: nombre de la columna target en df_mes

    incremento_pts: ver _standardize_increment para reglas

    tipo_variable_func: función que devuelve 'porcentaje'|'tiempo'|'entero' dado el nombre de la variable

    obtener_base_func: función que devuelve la base para una columna (por defecto usa obtener_base interno)

    base_type: "mean" o "max" para la base_target y base_k

    save_path: si se pasa ruta, guarda CSV

    """

    if obtener_base_func is None:

        obtener_base_func = lambda d, c: obtener_base(d, c, tipo=base_type)

 

    base_target = obtener_base_func(df_mes, target)

    if pd.isna(base_target):

        raise RuntimeError(f"No se pudo calcular base del target '{target}'")

 

    tipo_t = tipo_variable_func(target)

    # delta_target en misma unidad que base_target

    delta_target = _standardize_increment(incremento_pts, tipo_t, base_target)

 

    objetivo = base_target + delta_target

 

    rows = []

    for feat, coef in coef_series.items():

        try:

            coef_val = float(coef)

        except Exception:

            continue

        if abs(coef_val) < 1e-12:

            continue

        base_k = obtener_base_func(df_mes, feat)

        if pd.isna(base_k):

            continue

 

        # Si coef muy pequeño, evitar división por casi cero

        if abs(coef_val) < 1e-8:

            # marcar pero seguir (no proporcione resultados fiables)

            delta_x = np.nan

            nuevo = np.nan

            feasible = False

            reason = "coef_demasiado_pequeño"

        else:

            delta_x = delta_target / coef_val

            nuevo = base_k + delta_x

 

            # hist_span simple

            serie_hist = pd.to_numeric(df_mes[feat], errors="coerce").dropna() if feat in df_mes.columns else pd.Series(dtype=float)

            hist_span = None

            if not serie_hist.empty:

                hist_span = serie_hist.max() - serie_hist.min()

 

            tipo_var = tipo_variable_func(feat)

            feasible, reason = _check_bounds(tipo_var, nuevo, base_k, hist_span)

 

        rows.append({

            "feature": feat,

            "coef": coef_val,

            "base": base_k,

            "delta_x": delta_x,

            "nuevo": nuevo,

            "feasible": feasible,

            "reason": reason

        })

 

    dfc = pd.DataFrame(rows)

    if dfc.empty:

        result = {"base_target": base_target, "objetivo": objetivo, "candidates": dfc}

        if save_path:

            dfc.to_csv(save_path, index=False)

        return result

 

    # ordenar por impacto absoluto (coef * efecto esperable) o por coef abs

    dfc = dfc.sort_values(by="coef", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

 

    if save_path:

        try:

            dfc.to_csv(save_path, index=False)

        except Exception:

            pass

 

    return {"base_target": base_target, "objetivo": objetivo, "candidates": dfc}
