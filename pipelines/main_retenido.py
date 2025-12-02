"""

Main ejecutable: orquesta la carga, preprocesamiento, selección, entrenamiento, evaluación y simulación.

 

Notas:

- interactive: si False evita pedir input() y deja ejemplos no-bloqueantes.

- reemplaza ruta a tu Excel/CSV en la variable 'ruta' al final o pásala por argumentos.

"""

import os

import pandas as pd

import numpy as np

 

from load_data import load_from_excel

from preprocess import normalize_columns, detect_date_column, coerce_date, scale_possible_percentages, select_numeric, filter_month

from feature_selection import correlation_clustering_filter, lasso_elastic_selection, compute_vif

from model_training import train_ridgecv

from model_evaluation import permutation_importances, plot_coefs

from bootstrap import bootstrap_coefs

from simulate import apply_simulation

from utils import round_by_type, tipo_variable

 

def sanitize_filename(s: str) -> str:

    return s.replace("/", "_").replace(" ", "_")

 

def run_all(ruta_excel, sheet="DATA", month="auto", targets=None, interactive: bool = True):

    os.makedirs("outputs", exist_ok=True)

    print("Cargando...")

    # carga (soporta xlsx y csv)

    if isinstance(ruta_excel, str) and ruta_excel.lower().endswith(".csv"):

        df = pd.read_csv(ruta_excel)

    else:

        df = load_from_excel(ruta_excel, sheet)

 

    # preprocess

    df = normalize_columns(df)

    date_col = detect_date_column(df)

    if date_col is None:

        raise RuntimeError("No date column detected")

    df = coerce_date(df, date_col, new_col="fecha_completa")

    if month != "auto":

        df = filter_month(df, month, date_col="fecha_completa")

 

    # detect unidades de porcentajes (no transforma los valores)

    df, metadata_units = scale_possible_percentages(df)

 

    df_num = select_numeric(df)

    if df_num.empty:

        raise RuntimeError("No numeric columns found after preprocessing. Aborting run.")

 

    # default targets: verificar existencia exacta en df_num.columns

    if targets is None:

        candidate_targets = ["retenido/contacto_efectivo"]
      
        targets = [c for c in candidate_targets if c in df_num.columns]

 

    if not targets:

        print("No targets found in numeric columns. Aborting run.")

        return {}

 

    print("Targets:", targets)

 

    # --- feature engineering / selection prep

    features = [c for c in df_num.columns if c not in targets]

    df_num = df_num[features + targets].dropna(how="all")

    df_num = df_num.dropna(axis=1, how="all")

    X_all = df_num[features].copy()

 

    # correlation clustering filter (reduce multicollinearity groups)

    X_all2, dropped_by_cluster = correlation_clustering_filter(X_all, corr_threshold=0.85)

    if dropped_by_cluster:

        print(f"Dropped by correlation clustering: {len(dropped_by_cluster)} features")

 

    # VIF report (on remaining)

    if not X_all2.empty:

        vif = compute_vif(X_all2)

        print("VIF top 10:\n", vif.head(10))

 

    results = {}

 

    for target in targets:

        print(f"\n--- Processing target: {target} ---\n")

        y = df_num[target].astype(float)

 

        selected, scaler_lasso, _, _ = lasso_elastic_selection(X_all2, y)

        if not selected:

            print("No features selected for", target)

            continue

 

        X_sel = X_all2[selected].astype(float).copy()

 

        train_res = train_ridgecv(X_sel, y)

        model = train_res["model"]

        scaler_final = train_res["scaler"]

        coef_unscaled = train_res["coef_unscaled"]

        scores = train_res["scores"]

 

        # evaluation

        perm = permutation_importances(model, scaler_final.transform(X_sel), y, X_sel.columns, n_repeats=30, random_state=1, out_path="outputs")

        plot_coefs(coef_unscaled, out_path="outputs")

 

        # bootstrap

        try:

            bs = bootstrap_coefs(scaler_final.transform(X_sel), y, model.alpha_, n_boot=1000, random_state=1)

            bs_df = pd.DataFrame({

                "feature": X_sel.columns,

                "coef_median": bs["median"],

                "ci_lower": bs["lower"],

                "ci_upper": bs["upper"]

            }).set_index("feature")

        except Exception:

            bs_df = pd.DataFrame(columns=["coef_median","ci_lower","ci_upper"])

 

        coef_unscaled.to_csv(os.path.join("outputs", f"coef_{sanitize_filename(target)}.csv"))

        try:

            perm.to_csv(os.path.join("outputs", f"perm_{sanitize_filename(target)}.csv"))

        except Exception:

            pass

 

        results[target] = {

            "model": model,

            "scaler": scaler_final,

            "coef_unscaled": coef_unscaled,

            "perm": perm,

            "bootstrap": bs_df,

            "scores": scores,

            "selected": selected

        }

 

    # Basic simulation example for first target (non-blocking)

    if results:

        first_target = list(results.keys())[0]

        coef_series = results[first_target]["coef_unscaled"]

 

        sim_save = os.path.join("outputs", f"simulaciones_{sanitize_filename(first_target)}.csv")

        try:

            sim_res = apply_simulation(

                coef_series,

                df,  # pasar df original (con columnas y fechas)

                first_target,

                incremento_pts=2.0,  # si quieres +2 puntos: cuando base en 0..1 conviene usar 0.02 o dejar 2 y la función lo interpreta

                tipo_variable_func=tipo_variable,

                obtener_base_func=lambda d,c: pd.to_numeric(d[c], errors="coerce").dropna().mean() if c in d.columns else np.nan,

                save_path=sim_save

            )

            if "candidates" in sim_res and not sim_res["candidates"].empty:

                print("Resultado simulación (top candidates):\n", sim_res["candidates"].head(10))

        except Exception as e:

            print("Simulación falló:", str(e))

 

   

 

    # asumiendo X_sel (DataFrame), y (Series), coef_unscaled (Series)

    # coef_unscaled está en unidades originales (coef_desescalado)

    std_x = X_sel.std(ddof=0)

    std_y = y.std(ddof=0)

    beta_std = coef_unscaled * (std_x / (std_y + 1e-12))

    beta_std = beta_std.sort_values(key=lambda s: s.abs(), ascending=False)

    beta_std.to_csv("outputs/coef_beta_estandarizado.csv", header=True)

    print("Top betas estandarizados:\n", beta_std.head(10).round(4))

 

    print("Pipeline completo. Revisa carpeta outputs/")

   

    return results

 

if __name__ == "__main__":

    ruta = r"C:\Users\76566405\Documents\Victor\Analisis\Analisis PY\Estadisiticos\BASE_WSP.xlsx"

    run_all(ruta_excel=ruta, sheet="DATA", month="auto", targets=None, interactive=False)
