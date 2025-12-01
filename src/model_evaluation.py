"""
Evaluación: permutation importance, plots básicos, guardar coeficientes
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os

def permutation_importances(model, Xs, y, feature_names, n_repeats=30, random_state=1, out_path="outputs"):
    res = permutation_importance(model, Xs, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    perm_mean = pd.Series(res.importances_mean, index=feature_names).sort_values(ascending=False)
    os.makedirs(out_path, exist_ok=True)
    # plot top 20
    top = perm_mean.head(20)
    plt.figure(figsize=(8,6)); sns.barplot(x=top.values, y=top.index); plt.title("Permutation Importance (top20)")
    plt.tight_layout(); plt.savefig(os.path.join(out_path, "permutation_importance.png"), dpi=300); plt.close()
    return perm_mean

def plot_coefs(coef_series, out_path="outputs"):
    os.makedirs(out_path, exist_ok=True)
    top = coef_series.abs().sort_values(ascending=False).head(20).index
    plt.figure(figsize=(8,6)); sns.barplot(x=coef_series.loc[top].values, y=top); plt.title("Coeficientes (desescalados) top20")
    plt.tight_layout(); plt.savefig(os.path.join(out_path, "coef_plot.png"), dpi=300); plt.close()
    coef_series.to_csv(os.path.join(out_path, "coeficientes.csv"))
    return True
