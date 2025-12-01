"""
Entrenamiento de RidgeCV con cross-validation y guardado de resultados.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def train_ridgecv(X: pd.DataFrame, y: pd.Series, alphas=None, cv_splits=5, n_repeats=5):
    if alphas is None:
        alphas = np.logspace(-3,3,25)
    rkf = RepeatedKFold(n_splits=cv_splits, n_repeats=n_repeats, random_state=1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RidgeCV(alphas=alphas, cv=rkf, scoring='r2')
    model.fit(Xs, y)
    scores = cross_val_score(model, Xs, y, cv=rkf, scoring='r2', n_jobs=-1)
    # unscale coefs approx: coef / scaler.scale_
    coef_unscaled = pd.Series(model.coef_, index=X.columns) / scaler.scale_
    return {"model": model, "scaler": scaler, "scores": scores, "coef_unscaled": coef_unscaled}
