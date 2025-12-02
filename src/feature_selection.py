import numpy as np

import pandas as pd

from sklearn.linear_model import LassoCV, ElasticNetCV

from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.cluster.hierarchy import linkage, fcluster

from scipy.spatial.distance import squareform

 

def compute_vif(X: pd.DataFrame) -> pd.Series:

    Xc = X.copy().assign(const=1.0)

    vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1]-1)],

                    index=X.columns)

    return vif.sort_values(ascending=False)

 

def drop_high_vif_iterative(X: pd.DataFrame, vif_threshold: float = 10.0, max_iter: int = 20):

    X_work = X.copy()

    removed = []

    for i in range(max_iter):

        vif = compute_vif(X_work)

        max_vif = vif.iloc[0]

        if max_vif <= vif_threshold:

            break

        col = vif.index[0]

        removed.append((col, float(max_vif)))

        X_work = X_work.drop(columns=[col])

        if X_work.shape[1] <= 1:

            break

    return X_work, removed, compute_vif(X_work)

 

def correlation_clustering_filter(X: pd.DataFrame, corr_threshold: float = 0.85):

    """

    Use hierarchical clustering on feature correlations to drop one feature per highly-correlated cluster.

    Keeps the feature with highest variance in the cluster.

    """

    corr = X.corr().abs()

    # convert to distance

    dist = 1 - corr

    # make condensed

    condensed = squareform(dist.values, checks=False)

    Z = linkage(condensed, method='average')

    # cluster by distance threshold

    clusters = fcluster(Z, t=1 - corr_threshold, criterion='distance')

    to_keep = []

    dropped = []

    dfc = pd.DataFrame({'feature': X.columns, 'cluster': clusters})

    for cl in np.unique(clusters):

        members = dfc[dfc['cluster'] == cl]['feature'].tolist()

        if len(members) == 1:

            to_keep.append(members[0])

        else:

            # choose feature with max std (most informative)

            stds = X[members].std().abs()

            keep = stds.idxmax()

            to_keep.append(keep)

            for m in members:

                if m != keep:

                    dropped.append((m, cl))

    return X[to_keep].copy(), dropped

 

def lasso_elastic_selection(X: pd.DataFrame, y: pd.Series, random_state=1):

    scaler = StandardScaler()

    Xs = scaler.fit_transform(X)

    lasso = LassoCV(cv=5, n_jobs=-1, random_state=random_state).fit(Xs, y)

    coef_lasso = pd.Series(lasso.coef_, index=X.columns)

    enet = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, n_jobs=-1, random_state=random_state).fit(Xs, y)

    coef_enet = pd.Series(enet.coef_, index=X.columns)

    selected = sorted(set(coef_lasso[coef_lasso.abs() > 1e-8].index.tolist() +

                         coef_enet[coef_enet.abs() > 1e-8].index.tolist()))

    return selected, scaler, lasso, enet
