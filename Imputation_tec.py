

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer                # MICE
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. LOAD ONE DATA SOURCE -------------------------------------------------
iris = load_iris(as_frame=True)
X = iris.data.copy()                       # 150×4  (sepal, petal …)
y = iris.target

# 2. ARTIFICIALLY INJECT 20 % MCAR MISSING --------------------------------
rng = np.random.RandomState(42)
mask = rng.rand(*X.shape) < 0.20
X_missing = X.mask(mask)

print("Original shape:", X.shape)
print("Missing cells   :", X_missing.isna().sum().sum())

# 3. BASE-LINE IMPUTERS ---------------------------------------------------
# 3a. Mean (numeric only)
mean_imp = SimpleImputer(strategy='mean')
X_mean = pd.DataFrame(mean_imp.fit_transform(X_missing), columns=X.columns)

# 3b. Mode = most frequent (works for numeric too)
mode_imp = SimpleImputer(strategy='most_frequent')
X_mode = pd.DataFrame(mode_imp.fit_transform(X_missing), columns=X.columns)

# 3c. Zero / constant
zero_imp = SimpleImputer(strategy='constant', fill_value=0.)
X_zero = pd.DataFrame(zero_imp.fit_transform(X_missing), columns=X.columns)

# 4. K-Nearest NEIGHBOURS -------------------------------------------------
knn_imp = KNNImputer(n_neighbors=5)
X_knn = pd.DataFrame(knn_imp.fit_transform(X_missing), columns=X.columns)

# 5. MICE (Multivariate Imputation by Chained Equations) ------------------
mice = IterativeImputer(estimator=BayesianRidge(),
                        max_iter=10, random_state=0)
X_mice = pd.DataFrame(mice.fit_transform(X_missing), columns=X.columns)

# 6. STOCHASTIC REGRESSION (add noise to regression prediction) -----------
# Fit on complete rows, predict missing col with noise
def stochastic_regression(df, target_col, noise_scale=0.05):
    df = df.copy()
    missing_idx = df[target_col].isna()
    if missing_idx.sum() == 0:
        return df
    train = df.dropna()
    Xt, yt = train.drop(columns=[target_col]), train[target_col]
    model = BayesianRidge().fit(Xt, yt)
    preds = model.predict(df.loc[missing_idx].drop(columns=[target_col]))
    df.loc[missing_idx, target_col] = preds + rng.normal(
        scale=noise_scale, size=preds.shape)
    return df

X_stoch = X_missing.copy()
for col in X.columns:
    X_stoch = stochastic_regression(X_stoch, col)

# 7. HOT-DECK (nearest neighbour donor) ----------------------------------
# For each missing row, copy values from closest complete row
def hotdeck(df):
    df = df.copy()
    incomplete = df.isna().any(axis=1)
    complete   = ~incomplete
    if incomplete.sum() == 0:
        return df
    nn = NearestNeighbors(n_neighbors=1).fit(df[complete])
    idx_map = df[complete].index
    for rid in df[incomplete].index:
        dist, match = nn.kneighbors(df.loc[[rid]])
        donor = idx_map[match[0][0]]
        fill_vals = df.loc[donor, df.loc[rid].isna()]
        df.loc[rid, fill_vals.index] = fill_vals
    return df

X_hot = hotdeck(X_missing)

# 8. SIMPLE EXTRAPOLATION (linear trend last 2 points) -------------------
# Works column-wise; trivial demo
def extrapolate_linear(s):
    s = s.copy()
    na_idx = s.isna()
    if na_idx.sum() == 0:
        return s
    # fill first Na with last observed, others with last+diff
    last = s[s.notna()].iloc[-1]
    diff = s[s.notna()].diff().iloc[-1] if len(s[s.notna()])>1 else 0
    s[na_idx] = last + diff*np.arange(1, na_idx.sum()+1)
    return s

X_extra = X_missing.apply(extrapolate_linear, axis=0)

# 9. DEEP-LEARNING IMPUTATION (tiny auto-encoder) -------------------------
def dl_impute(df, epochs=100):
    # normalise to 0-1
    df_min, df_max = df.min(), df.max()
    normed = (df - df_min) / (df_max - df_min)
    mask = normed.isna()
    # fill NaNs with 0 for training input
    X_in = normed.fillna(0.).values
    # target = same values, but original NaNs become 0 target
    X_out = normed.fillna(0.).values

    # tiny AE
    inp = layers.Input(shape=(df.shape[1],))
    x = layers.Dense(16, activation='relu')(inp)
    x = layers.Dense(df.shape[1], activation='sigmoid')(x)
    model = models.Model(inp, x)
    model.compile('adam', 'mse')
    model.fit(X_in, X_out, epochs=epochs, verbose=0)

    preds = model.predict(X_in)
    # replace only missing positions
    filled = normed.values.copy()
    filled[mask] = preds[mask]
    filled = filled * (df_max - df_min) + df_min
    return pd.DataFrame(filled, columns=df.columns)

X_dl = dl_impute(X_missing)

# 10. QUICK SANITY CHECK --------------------------------------------------
print("\nImputed column 'sepal length (cm)' sample values:")
for tag, imputed in [('mean', X_mean), ('mode', X_mode),
                     ('zero', X_zero), ('knn', X_knn),
                     ('mice', X_mice), ('stoch', X_stoch),
                     ('hotdeck', X_hot), ('extrap', X_extra),
                     ('deep', X_dl)]:
    print(f"{tag:10}: {imputed.iloc[:3, 0].values}")
