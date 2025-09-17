import numpy as np
import pandas as pd
from sklearn import preprocessing

# ----------  toy data  -----------------------------------------------------
X = np.array([[1,  200,  0.5],
              [2,  300,  1.5],
              [3,  400,  2.5],
              [4, 500, 3]])

cols = ['feat1', 'feat2', 'feat3']
df_raw = pd.DataFrame(X, columns=cols)

# ----------  1. Z-score  (zero mean, unit variance)  ---------------------
zscore = preprocessing.StandardScaler().fit_transform(X)
df_z = pd.DataFrame(zscore, columns=cols)

# ----------  2. Min-Max  (0-1 range)  -------------------------------------
minmax = preprocessing.MinMaxScaler().fit_transform(X)
df_mm = pd.DataFrame(minmax, columns=cols)

# ----------  3. Max-Abs  (-1 to 1, keeps sparsity)  -----------------------
maxabs = preprocessing.MaxAbsScaler().fit_transform(X)
df_ma = pd.DataFrame(maxabs, columns=cols)

# ----------  4. Robust  (median & IQR, ignores outliers)  -----------------
robust = preprocessing.RobustScaler().fit_transform(X)
df_r = pd.DataFrame(robust, columns=cols)

# ----------  5. Quantile  (uniform 0-1, rank-based)  ----------------------
quantile = preprocessing.QuantileTransformer(output_distribution='uniform',
                                             random_state=0).fit_transform(X)
df_q = pd.DataFrame(quantile, columns=cols)

# ----------  6. Power  (Yeo-Johnson, removes skew)  -----------------------
power = preprocessing.PowerTransformer(method='yeo-johnson',
                                       standardize=True).fit_transform(X)
df_p = pd.DataFrame(power, columns=cols)

# ----------  7. Log / Box-Cox  (manual)  ----------------------------------
# Box-Cox needs strictly positive
X_pos = X + np.abs(X.min()) + 1   # shift positive
log_tr = np.log(X_pos)
df_log = pd.DataFrame(log_tr, columns=cols)

# ----------  8. Unit-vector  (L2 norm = 1)  -------------------------------
l2norm = preprocessing.Normalizer(norm='l2').fit_transform(X)
df_l2 = pd.DataFrame(l2norm, columns=cols)

# ----------  9. Mean-normalisation  (mean 0, keeps original scale)  -------
mean_norm = X - X.mean(axis=0)

# ----------  quick lookup table  ------------------------------------------
lookup = {
    'Raw': df_raw,
    'Z-score': df_z,
    'Min-Max': df_mm,
    'MaxAbs': df_ma,
    'Robust': df_r,
    'Quantile-unif': df_q,
    'Power-YJ': df_p,
    'Log-shift': df_log,
    'Unit-vector': df_l2,
    'Mean-norm': mean_norm
}

print('Original vs transformed (first row only):\n')
for k, v in lookup.items():
    print(f'{k:15} : {np.array(v.iloc[0])}')
