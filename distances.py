import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.neighbors import DistanceMetric
import warnings
warnings.filterwarnings("ignore")

# ----------  two example vectors  -----------------------------------------
u = np.array([1, 2, 3, 0])
v = np.array([4, 0, 3, 2])

# ----------  1. EUCLIDEAN  (L₂)  ------------------------------------------
euclidean = np.linalg.norm(u - v)                     # straight-line
# or  distance.euclidean(u, v)

# ----------  2. MANHATTAN  (L₁, City-block)  ------------------------------
manhattan = np.sum(np.abs(u - v))                     # taxi-cab
# or  distance.cityblock(u, v)

# ----------  3. MAHALANOBIS  (accounts for covariance)  -------------------
# need covariance of the *data* (here we fake a tiny data set)
X = np.random.rand(100, 4)
VI = np.linalg.inv(np.cov(X.T))                       # inverse covariance
mahal = distance.mahalanobis(u, v, VI)

# ----------  4. MINKOWSKI  (generalises L₁ and L₂)  -----------------------
p = 3                                                  # any p ≥ 1
minkowski = distance.minkowski(u, v, p=p)

# ----------  5. CHEBYSHEV  (L∞)  ------------------------------------------
chebyshev = distance.chebyshev(u, v)                  # max coordinate diff

# ----------  6. HAMMING  (for equal-length *symbols*)  --------------------
hamming = distance.hamming(u, v)                      # ratio of diffs

# ----------  7. JACCARD  (sets, 0/1 or bool)  -----------------------------
jaccard = distance.jaccard(u.astype(bool), v.astype(bool))

# ----------  8. COSINE  (angle, ignores length)  --------------------------
cosine = distance.cosine(u, v)                        # 1 - cos(θ)

# ----------  9. CORRELATION  (1 - |Pearson r|)  --------------------------
corr = distance.correlation(u, v)

# ---------- 10. HAUSDORFF  (shapes / point-clouds)  -----------------------
# two tiny clouds
A = np.random.rand(5, 2)
B = np.random.rand(4, 2)
hausdorff = distance.directed_hausdorff(A, B)[0]

# ---------- 11. EDIT / LEVENSHTEIN  (strings)  ----------------------------
import Levenshtein
edit_dist = Levenshtein.distance('kitten', 'sitting')

# ----------  quick lookup table -------------------------------------------
lookup = {
 'Euclidean': euclidean,
 'Manhattan': manhattan,
 'Mahalanobis': mahal,
 'Minkowski(p=3)': minkowski,
 'Chebyshev': chebyshev,
 'Hamming': hamming,
 'Jaccard': jaccard,
 'Cosine': cosine,
 'Correlation': corr,
 'Hausdorff': hausdorff,
 'Levenshtein': edit_dist
}

print('All distances between u and v (or clouds/strings):\n')
for k, v in lookup.items():
    print(f'{k:15} : {v:.4f}')


"""Euclidean – default geometry
Manhattan – grid / sparse features
Mahalanobis – correlated features / elliptic clusters
Minkowski – general Lp geometry (p=1→Manhattan, p=2→Euclidean)
Chebyshev – chess-board moves / max coordinate error
Hamming – bit-flips / categorical symbols
Jaccard – set similarity (0/1 baskets)
Cosine – text / embeddings (ignore magnitude)
Correlation – shape similarity of time-series
Hausdorff – shape matching between clouds
Levenshtein – spell-check / DNA / string similarity"""
