from pyod.models.cmgos import CMGOS
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=1000, centers=3, n_features=5,
                  cluster_std=0.5, random_state=42)
X = StandardScaler().fit_transform(X)

# choose variant: 'reg', 'red', 'mcd'
clf = CMGOS(n_clusters=3, contamination=0.05, version='reg')
clf.fit(X)
scores = clf.decision_scores_          # CMGOS values
labels = clf.labels_                   # 0 = normal, 1 = outlier

print('Top-5 outlier scores:', np.sort(scores)[-5:])
