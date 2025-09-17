import numpy as np
pop = np.arange(1000)          # 0 … 999
sample = np.random.choice(pop, size=100, replace=False)

import pandas as pd
df = pd.DataFrame({'group': ['A']*300 + ['B']*700, 'value': range(1000)})
strat_sample = df.groupby('group', group_keys=False).apply(
               lambda x: x.sample(frac=.1))   # 10 % from each group

clusters = {i: list(range(i*10, (i+1)*10)) for i in range(100)}  # 100 clusters of 10
picked = np.random.choice(list(clusters.keys()), size=5, replace=False)
cluster_sample = [v for c in picked for v in clusters[c]]

k = 10
start = np.random.randint(0, k)
systematic_sample = list(range(start, 1000, k))

# literally “first 50 rows” or “people who walked by”
convenience_sample = df.head(50)

"""Memory hook
Random = fair dice
Stratified = dice inside each locker
Cluster = dice to pick a locker, then open it completely
Systematic = clockwork every k-th tick
Convenience = whatever is within arm’s reach"""
