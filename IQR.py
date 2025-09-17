import pandas as pd, numpy as np

# 1. toy sensor data ----------------------------------------------------
df = pd.DataFrame({f'sensor_{i}': np.random.normal(10, 2, 200) for i in range(3)})
# inject a couple of extreme spikes
df.iloc[10, 0] = 25
df.iloc[20, 1] = 0

# 2. IQR fence calculation ----------------------------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
low_fence  = Q1 - 1.5 * IQR
high_fence = Q3 + 1.5 * IQR

# 3. flag outliers ------------------------------------------------------
outliers = (df < low_fence) | (df > high_fence)
print("Low fences:\n", low_fence, "\nHigh fences:\n", high_fence)
print("\nOutlier count per sensor:\n", outliers.sum())
