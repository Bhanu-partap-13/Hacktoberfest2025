# ------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import io, tempfile, os

# tiny toy data set (CSV in-memory) --------------------------
csv = """name,team,age,score,bonus,joined
Alice,A,25,81,3.2,2020-01-01
Bob,A,23,79,2.8,2020-02-01
Charlie,B,30,85,4.1,2019-03-01
Diana,B,np,90,4.0,2019-04-01
Eve,C,22,88,3.5,2021-05-01
Frank,A,24,83,3.0,2020-06-01
Grace,C,26,87,3.7,2020-07-01
Henry,B,29,84,3.9,2018-08-01"""

df = pd.read_csv(io.StringIO(csv.replace('np', str(np.nan))))
df['joined'] = pd.to_datetime(df['joined'])

print('=== 1. Basic info ===')
print(df.head())                 # first 5
print(df.tail(3))                # last 3
print(df.shape, df.columns.tolist(), df.dtypes)
print(df.info())                 # concise summary
print(df.describe())             # numeric summary

# ------------------------------------------------------------
# 2. Missing data
# ------------------------------------------------------------
print('\n=== 2. Missing ===')
print(df.isna().sum())           # count per col
df = df.dropna()                 # drop rows with ANY NaN
# df = df.fillna(df.mean())      # alternative

# ------------------------------------------------------------
# 3. Column ops
# ------------------------------------------------------------
print('\n=== 3. Column ops ===')
df = df.rename(columns={'name': 'employee'})        # rename
df['total'] = df['score'] + df['bonus']             # new col (vectorised)
df['grade'] = np.where(df['score'] >= 85, 'A', 'B') # conditional
df = df.assign(age_plus=df.age + 1)                 # assign
df = df.drop(columns=['bonus'])                     # drop col

# ------------------------------------------------------------
# 4. Filtering / slicing
# ------------------------------------------------------------
print('\n=== 4. Filter ===')
print(df[df.team == 'A'])                        # boolean mask
print(df.query("age > 23 & grade == 'A'"))       # SQL-like
print(df.loc[df['score'].idxmax()])              # max score row
print(df.iloc[2:5])                              # positional slice

# ------------------------------------------------------------
# 5. Sorting
# ------------------------------------------------------------
print('\n=== 5. Sort ===')
print(df.sort_values(['team', 'score'], ascending=[True, False]))

# ------------------------------------------------------------
# 6. Group-by aggregations
# ------------------------------------------------------------
print('\n=== 6. Group-by ===')
agg = (df.groupby('team')
         .agg(avg_score=('score', 'mean'),
              max_age=('age', 'max'),
              count=('employee', 'count'))
         .reset_index())
print(agg)

# pivot table
pivot = df.pivot_table(values='score', index='team',
                       columns='grade', aggfunc='mean')
print(pivot)

# ------------------------------------------------------------
# 7. Apply / map / transform
# ------------------------------------------------------------
print('\n=== 7. Apply ===')
df['age_sq'] = df['age'].apply(lambda x: x**2)          # element-wise
df[['score_norm']] = (df[['score']]
                        .transform(lambda x: (x - x.mean()) / x.std()))
print(df.head())

# ------------------------------------------------------------
# 8. String & datetime
# ------------------------------------------------------------
print('\n=== 8. String / Date ===')
df['employee_lower'] = df['employee'].str.lower()
df['year'] = df['joined'].dt.year
df['months_ago'] = (pd.Timestamp('today') - df['joined']).dt.days // 30
print(df[['employee_lower', 'year', 'months_ago']].head())

# ------------------------------------------------------------
# 9. Dummy encoding
# ------------------------------------------------------------
print('\n=== 9. Dummies ===')
dummies = pd.get_dummies(df.team, prefix='team')
df = pd.concat([df, dummies], axis=1)
print(df.head())

# ------------------------------------------------------------
# 10. Merging / joining
# ------------------------------------------------------------
print('\n=== 10. Merge ===')
team_info = pd.DataFrame({'team': ['A', 'B', 'C'],
                          'manager': ['M1', 'M2', 'M3']})
merged = df.merge(team_info, on='team', how='left')
print(merged[['employee', 'team', 'manager']].head())

# ------------------------------------------------------------
# 11. Concatenation
# ------------------------------------------------------------
print('\n=== 11. Concat ===')
extra = pd.DataFrame({'employee': ['Ivy'], 'team': ['A'],
                      'age': [27], 'score': [89], 'total': [92],
                      'grade': ['A'], 'age_plus': [28],
                      'age_sq': [729], 'score_norm': [np.nan],
                      'employee_lower': ['ivy'], 'year': [2021],
                      'months_ago': [30], 'team_A': [1], 'team_B': [0], 'team_C': [0]})
df = pd.concat([df, extra], ignore_index=True)
print(df.tail())

# ------------------------------------------------------------
# 12. Rolling / window functions
# ------------------------------------------------------------
print('\n=== 12. Rolling ===')
df_sorted = df.sort_values('joined')
df_sorted['score_mov_avg'] = (df_sorted['score']
                                .rolling(window=3, min_periods=1)
                                .mean())
print(df_sorted[['joined', 'score', 'score_mov_avg']])

# ------------------------------------------------------------
# 13. Sampling & shuffling
# ------------------------------------------------------------
print('\n=== 13. Sample ===')
print(df.sample(n=3, random_state=1))

# ------------------------------------------------------------
# 14. Export / Import
# ------------------------------------------------------------
print('\n=== 14. I/O ===')
# to CSV
csv_buf = df.to_csv(index=False)
# to parquet (needs pyarrow)
# df.to_parquet('tmp.parquet')
# to Excel
# df.to_excel('tmp.xlsx', index=False)
# read back
df2 = pd.read_csv(io.StringIO(csv_buf))
print('round-trip shape:', df2.shape)

# ------------------------------------------------------------
# 15. Speed tips & extras
# ------------------------------------------------------------
print('\n=== 15. Extras ===')
# vectorised string contains
mask = df['employee'].str.contains('li', na=False, case=False)
print('Names with "li":', df.loc[mask, 'employee'].tolist())

# query with external variable
min_score = 85
subset = df.query('score > @min_score')
print('High scorers:\n', subset[['employee', 'score']])

# memory usage
print('Memory (MB):', df.memory_usage(deep=True).sum() / 1024**2)

# convert categorical to save RAM
df['team'] = df['team'].astype('category')
print('After category dtype:\n', df.dtypes['team'])

# ------------------------------------------------------------
# 16. One-liner cheat sheet (copy-paste reminder) --------------------------
print('\n=== 16. One-liner reminder ===')
reminder = """
head(), tail(), info(), describe()
isna(), fillna(), dropna()
rename(), assign(), drop()
loc[], iloc[], query()
sort_values(), sort_index()
groupby().agg(), pivot_table()
apply(), map(), transform()
str.*, dt.*
get_dummies(), merge(), concat()
rolling().*, expanding().*
sample(), to_csv()/read_csv()
astype('category') for memory
"""
print(reminder)
