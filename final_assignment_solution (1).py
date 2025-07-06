
# Data Science Final Assignment Solutions (NumPy + Pandas)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Data Import and Cleaning
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'gender': ['F', 'M', np.nan, 'M', 'F'],
    'age': [25, np.nan, 30, 22, 28],
    'math': [88, 92, 85, 70, 90],
    'science': [78, 85, 80, 75, 88],
    'english': [82, 88, 79, 69, 91],
    'history': [75, 80, 78, 72, 85],
    'class': ['A', 'A', 'B', 'B', 'A']
})

# Drop missing rows
df_cleaned = df.dropna()

# Fill numerical NaN with mean
df['age'] = df['age'].fillna(df['age'].mean())

# Fill categorical NaN with mode
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# Data Transformation
df['total_score'] = df['math'] + df['science']
df['sqrt_math'] = np.sqrt(df['math'])
scaler = MinMaxScaler()
df[['math_norm']] = scaler.fit_transform(df[['math']])

# Merging DataFrames
df1 = pd.DataFrame({'id': [1, 2], 'val1': [10, 20]})
df2 = pd.DataFrame({'id': [1, 2], 'val2': [100, 200]})
merged = pd.merge(df1, df2, on='id')
left_joined = pd.merge(df1, df2, how='left', on='id')
concat_df = pd.concat([df1, df2], axis=1)

# Grouping and Pivoting
grouped = df.groupby('gender')['math'].agg(['mean', 'std'])
pivot = df.pivot_table(index='gender', columns='class', values='math', aggfunc='mean')

# NumPy Array Ops
arr = df['math'].values
reshaped = arr.reshape(-1, 1)
filtered_df = df[df['math'] > 70]

# Broadcasting
df['math_plus_10'] = df['math'] + 10
df['average'] = (df['math'] + df['science']) / 2

# Linear Algebra
A = np.array([[2, 1], [1, 3]])
b = np.array([8, 13])
x = np.linalg.solve(A, b)
df_solution = pd.DataFrame(x, columns=['solution'])

df['dot'] = df['math'] * df['science']

# Matrix multiplication
A = np.random.randint(1, 5, (2, 3))
B = np.random.randint(1, 5, (3, 2))
product = A @ B

# Handling Missing Data
df_missing = pd.DataFrame({
    'A': [1, np.nan, 3, 4, np.nan],
    'B': [10, 20, np.nan, 40, 50]
})
df_linear = df_missing.interpolate()
df_filled = df_missing.fillna(-1)
df_missing['B_no_outliers'] = np.where((df_missing['B'] < 15) | (df_missing['B'] > 45), df_missing['B'].median(), df_missing['B'])

# Advanced Analysis
grouped = df.groupby(['gender', 'class'])['math'].agg([np.mean, np.std])
corr_df = df.corr(numeric_only=True)

# Rolling mean
ts = pd.Series(np.random.randn(100).cumsum())
ts.rolling(window=10).mean().plot(label='Rolling Mean')
ts.plot(alpha=0.5, label='Original')
plt.legend()
plt.title("Rolling Mean vs Original")
plt.savefig("/mnt/data/rolling_mean_plot.png")
plt.close()

# Reshaping & MultiIndex
array = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
df1 = pd.DataFrame(np.random.randint(1, 10, (2, 3)))
df2 = pd.DataFrame(np.random.randint(1, 10, (2, 3)))
stacked = pd.concat([df1, df2])
arr3d = np.random.randint(1, 5, (2, 2, 2))
df_multi = pd.DataFrame(arr3d.reshape(4, 2))
df_multi.index = pd.MultiIndex.from_product([[0,1],[0,1]])
grouped = df_multi.groupby(level=0).sum()

# Time Series
time_df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'value': [10, 15, 20]
})
time_df['date'] = pd.to_datetime(time_df['date'])
time_df['rolling'] = time_df['value'].rolling(window=2).mean()
time_df['time_diff'] = time_df['date'].diff().dt.days

# Final NumPy Tasks
arr = np.random.randint(1, 100, 10)
norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
matrix = np.random.randint(1, 6, (5, 5))
matrix[matrix == 3] = 0

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)
sub = a - b
mul = a * b
div = a / b

A = np.array([[3, 4], [5, 2]])
B = np.array([7, 8])
sol = np.linalg.solve(A, B)

mat = np.ones((3,3))
vec = np.array([1,2,3])
broadcasted = mat + vec

identity = np.eye(3)

A = np.array([[1,2],[3,4]])
B = np.array([[2,0],[1,2]])
dot_prod = np.dot(A, B)

u = np.array([1,2,3])
v = np.array([4,5,6])
dot_uv = np.dot(u, v)
cross_uv = np.cross(u, v)

rand_arr = np.random.randint(1, 10, 20)
unique_vals = np.unique(rand_arr)

inv = np.linalg.inv(np.array([[1, 2], [3, 4]]))
