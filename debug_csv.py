import pandas as pd

# ➜ مسیر دقیق فایل شما
path = "data/raw/aps_failure_training_set.csv"

# 1) جداکننده را اتوماتیک شناسایی می‌کند
df = pd.read_csv(path, sep=None, engine="python")

print("Columns:")
print(df.columns)
print("\nNumber of columns:", len(df.columns))

print("\nFirst 5 rows:")
print(df.head())
