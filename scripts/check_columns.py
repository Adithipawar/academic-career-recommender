# scripts/check_features_columns.py
import pandas as pd

df = pd.read_csv("../data/features.csv")
print("📋 Columns in features.csv:\n")
print(df.columns.tolist())
