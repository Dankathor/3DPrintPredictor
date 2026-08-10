"""
explore_data.py
---------------
Exploratory data analysis for the 3D Print Quality Predictor capstone project.

Reads the raw dataset, prints structural and statistical summaries, and makes
no modifications to the source file.

Run from the project root:
    python src/explore_data.py
"""

import sys
import pandas as pd

# Force UTF-8 output so that column names containing Unicode characters
# (e.g. zero-width spaces found in the 'Speed' column header) do not cause
# a UnicodeEncodeError on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load

DATA_PATH = "data/raw/3d printing parametres dataset.xlsx"

df = pd.read_excel(DATA_PATH)

# 1. Dimensions

print("=" * 60)
print("DATASET DIMENSIONS")
print("=" * 60)
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

# 2. Column names (repr reveals hidden whitespace / non-printable characters)

print("\n" + "=" * 60)
print("COLUMN NAMES (repr)")
print("=" * 60)
for col in df.columns:
    print(repr(col))

# 3. First five rows

print("\n" + "=" * 60)
print("FIRST FIVE ROWS")
print("=" * 60)
print(df.head())

# 4. Data types

print("\n" + "=" * 60)
print("COLUMN DATA TYPES")
print("=" * 60)
print(df.dtypes.to_string())

# 5. Missing values per column

print("\n" + "=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)
missing = df.isnull().sum()
print(missing.to_string())
print(f"\nTotal missing cells: {missing.sum()}")

# 6. Duplicate rows

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(f"Total duplicate rows: {df.duplicated().sum()}")

# 7. Categorical columns — unique values and counts

categorical_cols = df.select_dtypes(include=["str", "category"]).columns.tolist()

print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS — UNIQUE VALUES AND COUNTS")
print("=" * 60)

if categorical_cols:
    for col in categorical_cols:
        print(f"\n--- {repr(col)} ---")
        print(df[col].value_counts(dropna=False).to_string())
else:
    print("No categorical columns detected.")

# 8. Descriptive statistics for numeric columns

numeric_cols = df.select_dtypes(include="number").columns.tolist()

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS (NUMERIC COLUMNS)")
print("=" * 60)

if numeric_cols:
    print(df[numeric_cols].describe().to_string())
else:
    print("No numeric columns detected.")

print("\n" + "=" * 60)
print("Exploration complete. No data was modified or saved.")
print("=" * 60)
