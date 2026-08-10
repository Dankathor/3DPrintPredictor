"""
SQLite is used here to give the Streamlit application a structured,
queryable store for the cleaned print records. Using a database instead
of loading the CSV on every page interaction improves responsiveness and
allows the app to run filtered SQL queries without loading the full
dataset into memory each time.
"""

import pathlib
import sqlite3

import pandas as pd

CLEANED_PATH = pathlib.Path("data/cleaned/cleaned_3d_printing_data.csv")
DB_PATH      = pathlib.Path("data/printing_data.db")

# All columns that exist in the cleaned CSV, in the order they appear.
COLUMNS = [
    "sample_no",
    "printer_type",
    "material",
    "layer_thickness_mm",
    "infill_density_pct",
    "infill_pattern",
    "speed_mm_s",
    "roughness_avg_um",
    "hardness_avg",
    "peak_load_n",
    "peak_stress_kpa",
    "strain_at_break",
    "modulus_mpa",
    "width_mm",
    "thickness_mm",
    "roughness_valid",
]

# Load cleaned data — the CSV is never modified.
df = pd.read_csv(CLEANED_PATH)
csv_row_count = len(df)
print(f"Rows loaded from CSV : {csv_row_count}")

# Open (or create) the database file.
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

# Write all rows to the print_records table, replacing it if it already
# exists so that re-running this script always produces a clean, reproducible
# database.
df[COLUMNS].to_sql("print_records", conn, if_exists="replace", index=False)

# Create indexes on the columns the Streamlit app is most likely to filter
# on. IF NOT EXISTS prevents errors when the script is run more than once.
cur.execute("CREATE INDEX IF NOT EXISTS idx_material      ON print_records (material)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_printer_type  ON print_records (printer_type)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_infill_pattern ON print_records (infill_pattern)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_roughness_valid ON print_records (roughness_valid)")

conn.commit()

# Validation queries
print("\n" + "=" * 60)
print("DATABASE VALIDATION")
print("=" * 60)

total_rows = cur.execute("SELECT COUNT(*) FROM print_records").fetchone()[0]
print(f"\nTotal rows in database : {total_rows}")

if total_rows != csv_row_count:
    conn.close()
    raise ValueError(
        f"Row count mismatch: CSV has {csv_row_count} rows "
        f"but database has {total_rows} rows."
    )

valid_count   = cur.execute(
    "SELECT COUNT(*) FROM print_records WHERE roughness_valid = 1"
).fetchone()[0]
invalid_count = cur.execute(
    "SELECT COUNT(*) FROM print_records WHERE roughness_valid = 0"
).fetchone()[0]

print(f"Valid roughness records   : {valid_count}")
print(f"Invalid roughness records : {invalid_count}")

print("\nRecords by material:")
for row in cur.execute(
    "SELECT material, COUNT(*) AS n FROM print_records GROUP BY material ORDER BY n DESC"
):
    print(f"  {row[0]:<20} {row[1]}")

print("\nRecords by printer type:")
for row in cur.execute(
    "SELECT printer_type, COUNT(*) AS n FROM print_records GROUP BY printer_type ORDER BY n DESC"
):
    print(f"  {row[0]:<20} {row[1]}")

print("\nRecords by infill pattern:")
for row in cur.execute(
    "SELECT infill_pattern, COUNT(*) AS n FROM print_records GROUP BY infill_pattern ORDER BY n DESC"
):
    print(f"  {row[0]:<20} {row[1]}")

conn.close()

print(f"\nDatabase saved to: {DB_PATH.resolve()}")
print("\n" + "=" * 60)
print("Database build complete.")
print("=" * 60)
