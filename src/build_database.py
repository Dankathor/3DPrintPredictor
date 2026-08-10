"""
build_database.py
-----------------
Builds the SQLite database used by the 3D Print Quality Predictor.

The script loads the cleaned CSV, writes all cleaned records to print_records,
creates indexes used by the Streamlit filters, and validates the resulting
database.

Run from the project root:
    python src/build_database.py
"""

import pathlib
import sqlite3

import pandas as pd

CLEANED_PATH = pathlib.Path("data/cleaned/cleaned_3d_printing_data.csv")
DB_PATH = pathlib.Path("data/printing_data.db")

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

df = pd.read_csv(CLEANED_PATH)
csv_row_count = len(df)
print(f"Rows loaded from CSV : {csv_row_count}")

missing_columns = [column for column in COLUMNS if column not in df.columns]
if missing_columns:
    raise ValueError(f"Cleaned CSV is missing required columns: {missing_columns}")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()

    # Replacing the table makes repeated builds reproducible.
    df[COLUMNS].to_sql("print_records", conn, if_exists="replace", index=False)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_material "
        "ON print_records (material)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_printer_type "
        "ON print_records (printer_type)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_infill_pattern "
        "ON print_records (infill_pattern)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_roughness_valid "
        "ON print_records (roughness_valid)"
    )
    conn.commit()

    print("\n" + "=" * 60)
    print("DATABASE VALIDATION")
    print("=" * 60)

    total_rows = cursor.execute(
        "SELECT COUNT(*) FROM print_records"
    ).fetchone()[0]
    print(f"\nTotal rows in database : {total_rows}")

    if total_rows != csv_row_count:
        raise ValueError(
            f"Row count mismatch: CSV has {csv_row_count} rows "
            f"but database has {total_rows} rows."
        )

    valid_count = cursor.execute(
        "SELECT COUNT(*) FROM print_records WHERE roughness_valid = 1"
    ).fetchone()[0]
    flagged_count = cursor.execute(
        "SELECT COUNT(*) FROM print_records WHERE roughness_valid = 0"
    ).fetchone()[0]

    print(f"Valid roughness records  : {valid_count}")
    print(f"Flagged roughness records: {flagged_count}")

    print("\nRecords by material:")
    for row in cursor.execute(
        """
        SELECT material, COUNT(*) AS n
        FROM print_records
        GROUP BY material
        ORDER BY n DESC
        """
    ):
        print(f"  {row[0]:<20} {row[1]}")

    print("\nRecords by printer type:")
    for row in cursor.execute(
        """
        SELECT printer_type, COUNT(*) AS n
        FROM print_records
        GROUP BY printer_type
        ORDER BY n DESC
        """
    ):
        print(f"  {row[0]:<20} {row[1]}")

    print("\nRecords by infill pattern:")
    for row in cursor.execute(
        """
        SELECT infill_pattern, COUNT(*) AS n
        FROM print_records
        GROUP BY infill_pattern
        ORDER BY n DESC
        """
    ):
        print(f"  {row[0]:<20} {row[1]}")

print(f"\nDatabase saved to: {DB_PATH.resolve()}")
print("\n" + "=" * 60)
print("Database build complete.")
print("=" * 60)
