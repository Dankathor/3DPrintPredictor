"""
clean_data.py
-------------
Data cleaning for the 3D Print Quality Predictor capstone project.

Loads the raw Excel dataset, applies targeted cleaning, and saves the result
to data/cleaned/cleaned_3d_printing_data.csv. The original Excel file is
never modified.

Run from the project root:
    python src/clean_data.py
"""

import pathlib
import sys
import unicodedata

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_PATH = pathlib.Path("data/raw/3d printing parametres dataset.xlsx")
CLEANED_DIR = pathlib.Path("data/cleaned")
CLEANED_PATH = CLEANED_DIR / "cleaned_3d_printing_data.csv"


def strip_unicode_junk(text: str) -> str:
    """Remove invisible/control Unicode characters from a string."""
    return "".join(
        ch for ch in text if not unicodedata.category(ch).startswith("C")
    )


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove hidden Unicode characters and surrounding whitespace in headers."""
    df.columns = [strip_unicode_junk(col).strip() for col in df.columns]
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical snake_case names used by the project."""
    rename_map = {
        "Sample No": "sample_no",
        "Printer Type": "printer_type",
        "Material": "material",
        "Layer Thickness (mm)": "layer_thickness_mm",
        "Infill Density (%)": "infill_density_pct",
        "Speed (mm/s)": "speed_mm_s",
        "Infill Pattern": "infill_pattern",
        "Roughness AVG": "roughness_avg_um",
        "Hardness AVG": "hardness_avg",
        "Peak Load (N)": "peak_load_n",
        "Peak Stress (kPa)": "peak_stress_kpa",
        "Strain at Break (mm/mm)": "strain_at_break",
        "Modulus (MPa)": "modulus_mpa",
        "Width (mm)": "width_mm",
        "Thickness (mm)": "thickness_mm",
    }
    return df.rename(columns=rename_map)


def normalize_printer_type(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize printer_type to lowercase trimmed strings."""
    df["printer_type"] = df["printer_type"].str.strip().str.lower()
    return df


def normalize_material(df: pd.DataFrame) -> pd.DataFrame:
    """Remove incidental leading/trailing whitespace from material values."""
    df["material"] = df["material"].str.strip()
    return df


def normalize_infill_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize capitalization/spacing and merge zig zag with zigzag."""
    df["infill_pattern"] = (
        df["infill_pattern"]
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("zig zag", "zigzag", regex=False)
    )
    return df


def add_roughness_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Flag positive roughness measurements for roughness modeling/analysis.

    The source dataset contains roughness values of zero, but its documentation
    does not explain what those values represent. The rows are preserved in the
    cleaned dataset and flagged so downstream roughness modeling can exclude
    them without deleting source records.
    """
    df["roughness_valid"] = df["roughness_avg_um"] > 0
    return df


raw_df = pd.read_excel(RAW_PATH)
original_row_count = len(raw_df)
df = raw_df.copy()

df = sanitize_column_names(df)
df = rename_columns(df)

required_columns = {
    "sample_no",
    "printer_type",
    "material",
    "layer_thickness_mm",
    "infill_density_pct",
    "speed_mm_s",
    "infill_pattern",
    "roughness_avg_um",
    "hardness_avg",
    "peak_load_n",
    "peak_stress_kpa",
    "strain_at_break",
    "modulus_mpa",
    "width_mm",
    "thickness_mm",
}
missing_columns = sorted(required_columns - set(df.columns))
if missing_columns:
    raise ValueError(f"Expected columns were not found after renaming: {missing_columns}")

df = normalize_printer_type(df)
df = normalize_material(df)
df = normalize_infill_pattern(df)
df = add_roughness_valid(df)

CLEANED_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEANED_PATH, index=False)

print("=" * 60)
print("CLEANING VALIDATION REPORT")
print("=" * 60)

print(f"\nOriginal rows : {original_row_count}")
print(f"Cleaned rows  : {len(df)}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Missing values: {df.isnull().sum().sum()}")

print(f"\nUnique printer types ({df['printer_type'].nunique()}):")
for value in sorted(df["printer_type"].dropna().unique()):
    print(f"  {value!r}  ({(df['printer_type'] == value).sum()} records)")

print(f"\nUnique materials ({df['material'].nunique()}):")
for value in sorted(df["material"].dropna().unique()):
    print(f"  {value!r}  ({(df['material'] == value).sum()} records)")

print(f"\nUnique infill patterns ({df['infill_pattern'].nunique()}):")
for value in sorted(df["infill_pattern"].dropna().unique()):
    print(f"  {value!r}  ({(df['infill_pattern'] == value).sum()} records)")

flagged_roughness = (~df["roughness_valid"]).sum()
print(f"\nRecords where roughness_valid is False : {flagged_roughness}")
print(f"Roughness min : {df['roughness_avg_um'].min():.6f}")
print(f"Roughness max : {df['roughness_avg_um'].max():.6f}")

if len(df) != original_row_count:
    raise ValueError(
        f"Row count changed during cleaning: {original_row_count} -> {len(df)}"
    )

print(f"\nSaved to: {CLEANED_PATH.resolve()}")
print("\n" + "=" * 60)
print("Cleaning complete.")
print("=" * 60)
