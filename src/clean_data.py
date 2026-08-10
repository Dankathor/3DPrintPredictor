"""
clean_data.py
-------------
Data cleaning for the 3D Print Quality Predictor capstone project.

Loads the raw Excel dataset, applies targeted cleaning, and saves the result
to data/cleaned/cleaned_3d_printing_data.csv.  The original Excel file is
never modified.

Run from the project root:
    python src/clean_data.py
"""

import sys
import unicodedata
import pathlib
import pandas as pd

# Force UTF-8 output so Unicode column names (e.g. zero-width spaces in the
# Speed column) do not cause a UnicodeEncodeError on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_PATH = pathlib.Path("data/raw/3d printing parametres dataset.xlsx")
CLEANED_DIR = pathlib.Path("data/cleaned")
CLEANED_PATH = CLEANED_DIR / "cleaned_3d_printing_data.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_unicode_junk(text: str) -> str:
    """Remove zero-width and other invisible Unicode characters from a string.

    The raw dataset contains zero-width spaces (U+200B) in the Speed column
    header.  Relying on the exact byte sequence would be brittle, so we
    normalise every column name by stripping all Unicode characters whose
    category starts with 'C' (control / format / surrogate / private-use)
    before applying the rename map.
    """
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("C"))


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip invisible Unicode characters and surrounding whitespace from
    every column name, returning a new DataFrame with cleaned names."""
    df.columns = [strip_unicode_junk(col).strip() for col in df.columns]
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical snake_case rename map for all 15 columns."""
    rename_map = {
        "Sample No":              "sample_no",
        "Printer Type":           "printer_type",
        "Material":               "material",
        "Layer Thickness (mm)":   "layer_thickness_mm",
        "Infill Density (%)":     "infill_density_pct",
        "Speed (mm/s)":           "speed_mm_s",       # cleaned of zero-width spaces above
        "Infill Pattern":         "infill_pattern",
        "Roughness AVG":          "roughness_avg_um",
        "Hardness AVG":           "hardness_avg",
        "Peak Load (N)":          "peak_load_n",
        "Peak Stress (kPa)":      "peak_stress_kpa",
        "Strain at Break (mm/mm)":"strain_at_break",
        "Modulus (MPa)":          "modulus_mpa",
        "Width (mm)":             "width_mm",
        "Thickness (mm)":         "thickness_mm",
    }
    return df.rename(columns=rename_map)


def normalize_printer_type(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise printer_type to lowercase trimmed strings.

    The raw data contains at least one record with a trailing space ('closed ')
    that would otherwise be counted as a third category.  Standardising to
    lowercase and stripping whitespace ensures only two valid values remain:
    'open' and 'closed'.
    """
    df["printer_type"] = df["printer_type"].str.strip().str.lower()
    return df


def normalize_material(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from material values.

    Material names are preserved exactly as supplied; only incidental padding
    is removed so that identical materials are not split into separate groups.
    """
    df["material"] = df["material"].str.strip()
    return df


def normalize_infill_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise infill_pattern to lowercase with whitespace stripped, and
    merge 'zig zag' into 'zigzag'.

    The raw data contains the same physical patterns recorded with different
    capitalisation ('Grid' vs 'grid') and spacing ('zig zag' vs 'Zigzag').
    Collapsing these duplicates gives a consistent set of categories for
    modelling without changing the meaning of any original value.
    """
    df["infill_pattern"] = (
        df["infill_pattern"]
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)   # collapse internal spaces
        .str.replace("zig zag", "zigzag", regex=False)
    )
    return df


def add_roughness_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Boolean flag indicating whether the roughness measurement is
    physically meaningful (i.e. greater than zero).

    A roughness reading of exactly 0 is not physically possible for FDM prints
    and likely indicates a failed or missing measurement.  The flag allows
    downstream modelling to exclude these records from roughness-related
    analysis without permanently deleting any rows.
    """
    df["roughness_valid"] = df["roughness_avg_um"] > 0
    return df


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

raw_df = pd.read_excel(RAW_PATH)
original_row_count = len(raw_df)

# Work on an explicit copy so the raw DataFrame is never mutated.
df = raw_df.copy()

# ---------------------------------------------------------------------------
# Column cleaning
# ---------------------------------------------------------------------------

df = sanitize_column_names(df)
df = rename_columns(df)

# ---------------------------------------------------------------------------
# Categorical normalisation
# ---------------------------------------------------------------------------

df = normalize_printer_type(df)
df = normalize_material(df)
df = normalize_infill_pattern(df)

# ---------------------------------------------------------------------------
# Derived column
# ---------------------------------------------------------------------------

df = add_roughness_valid(df)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

CLEANED_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEANED_PATH, index=False)

# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

print("=" * 60)
print("CLEANING VALIDATION REPORT")
print("=" * 60)

print(f"\nOriginal rows : {original_row_count}")
print(f"Cleaned rows  : {len(df)}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Missing values: {df.isnull().sum().sum()}")

print(f"\nUnique printer types ({df['printer_type'].nunique()}):")
for val in sorted(df["printer_type"].unique()):
    print(f"  {repr(val)}  ({(df['printer_type'] == val).sum()} records)")

print(f"\nUnique materials ({df['material'].nunique()}):")
for val in sorted(df["material"].unique()):
    print(f"  {repr(val)}  ({(df['material'] == val).sum()} records)")

print(f"\nUnique infill patterns ({df['infill_pattern'].nunique()}):")
for val in sorted(df["infill_pattern"].unique()):
    print(f"  {repr(val)}  ({(df['infill_pattern'] == val).sum()} records)")

invalid_roughness = (~df["roughness_valid"]).sum()
print(f"\nRecords where roughness_valid is False : {invalid_roughness}")
print(f"Roughness min : {df['roughness_avg_um'].min():.6f}")
print(f"Roughness max : {df['roughness_avg_um'].max():.6f}")

print(f"\nSaved to: {CLEANED_PATH.resolve()}")
print("\n" + "=" * 60)
print("Cleaning complete.")
print("=" * 60)
