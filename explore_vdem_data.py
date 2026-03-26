"""
Exploration of V-Dem Kenya data sources.

Analyzes all Kenya-specific CSV files in this repo to document:
1. Two distinct series: DS-CY (v5-v7.1) and CY-Core (v8-v16)
2. Column evolution across versions
3. Year coverage differences
4. How Kenya CSVs are derived from parent files
5. Automated extraction utility

Usage:
  python3 explore_vdem_data.py                          # Run full analysis
  python3 explore_vdem_data.py --derive <parent.csv>    # Extract Kenya from parent
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KENYA_VDEM_ID: str = "KEN"

BASE_DIR = Path(os.environ.get("VDEM_DATA_DIR", Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Indicator definitions — mirrored from V-DEM.py for cross-reference
# (vdem_source_col, db_col, low_col, high_col)
# ---------------------------------------------------------------------------

VDEM_INDICATORS: list[tuple[str, str, str, str]] = [
    ("v2x_polyarchy",       "polyarchy",                "v2x_polyarchy_codelow",        "v2x_polyarchy_codehigh"),
    ("v2x_libdem",          "libdem",                   "v2x_libdem_codelow",           "v2x_libdem_codehigh"),
    ("v2x_partipdem",       "partipdem",                "v2x_partipdem_codelow",        "v2x_partipdem_codehigh"),
    ("v2x_delibdem",        "delibdem",                 "v2x_delibdem_codelow",         "v2x_delibdem_codehigh"),
    ("v2x_egaldem",         "egaldem",                  "v2x_egaldem_codelow",          "v2x_egaldem_codehigh"),
    ("v2xel_frefair",       "elections_free_fair",      "v2xel_frefair_codelow",        "v2xel_frefair_codehigh"),
    ("v2x_jucon",           "judicial_constraints",     "v2x_jucon_codelow",            "v2x_jucon_codehigh"),
    ("v2x_legcon",          "legislative_constraints",  "v2x_legcon_codelow",           "v2x_legcon_codehigh"),
    ("v2x_corr",            "corruption_index",         "v2x_corr_codelow",             "v2x_corr_codehigh"),
    ("v2x_pubcorr",         "public_corruption",        "v2x_pubcorr_codelow",          "v2x_pubcorr_codehigh"),
    ("v2x_execorr",         "exec_corruption",          "v2x_execorr_codelow",          "v2x_execorr_codehigh"),
    ("v2xcs_ccsi",          "civil_society",            "v2xcs_ccsi_codelow",           "v2xcs_ccsi_codehigh"),
    ("v2x_freexp_altinf",   "freedom_expression",       "v2x_freexp_altinf_codelow",    "v2x_freexp_altinf_codehigh"),
]

_ETL_SOURCE_COLS: list[str] = [t[0] for t in VDEM_INDICATORS]

ETL_INDICATOR_NOTES: dict[str, str] = {
    "v2x_legcon": (
        "NOT present in any file. Actual V-Dem column is 'v2xlg_legcon'. "
        "The ETL handles this gracefully (stores NULL)."
    ),
    "v2x_freexp_altinf": (
        "Present in CY-Core v8-v16 only. "
        "DS-CY series (v5-v7.1) uses 'v2x_freexp' and 'v2x_freexp_thick' instead."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_version(filename: str) -> str:
    """Extract version string from filename (e.g. 'Core-v14 Kenya.csv' → '14')."""
    m = re.search(r"v([\d.]+)", filename, re.IGNORECASE)
    return m.group(1) if m else "?"


def _detect_series(filename: str) -> str:
    """Classify a V-Dem CSV into its release series."""
    name = filename.lower()
    if "ds-cy" in name:
        return "DS-CY"
    if "cy-core" in name:
        return "CY-Core"
    return "Unknown"

# ---------------------------------------------------------------------------
# Discover — scan directory for V-Dem CSVs
# ---------------------------------------------------------------------------

def discover_kenya_csvs(base: Path) -> list[dict[str, Any]]:
    """Find all Kenya-filtered V-Dem CSVs and extract metadata via pandas."""
    results: list[dict[str, Any]] = []

    for f in sorted(base.iterdir()):
        if f.suffix != ".csv" or "kenya" not in f.name.lower():
            continue

        log.info("Scanning Kenya file: %s", f.name)

        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as exc:
            log.warning("Failed to read %s: %s", f.name, exc)
            continue

        if "year" not in df.columns:
            log.warning("No 'year' column in %s — skipping.", f.name)
            continue

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        valid_years = df["year"].dropna().astype(int)

        country_ids: set[str] = set()
        if "country_text_id" in df.columns:
            country_ids = set(
                df["country_text_id"].dropna().str.strip().str.upper().unique()
            )

        results.append({
            "file":        f.name,
            "series":      _detect_series(f.name),
            "version":     _extract_version(f.name),
            "num_cols":    len(df.columns),
            "num_rows":    len(df),
            "year_min":    int(valid_years.min()) if len(valid_years) else None,
            "year_max":    int(valid_years.max()) if len(valid_years) else None,
            "num_years":   len(valid_years),
            "country_ids": country_ids,
            "columns":     set(df.columns),
        })

    log.info("Discovered %d Kenya CSV files.", len(results))
    return results


def discover_parent_csvs(base: Path) -> list[dict[str, Any]]:
    """Find parent (non-Kenya) V-Dem CSVs."""
    results: list[dict[str, Any]] = []

    for f in sorted(base.iterdir()):
        if f.suffix != ".csv" or "kenya" in f.name.lower():
            continue

        size = f.stat().st_size
        results.append({
            "file":       f.name,
            "version":    _extract_version(f.name),
            "size_bytes": size,
            "empty":      size == 0,
        })

    log.info("Discovered %d parent CSV files.", len(results))
    return results

# ---------------------------------------------------------------------------
# Extract — derive Kenya CSV from a full V-Dem parent file
# ---------------------------------------------------------------------------

def extract_kenya_from_parent(
    parent_csv: Path,
    output_csv: Path,
    country_code: str = KENYA_VDEM_ID,
) -> int:
    """
    Derive a Kenya-specific CSV from a full V-Dem parent CSV.

    Strategy (matches how the existing Kenya CSVs were created):
    1. Load the full V-Dem CSV (all countries, all years)
    2. Filter rows where country_text_id == country_code (case-insensitive)
    3. Write all columns unchanged to the output CSV

    Returns the number of Kenya rows written.
    """
    if not parent_csv.exists():
        log.error("Parent CSV not found: %s", parent_csv)
        sys.exit(1)

    log.info("Loading parent V-Dem file: %s  (may take a moment for large CSV)", parent_csv)

    try:
        header_df = pd.read_csv(parent_csv, nrows=0)
        available = set(header_df.columns)
    except Exception as exc:
        log.error("Failed to read header from %s: %s", parent_csv, exc)
        sys.exit(1)

    if "country_text_id" not in available:
        log.error("'country_text_id' column not found in %s.", parent_csv)
        sys.exit(1)

    try:
        df = pd.read_csv(parent_csv, low_memory=False)
    except Exception as exc:
        log.error("Failed to load %s: %s", parent_csv, exc)
        sys.exit(1)

    kenya = df[df["country_text_id"].str.upper().eq(country_code)].copy()

    if kenya.empty:
        log.error(
            "No rows with country_text_id='%s' in %s.", country_code, parent_csv.name
        )
        sys.exit(1)

    kenya.to_csv(output_csv, index=False)

    log.info(
        "Wrote %d Kenya rows (%d columns) to %s.",
        len(kenya), len(kenya.columns), output_csv.name,
    )
    return len(kenya)

# ---------------------------------------------------------------------------
# Analysis — print exploration report
# ---------------------------------------------------------------------------

def report_inventory(
    kenya_files: list[dict[str, Any]],
    parent_files: list[dict[str, Any]],
) -> None:
    """Print inventory of all discovered files."""
    log.info("── Kenya-specific CSVs ─────────────────────────────────────────")
    header = f"{'File':<40s} {'Series':<10s} {'Ver':<6s} {'Cols':>5s} {'Rows':>5s} {'Years':>12s}"
    log.info(header)
    log.info("-" * 80)
    for f in kenya_files:
        yr = f"{f['year_min']}-{f['year_max']}" if f["year_min"] else "N/A"
        log.info(
            "%s %s %s %5d %5d %12s",
            f["file"].ljust(40), f["series"].ljust(10), f["version"].ljust(6),
            f["num_cols"], f["num_rows"], yr,
        )

    log.info("── Parent CSVs (for automated derivation) ──────────────────────")
    if parent_files:
        for p in parent_files:
            status = "EMPTY (0 bytes)" if p["empty"] else f"{p['size_bytes']:,} bytes"
            log.info("  %s v%-6s %s", p["file"].ljust(40), p["version"], status)
    else:
        log.info("  None found.")


def report_series(kenya_files: list[dict[str, Any]]) -> None:
    """Analyze the two V-Dem release series."""
    ds_files = [f for f in kenya_files if f["series"] == "DS-CY"]
    core_files = [f for f in kenya_files if f["series"] == "CY-Core"]

    log.info("── Two distinct V-Dem release series ───────────────────────────")

    if ds_files:
        log.info(
            "  DS-CY (Dataset, v5-v7.1): %d files, %d–%d columns",
            len(ds_files),
            min(f["num_cols"] for f in ds_files),
            max(f["num_cols"] for f in ds_files),
        )
        log.info("    Older, larger dataset format with more variables.")

    if core_files:
        log.info(
            "  CY-Core (Country-Year Core, v8-v16): %d files, %d–%d columns",
            len(core_files),
            min(f["num_cols"] for f in core_files),
            max(f["num_cols"] for f in core_files),
        )
        log.info("    Current standard release format. More focused variable set.")


def report_column_evolution(kenya_files: list[dict[str, Any]]) -> None:
    """Compare columns between earliest and latest CY-Core versions."""
    core_files = [f for f in kenya_files if f["series"] == "CY-Core"]
    if len(core_files) < 2:
        return

    log.info("── Column evolution within CY-Core ─────────────────────────────")
    first = min(core_files, key=lambda x: float(x["version"]))
    last = max(core_files, key=lambda x: float(x["version"]))
    common = first["columns"] & last["columns"]
    only_first = first["columns"] - last["columns"]
    only_last = last["columns"] - first["columns"]

    log.info(
        "  v%s → v%s:  %d cols → %d cols",
        first["version"], last["version"], first["num_cols"], last["num_cols"],
    )
    log.info("  Shared: %d  |  Removed: %d  |  Added: %d", len(common), len(only_first), len(only_last))

    if only_first:
        log.info("  Removed (sample): %s", sorted(only_first)[:10])
    if only_last:
        log.info("  Added (sample):   %s", sorted(only_last)[:10])


def report_year_coverage(kenya_files: list[dict[str, Any]]) -> None:
    """Print year ranges for each version."""
    log.info("── Year coverage growth ────────────────────────────────────────")
    for f in sorted(kenya_files, key=lambda x: (x["series"], x["version"])):
        yr = f"{f['year_min']}-{f['year_max']}" if f["year_min"] else "N/A"
        log.info(
            "  v%-6s (%s): %s  (%d data-years)",
            f["version"], f["series"], yr, f["num_years"],
        )


def report_etl_indicators(kenya_files: list[dict[str, Any]]) -> None:
    """Check availability of the 13 ETL indicators across all versions."""
    log.info("── ETL indicator availability (from V-DEM.py) ──────────────────")
    for ind in _ETL_SOURCE_COLS:
        present_in = [f["version"] for f in kenya_files if ind in f["columns"]]
        absent_in  = [f["version"] for f in kenya_files if ind not in f["columns"]]

        if not absent_in:
            mark = "     OK"
            status = "ALL"
        elif present_in:
            mark = "PARTIAL"
            status = "v" + ", v".join(present_in)
        else:
            mark = "MISSING"
            status = "NONE"

        log.info("  [%s] %-25s — %s", mark, ind, status)

        note = ETL_INDICATOR_NOTES.get(ind)
        if note:
            log.info("           NOTE: %s", note)


def report_derivation_method() -> None:
    """Explain how Kenya CSVs are derived."""
    log.info("── How Kenya CSVs are derived ──────────────────────────────────")
    log.info("  Each Kenya CSV is a simple row filter of its parent V-Dem release:")
    log.info("    1. Load the full V-Dem CSV (all countries, all years)")
    log.info("    2. Filter: country_text_id == '%s'", KENYA_VDEM_ID)
    log.info("    3. Keep ALL columns unchanged")
    log.info("    4. Write to '{original_name} Kenya.csv'")
    log.info("")
    log.info("  Automated derivation command:")
    log.info("    python3 explore_vdem_data.py --derive <parent_csv_path>")


def report_series_differences() -> None:
    """Summarise key differences between DS-CY and CY-Core."""
    log.info("── Key differences: DS-CY vs CY-Core ──────────────────────────")
    log.info("  DS-CY (v5-v7.1):")
    log.info("    - Larger variable set (2097-3153 columns)")
    log.info("    - Includes 'thick' variants (e.g. v2x_freexp_thick)")
    log.info("    - Simpler metadata (codingstart/gapstart/gapend/codingend)")
    log.info("    - v2x_freexp_altinf NOT available (use v2x_freexp instead)")
    log.info("")
    log.info("  CY-Core (v8-v16):")
    log.info("    - Focused variable set (1730-1908 columns)")
    log.info("    - Richer metadata (historical_date, project, gap_index)")
    log.info("    - v2x_freexp_altinf IS available")
    log.info("    - Stable ~1818 cols v10-v15, expanded to 1908 in v16")
    log.info("")
    log.info("  Both series:")
    log.info("    - Filter by country_text_id = '%s'", KENYA_VDEM_ID)
    log.info("    - 12/13 ETL indicators present (v2x_legcon should be v2xlg_legcon)")
    log.info("    - Year coverage starts at 1900, grows ~1 year per release")


def report_recommendations() -> None:
    """Print actionable recommendations for the ETL pipeline."""
    log.info("── Recommendations for V-DEM.py ETL ────────────────────────────")
    log.info("  1. FIX: Change 'v2x_legcon' → 'v2xlg_legcon' in VDEM_INDICATORS")
    log.info("     (currently always NULL due to column name mismatch)")
    log.info("")
    log.info("  2. CONSIDER: For DS-CY (v5-v7.1), map 'v2x_freexp' or")
    log.info("     'v2x_freexp_thick' → freedom_expression (v2x_freexp_altinf missing)")
    log.info("")
    log.info("  3. ADD: Use --derive flag to auto-generate Kenya CSVs from parent files")
    log.info("")
    log.info("  4. NOTE: V-Dem-CY-Core-v10.csv is empty (0 bytes) — needs real data")

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    """Full exploration analysis — mirrors V-DEM.py's main() flow."""
    log.info("=" * 80)
    log.info("V-DEM KENYA DATA EXPLORATION")
    log.info("=" * 80)

    # Extract — discover files
    kenya_files = discover_kenya_csvs(BASE_DIR)
    parent_files = discover_parent_csvs(BASE_DIR)

    if not kenya_files:
        log.error("No Kenya CSV files found in %s.", BASE_DIR)
        sys.exit(1)

    # Transform — analyse
    report_inventory(kenya_files, parent_files)
    report_series(kenya_files)
    report_column_evolution(kenya_files)
    report_year_coverage(kenya_files)
    report_etl_indicators(kenya_files)

    # Load — output findings
    report_derivation_method()
    report_series_differences()
    report_recommendations()

    log.info("=" * 80)
    log.info(
        "Exploration complete — %d Kenya files, %d parent files analysed.",
        len(kenya_files), len(parent_files),
    )


def run_derive(parent_path: str) -> None:
    """Derive a Kenya CSV from a parent V-Dem file — mirrors V-DEM.py's ETL."""
    parent = Path(parent_path)
    if not parent.is_absolute():
        parent = BASE_DIR / parent

    version = _extract_version(parent.name)
    stem = parent.stem
    output = parent.parent / f"{stem} Kenya.csv"

    log.info(
        "Kenya derive — V-Dem v%s  country=%s  file=%s",
        version, KENYA_VDEM_ID, parent,
    )

    n = extract_kenya_from_parent(parent, output)

    log.info("Kenya derive complete — %d rows written to %s.", n, output.name)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--derive":
        if len(sys.argv) < 3:
            log.error("Usage: python3 explore_vdem_data.py --derive <parent_csv>")
            sys.exit(1)
        run_derive(sys.argv[2])
    else:
        run_analysis()


if __name__ == "__main__":
    main()
