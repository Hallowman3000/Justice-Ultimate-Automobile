"""
Exploration of V-Dem Kenya data sources.

Analyzes all Kenya-specific CSV files in this repo to document:
1. Two distinct series: DS-CY (v5-v7.1) and CY-Core (v8-v16)
2. Column evolution across versions
3. Year coverage differences
4. How Kenya CSVs are derived from parent files
5. Automated extraction utility

Run: python3 explore_vdem_data.py
"""

import csv
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


# ── Discover all Kenya CSV files ──────────────────────────────────────────────

def discover_kenya_csvs() -> list[dict]:
    """Find all Kenya-filtered V-Dem CSVs and extract metadata."""
    results = []
    for f in sorted(BASE.iterdir()):
        if not f.suffix == ".csv":
            continue
        name = f.name.lower()
        if "kenya" not in name:
            continue

        # Determine series
        if "ds-cy" in name:
            series = "DS-CY"
        elif "cy-core" in name:
            series = "CY-Core"
        else:
            series = "Unknown"

        # Extract version
        m = re.search(r"v([\d.]+)", f.name, re.IGNORECASE)
        version = m.group(1) if m else "?"

        with open(f) as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            rows = list(reader)

        years = sorted(int(r["year"]) for r in rows if r.get("year", "").strip())
        country_ids = set(r.get("country_text_id", "") for r in rows if r.get("country_text_id", "").strip())

        results.append({
            "file": f.name,
            "series": series,
            "version": version,
            "num_cols": len(cols),
            "num_rows": len(rows),
            "year_min": years[0] if years else None,
            "year_max": years[-1] if years else None,
            "num_years": len(years),
            "country_ids": country_ids,
            "columns": set(cols),
        })
    return results


def discover_parent_csvs() -> list[dict]:
    """Find parent (non-Kenya) V-Dem CSVs."""
    results = []
    for f in sorted(BASE.iterdir()):
        if not f.suffix == ".csv":
            continue
        name = f.name.lower()
        if "kenya" in name:
            continue
        size = f.stat().st_size
        m = re.search(r"v([\d.]+)", f.name, re.IGNORECASE)
        version = m.group(1) if m else "?"
        results.append({
            "file": f.name,
            "version": version,
            "size_bytes": size,
            "empty": size == 0,
        })
    return results


# ── ETL indicator columns used by V-DEM.py ────────────────────────────────────

ETL_INDICATORS = [
    "v2x_polyarchy", "v2x_libdem", "v2x_partipdem", "v2x_delibdem",
    "v2x_egaldem", "v2xel_frefair", "v2x_jucon", "v2x_legcon",
    "v2x_corr", "v2x_pubcorr", "v2x_execorr", "v2xcs_ccsi",
    "v2x_freexp_altinf",
]

# The ETL script references v2x_legcon, but V-Dem actually uses v2xlg_legcon.
# v2x_freexp_altinf exists only in CY-Core (v8+), not in DS-CY (v5-v7.1).
ETL_INDICATOR_NOTES = {
    "v2x_legcon": "NOT present in any file. Actual V-Dem column is 'v2xlg_legcon'. "
                  "The ETL handles this gracefully (stores NULL).",
    "v2x_freexp_altinf": "Present in CY-Core v8-v16 only. "
                         "DS-CY series (v5-v7.1) uses 'v2x_freexp' and 'v2x_freexp_thick' instead.",
}


# ── Automated Kenya extraction from parent ────────────────────────────────────

def extract_kenya_from_parent(parent_csv: Path, output_csv: Path,
                              country_code: str = "KEN") -> int:
    """
    Derive a Kenya-specific CSV from a full V-Dem parent CSV.

    Strategy (matches how the existing Kenya CSVs were created):
    1. Read the parent CSV
    2. Filter rows where country_text_id == 'KEN' (case-insensitive)
    3. Write all columns to the output CSV

    Returns the number of Kenya rows written.
    """
    if not parent_csv.exists():
        raise FileNotFoundError(f"Parent CSV not found: {parent_csv}")

    with open(parent_csv, newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No header found in {parent_csv}")

        # Identify the country column (country_text_id in all versions)
        if "country_text_id" not in fieldnames:
            raise ValueError(f"'country_text_id' column not found in {parent_csv}")

        kenya_rows = []
        for row in reader:
            if row.get("country_text_id", "").strip().upper() == country_code:
                kenya_rows.append(row)

    if not kenya_rows:
        print(f"  WARNING: No rows found for {country_code} in {parent_csv.name}")
        return 0

    with open(output_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kenya_rows)

    return len(kenya_rows)


def auto_derive_kenya_csv(parent_path: str) -> None:
    """Given a parent V-Dem CSV path, auto-generate the Kenya derivative."""
    parent = Path(parent_path)
    # Determine output name: insert " Kenya" before .csv
    stem = parent.stem  # e.g. "V-Dem-CY-Core-v10"
    output = parent.parent / f"{stem} Kenya.csv"

    print(f"\nExtracting Kenya data from: {parent.name}")
    print(f"Output: {output.name}")
    n = extract_kenya_from_parent(parent, output)
    print(f"Wrote {n} Kenya rows.")


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("V-DEM KENYA DATA EXPLORATION")
    print("=" * 80)

    # 1. Inventory
    kenya_files = discover_kenya_csvs()
    parent_files = discover_parent_csvs()

    print("\n── Kenya-specific CSVs ─────────────────────────────────────────")
    print(f"{'File':<40s} {'Series':<10s} {'Ver':<6s} {'Cols':>5s} {'Rows':>5s} {'Years':>12s}")
    print("-" * 80)
    for f in kenya_files:
        yr = f"{f['year_min']}-{f['year_max']}" if f["year_min"] else "N/A"
        print(f"{f['file']:<40s} {f['series']:<10s} {f['version']:<6s} {f['num_cols']:>5d} {f['num_rows']:>5d} {yr:>12s}")

    print("\n── Parent CSVs (for automated derivation) ──────────────────────")
    if parent_files:
        for p in parent_files:
            status = "EMPTY (0 bytes)" if p["empty"] else f"{p['size_bytes']:,} bytes"
            print(f"  {p['file']:<40s} v{p['version']:<6s} {status}")
    else:
        print("  None found.")

    # 2. Series analysis
    ds_files = [f for f in kenya_files if f["series"] == "DS-CY"]
    core_files = [f for f in kenya_files if f["series"] == "CY-Core"]

    print("\n── Two distinct V-Dem release series ───────────────────────────")
    print(f"\n  DS-CY (Dataset, v5-v7.1): {len(ds_files)} files")
    print(f"    Columns range: {min(f['num_cols'] for f in ds_files)}-{max(f['num_cols'] for f in ds_files)}")
    print(f"    These are the older, larger dataset format with more variables.")

    print(f"\n  CY-Core (Country-Year Core, v8-v16): {len(core_files)} files")
    print(f"    Columns range: {min(f['num_cols'] for f in core_files)}-{max(f['num_cols'] for f in core_files)}")
    print(f"    The current standard release format. More focused variable set.")

    # 3. Column evolution
    print("\n── Column evolution within CY-Core (v8 → v16) ─────────────────")
    if len(core_files) >= 2:
        first = min(core_files, key=lambda x: float(x["version"]))
        last = max(core_files, key=lambda x: float(x["version"]))
        common = first["columns"] & last["columns"]
        only_first = first["columns"] - last["columns"]
        only_last = last["columns"] - first["columns"]
        print(f"  v{first['version']} has {first['num_cols']} cols, v{last['version']} has {last['num_cols']} cols")
        print(f"  Shared columns: {len(common)}")
        print(f"  Removed in v{last['version']}: {len(only_first)} — {sorted(only_first)[:10]}")
        print(f"  Added in v{last['version']}: {len(only_last)} — {sorted(only_last)[:10]}...")

    # 4. Year coverage
    print("\n── Year coverage growth ────────────────────────────────────────")
    for f in sorted(kenya_files, key=lambda x: (x["series"], x["version"])):
        yr = f"{f['year_min']}-{f['year_max']}" if f["year_min"] else "N/A"
        print(f"  v{f['version']:<6s} ({f['series']:<8s}): {yr}  ({f['num_years']} data-years)")

    # 5. ETL indicator availability
    print("\n── ETL indicator availability (from V-DEM.py) ──────────────────")
    for ind in ETL_INDICATORS:
        present_in = [f["version"] for f in kenya_files if ind in f["columns"]]
        absent_in = [f["version"] for f in kenya_files if ind not in f["columns"]]
        status = "ALL" if not absent_in else f"v{', v'.join(present_in)}"
        note = ETL_INDICATOR_NOTES.get(ind, "")
        mark = "OK" if not absent_in else "PARTIAL" if present_in else "MISSING"
        print(f"  [{mark:>7s}] {ind:<25s} — {status}")
        if note:
            print(f"           NOTE: {note}")

    # 6. Derivation method
    print("\n── How Kenya CSVs are derived ──────────────────────────────────")
    print("""
  Each Kenya CSV is a simple row filter of its parent V-Dem release:
    1. Load the full V-Dem CSV (all countries, all years)
    2. Filter: country_text_id == 'KEN'
    3. Keep ALL columns unchanged
    4. Write to "{original_name} Kenya.csv"

  The parent CSV for v10 (V-Dem-CY-Core-v10.csv) is present but empty (0 bytes).
  To populate it, download the full V-Dem v10 dataset from https://v-dem.net/

  Automated derivation command:
    python3 explore_vdem_data.py --derive V-Dem-CY-Core-v10.csv
    """)

    # 7. Key differences between series
    print("── Key differences: DS-CY vs CY-Core ──────────────────────────")
    print("""
  DS-CY (v5-v7.1):
  - Larger variable set (2097-3153 columns)
  - Includes "thick" variants of some indices (e.g. v2x_freexp_thick)
  - Uses simpler metadata columns (codingstart/gapstart/gapend/codingend)
  - Column order: country_name, country_id, country_text_id, year, ...
  - v2x_freexp_altinf NOT available (use v2x_freexp or v2x_freexp_thick)

  CY-Core (v8-v16):
  - More focused variable set (1730-1908 columns)
  - Richer metadata (historical_date, project, histname, gap_index in v16)
  - Column order: country_name, country_text_id, country_id, year, ...
  - v2x_freexp_altinf IS available
  - Stable ~1818 cols from v10-v15, expanded to 1908 in v16

  Both series:
  - Filter by country_text_id = 'KEN'
  - All 13 ETL indicators present EXCEPT v2x_legcon (should be v2xlg_legcon)
  - Year coverage starts at 1900, grows by ~1 year per release
    """)

    # 8. Recommendations
    print("── Recommendations for V-DEM.py ETL ────────────────────────────")
    print("""
  1. FIX: Change 'v2x_legcon' to 'v2xlg_legcon' in VDEM_INDICATORS
     (currently always NULL due to column name mismatch)

  2. CONSIDER: For DS-CY (v5-v7.1), map 'v2x_freexp' or 'v2x_freexp_thick'
     to the freedom_expression field, since v2x_freexp_altinf doesn't exist

  3. ADD: An extract_kenya_csv() utility (included in this script) to
     auto-derive Kenya CSVs from any new parent V-Dem release

  4. NOTE: The parent V-Dem-CY-Core-v10.csv is empty (0 bytes). Either
     populate it with the real data or remove it to avoid confusion.
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--derive":
        if len(sys.argv) < 3:
            print("Usage: python3 explore_vdem_data.py --derive <parent_csv>")
            sys.exit(1)
        auto_derive_kenya_csv(sys.argv[2])
    else:
        main()
