"""
ETL Step 4g — V-Dem democracy indicators → kenya_democracy_vdem.
Loads 13 Kenya democracy, governance, and civil-society indicators (1963–present).
See ETL_GUIDE.txt (FILE 8) for full documentation.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

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

KENYA_VDEM_ID: str  = "KEN"
KENYA_MIN_YEAR: int = 1963
KENYA_MAX_YEAR: int = datetime.now().year - 1

VDEM_FILE_PATH = Path(os.environ.get("VDEM_FILE_PATH", "data/vdem/V-Dem-CY-Core.csv"))

# ---------------------------------------------------------------------------
# Indicator definitions: (vdem_source_col, db_col, low_col, high_col)
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

_VDEM_SOURCE_COLS: list[str] = (
    ["year", "country_text_id"]
    + [t[0] for t in VDEM_INDICATORS]
    + [t[2] for t in VDEM_INDICATORS]
    + [t[3] for t in VDEM_INDICATORS]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(v: Any) -> Any:
    """Coerce pandas NA / NaN to Python None and numpy scalars to native types."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    # Convert numpy scalars to native Python types for SQLAlchemy compatibility
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _vdem_version() -> str:
    """Extract version number from filename (e.g. 'Core-v14.csv' → '14')."""
    m = re.search(r"v(\d+)", VDEM_FILE_PATH.name, re.IGNORECASE)
    return m.group(1) if m else "current"

# ---------------------------------------------------------------------------
# Extract — load file
# ---------------------------------------------------------------------------

def load_vdem_file(path: Path) -> pd.DataFrame:
    """Load V-Dem CSV (usecols for memory), filter to Kenya."""
    if not path.exists():
        log.error(
            "V-Dem file not found: %s\n"
            "Download from https://v-dem.net/data/the-v-dem-dataset/ "
            "and set VDEM_FILE_PATH.",
            path,
        )
        sys.exit(1)

    log.info("Loading V-Dem file: %s  (may take a moment for large CSV)", path)

    try:
        header_df = pd.read_csv(path, nrows=0)
        available = set(header_df.columns)

        missing = [c for c in _VDEM_SOURCE_COLS if c not in available]
        present = [c for c in _VDEM_SOURCE_COLS if c in available]

        if missing:
            log.warning(
                "%d V-Dem source columns absent from file — will store as NULL: %s",
                len(missing), missing,
            )

        df = pd.read_csv(path, usecols=present, low_memory=False)

    except Exception as exc:
        log.error("Failed to load V-Dem file: %s", exc)
        sys.exit(1)

    if "country_text_id" not in df.columns:
        log.error("'country_text_id' column not found in V-Dem file.")
        sys.exit(1)

    kenya = df[df["country_text_id"].str.upper().eq(KENYA_VDEM_ID)].copy()

    if kenya.empty:
        log.error(
            "No rows with country_text_id='%s' in V-Dem file.", KENYA_VDEM_ID
        )
        sys.exit(1)

    # Add absent columns as None
    for col in missing:
        kenya[col] = None

    log.info(
        "Loaded %d Kenya V-Dem rows (%d–%d).",
        len(kenya),
        int(kenya["year"].min()),
        int(kenya["year"].max()),
    )
    return kenya

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def extract_kenya_vdem(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename V-Dem columns to DB names, round, filter to valid year window."""
    raw = raw.copy()
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw = raw.dropna(subset=["year"])
    raw["year"] = raw["year"].astype(int)
    raw = raw[
        (raw["year"] >= KENYA_MIN_YEAR) &
        (raw["year"] <= KENYA_MAX_YEAR)
    ].copy()

    out = pd.DataFrame({"year": raw["year"].values})

    for src_col, db_col, low_col, high_col in VDEM_INDICATORS:
        out[db_col] = (
            pd.to_numeric(raw[src_col].values, errors="coerce").round(4)
            if src_col in raw.columns else None
        )
        out[f"{db_col}_low"] = (
            pd.to_numeric(raw[low_col].values, errors="coerce").round(4)
            if low_col in raw.columns else None
        )
        out[f"{db_col}_high"] = (
            pd.to_numeric(raw[high_col].values, errors="coerce").round(4)
            if high_col in raw.columns else None
        )

    out = out.sort_values("year").reset_index(drop=True)

    poly_min = out["polyarchy"].dropna().min() if "polyarchy" in out else float("nan")
    poly_max = out["polyarchy"].dropna().max() if "polyarchy" in out else float("nan")
    log.info(
        "Extracted %d Kenya V-Dem years (%d–%d). Polyarchy range: %.3f–%.3f.",
        len(out),
        int(out["year"].min()),
        int(out["year"].max()),
        poly_min,
        poly_max,
    )
    return out

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

_VDEM_UPSERT = text("""
INSERT INTO kenya_democracy_vdem (
    year,
    polyarchy,              polyarchy_low,              polyarchy_high,
    libdem,                 libdem_low,                 libdem_high,
    partipdem,              partipdem_low,              partipdem_high,
    delibdem,               delibdem_low,               delibdem_high,
    egaldem,                egaldem_low,                egaldem_high,
    elections_free_fair,    elections_free_fair_low,    elections_free_fair_high,
    judicial_constraints,   judicial_constraints_low,   judicial_constraints_high,
    legislative_constraints, legislative_constraints_low, legislative_constraints_high,
    corruption_index,       corruption_index_low,       corruption_index_high,
    public_corruption,      public_corruption_low,      public_corruption_high,
    exec_corruption,        exec_corruption_low,        exec_corruption_high,
    civil_society,          civil_society_low,          civil_society_high,
    freedom_expression,     freedom_expression_low,     freedom_expression_high,
    data_source, etl_run_at
)
VALUES (
    :year,
    :polyarchy,             :polyarchy_low,             :polyarchy_high,
    :libdem,                :libdem_low,                :libdem_high,
    :partipdem,             :partipdem_low,             :partipdem_high,
    :delibdem,              :delibdem_low,              :delibdem_high,
    :egaldem,               :egaldem_low,               :egaldem_high,
    :elections_free_fair,   :elections_free_fair_low,   :elections_free_fair_high,
    :judicial_constraints,  :judicial_constraints_low,  :judicial_constraints_high,
    :legislative_constraints, :legislative_constraints_low, :legislative_constraints_high,
    :corruption_index,      :corruption_index_low,      :corruption_index_high,
    :public_corruption,     :public_corruption_low,     :public_corruption_high,
    :exec_corruption,       :exec_corruption_low,       :exec_corruption_high,
    :civil_society,         :civil_society_low,         :civil_society_high,
    :freedom_expression,    :freedom_expression_low,    :freedom_expression_high,
    :data_source, :etl_run_at
)
ON CONFLICT (year) DO UPDATE SET
    polyarchy                    = EXCLUDED.polyarchy,
    polyarchy_low                = EXCLUDED.polyarchy_low,
    polyarchy_high               = EXCLUDED.polyarchy_high,
    libdem                       = EXCLUDED.libdem,
    libdem_low                   = EXCLUDED.libdem_low,
    libdem_high                  = EXCLUDED.libdem_high,
    partipdem                    = EXCLUDED.partipdem,
    partipdem_low                = EXCLUDED.partipdem_low,
    partipdem_high               = EXCLUDED.partipdem_high,
    delibdem                     = EXCLUDED.delibdem,
    delibdem_low                 = EXCLUDED.delibdem_low,
    delibdem_high                = EXCLUDED.delibdem_high,
    egaldem                      = EXCLUDED.egaldem,
    egaldem_low                  = EXCLUDED.egaldem_low,
    egaldem_high                 = EXCLUDED.egaldem_high,
    elections_free_fair          = EXCLUDED.elections_free_fair,
    elections_free_fair_low      = EXCLUDED.elections_free_fair_low,
    elections_free_fair_high     = EXCLUDED.elections_free_fair_high,
    judicial_constraints         = EXCLUDED.judicial_constraints,
    judicial_constraints_low     = EXCLUDED.judicial_constraints_low,
    judicial_constraints_high    = EXCLUDED.judicial_constraints_high,
    legislative_constraints      = EXCLUDED.legislative_constraints,
    legislative_constraints_low  = EXCLUDED.legislative_constraints_low,
    legislative_constraints_high = EXCLUDED.legislative_constraints_high,
    corruption_index             = EXCLUDED.corruption_index,
    corruption_index_low         = EXCLUDED.corruption_index_low,
    corruption_index_high        = EXCLUDED.corruption_index_high,
    public_corruption            = EXCLUDED.public_corruption,
    public_corruption_low        = EXCLUDED.public_corruption_low,
    public_corruption_high       = EXCLUDED.public_corruption_high,
    exec_corruption              = EXCLUDED.exec_corruption,
    exec_corruption_low          = EXCLUDED.exec_corruption_low,
    exec_corruption_high         = EXCLUDED.exec_corruption_high,
    civil_society                = EXCLUDED.civil_society,
    civil_society_low            = EXCLUDED.civil_society_low,
    civil_society_high           = EXCLUDED.civil_society_high,
    freedom_expression           = EXCLUDED.freedom_expression,
    freedom_expression_low       = EXCLUDED.freedom_expression_low,
    freedom_expression_high      = EXCLUDED.freedom_expression_high,
    data_source                  = EXCLUDED.data_source,
    etl_run_at                   = EXCLUDED.etl_run_at,
    updated_at                   = NOW()
""")


def _row_to_params(row: pd.Series, run_at: datetime) -> dict[str, Any]:
    p: dict[str, Any] = {
        "year":        int(row["year"]),
        "data_source": f"V-Dem v{_vdem_version()} — Kenya (KEN)",
        "etl_run_at":  run_at,
    }
    for _, db_col, _, _ in VDEM_INDICATORS:
        p[db_col]              = clean(row.get(db_col))
        p[f"{db_col}_low"]     = clean(row.get(f"{db_col}_low"))
        p[f"{db_col}_high"]    = clean(row.get(f"{db_col}_high"))
    return p


def upsert_vdem(df: pd.DataFrame, engine, run_at: datetime) -> None:
    n = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(_VDEM_UPSERT, _row_to_params(row, run_at))
            n += 1
    log.info("Upserted %d rows into kenya_democracy_vdem.", n)

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        log.error("SUPABASE_DB_URL is not set.")
        sys.exit(1)

    log.info(
        "Kenya ETL Step 4g — V-Dem v%s  country=%s  file=%s",
        _vdem_version(), KENYA_VDEM_ID, VDEM_FILE_PATH,
    )
    engine = create_engine(db_url, future=True)
    run_at = datetime.now(tz=timezone.utc)

    raw = load_vdem_file(VDEM_FILE_PATH)
    df  = extract_kenya_vdem(raw)
    upsert_vdem(df, engine, run_at)

    log.info("Kenya ETL Step 4g complete — %d years upserted.", len(df))


if __name__ == "__main__":
    main()