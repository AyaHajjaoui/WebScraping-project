from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from processing.clean_data import (
        STANDARD_SCHEMA,
        clean_dataframe,
        remove_duplicates,
        summarize_cleaning,
    )
except ModuleNotFoundError:
    from clean_data import (  # type: ignore
        STANDARD_SCHEMA,
        clean_dataframe,
        remove_duplicates,
        summarize_cleaning,
    )

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "weather_data.csv"

RAW_FILES = {
    "openmeteo_raw.csv": "Open-Meteo",
    "timeanddate_raw.csv": "TimeAndDate",
    "wunderground_raw.csv": "WeatherUnderground",
}


def load_raw_file(filepath: Path, source_name: str) -> pd.DataFrame | None:
    """Load one raw file safely. Missing files are skipped with a message."""
    if not filepath.exists():
        print(f"[SKIP] Missing file: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        print(f"[SKIP] Could not read {filepath}: {exc}")
        return None

    # Ensure source label is consistent for this file.
    df["SourceWebsite"] = source_name
    print(f"[LOAD] {source_name}: {len(df)} rows from {filepath.name}")
    return df


def preprocess_all() -> None:
    """Load raw files, clean each source, combine, dedupe, sort, and save."""
    cleaned_frames: list[pd.DataFrame] = []

    print("=" * 70)
    print("Weather Data Preprocessing Pipeline")
    print("=" * 70)

    for filename, source_name in RAW_FILES.items():
        filepath = RAW_DIR / filename
        raw_df = load_raw_file(filepath, source_name)

        if raw_df is None:
            continue

        cleaned_df = clean_dataframe(raw_df)
        summarize_cleaning(raw_df, cleaned_df, source_name)
        cleaned_frames.append(cleaned_df)

    if not cleaned_frames:
        print("\n[STOP] No valid raw files were loaded. Nothing to process.")
        return

    combined = pd.concat(cleaned_frames, ignore_index=True)

    before_final_dedupe = len(combined)
    combined = remove_duplicates(combined)
    removed_final = before_final_dedupe - len(combined)

    combined = combined[STANDARD_SCHEMA]
    combined = combined.sort_values(
        by=["ScrapeDateTime", "SourceWebsite", "City", "Country"]
    ).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("Final Output Summary")
    print("=" * 70)
    print(f"Saved output path: {OUTPUT_FILE}")
    print(f"Final shape: {combined.shape}")
    print(f"Final duplicate rows removed after combining: {removed_final}")

    rows_per_source = combined["SourceWebsite"].value_counts(dropna=False).sort_index()
    print("Rows per source:")
    for source, count in rows_per_source.items():
        print(f"  - {source}: {int(count)}")

    if combined["ScrapeDateTime"].notna().any():
        date_min = combined["ScrapeDateTime"].min()
        date_max = combined["ScrapeDateTime"].max()
        print(f"Date range: {date_min} to {date_max}")
    else:
        print("Date range: not available (all ScrapeDateTime values are missing)")


if __name__ == "__main__":
    preprocess_all()
