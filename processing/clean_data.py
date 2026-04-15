from __future__ import annotations

import numpy as np
import pandas as pd

STANDARD_SCHEMA = [
    "SourceWebsite",
    "City",
    "Country",
    "ScrapeDateTime",
    "Temperature_C",
    "FeelsLike_C",
    "Humidity_%",
    "WindSpeed_kmh",
    "Condition",
]

CRITICAL_COLUMNS = [
    "SourceWebsite",
    "City",
    "Country",
    "ScrapeDateTime",
    "Temperature_C",
]


def ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns exist and return dataframe in standard order."""
    out = df.copy()
    for col in STANDARD_SCHEMA:
        if col not in out.columns:
            out[col] = np.nan
    return out[STANDARD_SCHEMA]


def _normalize_source_name(value: object) -> object:
    """Normalize source naming variants to one standard label."""
    if pd.isna(value):
        return pd.NA

    raw = str(value).strip()
    if not raw:
        return pd.NA

    key = raw.lower().replace("-", " ")
    key = " ".join(key.split())

    if key in {"openmeteo", "open meteo"}:
        return "Open-Meteo"
    if key in {"timeanddate", "time and date", "time & date"}:
        return "TimeAndDate"
    if key in {"wunderground", "weather underground", "weatherunderground"}:
        return "WeatherUnderground"

    return raw


def _normalize_country(value: object) -> object:
    """Clean country with simple, consistent formatting."""
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA

    if len(text) <= 3:
        return text.upper()
    return text.title()


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text fields and normalize case/style."""
    out = df.copy()

    out["SourceWebsite"] = out["SourceWebsite"].apply(_normalize_source_name)

    out["City"] = (
        out["City"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .str.title()
    )

    out["Country"] = out["Country"].apply(_normalize_country)

    out["Condition"] = (
        out["Condition"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .str.lower()
    )

    return out


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime and numeric columns safely."""
    out = df.copy()

    try:
        out["ScrapeDateTime"] = pd.to_datetime(
            out["ScrapeDateTime"],
            errors="coerce",
            utc=True,
            format="mixed",
        )
    except TypeError:
        out["ScrapeDateTime"] = pd.to_datetime(
            out["ScrapeDateTime"],
            errors="coerce",
            utc=True,
        )

    numeric_cols = ["Temperature_C", "FeelsLike_C", "Humidity_%", "WindSpeed_kmh"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def fix_temperature_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert suspicious Fahrenheit-like values only for WeatherUnderground."""
    out = df.copy()

    source_mask = out["SourceWebsite"].eq("WeatherUnderground")

    for col in ["Temperature_C", "FeelsLike_C"]:
        if col not in out.columns:
            continue

        # Only convert values that are very likely Fahrenheit (safer threshold >55).
        fahrenheit_like = source_mask & out[col].notna() & (out[col] > 55)
        out.loc[fahrenheit_like, col] = (out.loc[fahrenheit_like, col] - 32) * 5.0 / 9.0

    return out


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Apply numeric sanity ranges and set invalid values to NaN."""
    out = df.copy()

    out.loc[(out["Humidity_%"] < 0) | (out["Humidity_%"] > 100), "Humidity_%"] = np.nan
    out.loc[out["WindSpeed_kmh"] < 0, "WindSpeed_kmh"] = np.nan

    for col in ["Temperature_C", "FeelsLike_C"]:
        out.loc[(out[col] < -50) | (out[col] > 60), col] = np.nan

    return out


def drop_critical_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing required core fields."""
    return df.dropna(subset=CRITICAL_COLUMNS)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ONLY exact duplicate rows."""
    return df.drop_duplicates(keep="last")


def fill_remaining_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill non-critical numeric gaps using progressively broader median fallbacks."""
    out = df.copy()
    fill_cols = ["FeelsLike_C", "Humidity_%", "WindSpeed_kmh"]

    for col in fill_cols:
        source_city_medians = out.groupby(["SourceWebsite", "City"])[col].transform("median")
        city_medians = out.groupby("City")[col].transform("median")
        source_medians = out.groupby("SourceWebsite")[col].transform("median")
        global_median = out[col].median()

        out[col] = out[col].fillna(source_city_medians)
        out[col] = out[col].fillna(city_medians)
        out[col] = out[col].fillna(source_medians)
        out[col] = out[col].fillna(global_median)

    return out


def summarize_cleaning(before_df: pd.DataFrame, after_df: pd.DataFrame, source_name: str) -> None:
    """Print source-level cleaning summary."""
    before_rows = len(before_df)
    after_rows = len(after_df)
    removed = before_rows - after_rows

    print(f"\n[{source_name}] Cleaning Summary")
    print(f"  Rows before cleaning: {before_rows}")
    print(f"  Rows after cleaning:  {after_rows}")
    print(f"  Rows removed:         {removed}")

    missing_counts = after_df[STANDARD_SCHEMA].isna().sum()
    print("  Missing values after cleaning:")
    for col, count in missing_counts.items():
        print(f"    - {col}: {int(count)}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run full reusable cleaning pipeline on one source dataframe."""
    out = ensure_standard_columns(df)
    out = clean_text_columns(out)
    out = convert_types(out)
    out = fix_temperature_units(out)
    out = validate_ranges(out)
    out = drop_critical_missing(out)
    out = remove_duplicates(out)
    out = fill_remaining_missing(out)
    out = out.sort_values(by=["ScrapeDateTime", "SourceWebsite", "City", "Country"]).reset_index(drop=True)
    return out
