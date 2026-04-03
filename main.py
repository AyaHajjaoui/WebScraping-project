# main.py
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from apscheduler.schedulers.blocking import BlockingScheduler

import config
import database
from scrapers.openmeteo_scraper import scrape_openmeteo
from scrapers.timeanddate_scraper import scrape_timeanddate
from scrapers.wunderground_scraper import scrape_wunderground
from utils import (
    append_raw_rows,
    create_session,
    ensure_directories,
    load_and_merge_raw_files,
    setup_logging,
    update_summary_report,
)

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
DELAY_BETWEEN_CITIES = 2
DELAY_BETWEEN_SOURCES = 30
RETRY_BASE_DELAY_SECONDS = 2
SCHEDULER_RESTART_DELAY_SECONDS = 10


def start_in_background_if_requested() -> bool:
    """
    Placeholder hook for background mode.
    Returns False by default so normal execution continues.
    """
    return False


def run_scraping_batch(cities: List[Dict]) -> int:
    """Run one batch of scraping - collects current weather from all sources."""
    total = 0

    total += collect_for_source(cities, scrape_openmeteo, 0, "Open-Meteo")
    time.sleep(DELAY_BETWEEN_SOURCES)

    total += collect_for_source(cities, scrape_timeanddate, 0, "TimeAndDate")
    time.sleep(DELAY_BETWEEN_SOURCES)

    total += collect_for_source(cities, scrape_wunderground, 0, "WeatherUnderground")
    return total


def load_cities() -> List[Dict]:
    """Load cities from CSV or use hardcoded fallback list."""
    if config.CITIES_CSV.exists():
        try:
            df = pd.read_csv(config.CITIES_CSV)
            cities = []

            for _, row in df.iterrows():
                city = {
                    "City": row.get("City", ""),
                    "Country": row.get("Country", ""),
                    "TimeAndDate URL": row.get("TimeAndDate URL", ""),
                    "WeatherUnderground URL": row.get("WeatherUnderground URL", ""),
                    "Meteostat URL": row.get("Meteostat URL", ""),
                }
                if city["City"]:
                    cities.append(city)

            if cities:
                logger.info("Loaded %s cities from %s", len(cities), config.CITIES_CSV)
                return cities
        except Exception as e:
            logger.warning("Could not load cities.csv: %s", e)

    cities = [
        {"City": "Beirut", "Country": "LB"},
        {"City": "New York", "Country": "US"},
        {"City": "Los Angeles", "Country": "US"},
        {"City": "Chicago", "Country": "US"},
        {"City": "Toronto", "Country": "CA"},
        {"City": "Mexico City", "Country": "MX"},
        {"City": "Sao Paulo", "Country": "BR"},
        {"City": "Buenos Aires", "Country": "AR"},
        {"City": "London", "Country": "GB"},
        {"City": "Paris", "Country": "FR"},
        {"City": "Berlin", "Country": "DE"},
        {"City": "Madrid", "Country": "ES"},
        {"City": "Rome", "Country": "IT"},
        {"City": "Cairo", "Country": "EG"},
        {"City": "Nairobi", "Country": "KE"},
        {"City": "Johannesburg", "Country": "ZA"},
        {"City": "Dubai", "Country": "AE"},
        {"City": "Mumbai", "Country": "IN"},
        {"City": "Tokyo", "Country": "JP"},
        {"City": "Seoul", "Country": "KR"},
        {"City": "Singapore", "Country": "SG"},
        {"City": "Sydney", "Country": "AU"},
        {"City": "Melbourne", "Country": "AU"},
        {"City": "Auckland", "Country": "NZ"},
    ]
    logger.info("Using hardcoded list of %s cities", len(cities))
    return cities


def count_rows_by_source() -> Dict[str, int]:
    """Count rows for each source in processed data."""
    if not config.WEATHER_CSV.exists():
        return {"Open-Meteo": 0, "TimeAndDate": 0, "WeatherUnderground": 0}

    try:
        df = pd.read_csv(config.WEATHER_CSV, low_memory=False)
    except Exception as e:
        logger.error("Error counting rows by source: %s", e)
        return {"Open-Meteo": 0, "TimeAndDate": 0, "WeatherUnderground": 0}

    counts = {}
    for source in ["Open-Meteo", "TimeAndDate", "WeatherUnderground"]:
        counts[source] = int((df.get("SourceWebsite") == source).sum()) if "SourceWebsite" in df.columns else 0
    return counts


def _normalize_scrape_datetime(series: pd.Series) -> pd.Series:
    """Parse mixed datetime formats into UTC timestamps."""
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)

    missing = parsed.isna()
    if missing.any():
        cleaned = (
            series[missing]
            .astype(str)
            .str.strip()
            .str.replace("/", "-", regex=False)
            .str.replace("T", " ", regex=False)
            .str.replace("Z", "+00:00", regex=False)
        )
        try:
            reparsed = pd.to_datetime(cleaned, errors="coerce", utc=True, format="mixed")
        except TypeError:
            reparsed = pd.to_datetime(cleaned, errors="coerce", utc=True)
        parsed.loc[missing] = reparsed

    return parsed


def collect_for_source(cities: List[Dict], scraper_func, history_days: int, source_name: str) -> int:
    """Collect data for all cities using a specific scraper with retry logic."""
    logger.info("\n%s\n%s: Collecting %s days of data\n%s", "=" * 60, source_name, history_days, "=" * 60)

    session = create_session()
    total_rows = 0
    successful = []
    failed = []

    for i, city in enumerate(cities, 1):
        city_name = city.get("City")
        logger.info("[%s/%s] %s...", i, len(cities), city_name)

        city_succeeded = False

        for attempt in range(MAX_RETRIES):
            try:
                rows = scraper_func(session, city, history_days, pass_index=0)

                if rows:
                    total_rows += len(rows)
                    successful.append(city_name)

                    raw_files = {
                        scrape_openmeteo: config.OPENMETEO_RAW_CSV,
                        scrape_timeanddate: config.TIMEANDDATE_RAW_CSV,
                        scrape_wunderground: config.WUNDERGROUND_RAW_CSV,
                    }
                    raw_file = raw_files.get(scraper_func)
                    if raw_file:
                        append_raw_rows(raw_file, rows)

                    temp = rows[0].get("Temperature_C", "N/A")
                    logger.info("  OK %s rows (Current: %s C)", len(rows), temp)
                    city_succeeded = True
                    break

                logger.warning("  No data (attempt %s/%s)", attempt + 1, MAX_RETRIES)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_SECONDS * (attempt + 1))

            except Exception as e:
                logger.error("  Error (attempt %s/%s): %s", attempt + 1, MAX_RETRIES, e)
                if attempt < MAX_RETRIES - 1:
                    is_connection_error = isinstance(
                        e,
                        (
                            requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout,
                            requests.exceptions.ChunkedEncodingError,
                        ),
                    )
                    delay = RETRY_BASE_DELAY_SECONDS * (attempt + 1)
                    if is_connection_error:
                        logger.info("    Connection issue detected. Retrying in %ss...", delay)
                    else:
                        logger.info("    Retrying in %ss...", delay)
                    time.sleep(delay)

        if not city_succeeded:
            failed.append(city_name)

        time.sleep(DELAY_BETWEEN_CITIES)

    logger.info("\n%s Summary: %s rows from %s/%s cities", source_name, total_rows, len(successful), len(cities))
    if failed:
        logger.warning("Failed: %s", ", ".join(failed))

    return total_rows


def run_scheduled_batch(cities: List[Dict]) -> int:
    """Run batch for scheduled job - collects current weather from all sources."""
    total = 0

    total += collect_for_source(cities, scrape_openmeteo, 0, "Open-Meteo")
    time.sleep(DELAY_BETWEEN_SOURCES)

    total += collect_for_source(cities, scrape_timeanddate, 0, "TimeAndDate")
    time.sleep(DELAY_BETWEEN_SOURCES)

    total += collect_for_source(cities, scrape_wunderground, 0, "WeatherUnderground")

    return total


def scheduled_job(cities: List[Dict], scheduler: Optional[BlockingScheduler] = None) -> None:
    """Scheduled job: scrape current weather and auto-merge."""
    before = count_rows_by_source()
    logger.info("\n%s\nSCHEDULED SCRAPE STARTED - %s\nBefore: %s\n%s", "=" * 80, datetime.now(), before, "=" * 80)

    run_scheduled_batch(cities)

    logger.info("\nAuto-merging raw data...")
    try:
        written = merge_raw_data()
        logger.info("Merge complete: %s rows", written)
    except Exception as e:
        logger.exception("Merge failed: %s", e)

    after = count_rows_by_source()
    logger.info("\n%s\nSCHEDULED SCRAPE COMPLETED\nAfter: %s\n%s\n", "=" * 80, after, "=" * 80)


def merge_raw_data() -> int:
    """
    Merge all raw CSV files into processed output.

    Requirements handled:
    - Merge all raw CSV files from data/raw
    - Normalize values and create Date from ScrapeDateTime when missing
    - Dedupe on ['SourceWebsite', 'City', 'Date', 'ScrapeDateTime']
    - Sort by Date before save
    - Update DB and summary report
    """
    raw_rows = load_and_merge_raw_files()
    logger.info("Loaded %s raw rows", len(raw_rows))

    if not raw_rows:
        logger.warning("No raw rows found to merge")
        return 0

    df = pd.DataFrame(raw_rows)
    if df.empty:
        logger.warning("Raw rows produced empty dataframe")
        return 0

    for col in config.STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = None

    for col in ["SourceWebsite", "City", "Country", "Condition", "ScrapeDateTime"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": None, "nan": None, "None": None})

    for col in ["Temperature_C", "FeelsLike_C", "Humidity_%", "WindSpeed_kmh", "Precipitation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    raw_scrape_dt = df["ScrapeDateTime"].astype(str).str.strip()
    blank_scrape_dt = raw_scrape_dt.str.lower().isin({"", "none", "nan"})

    if blank_scrape_dt.any():
        df.loc[blank_scrape_dt, "ScrapeDateTime"] = pd.Timestamp.now(tz="UTC").isoformat()
    df.loc[~blank_scrape_dt, "ScrapeDateTime"] = raw_scrape_dt.loc[~blank_scrape_dt]

    scrape_dt = _normalize_scrape_datetime(df["ScrapeDateTime"])

    if "Date" not in df.columns:
        df["Date"] = scrape_dt.dt.strftime("%Y-%m-%d")
    else:
        date_parsed = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"] = date_parsed.dt.strftime("%Y-%m-%d")
        missing_date = df["Date"].isna()
        df.loc[missing_date, "Date"] = scrape_dt.dt.strftime("%Y-%m-%d").loc[missing_date]

    missing_date = df["Date"].isna()
    if missing_date.any():
        extracted = df.loc[missing_date, "ScrapeDateTime"].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
        df.loc[missing_date, "Date"] = extracted

    dedupe_cols = ["SourceWebsite", "City", "Date", "ScrapeDateTime"]
    before = len(df)
    df = df.drop_duplicates(subset=dedupe_cols, keep="last")
    logger.info("Deduplicated: %s -> %s rows", before, len(df))

    sort_date = pd.to_datetime(df["Date"], errors="coerce")
    sort_scrape_dt = _normalize_scrape_datetime(df["ScrapeDateTime"])
    df = (
        df.assign(_sort_date=sort_date, _sort_scrape_dt=sort_scrape_dt)
        .sort_values(by=["_sort_date", "SourceWebsite", "City", "_sort_scrape_dt"], ascending=True)
        .drop(columns=["_sort_date", "_sort_scrape_dt"])
    )

    output_cols = ["Date"] + config.STANDARD_COLUMNS
    for col in output_cols:
        if col not in df.columns:
            df[col] = None
    df = df[output_cols]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.WEATHER_CSV, index=False)
    logger.info("Saved %s rows to %s", len(df), config.WEATHER_CSV)

    try:
        df.to_excel(config.WEATHER_XLSX, index=False)
        logger.info("Saved %s rows to %s", len(df), config.WEATHER_XLSX)
    except Exception as e:
        logger.warning("Could not save Excel output: %s", e)

    try:
        inserted = database.insert_rows(df.to_dict(orient="records"))
        logger.info("Database updated. Inserted %s new rows.", inserted)
    except Exception as e:
        logger.error("Database error: %s", e)

    update_summary_report()
    return len(df)


def initialize_app() -> List[Dict]:
    """Shared startup logic."""
    setup_logging()
    ensure_directories()
    database.init_db()

    cities = load_cities()
    logger.info("Loaded %s cities", len(cities))

    try:
        merge_raw_data()
    except Exception as e:
        logger.exception("Initial merge failed: %s", e)

    counts = count_rows_by_source()
    logger.info("\nCurrent data counts:")
    for source, count in counts.items():
        logger.info("  %s: %s rows", source, count)

    return cities


def run_once() -> None:
    """Run scraper one time only, then exit. Best for GitHub Actions."""
    cities = initialize_app()

    try:
        scheduled_job(cities=cities, scheduler=None)
    except Exception as e:
        logger.exception("Single-run job failed: %s", e)


def run_scheduler_forever() -> None:
    """Run scraper continuously with APScheduler. Best for local machine/server."""
    cities = initialize_app()

    logger.info("\n%s", "=" * 80)
    logger.info("SCHEDULER STARTING - Every %s hours", config.SCRAPE_INTERVAL_HOURS)
    logger.info("Auto-merge ENABLED | Press Ctrl+C to stop")
    logger.info("%s\n", "=" * 80)

    while True:
        scheduler = BlockingScheduler()
        scheduler.add_job(
            scheduled_job,
            "interval",
            hours=config.SCRAPE_INTERVAL_HOURS,
            kwargs={"cities": cities, "scheduler": scheduler},
            max_instances=1,
            coalesce=True,
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scraper stopped by user.")
            break
        except Exception as e:
            logger.exception(
                "Scheduler crashed unexpectedly (%s). Restarting in %ss...",
                e,
                SCHEDULER_RESTART_DELAY_SECONDS,
            )
            time.sleep(SCHEDULER_RESTART_DELAY_SECONDS)


def main() -> None:
    run_once()


if __name__ == "__main__":
    main()