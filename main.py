# main.py - Cleaned and Optimized
import logging
import time
from typing import Dict, List
from datetime import datetime

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

import config
import database
from scrapers.openmeteo_scraper import scrape_openmeteo
from scrapers.timeanddate_scraper import scrape_timeanddate
from scrapers.wunderground_scraper import scrape_wunderground
from utils import (
    append_raw_rows,
    count_processed_rows,
    create_session,
    ensure_directories,
    load_and_merge_raw_files,
    load_existing_rows,
    normalize_rows,
    replace_processed_outputs,
    setup_logging,
    update_summary_report,
)

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
DELAY_BETWEEN_CITIES = 2
DELAY_BETWEEN_SOURCES = 30


def load_cities() -> List[Dict]:
    """Load cities from CSV or use hardcoded fallback list"""
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
                logger.info(f"Loaded {len(cities)} cities from {config.CITIES_CSV}")
                return cities
        except Exception as e:
            logger.warning(f"Could not load cities.csv: {e}")
    
    # Hardcoded fallback cities
    cities = [
        {"City": "Beirut", "Country": "LB"}, {"City": "New York", "Country": "US"},
        {"City": "Los Angeles", "Country": "US"}, {"City": "Chicago", "Country": "US"},
        {"City": "Toronto", "Country": "CA"}, {"City": "Mexico City", "Country": "MX"},
        {"City": "Sao Paulo", "Country": "BR"}, {"City": "Buenos Aires", "Country": "AR"},
        {"City": "London", "Country": "GB"}, {"City": "Paris", "Country": "FR"},
        {"City": "Berlin", "Country": "DE"}, {"City": "Madrid", "Country": "ES"},
        {"City": "Rome", "Country": "IT"}, {"City": "Cairo", "Country": "EG"},
        {"City": "Nairobi", "Country": "KE"}, {"City": "Johannesburg", "Country": "ZA"},
        {"City": "Dubai", "Country": "AE"}, {"City": "Mumbai", "Country": "IN"},
        {"City": "Tokyo", "Country": "JP"}, {"City": "Seoul", "Country": "KR"},
        {"City": "Singapore", "Country": "SG"}, {"City": "Sydney", "Country": "AU"},
        {"City": "Melbourne", "Country": "AU"}, {"City": "Auckland", "Country": "NZ"},
    ]
    logger.info(f"Using hardcoded list of {len(cities)} cities")
    return cities


def count_rows_by_source() -> Dict[str, int]:
    """Count rows for each source in processed data"""
    source_counts = {"Open-Meteo": 0, "TimeAndDate": 0, "WeatherUnderground": 0}
    
    if config.WEATHER_CSV.exists():
        try:
            df = pd.read_csv(config.WEATHER_CSV)
            for source in source_counts:
                source_counts[source] = len(df[df['SourceWebsite'] == source])
        except Exception as e:
            logger.error(f"Error counting rows by source: {e}")
    
    return source_counts


def collect_for_source(cities: List[Dict], scraper_func, history_days: int, source_name: str) -> int:
    """Collect data for all cities using a specific scraper with retry logic"""
    logger.info(f"\n{'='*60}\n{source_name}: Collecting {history_days} days of data\n{'='*60}")
    
    session = create_session()
    total_rows = 0
    successful = []
    failed = []
    
    for i, city in enumerate(cities, 1):
        city_name = city.get("City")
        logger.info(f"[{i}/{len(cities)}] {city_name}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                rows = scraper_func(session, city, history_days, pass_index=0)
                
                if rows:
                    total_rows += len(rows)
                    successful.append(city_name)
                    
                    # Save to raw file
                    raw_files = {
                        scrape_openmeteo: config.OPENMETEO_RAW_CSV,
                        scrape_timeanddate: config.TIMEANDDATE_RAW_CSV,
                        scrape_wunderground: config.WUNDERGROUND_RAW_CSV,
                    }
                    if raw_file := raw_files.get(scraper_func):
                        append_raw_rows(raw_file, rows)
                    
                    temp = rows[0].get('Temperature_C', 'N/A')
                    logger.info(f"  ✓ {len(rows)} rows (Current: {temp}°C)")
                    break
                    
                else:
                    logger.warning(f"  ✗ No data (attempt {attempt + 1}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(5)
                        
            except Exception as e:
                logger.error(f"  ✗ Error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    logger.info(f"    Retrying in 10s...")
                    time.sleep(10)
                else:
                    failed.append(city_name)
        
        time.sleep(DELAY_BETWEEN_CITIES)
    
    # Summary
    logger.info(f"\n{source_name} Summary: {total_rows} rows from {len(successful)}/{len(cities)} cities")
    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")
    
    return total_rows


def run_scheduled_batch(cities: List[Dict]) -> int:
    """Run batch for scheduled job - collects current weather from all sources"""
    total = 0
    
    # Open-Meteo (with 1 day history for continuity)
    total += collect_for_source(cities, scrape_openmeteo, 1, "Open-Meteo")
    time.sleep(DELAY_BETWEEN_SOURCES)
    
    # TimeAndDate
    total += collect_for_source(cities, scrape_timeanddate, 0, "TimeAndDate")
    time.sleep(DELAY_BETWEEN_SOURCES)
    
    # WeatherUnderground
    total += collect_for_source(cities, scrape_wunderground, 0, "WeatherUnderground")
    
    return total


def scheduled_job(cities: List[Dict], scheduler: BlockingScheduler) -> None:
    """Scheduled job: scrape current weather and auto-merge"""
    before = count_rows_by_source()
    logger.info(f"\n{'='*80}\nSCHEDULED SCRAPE STARTED - {datetime.now()}\nBefore: {before}\n{'='*80}")
    
    # Scrape data
    run_scheduled_batch(cities)
    
    # Auto-merge
    logger.info("\n🔄 Auto-merging raw data...")
    try:
        written = merge_raw_data()
        logger.info(f"✅ Merge complete: {written} rows")
    except Exception as e:
        logger.error(f"❌ Merge failed: {e}")
    
    after = count_rows_by_source()
    logger.info(f"\n{'='*80}\nSCHEDULED SCRAPE COMPLETED\nAfter: {after}\n{'='*80}\n")


def merge_raw_data() -> int:
    """Load, normalize, deduplicate, and save all raw data"""
    # Load raw data
    raw_rows = load_and_merge_raw_files()
    logger.info(f"Loaded {len(raw_rows)} raw rows")
    
    # Load existing processed data
    existing = []
    if config.WEATHER_CSV.exists():
        try:
            existing = pd.read_csv(config.WEATHER_CSV).to_dict(orient='records')
            logger.info(f"Loaded {len(existing)} existing rows")
        except Exception as e:
            logger.error(f"Error loading existing file: {e}")
    
    # Combine and normalize
    all_rows = existing + raw_rows
    normalized = normalize_rows(all_rows)
    logger.info(f"Normalized: {len(normalized)} rows")
    
    if not normalized:
        return 0
    
    # Deduplicate
    df = pd.DataFrame(normalized)
    key_cols = [c for c in ['SourceWebsite', 'City', 'Date', 'ScrapeDateTime'] if c in df.columns]
    if key_cols:
        before = len(df)
        df = df.drop_duplicates(subset=key_cols, keep='last')
        logger.info(f"Deduplicated: {before} -> {len(df)} rows")
    
    # Save
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.WEATHER_CSV, index=False)
    logger.info(f"Saved {len(df)} rows to {config.WEATHER_CSV}")
    
    # Update database and summary
    try:
        database.insert_rows(df.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Database error: {e}")
    update_summary_report()
    
    return len(df)


def main():
    """Main entry point"""
    setup_logging()
    ensure_directories()
    database.init_db()
    
    cities = load_cities()
    logger.info(f"Loaded {len(cities)} cities")
    
    # Initial merge
    merge_raw_data()
    
    # Show current counts
    counts = count_rows_by_source()
    logger.info("\nCurrent data counts:")
    for source, count in counts.items():
        logger.info(f"  {source}: {count} rows")
    
    # Start scheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(
        scheduled_job,
        "interval",
        hours=config.SCRAPE_INTERVAL_HOURS,
        kwargs={"cities": cities, "scheduler": scheduler},
        max_instances=1,
        coalesce=True,
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SCHEDULER STARTED - Every {config.SCRAPE_INTERVAL_HOURS} hours")
    logger.info("Auto-merge ENABLED | Press Ctrl+C to stop")
    logger.info(f"{'='*80}\n")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scraper stopped by user.")


if __name__ == "__main__":
    main()