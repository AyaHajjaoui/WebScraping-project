# main.py (fixed imports)
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
from scrapers.wunderground_scraper import scrape_wunderground  # Fixed: wunderground, not underground
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
    write_processed_outputs,
)

logger = logging.getLogger(__name__)


def load_cities() -> List[Dict]:
    """Load cities from CSV or use hardcoded list"""
    
    # Try to load from CSV
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
    
    # Fallback to hardcoded list of 24 cities
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
    logger.info(f"Using hardcoded list of {len(cities)} cities")
    return cities


def count_rows_by_source() -> Dict[str, int]:
    """Count rows for each source in processed data"""
    source_counts = {
        "Open-Meteo": 0,
        "TimeAndDate": 0,
        "WeatherUnderground": 0
    }
    
    if config.WEATHER_CSV.exists():
        try:
            df = pd.read_csv(config.WEATHER_CSV)
            for source in source_counts.keys():
                source_counts[source] = len(df[df['SourceWebsite'] == source])
        except Exception as e:
            logger.error(f"Error counting rows by source: {e}")
    
    return source_counts


def collect_for_source(cities: List[Dict], source_name: str, scraper_func, target_rows: int = 2000) -> int:
    """
    Collect data for a specific source until target rows are reached
    
    Args:
        cities: List of city dictionaries
        source_name: Name of the source (for logging)
        scraper_func: Scraper function to use
        target_rows: Minimum rows to collect (default: 2000)
    
    Returns:
        Total rows collected for this source
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"COLLECTING DATA FOR {source_name.upper()}")
    logger.info(f"{'='*80}")
    
    # Check current rows
    current_counts = count_rows_by_source()
    current_rows = current_counts.get(source_name, 0)
    
    logger.info(f"Current {source_name} rows: {current_rows}")
    
    if current_rows >= target_rows:
        logger.info(f"✓ Already have {current_rows} rows, which meets the target of {target_rows}!")
        return current_rows
    
    needed_rows = target_rows - current_rows
    logger.info(f"Need {needed_rows} more rows to reach target")
    
    # Calculate how many days of history we need
    # Each city contributes (1 + history_days * 24) rows
    # For 24 cities, total rows = 24 * (1 + days * 24)
    
    days_needed = max(1, int(((needed_rows / len(cities)) - 1) / 24) + 1)
    logger.info(f"Will collect {days_needed} days of historical data")
    
    total_collected = 0
    
    # Pass 1: Current weather only
    if current_rows < target_rows:
        logger.info(f"\n📊 Pass 1: Collecting current weather from {source_name}...")
        rows = collect_historical_batch_for_source(cities, scraper_func, history_days=0, pass_name=f"{source_name} Pass 1")
        total_collected += rows
        current_rows = count_rows_by_source().get(source_name, 0)
        
        if current_rows >= target_rows:
            logger.info(f"✓ Target reached after Pass 1! {source_name} rows: {current_rows}")
            return current_rows
    
    # Pass 2: 1 day of history
    if current_rows < target_rows:
        logger.info(f"\n📊 Pass 2: Collecting 1 day of historical data from {source_name}...")
        rows = collect_historical_batch_for_source(cities, scraper_func, history_days=1, pass_name=f"{source_name} Pass 2")
        total_collected += rows
        current_rows = count_rows_by_source().get(source_name, 0)
        
        if current_rows >= target_rows:
            logger.info(f"✓ Target reached after Pass 2! {source_name} rows: {current_rows}")
            return current_rows
    
    # Pass 3: Additional 2 days (total 3 days)
    if current_rows < target_rows:
        logger.info(f"\n📊 Pass 3: Collecting 2 more days of historical data from {source_name}...")
        rows = collect_historical_batch_for_source(cities, scraper_func, history_days=2, pass_name=f"{source_name} Pass 3")
        total_collected += rows
        current_rows = count_rows_by_source().get(source_name, 0)
        
        if current_rows >= target_rows:
            logger.info(f"✓ Target reached after Pass 3! {source_name} rows: {current_rows}")
            return current_rows
    
    # Pass 4: Additional 2 days (total 5 days)
    if current_rows < target_rows:
        logger.info(f"\n📊 Pass 4: Collecting 2 more days of historical data from {source_name}...")
        rows = collect_historical_batch_for_source(cities, scraper_func, history_days=2, pass_name=f"{source_name} Pass 4")
        total_collected += rows
        current_rows = count_rows_by_source().get(source_name, 0)
        
        if current_rows >= target_rows:
            logger.info(f"✓ Target reached after Pass 4! {source_name} rows: {current_rows}")
            return current_rows
    
    # Pass 5: Additional 3 days (total 8 days)
    if current_rows < target_rows:
        logger.info(f"\n📊 Pass 5: Collecting 3 more days of historical data from {source_name}...")
        rows = collect_historical_batch_for_source(cities, scraper_func, history_days=3, pass_name=f"{source_name} Pass 5")
        total_collected += rows
        current_rows = count_rows_by_source().get(source_name, 0)
    
    # Final check
    logger.info(f"\n{source_name} Collection Summary:")
    logger.info(f"  Total collected in this session: {total_collected}")
    logger.info(f"  Total rows now: {current_rows}")
    
    if current_rows >= target_rows:
        logger.info(f"  ✅ SUCCESS! Achieved {current_rows} rows, exceeding target of {target_rows}")
    else:
        logger.warning(f"  ⚠️ Only achieved {current_rows} rows, target was {target_rows}")
        logger.warning(f"     Missing: {target_rows - current_rows} rows")
    
    return current_rows


def collect_historical_batch_for_source(cities: List[Dict], scraper_func, history_days: int, pass_name: str) -> int:
    """
    Collect historical data for all cities using a specific scraper
    
    Args:
        cities: List of city dictionaries
        scraper_func: Scraper function to use
        history_days: Number of days of historical data to collect
        pass_name: Name for this collection pass
    
    Returns:
        Total rows collected
    """
    logger.info(f"\n{pass_name}: Collecting {history_days} days of history")
    
    total_rows = 0
    successful_cities = []
    failed_cities = []
    
    expected_rows = len(cities) * (1 + history_days * 24)
    logger.info(f"Expected rows: ~{expected_rows}")

    # Create a shared session for scrapers that need one (e.g. scrape_timeanddate)
    session = create_session()

    for i, city in enumerate(cities, 1):
        city_name = city.get("City")
        logger.info(f"[{i}/{len(cities)}] Processing {city_name}...")
        
        try:
            # scrape_timeanddate expects (session, city_info, ...) as positional args
            if scraper_func is scrape_timeanddate:
                rows = scraper_func(
                    session,
                    city_info=city,
                    history_days=history_days,
                    pass_index=0
                )
            else:
                rows = scraper_func(
                    city_info=city,
                    history_days=history_days,
                    pass_index=0
                )
            
            if rows:
                total_rows += len(rows)
                successful_cities.append(city_name)
                
                # Get the appropriate raw file
                raw_file_map = {
                    scrape_openmeteo: config.OPENMETEO_RAW_CSV,
                    scrape_timeanddate: config.TIMEANDDATE_RAW_CSV,
                    scrape_wunderground: config.WUNDERGROUND_RAW_CSV,
                }
                raw_file = raw_file_map.get(scraper_func)
                
                if raw_file:
                    append_raw_rows(raw_file, rows)
                
                # Persist to processed data
                write_processed_outputs(rows)
                
                current_temp = rows[0]['Temperature_C'] if rows else 'N/A'
                historical_count = len(rows) - 1
                logger.info(f"  ✓ Current: {current_temp}°C, Historical: {historical_count}, Total: {len(rows)}")
                
            else:
                failed_cities.append(city_name)
                logger.warning(f"  ✗ No data received")
                
        except Exception as e:
            failed_cities.append(city_name)
            logger.error(f"  ✗ Error: {e}")
        
        # Small delay between cities
        time.sleep(1)
    
    # Log summary
    logger.info(f"\n{pass_name} Summary:")
    logger.info(f"  Total rows collected: {total_rows}")
    logger.info(f"  Successful cities: {len(successful_cities)}/{len(cities)}")
    
    if failed_cities:
        logger.warning(f"  Failed cities: {', '.join(failed_cities)}")
    
    return total_rows


def ensure_all_sources_target(cities: List[Dict], target_per_source: int = 2000) -> Dict[str, int]:
    """
    Ensure each source has at least target_per_source rows
    
    Args:
        cities: List of city dictionaries
        target_per_source: Minimum rows per source (default: 2000)
    
    Returns:
        Dictionary with final counts per source
    """
    logger.info("\n" + "="*80)
    logger.info("CHECKING DATA COLLECTION TARGET: 2,000+ ROWS PER WEBSITE")
    logger.info("="*80)
    
    # Initial counts
    initial_counts = count_rows_by_source()
    logger.info("\nInitial data counts:")
    for source, count in initial_counts.items():
        logger.info(f"  {source}: {count} rows")
    
    results = {}
    
    # Collect for Open-Meteo
    logger.info("\n" + "="*80)
    logger.info("TARGET 1: Open-Meteo - 2,000+ rows")
    logger.info("="*80)
    results['Open-Meteo'] = collect_for_source(
        cities, 
        "Open-Meteo", 
        scrape_openmeteo, 
        target_per_source
    )
    
    # Collect for TimeAndDate
    logger.info("\n" + "="*80)
    logger.info("TARGET 2: TimeAndDate - 2,000+ rows")
    logger.info("="*80)
    results['TimeAndDate'] = collect_for_source(
        cities, 
        "TimeAndDate", 
        scrape_timeanddate, 
        target_per_source
    )
    
    # Collect for WeatherUnderground
    logger.info("\n" + "="*80)
    logger.info("TARGET 3: WeatherUnderground - 2,000+ rows")
    logger.info("="*80)
    results['WeatherUnderground'] = collect_for_source(
        cities, 
        "WeatherUnderground", 
        scrape_wunderground, 
        target_per_source
    )
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL DATA COLLECTION SUMMARY")
    logger.info("="*80)
    
    all_targets_reached = True
    for source, count in results.items():
        status = "✅" if count >= target_per_source else "⚠️"
        logger.info(f"  {status} {source}: {count} rows (target: {target_per_source})")
        if count < target_per_source:
            all_targets_reached = False
    
    if all_targets_reached:
        logger.info(f"\n✅ SUCCESS! All sources have at least {target_per_source} rows!")
    else:
        logger.info(f"\n⚠️ Some sources still need more data. Consider running again or increasing collection days.")
    
    return results


def run_batch_for_scheduled(cities: List[Dict]) -> int:
    """Run batch for all sources (used for scheduled jobs)"""
    
    total_rows = 0
    
    # Run Open-Meteo
    logger.info("Running Open-Meteo scheduled scrape...")
    rows = collect_historical_batch_for_source(cities, scrape_openmeteo, history_days=0, pass_name="Scheduled-OpenMeteo")
    total_rows += rows
    
    # Run TimeAndDate
    logger.info("Running TimeAndDate scheduled scrape...")
    rows = collect_historical_batch_for_source(cities, scrape_timeanddate, history_days=0, pass_name="Scheduled-TimeAndDate")
    total_rows += rows
    
    # Run WeatherUnderground
    logger.info("Running WeatherUnderground scheduled scrape...")
    rows = collect_historical_batch_for_source(cities, scrape_wunderground, history_days=0, pass_name="Scheduled-WUnderground")
    total_rows += rows
    
    return total_rows


def scheduled_job(cities: List[Dict], scheduler: BlockingScheduler) -> None:
    """Scheduled job to fetch current weather from all sources"""
    
    before_counts = count_rows_by_source()
    logger.info(f"Scheduled scrape started.")
    logger.info(f"Before counts: {before_counts}")
    
    # Run batch for all sources (current weather only)
    run_batch_for_scheduled(cities)
    
    after_counts = count_rows_by_source()
    logger.info(f"Scheduled scrape completed.")
    logger.info(f"After counts: {after_counts}")
    

def initialize_merged_dataset() -> int:
    """Initialize the merged dataset from existing raw files"""
    
    raw_rows = load_and_merge_raw_files()
    existing_processed_rows = load_existing_rows(config.WEATHER_CSV)
    merged_rows = normalize_rows(raw_rows + existing_processed_rows)
    written = replace_processed_outputs(merged_rows)
    inserted = database.insert_rows(merged_rows)
    update_summary_report()
    
    logger.info(f"Merged dataset prepared. rows_written={written}, rows_inserted_db={inserted}")
    return written


def main() -> None:
    """Main entry point"""
    
    # Setup
    setup_logging()
    ensure_directories()
    database.init_db()
    
    # Load cities
    cities = load_cities()
    logger.info(f"Loaded {len(cities)} cities")
    
    # Initialize merged dataset
    initialize_merged_dataset()
    
    # Ensure each source has at least 2,000 rows
    results = ensure_all_sources_target(cities, target_per_source=2000)
    
    # Start scheduler for regular updates
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
    logger.info("SCHEDULER STARTED")
    logger.info(f"{'='*80}")
    logger.info(f"Final counts: {results}")
    logger.info(f"Scheduler running every {config.SCRAPE_INTERVAL_HOURS} hours")
    logger.info(f"Press Ctrl+C to stop")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scraper stopped by user.")


if __name__ == "__main__":
    main()