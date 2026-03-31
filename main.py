import logging
import time
from typing import Dict, List

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

import config
import database
from scrapers import scrape_timeanddate, scrape_wunderground
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
    if not config.CITIES_CSV.exists():
        raise FileNotFoundError(f"Missing cities file: {config.CITIES_CSV}")

    df = pd.read_csv(config.CITIES_CSV)
    expected_columns = {
        "City",
        "Country",
        "TimeAndDate URL",
        "WeatherUnderground URL",
        "Meteostat URL",
    }
    missing = expected_columns.difference(df.columns)
    if missing:
        raise ValueError(f"cities.csv missing required columns: {missing}")

    if len(df.index) < config.MIN_CITIES_REQUIRED:
        raise ValueError(
            f"cities.csv must include at least {config.MIN_CITIES_REQUIRED} cities; found {len(df.index)}"
        )
    return df.to_dict(orient="records")


def _run_scrapers_for_city(session, city_info: Dict, pass_index: int, history_days: int, history_hours: int) -> List[Dict]:
    rows_tad = scrape_timeanddate(session, city_info, history_days=history_days, pass_index=pass_index)
    rows_wu = scrape_wunderground(session, city_info, history_days=history_days, pass_index=pass_index)

    append_raw_rows(config.TIMEANDDATE_RAW_CSV, rows_tad)
    append_raw_rows(config.WUNDERGROUND_RAW_CSV, rows_wu)
    return normalize_rows(rows_tad + rows_wu)


def persist_rows(rows: List[Dict]) -> int:
    if not rows:
        return 0
    inserted_db = database.insert_rows(rows)
    write_processed_outputs(rows)
    update_summary_report()
    return inserted_db


def initialize_merged_dataset() -> int:
    raw_rows = load_and_merge_raw_files()
    existing_processed_rows = load_existing_rows(config.WEATHER_CSV)
    merged_rows = normalize_rows(raw_rows + existing_processed_rows)
    written = replace_processed_outputs(merged_rows)
    inserted = database.insert_rows(merged_rows)
    update_summary_report()
    logger.info(
        "Merged startup dataset prepared. rows_written=%s rows_inserted_db=%s",
        written,
        inserted,
    )
    return written


def run_batch(
    cities: List[Dict],
    pass_index: int = 0,
    history_days: int = config.DEFAULT_HISTORY_DAYS,
    history_hours: int = config.DEFAULT_HISTORY_HOURS,
) -> int:
    session = create_session()
    total_rows = 0
    for city in cities:
        try:
            rows = _run_scrapers_for_city(
                session=session,
                city_info=city,
                pass_index=pass_index,
                history_days=history_days,
                history_hours=history_hours,
            )
            inserted = persist_rows(rows)
            total_rows += max(inserted, len(rows))
            logger.info(
                "Batch pass=%s city=%s rows_scraped=%s rows_db_inserted=%s",
                pass_index,
                city.get("City"),
                len(rows),
                inserted,
            )
            time.sleep(0.6)
        except Exception as exc:
            logger.exception("City scrape failed for %s: %s", city.get("City"), exc)
    return total_rows


def run_initial_backfill(cities: List[Dict]) -> None:
    current_rows = max(database.count_rows(), count_processed_rows())
    if current_rows >= config.INITIAL_TARGET_ROWS:
        logger.info("Initial backfill skipped. Existing rows: %s", current_rows)
        return

    logger.info("Starting initial backfill. Current rows=%s target=%s", current_rows, config.INITIAL_TARGET_ROWS)

    for pass_index in range(config.MAX_INITIAL_PASSES):
        if current_rows >= config.INITIAL_TARGET_ROWS:
            break
        history_days = config.DEFAULT_HISTORY_DAYS + pass_index
        history_hours = config.DEFAULT_HISTORY_HOURS + pass_index * 24
        scraped = run_batch(
            cities=cities,
            pass_index=pass_index,
            history_days=history_days,
            history_hours=history_hours,
        )
        current_rows = max(database.count_rows(), count_processed_rows())
        logger.info(
            "Initial backfill pass %s done. scraped=%s current_total_rows=%s",
            pass_index + 1,
            scraped,
            current_rows,
        )

    if current_rows < config.INITIAL_TARGET_ROWS:
        logger.warning(
            "Initial backfill stopped at %s rows, below target %s. The scheduler will continue appending.",
            current_rows,
            config.INITIAL_TARGET_ROWS,
        )
    else:
        logger.info("Initial backfill completed with %s rows.", current_rows)


def scheduled_job(cities: List[Dict], scheduler: BlockingScheduler) -> None:
    before = max(database.count_rows(), count_processed_rows())
    logger.info("Scheduled scrape started. Existing rows=%s", before)
    run_batch(cities=cities, pass_index=0, history_days=1, history_hours=8)
    after = max(database.count_rows(), count_processed_rows())
    logger.info("Scheduled scrape completed. Total rows=%s", after)

    if after >= config.AUTO_TARGET_ROWS:
        logger.info(
            "Auto target reached (%s rows >= %s). Stopping scheduler.",
            after,
            config.AUTO_TARGET_ROWS,
        )
        scheduler.shutdown(wait=False)


def main() -> None:
    setup_logging()
    ensure_directories()
    database.init_db()
    cities = load_cities()
    initialize_merged_dataset()

    run_initial_backfill(cities)

    scheduler = BlockingScheduler()
    scheduler.add_job(
        scheduled_job,
        "interval",
        hours=config.SCRAPE_INTERVAL_HOURS,
        kwargs={"cities": cities, "scheduler": scheduler},
        max_instances=1,
        coalesce=True,
        next_run_time=None,
    )

    logger.info(
        "Scheduler started. Running every %s hours. Press Ctrl+C to stop.",
        config.SCRAPE_INTERVAL_HOURS,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scraper stopped by user.")


if __name__ == "__main__":
    main()
