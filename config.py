from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, DATABASE_DIR, LOGS_DIR, DASHBOARD_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Raw data files - Open-Meteo (primary)
OPENMETEO_RAW_CSV = RAW_DIR / "openmeteo_raw.csv"

# Legacy raw files (kept for compatibility, but not actively used)
TIMEANDDATE_RAW_CSV = RAW_DIR / "timeanddate_raw.csv"
WUNDERGROUND_RAW_CSV = RAW_DIR / "wunderground_raw.csv"
OPENWEATHER_RAW_CSV = RAW_DIR / "openweather_raw.csv"  # Legacy

# Processed data files
WEATHER_CSV = PROCESSED_DIR / "weather_data.csv"
WEATHER_XLSX = PROCESSED_DIR / "weather_data.xlsx"
SUMMARY_CSV = PROCESSED_DIR / "summary_report.csv"

# Database
SQLITE_DB = DATABASE_DIR / "weather.db"

# Cities CSV
CITIES_CSV = BASE_DIR / "cities.csv"

# Log file
LOG_FILE = LOGS_DIR / "scraping.log"

# Scraping settings
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
BACKOFF_FACTOR = 1.0

# Target rows for data collection
INITIAL_TARGET_ROWS = 1000  # Target rows for initial backfill
AUTO_TARGET_ROWS = 5000     # Auto-stop when reaching this many rows
SCRAPE_INTERVAL_HOURS = 6   # How often to run scheduled scraping

MIN_CITIES_REQUIRED = 20

# History settings for backfill
DEFAULT_HISTORY_DAYS = 1    # Start with 1 day of history
DEFAULT_HISTORY_HOURS = 24  # 24 hours of history
MAX_INITIAL_PASSES = 7      # Build up to 7 days of history

# Source websites
SOURCE_WEBSITES = ("Open-Meteo",)

# Standard columns for output
STANDARD_COLUMNS = [
    "SourceWebsite",
    "City",
    "Country",
    "ScrapeDateTime",
    "Temperature_C",
    "FeelsLike_C",
    "Humidity_%",
    "WindSpeed_kmh",
    "Condition",
    "Precipitation",
]

# User agents for requests
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]