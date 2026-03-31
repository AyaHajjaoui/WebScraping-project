from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
DASHBOARD_DIR = BASE_DIR / "dashboard"

CITIES_CSV = BASE_DIR / "cities.csv"

WEATHER_CSV = PROCESSED_DIR / "weather_data.csv"
WEATHER_XLSX = PROCESSED_DIR / "weather_data.xlsx"
SUMMARY_CSV = PROCESSED_DIR / "summary_report.csv"
SQLITE_DB = DATABASE_DIR / "weather.db"

TIMEANDDATE_RAW_CSV = RAW_DIR / "timeanddate_raw.csv"
WUNDERGROUND_RAW_CSV = RAW_DIR / "wunderground_raw.csv"
METEOSTAT_RAW_CSV = RAW_DIR / "meteostat_raw.csv"

LOG_FILE = LOGS_DIR / "scraping.log"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
BACKOFF_FACTOR = 1.0

INITIAL_TARGET_ROWS = 2000
AUTO_TARGET_ROWS = 5000
SCRAPE_INTERVAL_HOURS = 3

MIN_CITIES_REQUIRED = 20

DEFAULT_HISTORY_DAYS = 2
DEFAULT_HISTORY_HOURS = 48
MAX_INITIAL_PASSES = 10

SOURCE_WEBSITES = ("TimeAndDate", "WeatherUnderground", "Meteostat")

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

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]
