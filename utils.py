import logging
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

logger = logging.getLogger(__name__)


def ensure_directories() -> None:
    """Ensure all required directories exist"""
    dirs = [
        config.DATA_DIR,
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.DATABASE_DIR,
        config.LOGS_DIR,
        config.DASHBOARD_DIR,
    ]
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Setup logging configuration"""
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def pick_user_agent() -> str:
    """Pick a random user agent from the list"""
    return random.choice(config.USER_AGENTS)


def create_session() -> requests.Session:
    """Create a requests session with retry logic"""
    retry = Retry(
        total=config.MAX_RETRIES,
        connect=config.MAX_RETRIES,
        read=config.MAX_RETRIES,
        backoff_factor=config.BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_url(
    session: requests.Session,
    url: str,
    timeout: int = config.REQUEST_TIMEOUT,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Fetch a URL with retry logic"""
    request_headers = {"User-Agent": pick_user_agent()}
    if headers:
        request_headers.update(headers)
    response = session.get(url, timeout=timeout, headers=request_headers)
    response.raise_for_status()
    return response.text


def now_utc_iso() -> str:
    """Return current UTC time in ISO format"""
    return datetime.now(timezone.utc).isoformat()


def parse_numeric(value) -> Optional[float]:
    """Parse numeric value from string"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "--"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def to_kmh_from_mph(value) -> Optional[float]:
    """Convert mph to km/h"""
    num = parse_numeric(value)
    if num is None:
        return None
    return round(num * 1.60934, 2)


def safe_text(value) -> Optional[str]:
    """Safely convert value to text"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_row(row: Dict) -> Dict:
    """Normalize a single row to standard format"""
    normalized = {col: None for col in config.STANDARD_COLUMNS}
    normalized.update(row)
    normalized["Temperature_C"] = parse_numeric(normalized.get("Temperature_C"))
    normalized["FeelsLike_C"] = parse_numeric(normalized.get("FeelsLike_C"))
    normalized["Humidity_%"] = parse_numeric(normalized.get("Humidity_%"))
    normalized["WindSpeed_kmh"] = parse_numeric(normalized.get("WindSpeed_kmh"))
    normalized["Precipitation"] = parse_numeric(normalized.get("Precipitation"))
    normalized["Condition"] = safe_text(normalized.get("Condition"))
    normalized["ScrapeDateTime"] = safe_text(normalized.get("ScrapeDateTime")) or now_utc_iso()
    return normalized


def normalize_rows(rows: Iterable[Dict]) -> List[Dict]:
    """Normalize multiple rows"""
    normalized = [normalize_row(row) for row in rows]
    if not normalized:
        return []
    frame = pd.DataFrame(normalized, columns=config.STANDARD_COLUMNS)
    frame.drop_duplicates(
        subset=["SourceWebsite", "City", "Country", "ScrapeDateTime"],
        inplace=True,
    )
    return frame.to_dict(orient="records")


def append_dataframe_csv(file_path: Path, frame: pd.DataFrame) -> None:
    """Append DataFrame to CSV file"""
    file_exists = file_path.exists()
    frame.to_csv(file_path, mode="a", index=False, header=not file_exists)


def rows_to_frame(rows: Iterable[Dict]) -> pd.DataFrame:
    """Convert rows to DataFrame with standard columns"""
    normalized_rows = normalize_rows(rows)
    if not normalized_rows:
        return pd.DataFrame(columns=config.STANDARD_COLUMNS)
    frame = pd.DataFrame(normalized_rows, columns=config.STANDARD_COLUMNS)
    return frame


def load_existing_rows(file_path: Path) -> List[Dict]:
    """Load existing rows from a CSV file"""
    if not file_path.exists():
        return []
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []



def load_and_merge_raw_files() -> List[Dict]:
    """Load and merge all raw CSV files from the raw directory."""
    all_rows = []

    raw_files = sorted(config.RAW_DIR.glob("*.csv"))
    if not raw_files:
        logger.warning("No raw CSV files found in %s", config.RAW_DIR)
        return all_rows

    source_by_file = {
        config.OPENMETEO_RAW_CSV.name.lower(): "Open-Meteo",
        config.TIMEANDDATE_RAW_CSV.name.lower(): "TimeAndDate",
        config.WUNDERGROUND_RAW_CSV.name.lower(): "WeatherUnderground",
    }

    for file_path in raw_files:
        try:
            df = pd.read_csv(file_path, dtype=str, low_memory=False, on_bad_lines="skip")
            if df.empty:
                logger.info("Loaded 0 rows from %s", file_path.name)
                continue

            if "SourceWebsite" not in df.columns:
                guessed_source = source_by_file.get(file_path.name.lower(), file_path.stem)
                df["SourceWebsite"] = guessed_source

            all_rows.extend(df.to_dict(orient="records"))
            logger.info("Loaded %s rows from %s", len(df), file_path.name)
        except Exception as e:
            logger.error("Could not load %s: %s", file_path, e)

    return all_rows


def replace_processed_outputs(rows: List[Dict]) -> int:
    """Replace processed outputs with new data"""
    frame = rows_to_frame(rows)
    if frame.empty:
        logger.warning("No rows to replace processed outputs")
        return 0
    
    # Save to CSV
    frame.to_csv(config.WEATHER_CSV, index=False)
    logger.info(f"Saved {len(frame)} rows to {config.WEATHER_CSV}")
    
    # Save to Excel
    try:
        frame.to_excel(config.WEATHER_XLSX, index=False)
        logger.info(f"Saved {len(frame)} rows to {config.WEATHER_XLSX}")
    except Exception as e:
        logger.warning(f"Could not save Excel file: {e}")
    
    return len(frame)


def write_processed_outputs(rows: List[Dict]) -> int:
    """Append rows to processed outputs"""
    if not rows:
        return 0
    
    frame = rows_to_frame(rows)
    if frame.empty:
        return 0
    
    # Append to CSV
    append_dataframe_csv(config.WEATHER_CSV, frame)
    
    # Update Excel file
    try:
        if config.WEATHER_XLSX.exists():
            existing = pd.read_excel(config.WEATHER_XLSX)
            combined = pd.concat([existing, frame], ignore_index=True).drop_duplicates(
                subset=["SourceWebsite", "City", "Country", "ScrapeDateTime"],
                keep="last",
            )
        else:
            combined = frame.copy()
        combined.to_excel(config.WEATHER_XLSX, index=False)
    except Exception as e:
        logger.warning(f"Could not update Excel file: {e}")
    
    logger.info(f"Appended {len(frame)} rows to processed outputs")
    return len(frame)


def append_raw_rows(file_path: Path, rows: List[Dict]) -> None:
    """Append rows to raw data file"""
    if not rows:
        logger.debug("No rows to append")
        return
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with standard columns
    frame = pd.DataFrame(rows)
    
    # Ensure all standard columns exist
    for col in config.STANDARD_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    
    # Reorder to match STANDARD_COLUMNS
    frame = frame[config.STANDARD_COLUMNS]
    
    # Append to CSV
    file_exists = file_path.exists()
    frame.to_csv(file_path, mode='a', index=False, header=not file_exists)
    
    logger.info(f"Appended {len(rows)} rows to {file_path.name}")
    if not file_exists:
        logger.info(f"Created new file: {file_path}")


def count_processed_rows() -> int:
    """Count total rows in processed CSV"""
    if not config.WEATHER_CSV.exists():
        return 0
    try:
        frame = pd.read_csv(config.WEATHER_CSV)
        return len(frame.index)
    except Exception as e:
        logger.error(f"Error counting processed rows: {e}")
        return 0


def update_summary_report() -> None:
    """Write a compact summary report from the processed CSV."""
    if not config.WEATHER_CSV.exists():
        logger.debug("Weather CSV not found, skipping summary report")
        return

    try:
        df = pd.read_csv(config.WEATHER_CSV, low_memory=False)
        if df.empty:
            return

        generated_at = now_utc_iso()
        report_rows = []

        if "Date" not in df.columns and "ScrapeDateTime" in df.columns:
            try:
                dt = pd.to_datetime(df["ScrapeDateTime"], errors="coerce", utc=True, format="mixed")
            except TypeError:
                dt = pd.to_datetime(df["ScrapeDateTime"], errors="coerce", utc=True)
            df["Date"] = dt.dt.strftime("%Y-%m-%d")

        date_series = df["Date"] if "Date" in df.columns else pd.Series(dtype="object")
        valid_dates = pd.to_datetime(date_series, errors="coerce")
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        unique_dates = int(valid_dates.dt.date.nunique()) if not valid_dates.empty else 0

        report_rows.append(
            {
                "ReportGeneratedAt": generated_at,
                "Metric": "TotalRows",
                "Value": int(len(df)),
            }
        )

        by_source = (
            df.groupby("SourceWebsite", dropna=False)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        for _, row in by_source.iterrows():
            report_rows.append(
                {
                    "ReportGeneratedAt": generated_at,
                    "Metric": f"RowsBySource:{row['SourceWebsite']}",
                    "Value": int(row["Count"]),
                }
            )

        report_rows.append(
            {
                "ReportGeneratedAt": generated_at,
                "Metric": "DateRangeStart",
                "Value": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None,
            }
        )
        report_rows.append(
            {
                "ReportGeneratedAt": generated_at,
                "Metric": "DateRangeEnd",
                "Value": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None,
            }
        )
        report_rows.append(
            {
                "ReportGeneratedAt": generated_at,
                "Metric": "UniqueDates",
                "Value": unique_dates,
            }
        )

        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(config.SUMMARY_CSV, index=False)
        logger.info("Updated summary report with %s entries", len(report_rows))

    except Exception as e:
        logger.error("Error updating summary report: %s", e)
