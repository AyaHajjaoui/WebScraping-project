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


def ensure_directories() -> None:
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
    return random.choice(config.USER_AGENTS)


def create_session() -> requests.Session:
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
    request_headers = {"User-Agent": pick_user_agent()}
    if headers:
        request_headers.update(headers)
    response = session.get(url, timeout=timeout, headers=request_headers)
    response.raise_for_status()
    return response.text


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_numeric(value) -> Optional[float]:
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
    num = parse_numeric(value)
    if num is None:
        return None
    return round(num * 1.60934, 2)


def safe_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_row(row: Dict) -> Dict:
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
    file_exists = file_path.exists()
    frame.to_csv(file_path, mode="a", index=False, header=not file_exists)


def rows_to_frame(rows: Iterable[Dict]) -> pd.DataFrame:
    normalized_rows = normalize_rows(rows)
    if not normalized_rows:
        return pd.DataFrame(columns=config.STANDARD_COLUMNS)
    frame = pd.DataFrame(normalized_rows, columns=config.STANDARD_COLUMNS)
    return frame


def load_existing_rows(file_path: Path) -> List[Dict]:
    if not file_path.exists():
        return []
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except Exception:
        return []


def load_and_merge_raw_files() -> List[Dict]:
    raw_files = [
        config.TIMEANDDATE_RAW_CSV,
        config.WUNDERGROUND_RAW_CSV,
        config.METEOSTAT_RAW_CSV,
    ]
    all_rows: List[Dict] = []
    for file_path in raw_files:
        all_rows.extend(load_existing_rows(file_path))
    return normalize_rows(all_rows)


def replace_processed_outputs(rows: List[Dict]) -> int:
    frame = rows_to_frame(rows)
    frame.to_csv(config.WEATHER_CSV, index=False)
    frame.to_excel(config.WEATHER_XLSX, index=False)
    return len(frame)


def write_processed_outputs(rows: List[Dict]) -> int:
    if not rows:
        return 0
    frame = rows_to_frame(rows)
    if frame.empty:
        return 0
    append_dataframe_csv(config.WEATHER_CSV, frame)

    if config.WEATHER_XLSX.exists():
        existing = pd.read_excel(config.WEATHER_XLSX)
        combined = pd.concat([existing, frame], ignore_index=True).drop_duplicates(
            subset=["SourceWebsite", "City", "Country", "ScrapeDateTime"],
            keep="last",
        )
    else:
        combined = frame.copy()
    combined.to_excel(config.WEATHER_XLSX, index=False)
    return len(frame)


def append_raw_rows(file_path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    append_dataframe_csv(file_path, frame)


def count_processed_rows() -> int:
    if not config.WEATHER_CSV.exists():
        return 0
    try:
        frame = pd.read_csv(config.WEATHER_CSV)
        return len(frame.index)
    except Exception:
        return 0


def update_summary_report() -> None:
    if not config.WEATHER_CSV.exists():
        return
    df = pd.read_csv(config.WEATHER_CSV)
    if df.empty:
        return

    for col in ("Temperature_C", "WindSpeed_kmh"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    report_rows = []
    generated_at = now_utc_iso()

    avg_temp = (
        df.groupby(["City", "Country"], dropna=False)["Temperature_C"]
        .mean()
        .reset_index()
        .sort_values("Temperature_C", ascending=False)
    )
    for _, row in avg_temp.iterrows():
        report_rows.append(
            {
                "ReportGeneratedAt": generated_at,
                "Metric": "AverageTemperature_C",
                "City": row["City"],
                "Country": row["Country"],
                "Value": round(float(row["Temperature_C"]), 2)
                if pd.notna(row["Temperature_C"])
                else None,
            }
        )

    if not avg_temp.empty:
        hottest = avg_temp.iloc[0]
        coldest = avg_temp.iloc[-1]
        report_rows.append(
            {
                "ReportGeneratedAt": generated_at,
                "Metric": "HottestCityByAvgTemp",
                "City": hottest["City"],
                "Country": hottest["Country"],
                "Value": round(float(hottest["Temperature_C"]), 2)
                if pd.notna(hottest["Temperature_C"])
                else None,
            }
        )
        report_rows.append(
            {
                "ReportGeneratedAt": generated_at,
                "Metric": "ColdestCityByAvgTemp",
                "City": coldest["City"],
                "Country": coldest["Country"],
                "Value": round(float(coldest["Temperature_C"]), 2)
                if pd.notna(coldest["Temperature_C"])
                else None,
            }
        )

    top_wind = (
        df.groupby(["City", "Country"], dropna=False)["WindSpeed_kmh"]
        .mean()
        .reset_index()
        .sort_values("WindSpeed_kmh", ascending=False)
        .head(5)
    )
    for rank, (_, row) in enumerate(top_wind.iterrows(), start=1):
        report_rows.append(
            {
                "ReportGeneratedAt": generated_at,
                "Metric": f"TopWindiestCity_{rank}",
                "City": row["City"],
                "Country": row["Country"],
                "Value": round(float(row["WindSpeed_kmh"]), 2)
                if pd.notna(row["WindSpeed_kmh"])
                else None,
            }
        )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(config.SUMMARY_CSV, index=False)
