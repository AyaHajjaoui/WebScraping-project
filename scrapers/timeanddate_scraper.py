import logging
import re
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Dict, List, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
DEFAULT_HISTORY_DAYS = 2
REQUEST_TIMEOUT = 120


def _safe_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_numeric(value) -> Optional[float]:
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


def _fetch_url(session: requests.Session, url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = session.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.text


def _normalize_rows(rows: List[Dict]) -> List[Dict]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    expected_cols = [
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
    for col in expected_cols:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[expected_cols]
    frame["ScrapeDateTime"] = frame["ScrapeDateTime"].fillna(datetime.now(timezone.utc).isoformat())
    frame.drop_duplicates(
        subset=["SourceWebsite", "City", "Country", "ScrapeDateTime"],
        inplace=True,
    )
    return frame.to_dict(orient="records")


def _ensure_session(session=None) -> requests.Session:
    if session is not None:
        return session
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    new_session = requests.Session()
    new_session.mount("http://", adapter)
    new_session.mount("https://", adapter)
    return new_session


def _extract_current_details(soup: BeautifulSoup) -> Dict[str, Optional[object]]:
    feels_like = None
    wind_speed = None
    condition = None
    humidity = None


    qlook = soup.select_one("#qlook")
    if qlook:
        cond_candidates = qlook.find_all("p", recursive=False)
        if cond_candidates:
            condition = _safe_text(cond_candidates[0].get_text(" ", strip=True))


    if not condition and qlook:
        for p in qlook.find_all("p"):
            txt = _safe_text(p.get_text(" ", strip=True))
            if txt and len(txt) < 120 and not re.search(r"\d+\s*°", txt):
                condition = txt
                break


    detail_p = None
    if qlook:
        all_p = qlook.find_all("p", recursive=False)
        if len(all_p) >= 2:
            detail_p = all_p[1]
        elif len(all_p) == 1:

            detail_p = all_p[0]

    if detail_p:
        detail_text = detail_p.get_text(" ", strip=True)


        fl_match = re.search(r"Feels\s+Like[:\s]+(-?\d+(?:\.\d+)?)\s*°?", detail_text, re.I)
        if fl_match:
            feels_like = _parse_numeric(fl_match.group(1))


        wind_match = re.search(
            r"Wind[:\s]+(?:from\s+\w+\s+at\s+)?(-?\d+(?:\.\d+)?)\s*(km/h|kph|mph)",
            detail_text,
            re.I,
        )
        if wind_match:
            speed = _parse_numeric(wind_match.group(1))
            unit = wind_match.group(2).lower()
            if speed is not None:
                wind_speed = round(speed * 1.60934, 2) if unit == "mph" else speed


    if feels_like is None or wind_speed is None:
        full_text = soup.get_text(" ", strip=True)

        if feels_like is None:
            fl_fb = re.search(r"Feels\s+Like[:\s]+(-?\d+(?:\.\d+)?)\s*°?", full_text, re.I)
            if fl_fb:
                feels_like = _parse_numeric(fl_fb.group(1))

        if wind_speed is None:
            wind_fb = re.search(
                r"Wind[:\s]+(?:from\s+\w+\s+at\s+)?(-?\d+(?:\.\d+)?)\s*(km/h|kph|mph)",
                full_text,
                re.I,
            )
            if wind_fb:
                speed = _parse_numeric(wind_fb.group(1))
                unit = wind_fb.group(2).lower()
                if speed is not None:
                    wind_speed = round(speed * 1.60934, 2) if unit == "mph" else speed


    for row in soup.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        if not th or not td:
            continue
        label = th.get_text(" ", strip=True).lower()
        value_text = td.get_text(" ", strip=True)

        if "humidity" in label and humidity is None:
            humidity = _parse_numeric(value_text)


    if humidity is None:
        full_text = soup.get_text(" ", strip=True)

        if humidity is None:
            hum_match = re.search(r"Humidity[:\s]+(\d+(?:\.\d+)?)\s*%", full_text, re.I)
            if hum_match:
                humidity = _parse_numeric(hum_match.group(1))

    return {
        "FeelsLike_C": feels_like,
        "Humidity_%": humidity,
        "WindSpeed_kmh": wind_speed,
        "Condition": condition,
    }


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            " ".join(str(c).strip() for c in col if str(c).strip() and str(c) != "Unnamed: 0 level_0")
            for col in frame.columns
        ]
    else:
        frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _parse_table_rows(
    frame: pd.DataFrame,
    city: str,
    country: str,
    scrape_dt_prefix: str,
) -> List[Dict]:

    frame = _flatten_columns(frame)

    records = []

    def _find_col(keywords: List[str]) -> Optional[str]:
        col_lower = {c: c.lower() for c in frame.columns}
        for kw in keywords:
            for col, cl in col_lower.items():
                if kw in cl:
                    return col
        return None

    temp_col      = _find_col(["temp"])
    feels_col     = _find_col(["feels", "feel"])
    humidity_col  = _find_col(["humid"])
    wind_col      = _find_col(["wind"])
    condition_col = _find_col(["weather", "condition", "desc"])
    time_col      = _find_col(["time", "hour"])

    if not temp_col:
        return []

    for _, row in frame.iterrows():
        row_time = _safe_text(row.get(time_col)) if time_col else None
        records.append(
            {
                "SourceWebsite": "TimeAndDate",
                "City": city,
                "Country": country,
                "ScrapeDateTime": f"{scrape_dt_prefix}T{row_time}"
                if row_time
                else scrape_dt_prefix,
                "Temperature_C": _parse_numeric(row.get(temp_col)),
                "FeelsLike_C": _parse_numeric(row.get(feels_col)) if feels_col else None,
                "Humidity_%": _parse_numeric(row.get(humidity_col)) if humidity_col else None,
                "WindSpeed_kmh": _parse_numeric(row.get(wind_col)) if wind_col else None,
                "Condition": _safe_text(row.get(condition_col)) if condition_col else None,
            }
        )
    return records


def scrape_timeanddate(
    session,
    city_info: Dict,
    history_days: int = DEFAULT_HISTORY_DAYS,
    pass_index: int = 0,
) -> List[Dict]:
    local_session = _ensure_session(session)
    city = city_info["City"]
    country = city_info["Country"]
    base_url = city_info["TimeAndDate URL"].rstrip("/")
    all_records: List[Dict] = []


    try:
        html = _fetch_url(local_session, base_url)
        soup = BeautifulSoup(html, "lxml")


        temp_box = soup.select_one("#qlook .h2") or soup.select_one(".h2")

        details = _extract_current_details(soup)
        current_record = {
            "SourceWebsite": "TimeAndDate",
            "City": city,
            "Country": country,
            "ScrapeDateTime": datetime.now(timezone.utc).isoformat(),
            "Temperature_C": _parse_numeric(temp_box.text if temp_box else None),
            "FeelsLike_C": details["FeelsLike_C"],
            "Humidity_%": details["Humidity_%"],
            "WindSpeed_kmh": details["WindSpeed_kmh"],
            "Condition": details["Condition"],
        }
        if any(
            current_record.get(field) is not None
            for field in ["Temperature_C", "FeelsLike_C", "Humidity_%", "WindSpeed_kmh", "Condition"]
        ):
            all_records.append(current_record)
            logger.debug(
                "Current record for %s: FL=%s Wind=%s Cond=%s",
                city,
                details["FeelsLike_C"],
                details["WindSpeed_kmh"],
                details["Condition"],
            )
    except Exception as exc:
        logger.warning("TimeAndDate current scrape failed for %s: %s", city, exc)


    historic_url = f"{base_url}/historic"
    start_days_ago = pass_index * history_days + 1
    for day_offset in range(start_days_ago, start_days_ago + history_days):
        date_obj = datetime.now(timezone.utc) - timedelta(days=day_offset)
        date_code = date_obj.strftime("%Y%m%d")
        url = f"{historic_url}?hd={date_code}"
        try:
            html = _fetch_url(local_session, url)
            tables = pd.read_html(StringIO(html))
            for table in tables:

                flat = _flatten_columns(table.copy())
                table_columns_str = " ".join([str(c).lower() for c in flat.columns])
                if "temp" not in table_columns_str:
                    continue
                all_records.extend(
                    _parse_table_rows(
                        flat,
                        city=city,
                        country=country,
                        scrape_dt_prefix=date_obj.strftime("%Y-%m-%d"),
                    )
                )
                break
        except Exception as exc:
            logger.warning("TimeAndDate historic scrape failed for %s (%s): %s", city, date_code, exc)

    return _normalize_rows(all_records)
