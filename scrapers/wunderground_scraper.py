import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
DEFAULT_HISTORY_DAYS = 5
REQUEST_TIMEOUT = 30
MAX_INITIAL_PASSES = 5
INITIAL_TARGET_ROWS = 1200


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _safe_text(value) -> Optional[str]:
    if isinstance(value, list):
        for item in value:
            text = _safe_text(item)
            if text:
                return text
        return None
    if isinstance(value, dict):
        return None
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


def _to_kmh_from_mph(value) -> Optional[float]:
    speed = _parse_numeric(value)
    if speed is None:
        return None
    return round(speed * 1.60934, 2)


def _to_celsius(value) -> Optional[float]:
    temp = _parse_numeric(value)
    if temp is None:
        return None
    if temp > 60 and temp <= 140:
        return round((temp - 32) * 5.0 / 9.0, 2)
    return temp


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
        "Precipitation",
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


def _walk_dict(obj: Any) -> Iterable[Dict]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dict(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_dict(item)


def _extract_from_next_data(html: str, city: str, country: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    scripts = soup.find_all("script")
    for script in scripts:
        text = script.string or script.text or ""
        if "__NEXT_DATA__" not in text and "observations" not in text:
            continue

        try:
            if script.get("id") == "__NEXT_DATA__":
                data = json.loads(text)
            else:
                match = re.search(r"({.*})", text, flags=re.DOTALL)
                if not match:
                    continue
                data = json.loads(match.group(1))
        except Exception:
            continue

        rows = []
        for node in _walk_dict(data):
            obs_time = node.get("obsTimeUtc") or node.get("validTimeUtc")
            temperature = node.get("temperature")
            if not _is_scalar(obs_time) or not _is_scalar(temperature):
                continue
            if obs_time is None or temperature is None:
                continue

            # ✅ UPDATED PART
            humidity = (
                node.get("humidity")
                or node.get("relativeHumidity")
                or node.get("rh")
            )

            wind = (
                node.get("windSpeed")
                or node.get("wspd")
                or node.get("windKph")
            )

            feels_like = (
                     node.get("feelsLike")
                    or node.get("feelslike")
                    or node.get("feelsLikeTemperature")
                    or node.get("heatIndex")
                    or node.get("apparentTemperature")
                    or node.get("realFeel")
            )

            precip = (
                    node.get("precipRate")
                    or node.get("precipTotal")
                    or node.get("precip")
                    or node.get("precipAccumulation")
                    or node.get("precip1h")
                    or node.get("precip24h")
                    or node.get("qpf")
            )
            

            condition = node.get("wxPhraseLong")
            if condition is None:
                condition = node.get("condition")
            condition = _safe_text(condition)

            if condition is None:
                alt = node.get("phrase")
                condition = _safe_text(alt) if _is_scalar(alt) else None

            if (
                _to_celsius(temperature) is None
                and _parse_numeric(humidity) is None
                and _parse_numeric(wind) is None
            ):
                continue

            if not _is_scalar(humidity) or not _is_scalar(wind):
                continue

            rows.append(
                {
                    "SourceWebsite": "WeatherUnderground",
                    "City": city,
                    "Country": country,
                    "ScrapeDateTime": _safe_text(obs_time)
                    or datetime.now(timezone.utc).isoformat(),
                    "Temperature_C": _to_celsius(temperature),
                    "FeelsLike_C": _to_celsius(feels_like),
                    "Humidity_%": _parse_numeric(humidity),
                    "WindSpeed_kmh": _to_kmh_from_mph(wind),
                    "Condition": condition,
                    "Precipitation": _parse_numeric(precip),
                }
            )
        if rows:
            dedup = pd.DataFrame(rows).drop_duplicates(
                subset=["SourceWebsite", "City", "Country", "ScrapeDateTime"],
                keep="last",
            )
            return dedup.to_dict(orient="records")
    return []


def _extract_from_tables(html: str, city: str, country: str, date_text: Optional[str]) -> List[Dict]:
    rows = []
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return rows

    for table in tables:
        header_str = " ".join([str(c).lower() for c in table.columns])
        if not any(key in header_str for key in ("temp", "temperature")):
            continue

        time_col = next((c for c in table.columns if "time" in str(c).lower()), None)
        temp_col = next((c for c in table.columns if "temp" in str(c).lower()), None)
        feels_col = next((c for c in table.columns if "feels" in str(c).lower()), None)
        humidity_col = next((c for c in table.columns if "humid" in str(c).lower()), None)
        wind_col = next((c for c in table.columns if "wind" in str(c).lower()), None)
        cond_col = next((c for c in table.columns if "cond" in str(c).lower()), None)
        precip_col = next((c for c in table.columns if "precip" in str(c).lower()), None)

        if not temp_col:
            continue

        for _, row in table.iterrows():
            time_value = _safe_text(row.get(time_col)) if time_col else None
            scrape_dt = (
                f"{date_text}T{time_value}" if date_text and time_value else datetime.now(timezone.utc).isoformat()
            )
            temp_value = _to_celsius(row.get(temp_col))
            feels_value = _to_celsius(row.get(feels_col)) if feels_col else None
            humidity_value = _parse_numeric(row.get(humidity_col)) if humidity_col else None
            wind_value = _to_kmh_from_mph(row.get(wind_col)) if wind_col else None
            precip_value = _parse_numeric(row.get(precip_col)) if precip_col else None
            condition_value = _safe_text(row.get(cond_col)) if cond_col else None
            if (
                temp_value is None
                and feels_value is None
                and humidity_value is None
                and wind_value is None
                and precip_value is None
                and condition_value is None
            ):
                continue
            rows.append(
                {
                    "SourceWebsite": "WeatherUnderground",
                    "City": city,
                    "Country": country,
                    "ScrapeDateTime": scrape_dt,
                    "Temperature_C": temp_value,
                    "FeelsLike_C": feels_value,
                    "Humidity_%": humidity_value,
                    "WindSpeed_kmh": wind_value,
                    "Condition": condition_value,
                    "Precipitation": precip_value,
                }
            )
        if rows:
            return rows
    return rows


def _history_url(base_url: str, date_obj: datetime) -> str:
    transformed = base_url.replace("/weather/", "/history/daily/")
    return f"{transformed}/date/{date_obj.strftime('%Y-%m-%d')}"


def scrape_wunderground(
    session,
    city_info: Dict,
    history_days: int = DEFAULT_HISTORY_DAYS,
    pass_index: int = 0,
) -> List[Dict]:
    local_session = _ensure_session(session)
    city = city_info["City"]
    country = city_info["Country"]
    base_url = city_info["WeatherUnderground URL"].rstrip("/")
    all_records: List[Dict] = []

    try:
        html = _fetch_url(local_session, base_url)
        rows = _extract_from_next_data(html, city, country)
        if not rows:
            rows = _extract_from_tables(html, city, country, date_text=None)
        all_records.extend(rows[:20])
    except Exception as exc:
        logger.warning("Weather Underground current scrape failed for %s: %s", city, exc)

    start_days_ago = pass_index * history_days + 1
    for day_offset in range(start_days_ago, start_days_ago + history_days):
        date_obj = datetime.now(timezone.utc) - timedelta(days=day_offset)
        history_url = _history_url(base_url, date_obj)
        try:
            html = _fetch_url(local_session, history_url)
            rows = _extract_from_tables(html, city, country, date_text=date_obj.strftime("%Y-%m-%d"))
            if rows:
                all_records.extend(rows)
        except Exception as exc:
            logger.warning(
                "Weather Underground history scrape failed for %s (%s): %s",
                city,
                date_obj.strftime("%Y-%m-%d"),
                exc,
            )

    return _normalize_rows(all_records)

def load_cities():
    df = pd.read_csv("cities.csv")

    required = {"City", "Country", "WeatherUnderground URL"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"cities.csv missing columns: {missing}")

    return df.to_dict(orient="records")


def run_batch(cities, session, pass_index=0):
    all_data = []

    for city in cities:
        try:
            rows = scrape_wunderground(
                session=session,
                city_info=city,
                pass_index=pass_index,
            )

            all_data.extend(rows)

            print(f"✅ {city['City']} -> {len(rows)} rows")

        except Exception as e:
            print(f"❌ Failed {city['City']}: {e}")

        time.sleep(0.5)  # small delay

    return all_data


def run_backfill(cities):
    session = requests.Session()
    collected = []

    for pass_index in range(MAX_INITIAL_PASSES):
        print(f"\n🚀 Pass {pass_index + 1}/{MAX_INITIAL_PASSES}")

        batch = run_batch(cities, session, pass_index=pass_index)
        collected.extend(batch)

        print(f"📊 Total rows so far: {len(collected)}")

        if len(collected) >= INITIAL_TARGET_ROWS:
            break

    return collected


def main():
    print(" Starting full scrape using cities.csv...")

    cities = load_cities()

    data = run_backfill(cities)

    if not data:
        print(" No data scraped")
        return

    df = pd.DataFrame(data)

    df.drop_duplicates(inplace=True)

    output_file = "weather_data.csv"
    df.to_csv(output_file, index=False)

    print(f"\n Saved {len(df)} rows to {output_file}")
    print(df.head())


if __name__ == "__main__":
    main()