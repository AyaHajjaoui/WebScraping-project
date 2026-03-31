import sqlite3
from typing import Dict, Iterable, List

import config


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(config.SQLITE_DB)


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                SourceWebsite TEXT,
                City TEXT,
                Country TEXT,
                ScrapeDateTime TEXT,
                Temperature_C REAL,
                FeelsLike_C REAL,
                Humidity_pct REAL,
                WindSpeed_kmh REAL,
                Condition TEXT,
                Precipitation REAL,
                UNIQUE(SourceWebsite, City, Country, ScrapeDateTime)
            );
            """
        )
        conn.commit()


def insert_rows(rows: Iterable[Dict]) -> int:
    row_list: List[Dict] = list(rows)
    if not row_list:
        return 0
    with get_connection() as conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO weather_data (
                SourceWebsite, City, Country, ScrapeDateTime,
                Temperature_C, FeelsLike_C, Humidity_pct, WindSpeed_kmh,
                Condition, Precipitation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    row.get("SourceWebsite"),
                    row.get("City"),
                    row.get("Country"),
                    row.get("ScrapeDateTime"),
                    row.get("Temperature_C"),
                    row.get("FeelsLike_C"),
                    row.get("Humidity_%"),
                    row.get("WindSpeed_kmh"),
                    row.get("Condition"),
                    row.get("Precipitation"),
                )
                for row in row_list
            ],
        )
        conn.commit()
        return cursor.rowcount if cursor.rowcount is not None else 0


def count_rows() -> int:
    with get_connection() as conn:
        result = conn.execute("SELECT COUNT(*) FROM weather_data;").fetchone()
    return int(result[0]) if result else 0
