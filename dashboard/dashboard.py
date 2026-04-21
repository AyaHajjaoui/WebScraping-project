import os
import re
import importlib.util
import pandas as pd
import plotly.express as px
import streamlit as st
import random


COLORS = {
    "Ideal": "#86efac",      # pastel green
    "Good": "#93c5fd",       # pastel blue
    "Moderate": "#fef3c7",   # pastel yellow
    "Avoid": "#fca5a5",      # pastel red
    "Unknown": "#d1d5db"     # pastel gray
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "weather_data.csv")
SUMMARY_REPORT_PATH = os.path.join(BASE_DIR, "data", "processed", "summary_report.csv")
CONDITION_ANALYSIS_PATH = os.path.join(BASE_DIR, "data", "processed", "condition_analysis.csv")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV path with safe fallback."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_and_prepare_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read and preprocess the dashboard dataset once per file change."""
    return clean_data(load_data(path))


PLACEHOLDER_NA_VALUES = {"", "-", "N/A", "NA", "None", "nan", "null"}


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed-type dataframe columns so Streamlit can serialize them safely."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    safe_df = df.copy()

    for col in safe_df.columns:
        series = safe_df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized = series.replace(list(PLACEHOLDER_NA_VALUES), pd.NA)
            normalized = normalized.where(~normalized.isna(), pd.NA)
            normalized = normalized.apply(lambda value: value.strip() if isinstance(value, str) else value)
            normalized = normalized.replace(list(PLACEHOLDER_NA_VALUES), pd.NA)

            numeric_candidate = pd.to_numeric(normalized, errors="coerce")
            non_null_mask = normalized.notna()
            if non_null_mask.any() and numeric_candidate[non_null_mask].notna().all():
                safe_df[col] = numeric_candidate
            else:
                safe_df[col] = normalized

    return safe_df


def prepare_display_df(df: pd.DataFrame, na_columns: list[str] | None = None) -> pd.DataFrame:
    """Return an Arrow-compatible dataframe, optionally labeling missing values for display."""
    safe_df = make_arrow_compatible(df)
    if safe_df.empty or not na_columns:
        return safe_df

    display_df = safe_df.copy()
    for col in na_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype("string").fillna("N/A")
    return display_df


@st.cache_resource(show_spinner=False)
def load_analysis_module(module_name: str, relative_path: str):
    """Load a local analysis module by file path."""
    module_path = os.path.join(BASE_DIR, relative_path)
    if not os.path.exists(module_path):
        return None

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse datetimes and normalize to timezone-naive UTC for safe comparisons."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)

    # Some sources emit mixed timestamp formats in the same CSV column.
    # Retry failed non-null values individually so one source format does not
    # cause another source's timestamps to become NaT in the dashboard.
    failed_mask = series.notna() & parsed.isna()
    if failed_mask.any():
        reparsed = series.loc[failed_mask].apply(
            lambda value: pd.to_datetime(value, errors="coerce", utc=True)
        )
        parsed.loc[failed_mask] = reparsed

    return parsed.dt.tz_localize(None)


def comfort_score(temp, feels_like, humidity) -> float:
    """Simple comfort formula: 100 minus temperature/humidity/feels-like penalties."""
    score = 100.0

    if pd.notna(temp):
        score -= abs(float(temp) - 22.0) * 2.5

    if pd.notna(humidity):
        humidity = float(humidity)
        score -= abs(humidity - 50.0) * 0.5
        if humidity > 75:
            score -= (humidity - 75.0) * 0.7

    if pd.notna(feels_like):
        feels_like = float(feels_like)
        if feels_like > 30:
            score -= (feels_like - 30.0) * 2.0
        if feels_like < 5:
            score -= (5.0 - feels_like) * 1.2

    return round(max(0.0, min(100.0, score)), 1)


def travel_recommendation(score: float) -> str:
    """Turn numeric comfort score into a travel label."""
    if pd.isna(score):
        return "Unknown"
    if score >= 80:
        return "Ideal"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Moderate"
    return "Avoid"


def clean_data(_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and compute comfort metrics."""
    df = _df.copy()
    if df.empty:
        return df

    data = df.copy()

    if "ScrapeDateTime" in data.columns:
        data["ScrapeDateTime"] = parse_datetime(data["ScrapeDateTime"])

    if "Date" in data.columns:
        data["Date"] = parse_datetime(data["Date"])
    elif "ScrapeDateTime" in data.columns:
        data["Date"] = data["ScrapeDateTime"].dt.floor("D")

    numeric_cols = [
        "Temperature_C",
        "FeelsLike_C",
        "Humidity_%",
        "WindSpeed_kmh",
        "Precipitation",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in ["City", "SourceWebsite", "Country", "Condition"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()
            data.loc[data[col].isin(["", "nan", "None"]), col] = pd.NA

    required_cols = [c for c in ["City", "SourceWebsite"] if c in data.columns]
    if required_cols:
        data = data.dropna(subset=required_cols)

    if "Temperature_C" not in data.columns:
        data["Temperature_C"] = pd.NA
    if "FeelsLike_C" not in data.columns:
        data["FeelsLike_C"] = pd.NA
    if "Humidity_%" not in data.columns:
        data["Humidity_%"] = pd.NA

    data["Comfort Score"] = data.apply(
        lambda row: comfort_score(
            row.get("Temperature_C"),
            row.get("FeelsLike_C"),
            row.get("Humidity_%"),
        ),
        axis=1,
    )
    data["Travel Recommendation"] = data["Comfort Score"].apply(travel_recommendation)

    sort_cols = [
        col for col in ["Date", "SourceWebsite", "City", "ScrapeDateTime"] if col in data.columns
    ]
    if sort_cols:
        data = data.sort_values(sort_cols)

    return data


def filter_latest_per_city_source(_df: pd.DataFrame) -> pd.DataFrame:
    """Keep most recent row for each city/source pair."""
    df = _df.copy()
    if df.empty or "City" not in df.columns or "SourceWebsite" not in df.columns:
        return df

    temp = df.copy()
    if "ScrapeDateTime" in temp.columns:
        temp = temp.sort_values("ScrapeDateTime")
    elif "Date" in temp.columns:
        temp = temp.sort_values("Date")

    return temp.drop_duplicates(subset=["City", "SourceWebsite"], keep="last")


def get_filter_options(_weather_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Build reusable filter option lists from the prepared dataset."""
    weather_df = _weather_df.copy()
    city_options = sorted(weather_df["City"].dropna().unique().tolist()) if "City" in weather_df.columns else []
    country_options = sorted(weather_df["Country"].dropna().unique().tolist()) if "Country" in weather_df.columns else []
    source_options = (
        sorted(weather_df["SourceWebsite"].dropna().unique().tolist())
        if "SourceWebsite" in weather_df.columns
        else []
    )
    return city_options, country_options, source_options


def get_city_options(
    _df: pd.DataFrame,
    selected_countries: tuple[str, ...],
    selected_sources: tuple[str, ...],
) -> list[str]:
    """Return city options for the sidebar based on country/source filters."""
    df = _df.copy()
    if df.empty or "City" not in df.columns:
        return []

    city_pool_df = df
    if selected_countries and "Country" in city_pool_df.columns:
        city_pool_df = city_pool_df[city_pool_df["Country"].isin(selected_countries)]

    if selected_sources and "SourceWebsite" in city_pool_df.columns:
        city_pool_df = city_pool_df[city_pool_df["SourceWebsite"].isin(selected_sources)]

    return sorted(city_pool_df["City"].dropna().unique().tolist())


def apply_dashboard_filters(
    _df: pd.DataFrame,
    selected_countries: tuple[str, ...],
    selected_cities: tuple[str, ...],
    selected_sources: tuple[str, ...],
    use_latest_only: bool,
    min_comfort: int,
) -> pd.DataFrame:
    """Apply dashboard filters once and reuse the result across the app."""
    df = _df.copy()
    if df.empty:
        return df

    filtered_df = df

    if selected_countries and "Country" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

    if selected_cities and "City" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["City"].isin(selected_cities)]

    if selected_sources and "SourceWebsite" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["SourceWebsite"].isin(selected_sources)]

    if use_latest_only:
        filtered_df = filter_latest_per_city_source(filtered_df)

    if "Comfort Score" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Comfort Score"] >= min_comfort]

    return filtered_df


def build_city_ranking(_df: pd.DataFrame) -> pd.DataFrame:
    """Build city-level aggregation used by KPI cards and tables."""
    df = _df.copy()
    if df.empty or "City" not in df.columns:
        return pd.DataFrame()

    agg_map = {
        "Comfort Score": "mean",
        "Temperature_C": "mean",
        "FeelsLike_C": "mean",
        "Humidity_%": "mean",
    }
    if "WindSpeed_kmh" in df.columns:
        agg_map["WindSpeed_kmh"] = "mean"

    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}
    if not agg_map:
        return pd.DataFrame()

    ranking = (
        df.groupby("City", as_index=False)
        .agg(agg_map)
        .rename(
            columns={
                "Temperature_C": "Avg Temperature_C",
                "FeelsLike_C": "Avg FeelsLike_C",
                "Humidity_%": "Avg Humidity_%",
                "WindSpeed_kmh": "Avg WindSpeed_kmh",
            }
        )
    )

    if "Country" in df.columns:
        country_ref = (
            df.groupby("City", as_index=False)["Country"]
            .agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else pd.NA)
        )
        ranking = ranking.merge(country_ref, on="City", how="left")

    for col in [
        "Avg Temperature_C",
        "Avg FeelsLike_C",
        "Avg Humidity_%",
        "Avg WindSpeed_kmh",
        "Comfort Score",
    ]:
        if col in ranking.columns:
            ranking[col] = ranking[col].round(1)

    ranking["Travel Recommendation"] = ranking["Comfort Score"].apply(travel_recommendation)
    ranking = ranking.sort_values("Comfort Score", ascending=False)

    cols = [
        "City",
        "Country",
        "Avg Temperature_C",
        "Avg FeelsLike_C",
        "Avg Humidity_%",
        "Avg WindSpeed_kmh",
        "Comfort Score",
        "Travel Recommendation",
    ]
    cols = [c for c in cols if c in ranking.columns]
    return ranking[cols]


def build_source_summary(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return temperature by source, wind by source, and city disagreement table."""
    df = _df.copy()
    temp_by_source = pd.DataFrame()
    wind_by_source = pd.DataFrame()
    disagreement = pd.DataFrame()

    if df.empty or "SourceWebsite" not in df.columns:
        return temp_by_source, wind_by_source, disagreement

    if "Temperature_C" in df.columns:
        temp_by_source = (
            df.groupby("SourceWebsite", as_index=False)["Temperature_C"]
            .mean()
            .rename(columns={"Temperature_C": "Avg Temperature_C"})
        )
        temp_by_source["Avg Temperature_C"] = temp_by_source["Avg Temperature_C"].round(1)

    if "WindSpeed_kmh" in df.columns:
        wind_by_source = (
            df.groupby("SourceWebsite", as_index=False)["WindSpeed_kmh"]
            .mean()
            .rename(columns={"WindSpeed_kmh": "Avg Wind (km/h)"})
        )
        wind_by_source["Avg Wind (km/h)"] = wind_by_source["Avg Wind (km/h)"].round(1)

    if all(c in df.columns for c in ["City", "SourceWebsite", "Temperature_C"]):
        disagreement = (
            df.dropna(subset=["City", "SourceWebsite", "Temperature_C"])
            .groupby(["City", "SourceWebsite"], as_index=False)["Temperature_C"]
            .mean()
        )
        disagreement = (
            disagreement.groupby("City", as_index=False)["Temperature_C"]
            .agg(["max", "min", "count"])
            .reset_index()
            .rename(columns={"count": "Source Count"})
        )
        disagreement["Temp Disagreement (C)"] = (disagreement["max"] - disagreement["min"]).round(2)
        disagreement = disagreement.sort_values("Temp Disagreement (C)", ascending=False)

    return temp_by_source, wind_by_source, disagreement


def best_hour_analysis(_df: pd.DataFrame) -> pd.DataFrame:
    """Find best hour by city when timestamp detail is sufficient."""
    df = _df.copy()
    if df.empty or "ScrapeDateTime" not in df.columns or "Comfort Score" not in df.columns:
        return pd.DataFrame()

    temp = df.dropna(subset=["ScrapeDateTime", "City"]).copy()
    if temp.empty:
        return pd.DataFrame()

    temp["Hour"] = temp["ScrapeDateTime"].dt.hour
    if temp["Hour"].nunique() < 2:
        return pd.DataFrame()

    hourly = (
        temp.groupby(["City", "Hour"], as_index=False)["Comfort Score"]
        .mean()
        .rename(columns={"Comfort Score": "Avg Comfort Score"})
    )
    hourly["Avg Comfort Score"] = hourly["Avg Comfort Score"].round(1)

    idx = hourly.groupby("City")["Avg Comfort Score"].idxmax()
    best = hourly.loc[idx].sort_values("Avg Comfort Score", ascending=False).reset_index(drop=True)
    return best.rename(columns={"Hour": "Best Hour"})


def apply_sort(ranking: pd.DataFrame, option: str) -> pd.DataFrame:
    """Sort ranking table based on selected option."""
    if ranking.empty:
        return ranking

    if option == "Comfort Score (High to Low)" and "Comfort Score" in ranking.columns:
        return ranking.sort_values("Comfort Score", ascending=False)
    if option == "Comfort Score (Low to High)" and "Comfort Score" in ranking.columns:
        return ranking.sort_values("Comfort Score", ascending=True)
    if option == "Temperature (High to Low)" and "Avg Temperature_C" in ranking.columns:
        return ranking.sort_values("Avg Temperature_C", ascending=False)
    if option == "Humidity (Low to High)" and "Avg Humidity_%" in ranking.columns:
        return ranking.sort_values("Avg Humidity_%", ascending=True)
    if option == "City (A-Z)" and "City" in ranking.columns:
        return ranking.sort_values("City", ascending=True)
    return ranking


def get_high_level_insights(_ranking: pd.DataFrame) -> dict:
    """Create practical quick insights from ranking table."""
    ranking = _ranking.copy()
    insights = {
        "best_city": "N/A",
        "worst_city": "N/A",
        "most_humid_city": "N/A",
        "hottest_feels_like_city": "N/A",
        "good_cities": "None",
        "avoid_cities": "None",
    }

    if ranking.empty:
        return insights

    best = ranking.sort_values("Comfort Score", ascending=False).head(1)
    worst = ranking.sort_values("Comfort Score", ascending=True).head(1)

    if not best.empty:
        insights["best_city"] = str(best.iloc[0]["City"])
    if not worst.empty:
        insights["worst_city"] = str(worst.iloc[0]["City"])

    if "Avg Humidity_%" in ranking.columns:
        top_humidity = ranking.sort_values("Avg Humidity_%", ascending=False).head(1)
        if not top_humidity.empty:
            insights["most_humid_city"] = str(top_humidity.iloc[0]["City"])

    if "Avg FeelsLike_C" in ranking.columns:
        top_feels = ranking.sort_values("Avg FeelsLike_C", ascending=False).head(1)
        if not top_feels.empty:
            insights["hottest_feels_like_city"] = str(top_feels.iloc[0]["City"])

    good = ranking[ranking["Travel Recommendation"].isin(["Ideal", "Good"])]["City"].head(8).tolist()
    avoid = ranking[ranking["Travel Recommendation"] == "Avoid"]["City"].head(8).tolist()

    if good:
        insights["good_cities"] = ", ".join(good)
    if avoid:
        insights["avoid_cities"] = ", ".join(avoid)

    return insights


def add_quick_trip_tips(city_row: pd.Series) -> list[str]:
    """Generate practical travel tips based on city averages."""
    tips = []

    temp = city_row.get("Avg Temperature_C")
    humid = city_row.get("Avg Humidity_%")
    wind = city_row.get("Avg WindSpeed_kmh")
    rain = city_row.get("Avg Precipitation_mm") if "Avg Precipitation_mm" in city_row else None
    aqi = city_row.get("Avg AQI") if "Avg AQI" in city_row else None


# Temperature-based advice
    if pd.notna(temp):
        if temp >= 38:
            tips.append("Extreme heat expected — avoid outdoor activity during peak hours (12–4 PM).")
            tips.append("Use high SPF sunscreen and wear UV-protective clothing.")
        elif temp >= 30:
            tips.append("Hot weather — pack light, breathable cotton/linen clothes.")
            tips.append("Stay hydrated and carry a reusable water bottle.")
        elif temp >= 20:
            tips.append("Warm and comfortable — light layers are ideal.")
        elif temp >= 10:
            tips.append("Cool weather — bring a light jacket or hoodie.")
        elif temp >= 0:
            tips.append("Cold conditions — wear warm layers and insulated outerwear.")
        else:
            tips.append("Freezing temperatures — heavy winter clothing, gloves, and thermal layers required.")

#  Humidity-based advice
    if pd.notna(humid):
        if humid >= 85:
            tips.append("Very high humidity — expect sticky conditions and reduced comfort.")
            tips.append("Choose ultra-breathable fabrics and avoid heavy meals.")
        elif humid >= 70:
            tips.append("High humidity — breathable clothes recommended.")
        elif humid <= 30:
            tips.append("Dry air — stay hydrated and consider moisturizer for skin protection.")
            tips.append("Dry throat possible — carry water or lozenges.")

#  Wind-based advice
    if pd.notna(wind):
        if wind >= 40:
            tips.append("Strong winds expected — avoid loose items and wear wind-resistant jacket.")
        elif wind >= 25:
            tips.append("Windy conditions — carry a light windbreaker.")
        elif wind >= 15:
            tips.append("Light breeze — comfortable outdoor conditions.")

#  Rain / weather risk layer (if you have precipitation data)
        if "rain" in globals() and pd.notna(rain):
            if rain >= 80:
                tips.append("Heavy rain expected — waterproof jacket and umbrella required.")
            elif rain >= 40:
                tips.append("Possible rain — carry an umbrella or light raincoat.")
            elif rain > 0:
                tips.append("Light rain possible — be prepared for brief showers.")

# Air quality (if available)
        if "aqi" in globals() and pd.notna(aqi):
            if aqi >= 150:
                tips.append("Unhealthy air quality — limit outdoor activity and consider a mask.")
            elif aqi >= 100:
                tips.append("Moderate pollution — sensitive individuals should reduce exposure.")
            elif aqi >= 50:
                tips.append("Acceptable air quality — generally safe for outdoor activities.")

# fallback
        if not tips:
            tips.append("Weather is stable — no special packing precautions needed.")

    return tips


def parse_trip_request(request: str) -> dict:
    """Extract simple travel preferences from a free-text request."""
    text = (request or "").strip().lower()
    prefs = {
        "target_temp": 22.0,
        "humidity_pref": "balanced",
        "wind_pref": "balanced",
        "strict_recommendation": False,
        "keywords": [],
    }

    if not text:
        return prefs

    if any(word in text for word in ["cool", "cooler", "cold", "chilly", "fresh"]):
        prefs["target_temp"] = 16.0
        prefs["keywords"].append("cool weather")
    elif any(word in text for word in ["warm", "hot", "summer", "beach", "sunny"]):
        prefs["target_temp"] = 28.0
        prefs["keywords"].append("warm weather")

    if any(word in text for word in ["dry", "not humid", "low humidity", "desert"]):
        prefs["humidity_pref"] = "dry"
        prefs["keywords"].append("lower humidity")
    elif any(word in text for word in ["humid", "tropical"]):
        prefs["humidity_pref"] = "humid"
        prefs["keywords"].append("humid atmosphere")

    if any(word in text for word in ["calm", "not windy", "low wind", "quiet"]):
        prefs["wind_pref"] = "calm"
        prefs["keywords"].append("calmer wind")
    elif any(word in text for word in ["windy", "breeze", "breezy"]):
        prefs["wind_pref"] = "windy"
        prefs["keywords"].append("more breeze")

    if any(word in text for word in ["best", "ideal", "perfect", "top", "excellent"]):
        prefs["strict_recommendation"] = True

    temp_match = re.search(r"(\d{1,2})\s*(?:c|°|degrees?)", text)
    if temp_match:
        prefs["target_temp"] = float(temp_match.group(1))
        prefs["keywords"].append(f"around {int(prefs['target_temp'])}C")

    return prefs


def build_ai_recommendations(ranking_df: pd.DataFrame, best_hour_df: pd.DataFrame, request: str, top_k: int = 3) -> pd.DataFrame:
    """Generate an AI-style ranked shortlist from a natural-language travel request."""
    if ranking_df.empty:
        return pd.DataFrame()

    prefs = parse_trip_request(request)
    rec_df = ranking_df.copy()

    if prefs["strict_recommendation"] and "Travel Recommendation" in rec_df.columns:
        rec_df = rec_df[rec_df["Travel Recommendation"].isin(["Ideal", "Good"])]
        if rec_df.empty:
            rec_df = ranking_df.copy()

    rec_df["AI Match Score"] = rec_df["Comfort Score"].fillna(0).astype(float)

    if "Avg Temperature_C" in rec_df.columns:
        rec_df["AI Match Score"] -= (rec_df["Avg Temperature_C"] - prefs["target_temp"]).abs() * 1.6

    if "Avg Humidity_%" in rec_df.columns:
        if prefs["humidity_pref"] == "dry":
            rec_df["AI Match Score"] -= rec_df["Avg Humidity_%"].fillna(50) * 0.12
        elif prefs["humidity_pref"] == "humid":
            rec_df["AI Match Score"] -= (rec_df["Avg Humidity_%"].fillna(50) - 70).abs() * 0.08
        else:
            rec_df["AI Match Score"] -= (rec_df["Avg Humidity_%"].fillna(50) - 50).abs() * 0.05

    if "Avg WindSpeed_kmh" in rec_df.columns:
        if prefs["wind_pref"] == "calm":
            rec_df["AI Match Score"] -= rec_df["Avg WindSpeed_kmh"].fillna(0) * 0.25
        elif prefs["wind_pref"] == "windy":
            rec_df["AI Match Score"] -= (rec_df["Avg WindSpeed_kmh"].fillna(0) - 22).abs() * 0.12
        else:
            rec_df["AI Match Score"] -= (rec_df["Avg WindSpeed_kmh"].fillna(0) - 14).abs() * 0.06

    if not best_hour_df.empty and "City" in best_hour_df.columns:
        rec_df = rec_df.merge(best_hour_df[["City", "Best Hour", "Avg Comfort Score"]], on="City", how="left")

    rec_df["AI Match Score"] = rec_df["AI Match Score"].round(1)
    rec_df = rec_df.sort_values(["AI Match Score", "Comfort Score"], ascending=[False, False]).head(top_k).copy()

    reasons = []
    for _, row in rec_df.iterrows():
        parts = []
        if pd.notna(row.get("Avg Temperature_C")):
            parts.append(f"{row['Avg Temperature_C']:.1f}C average temperature")
        if pd.notna(row.get("Avg Humidity_%")):
            parts.append(f"{row['Avg Humidity_%']:.1f}% humidity")
        if pd.notna(row.get("Comfort Score")):
            parts.append(f"comfort score {row['Comfort Score']:.1f}")
        if pd.notna(row.get("Best Hour")):
            parts.append(f"best hour around {int(row['Best Hour']):02d}:00")
        reasons.append("; ".join(parts))

    rec_df["Why it matches"] = reasons
    return rec_df


def build_ai_summary(request: str, rec_df: pd.DataFrame) -> str:
    """Create a concise recommendation narrative."""
    prefs = parse_trip_request(request)
    pref_bits = prefs["keywords"] if prefs["keywords"] else ["balanced weather"]

    if rec_df.empty:
        return f"I couldn't find a strong match for {', '.join(pref_bits)} in the current filtered data."

    city_list = ", ".join(rec_df["City"].tolist())
    return f"Based on your request for {', '.join(pref_bits)}, the strongest matches right now are {city_list}."


@st.cache_data(show_spinner=False)
def load_summary_report(path: str = SUMMARY_REPORT_PATH) -> pd.DataFrame:
    """Load saved summary report from processed outputs."""
    return make_arrow_compatible(load_data(path))


@st.cache_data(show_spinner=False)
def load_condition_analysis(path: str = CONDITION_ANALYSIS_PATH) -> pd.DataFrame:
    """Load saved normalized-condition analysis output."""
    df = load_data(path)
    if "ScrapeDateTime" in df.columns:
        df["ScrapeDateTime"] = parse_datetime(df["ScrapeDateTime"])
    return make_arrow_compatible(df)


@st.cache_data(show_spinner=False)
def run_ml_analysis_dashboard(data_path: str = DATA_PATH) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Run ML analysis module and return model comparison, best model, and top feature importances."""
    ml_module = load_analysis_module("analysis_ml_dashboard", os.path.join("analysis", "ml_analysis.py"))
    if ml_module is None:
        return pd.DataFrame(), "Unavailable", pd.DataFrame()

    df = ml_module.load_data(data_path)
    X, y = ml_module.prepare_features(df)
    if len(X) < 10:
        return pd.DataFrame(), "Not enough data", pd.DataFrame()

    X_train, X_test, y_train, y_test = ml_module.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results_df, best_model_name, best_pipeline = ml_module.train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    if results_df is None:
        results_df = pd.DataFrame()
    else:
        results_df = make_arrow_compatible(results_df.copy().round(4))

    importance_df = pd.DataFrame()
    if best_pipeline is None:
        return results_df, str(best_model_name), importance_df

    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    if hasattr(model, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(12)
            .reset_index(drop=True)
        )
        importance_df["Importance"] = importance_df["Importance"].round(4)
        importance_df = make_arrow_compatible(importance_df)

    return results_df, str(best_model_name), importance_df


def run_ml_analysis_dashboard_from_df(_df: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Run ML analysis against the currently filtered dashboard dataframe."""
    df = _df.copy()
    ml_module = load_analysis_module("analysis_ml_dashboard", os.path.join("analysis", "ml_analysis.py"))
    if ml_module is None:
        return pd.DataFrame(), "Unavailable", pd.DataFrame()

    X, y = ml_module.prepare_features(df)
    if len(X) < 10:
        return pd.DataFrame(), "Not enough data", pd.DataFrame()

    X_train, X_test, y_train, y_test = ml_module.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results_df, best_model_name, best_pipeline = ml_module.train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    if results_df is None:
        results_df = pd.DataFrame()
    else:
        results_df = make_arrow_compatible(results_df.copy().round(4))

    importance_df = pd.DataFrame()
    if best_pipeline is None:
        return results_df, str(best_model_name), importance_df

    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    if hasattr(model, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(12)
            .reset_index(drop=True)
        )
        importance_df["Importance"] = importance_df["Importance"].round(4)
        importance_df = make_arrow_compatible(importance_df)

    return results_df, str(best_model_name), importance_df


def build_filtered_summary_report(_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact summary table from the current filtered dashboard scope."""
    df = _df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Metric", "Value"])

    rows = [{"Metric": "TotalRows", "Value": len(df)}]

    if "SourceWebsite" in df.columns:
        for source, count in df["SourceWebsite"].value_counts().items():
            rows.append({"Metric": f"RowsBySource:{source}", "Value": int(count)})

    if "Country" in df.columns:
        rows.append({"Metric": "UniqueCountries", "Value": int(df["Country"].nunique())})

    if "City" in df.columns:
        rows.append({"Metric": "UniqueCities", "Value": int(df["City"].nunique())})

    if "Comfort Score" in df.columns:
        rows.append(
            {
                "Metric": "AverageComfortScore",
                "Value": round(pd.to_numeric(df["Comfort Score"], errors="coerce").mean(), 2),
            }
        )

    return pd.DataFrame(rows)


def filter_condition_analysis_by_scope(
    _condition_df: pd.DataFrame,
    _filtered_weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict condition-analysis rows to the same current dashboard scope."""
    condition_df = _condition_df.copy()
    filtered_weather_df = _filtered_weather_df.copy()

    if condition_df.empty or filtered_weather_df.empty:
        return pd.DataFrame()

    shared_cols = [
        col for col in ["City", "Country", "SourceWebsite", "ScrapeDateTime"]
        if col in condition_df.columns and col in filtered_weather_df.columns
    ]
    if not shared_cols:
        return condition_df

    scope_keys = filtered_weather_df[shared_cols].drop_duplicates()
    return condition_df.merge(scope_keys, on=shared_cols, how="inner")


def build_eda_snapshot(_weather_df: pd.DataFrame, _ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Create compact EDA metrics for dashboard display."""
    weather_df = _weather_df.copy()
    ranking_df = _ranking_df.copy()
    if weather_df.empty:
        return pd.DataFrame()

    rows = [
        {"Metric": "Total weather rows", "Value": len(weather_df)},
        {"Metric": "Cities covered", "Value": int(weather_df["City"].nunique()) if "City" in weather_df.columns else 0},
        {"Metric": "Countries covered", "Value": int(weather_df["Country"].nunique()) if "Country" in weather_df.columns else 0},
        {"Metric": "Sources covered", "Value": int(weather_df["SourceWebsite"].nunique()) if "SourceWebsite" in weather_df.columns else 0},
    ]

    if "Temperature_C" in weather_df.columns:
        rows.append({"Metric": "Average temperature (C)", "Value": round(pd.to_numeric(weather_df["Temperature_C"], errors="coerce").mean(), 2)})
    if "Humidity_%" in weather_df.columns:
        rows.append({"Metric": "Average humidity (%)", "Value": round(pd.to_numeric(weather_df["Humidity_%"], errors="coerce").mean(), 2)})
    if "Comfort Score" in weather_df.columns:
        rows.append({"Metric": "Average comfort score", "Value": round(pd.to_numeric(weather_df["Comfort Score"], errors="coerce").mean(), 2)})
    if not ranking_df.empty and "Travel Recommendation" in ranking_df.columns:
        rows.append({"Metric": "Ideal or Good cities", "Value": int(ranking_df["Travel Recommendation"].isin(["Ideal", "Good"]).sum())})

    return pd.DataFrame(rows)


def build_score_band_summary(_ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Bucket comfort scores into simple bands for overview distribution."""
    ranking_df = _ranking_df.copy()
    if ranking_df.empty or "Comfort Score" not in ranking_df.columns:
        return pd.DataFrame()

    score_bands = pd.cut(
        ranking_df["Comfort Score"],
        bins=[0, 40, 60, 80, 100],
        labels=["0-40", "40-60", "60-80", "80-100"],
        include_lowest=True,
    )
    return (
        score_bands.value_counts(sort=False)
        .rename_axis("Score Band")
        .reset_index(name="City Count")
    )


def build_source_health_summary(_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize source coverage and average conditions."""
    df = _df.copy()
    if df.empty or "SourceWebsite" not in df.columns:
        return pd.DataFrame()

    source_summary = df.groupby("SourceWebsite", as_index=False).agg(
        Records=("SourceWebsite", "size"),
        Cities=("City", pd.Series.nunique),
    )

    if "Country" in df.columns:
        country_counts = (
            df.groupby("SourceWebsite")["Country"]
            .nunique()
            .rename("Countries")
            .reset_index()
        )
        source_summary = source_summary.merge(country_counts, on="SourceWebsite", how="left")

    for metric in ["Temperature_C", "Humidity_%", "WindSpeed_kmh", "Comfort Score"]:
        if metric in df.columns:
            metric_summary = (
                df.groupby("SourceWebsite")[metric]
                .mean()
                .round(1)
                .rename(
                    {
                        "Temperature_C": "Avg Temp (C)",
                        "Humidity_%": "Avg Humidity (%)",
                        "WindSpeed_kmh": "Avg Wind (km/h)",
                        "Comfort Score": "Avg Comfort",
                    }[metric]
                )
                .reset_index()
            )
            source_summary = source_summary.merge(metric_summary, on="SourceWebsite", how="left")

    missing_condition = (
        df.groupby("SourceWebsite")["Condition"].apply(lambda s: s.isna().mean() * 100)
        .round(1)
        .rename("Condition Missing (%)")
        .reset_index()
        if "Condition" in df.columns
        else pd.DataFrame()
    )
    if not missing_condition.empty:
        source_summary = source_summary.merge(missing_condition, on="SourceWebsite", how="left")

    return make_arrow_compatible(
        source_summary.sort_values(["Cities", "Records"], ascending=[False, False])
    )


def build_country_summary(_ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate average comfort and city counts by country."""
    ranking_df = _ranking_df.copy()
    if ranking_df.empty or "Country" not in ranking_df.columns:
        return pd.DataFrame()

    country_summary = (
        ranking_df.dropna(subset=["Country"])
        .groupby("Country", as_index=False)
        .agg(
            Cities=("City", "nunique"),
            Avg_Comfort=("Comfort Score", "mean"),
            Avg_Temp=("Avg Temperature_C", "mean"),
        )
        .rename(
            columns={
                "Avg_Comfort": "Avg Comfort Score",
                "Avg_Temp": "Avg Temperature (C)",
            }
        )
    )
    for col in ["Avg Comfort Score", "Avg Temperature (C)"]:
        if col in country_summary.columns:
            country_summary[col] = country_summary[col].round(1)
    return country_summary.sort_values(["Avg Comfort Score", "Cities"], ascending=[False, False])


def build_all_cities_table(_filtered_df: pd.DataFrame, _ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Create an operational all-cities table with source coverage."""
    filtered_df = _filtered_df.copy()
    ranking_df = _ranking_df.copy()
    if ranking_df.empty:
        return pd.DataFrame()

    details = ranking_df.copy()

    if "City" in filtered_df.columns:
        row_counts = (
            filtered_df.groupby("City")
            .size()
            .rename("Records")
            .reset_index()
        )
        details = details.merge(row_counts, on="City", how="left")

    if "SourceWebsite" in filtered_df.columns:
        source_counts = (
            filtered_df.groupby("City")["SourceWebsite"]
            .nunique()
            .rename("Source Count")
            .reset_index()
        )
        details = details.merge(source_counts, on="City", how="left")

    return details


def build_source_coverage_matrix(_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot city/source coverage counts for a compact reliability matrix."""
    df = _df.copy()
    if df.empty or not all(c in df.columns for c in ["City", "SourceWebsite"]):
        return pd.DataFrame()

    coverage = (
        df.groupby(["City", "SourceWebsite"])
        .size()
        .rename("Rows")
        .reset_index()
        .pivot(index="City", columns="SourceWebsite", values="Rows")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    return coverage


def filter_all_cities_table(
    _all_cities_df: pd.DataFrame,
    city_query: str,
    recommendation_filter: tuple[str, ...],
    country_filter_quick: tuple[str, ...],
) -> pd.DataFrame:
    """Apply the All Cities tab filters to the prepared city table."""
    all_cities_df = _all_cities_df.copy()

    if city_query and "City" in all_cities_df.columns:
        all_cities_df = all_cities_df[
            all_cities_df["City"].astype(str).str.contains(city_query, case=False, na=False)
        ]
    if recommendation_filter and "Travel Recommendation" in all_cities_df.columns:
        all_cities_df = all_cities_df[all_cities_df["Travel Recommendation"].isin(recommendation_filter)]
    if country_filter_quick and "Country" in all_cities_df.columns:
        all_cities_df = all_cities_df[all_cities_df["Country"].isin(country_filter_quick)]

    return all_cities_df


def build_alerts(ranking_df: pd.DataFrame, disagreement_df: pd.DataFrame) -> list[str]:
    """Generate randomized but meaningful dashboard alerts."""

    if ranking_df.empty:
        return ["No data available for alerts."]

    pool = []

    # Temperature alerts
    if "Avg Temperature_C" in ranking_df.columns:
        hot_cities = ranking_df[ranking_df["Avg Temperature_C"] >= 32]["City"].dropna().head(5).tolist()
        if hot_cities:
            city_list = ", ".join(hot_cities)
            pool.append(f"High heat detected in: {city_list}.")
            pool.append(f"Rising temperatures affecting: {city_list}.")
            pool.append(f"Hot weather conditions reported in: {city_list}.")            
            pool.append(f"Extreme warmth expected in: {city_list}.")

    # Humidity alerts
    if "Avg Humidity_%" in ranking_df.columns:
        humid_cities = ranking_df[ranking_df["Avg Humidity_%"] >= 80]["City"].dropna().head(5).tolist()
        if humid_cities:
            city_list = ", ".join(humid_cities)
            pool.append(f"Very humid conditions in: {city_list}.")
            pool.append(f"Sticky air conditions affecting: {city_list}.")
            pool.append(f"High moisture levels detected in: {city_list}.")

    # Travel risk alerts
    if "Travel Recommendation" in ranking_df.columns:
        avoid_count = int((ranking_df["Travel Recommendation"] == "Avoid").sum())
        avoid_cities = []
        if avoid_count > 0:
            avoid_cities = ranking_df[ranking_df["Travel Recommendation"] == "Avoid"]["City"].dropna().head(5).tolist()

        if avoid_cities:
            city_list = ", ".join(avoid_cities)
            pool.append(f"Travel warnings active in: {city_list}.")
            pool.append(f"Travel warnings active for: {city_list}.")
            pool.append(f"Several regions flagged as unsafe for travel.")

    if (
        not disagreement_df.empty
        and "Temp Disagreement (C)" in disagreement_df.columns
    ):
        volatile = disagreement_df[
            disagreement_df["Temp Disagreement (C)"] >= 5
        ]["City"].dropna().head(5).tolist()

        if volatile:
            city_list = ", ".join(volatile)
            pool.append(f"Conflicting temperature data in: {city_list}.")
            pool.append(f"Source disagreement detected across: {city_list}.")
            pool.append(f"Inconsistent readings found in multiple sources.")

    # fallback safety net
    if not pool:
        return ["No major weather or data issues detected in the current view."]

    # random selection (max 3 alerts)
    return random.sample(pool, k=min(3, len(pool)))


PALETTE = {
    "paper": "#f8f5f1",
    "panel": "#fbfaf8",
    "sidebar": "#f1f0ef",
    "border": "#e8dfd7",
    "text": "#4c4a4a",
    "muted": "#8e8883",
    "accent": "#f05a5a",
    "accent_soft": "#fde6e4",
    "blue": "#48b5cc",
    "teal": "#56c5c1",
    "mint": "#95ccb2",
    "yellow": "#f7de8a",
    "stone": "#d9dfdf",
}


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Manrope', sans-serif;
                background-color: white !important;
                color: black !important;
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(255,255,255,0.95), transparent 32%),
                    linear-gradient(180deg, {PALETTE["paper"]} 0%, #f6f2ee 100%);
                color: {PALETTE["text"]};
            }}

            .block-container {{
                max-width: 1420px;
                padding-top: 1.35rem;
                padding-bottom: 2.5rem;
            }}

            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #f2f2f3 0%, {PALETTE["sidebar"]} 100%);
                border-right: 1px solid rgba(0, 0, 0, 0.04);
            }}

            section[data-testid="stSidebar"] .block-container {{
                padding-top: 1.2rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            h1, h2, h3, p, label, span, div {{
                color: {PALETTE["text"]};
            }}

            .hero-shell {{
                background: rgba(255, 252, 249, 0.78);
                border: 1px solid {PALETTE["border"]};
                border-radius: 28px;
                padding: 1.5rem 1.7rem 1.35rem 1.7rem;
                margin-bottom: 2rem;
                box-shadow: 0 18px 34px rgba(188, 180, 172, 0.12);
            }}

            .hero-kicker {{
                display: inline-block;
                border: 1px solid rgba(240, 90, 90, 0.35);
                background: {PALETTE["accent_soft"]};
                color: {PALETTE["accent"]};
                border-radius: 999px;
                padding: 0.28rem 0.72rem;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.01em;
                margin-bottom: 0.9rem;
            }}

            .hero-title {{
                font-size: 2.1rem;
                line-height: 1.04;
                font-weight: 800;
                margin: 0;
                color: #4a4949;
            }}

            .hero-sub {{
                margin-top: 0.55rem;
                font-size: 0.98rem;
                color: {PALETTE["muted"]};
                max-width: 900px;
            }}

            .pill-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.8rem;
                margin: 0.35rem 0 1rem 0;
            }}

            .pill {{
                border: 1px solid rgba(240, 90, 90, 0.42);
                color: {PALETTE["accent"]};
                background: rgba(255, 248, 247, 0.92);
                padding: 0.48rem 1rem;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 600;
            }}

            .metric-tile {{
                background: rgba(255, 251, 248, 0.9);
                border: 1px solid var(--border); /* adjust if needed */
                border-radius: 22px;
                padding: 1.2rem 1.25rem 1.1rem;
                
                min-height: 132px;
                height: 132px;          /* 🔥 forces uniform height */
                
                box-shadow: 0 14px 24px rgba(188, 180, 172, 0.10);

                display: flex;          /* 🔥 enables centering */
                flex-direction: column;
                justify-content: center; /* vertical centering */
                align-items: center;     /* horizontal centering */

                text-align: center;      /* fixes multi-line text alignment */
                        }}

            .metric-label {{
                color: {PALETTE["muted"]};
                font-size: 0.85rem;
                margin-bottom: 0.35rem;
            }}

            .metric-value {{
                color: #444241;
                font-size: 1.7rem;
                font-weight: 800;
                line-height: 1.1;
            }}

            .metric-note {{
                color: {PALETTE["muted"]};
                font-size: 0.82rem;
                margin-top: 0.35rem;
            }}

            .sidebar-card {{
                background: rgba(255, 252, 249, 0.86);
                border: 1px solid {PALETTE["border"]};
                border-radius: 18px;
                padding: 0.9rem 0.95rem;
                margin: 0.35rem 0 1rem 0;
            }}

            .sidebar-title {{
                font-size: 0.86rem;
                font-weight: 800;
                color: #504d4c;
                margin-bottom: 0.35rem;
            }}

            .sidebar-copy {{
                font-size: 0.82rem;
                color: {PALETTE["muted"]};
                line-height: 1.45;
            }}

            .panel {{
                background: rgba(255, 252, 249, 0.76);
                border: 1px solid {PALETTE["border"]};
                border-radius: 26px;
                padding: 1rem 1rem 0.7rem 1rem;
                box-shadow: 0 16px 28px rgba(188, 180, 172, 0.10);
                margin-bottom: 1rem;
            }}

            .panel-title {{
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.15rem;
                color: #4d4a49;
            }}

            .panel-copy {{
                color: {PALETTE["muted"]};
                font-size: 0.9rem;
                margin-bottom: 0.35rem;
            }}

            [data-testid="stTabs"] [data-baseweb="tab-list"] {{
                gap: 0.8rem;
                margin-top: 0.7rem;
                margin-bottom: 0.55rem;
            }}

            [data-testid="stTabs"] [data-baseweb="tab"] {{
                background: transparent;
                border: none;
                border-radius: 0;
                padding: 0.62rem 1.15rem;
                color: {PALETTE["muted"]};
            }}

            [data-testid="stTabs"] [aria-selected="true"] {{
                border-bottom: 2px solid rgba(240, 90, 90, 0.7) !important;
                color: {PALETTE["accent"]} !important;
                background: transparent !important;
            }}

            .stSelectbox div[data-baseweb="select"] > div,
            .stMultiSelect div[data-baseweb="select"] > div,
            .stDateInput > div > div,
            .stTextInput > div > div > input {{
                background: rgba(255,255,255,0.92);
                border-radius: 14px;
                border: 1px solid {PALETTE["border"]};
            }}

            .stSlider [data-baseweb="slider"] > div > div > div {{
                background: {PALETTE["accent"]};
            }}

            .stDownloadButton button,
            .stButton button {{
                background: {PALETTE["accent_soft"]};
                color: {PALETTE["accent"]};
                border: 1px solid rgba(240, 90, 90, 0.35);
                border-radius: 999px;
                font-weight: 700;
            }}

            div[data-testid="stDataFrame"] {{
                border: 1px solid {PALETTE["border"]};
                border-radius: 18px;
                overflow: hidden;
            }}

            div[data-testid="stMetric"] {{
                background: rgba(255, 251, 248, 0.86);
                border: 1px solid {PALETTE["border"]};
                border-radius: 18px;
                padding: 0.8rem 0.9rem;
            }}

            .active-filter-bar {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.75rem;
                margin: 0 0 1.5rem 0;
                justify-content: center;
            }}

            .active-filter {{
                background: rgba(255,255,255,0.78);
                 border: 1px solid var(--border);
                border-radius: 999px;
                color: {PALETTE["muted"]};
                font-size: 0.8rem;
                padding: 0.6rem 1.2rem;
            }}

            .alert-card {{
                background: rgba(255, 247, 244, 0.95);
                border: 1px solid rgba(240, 90, 90, 0.2);
                border-left: 4px solid {PALETTE["accent"]};
                border-radius: 18px;
                padding: 0.9rem 1rem;
                margin-bottom: 0.85rem;
            }}

            .section-note {{
                color: {PALETTE["muted"]};
                font-size: 0.88rem;
                margin-bottom: 0.4rem;
            }}

            div[data-testid="stHorizontalBlock"] {{
                gap: 1.35rem;
            }}

            div[data-testid="stTabs"] {{
                margin-top: 0.9rem;
            }}

            .stMarkdown ul {{
                padding-left: 1.2rem;
            }}

            .select-wrapper {{
                position: relative;
                top: -20px;  
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    


def metric_tile(title: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_start(title: str, copy: str = "") -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def style_figure(fig, *, height: int | None = None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(family="Manrope, sans-serif", color=PALETTE["text"]),
        title_font=dict(size=20, color=PALETTE["text"]),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#fffaf6", font_color=PALETTE["text"]),
        margin=dict(l=22, r=22, t=60, b=24),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=PALETTE["border"],
            tickfont=dict(color=PALETTE["muted"]),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(232, 223, 215, 0.65)",
            zeroline=False,
            tickfont=dict(color=PALETTE["muted"]),
        ),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


st.set_page_config(
    page_title="Smart Weather Travel & Comfort Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f8fafc;
    color: #000000;
}

/* Text colors */
body, p, h1, h2, h3, h4, h5, h6, span, div {
    color: #000000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f1f5f9;
}

/* Buttons */
.stButton > button {
    background-color: #fda4af;
    color: #000000;
    border-radius: 10px;
    border: none;
}
.stButton > button:hover {
    background-color: #fb7185;
}

/* Clear filters button - full width */
section[data-testid="stSidebar"] .stButton {
    width: 100%;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
}

/* Sliders - light red */
.stSlider > div[data-baseweb="slider"] > div {
    color: #fca5a5;
}
.stSlider [data-baseweb="slider"] {
    background-color: transparent !important;
}

/* Tabs - light red */
.stTabs [data-baseweb="tab"] {
    color: #000000;
    transition: all 0.3s ease;
    background-color: transparent;
    border: none;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #000000 !important;
    border-bottom: 2px solid #fca5a5;
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #000000;
    background-color: transparent;
}

/* Input boxes - light grey background */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div,
.stSelectbox input,
.stRadio > div > label > input,
.stRadio > div > label > span {
    background-color: #f3f4f6 !important;
    color: #000000 !important;
}

/* Radio buttons - light grey */
.stRadio > div {
    background-color: transparent !important;
}
.stRadio > div > label {
    background-color: #f3f4f6 !important;
    padding: 10px 12px;
    border-radius: 6px;
    margin-bottom: 6px;
    color: #000000 !important;
}

/* Dataframe - light backgrounds */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    background-color: #f5f9fc !important;
}
[data-testid="stDataFrame"] table {
    background-color: #ffffff !important;
}
[data-testid="stDataFrame"] tr {
    background-color: #ffffff !important;
}
[data-testid="stDataFrame"] th {
    background-color: #ecf0f5 !important;
    color: #000000 !important;
}
[data-testid="stDataFrame"] td {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Download button */
.stDownloadButton > button {
    background-color: #f0f9ff !important;
    color: #000000 !important;
    border: 1px solid #dbeafe !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* Expander - light red */
.streamlit-expanderHeader {
    background-color: #fecaca !important;
}
.streamlit-expanderHeader:hover {
    background-color: #fca5a5 !important;
}

/* Subheader hover - light red */
h2, h3 {
    transition: color 0.3s ease;
}
h2:hover, h3:hover {
    color: #fca5a5 !important;
}

/* Plotly graphs - light background and black text */
.plotly {
    background-color: #ffffff !important;
}
.plotly-graph {
    background-color: #ffffff !important;
}
[data-testid="plotly.modebar"] {
    background-color: #f8fafc !important;
}

/* Black text in all SVG text elements */
svg text {
    fill: #000000 !important;
    color: #000000 !important;
}
            
.travel-insights {
    display: block;
    width: 100%;
    flex-direction: column;
    align-items: center;     
    text-align: center !important;     
    gap: 14px;               
    padding-top: 10px;
    line-height: 1.9;
}
            
.travel-insights span {
    display: block;
    text-align: center;
}

.travel-item {
    font-size: 0.95rem;
    color: #111827;
    display: block;
    padding: 2px 0;
}

.travel-item b {
    color: #111827;
    font-weight: 600;
}

.stDivider {
    margin: 12px 0 !important;
}

/* Add spacing to subheaders */
h2, h3 {
    margin-top: 32px !important;
    margin-bottom: 8px !important;
}

/* Spacing for markdown content */
.stMarkdown {
    margin: 12px 0 !important;
}

.packing-tips .stMarkdown,
.packing-tips p,
.packing-tips ul {
    margin-bottom: 4px !important;
}
</style>
""", unsafe_allow_html=True)

inject_styles()

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-kicker">Weather Intelligence</div>
        <div class="hero-title">Smart Weather Travel & Comfort Dashboard</div>
        <div class="hero-sub">
            A working dashboard for our scraping project: evaluate travel comfort, compare weather sources,
            inspect city-level behavior, and make decisions from the processed dataset instead of raw tables.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

weather_df = load_and_prepare_data(DATA_PATH)
if weather_df.empty:
    st.error(
        f"Could not load processed weather data from `{DATA_PATH}`. "
        "Please make sure the file exists and contains data."
    )
    st.stop()

city_options, country_options, source_options = get_filter_options(weather_df)
default_top_n = max(5, min(25, len(city_options))) if city_options else 5

if "dashboard_filters_initialized" not in st.session_state:
    st.session_state.dashboard_filters_initialized = True
    st.session_state.country_filter = []
    st.session_state.source_filter = []
    st.session_state.city_scope = "All cities"
    st.session_state.city_search = ""
    st.session_state.city_filter = []
    st.session_state.use_latest_only = False
    st.session_state.min_comfort = 0
    st.session_state.top_n = default_top_n
    st.session_state.sort_option = "Comfort Score (High to Low)"
    st.session_state.ml_computed = False
    st.session_state.ml_results = pd.DataFrame()
    st.session_state.ml_best_model = "Not yet computed"
    st.session_state.ml_importance = pd.DataFrame()
    st.session_state.ai_submitted_request = "I want a cool, dry city with good comfort for walking outside."

st.sidebar.header("Dashboard Filters")

# Reset button BEFORE widgets are created
if st.sidebar.button("Reset filters", width="stretch"):
    st.session_state.country_filter = []
    st.session_state.source_filter = []
    st.session_state.city_scope = "All cities"
    st.session_state.city_search = ""
    st.session_state.city_filter = []
    st.session_state.use_latest_only = False
    st.session_state.min_comfort = 0
    st.session_state.top_n = default_top_n
    st.session_state.sort_option = "Comfort Score (High to Low)"
    st.rerun()

selected_countries = st.session_state.get("country_filter", [])
selected_sources = st.session_state.get("source_filter", [])
use_latest_only = st.session_state.get("use_latest_only", False)

with st.sidebar.expander("Coverage", expanded=False):
    selected_countries = st.multiselect(
        "Countries",
        country_options,
        key="country_filter",
        help="Limit the dashboard to one or more countries.",
    )
    selected_sources = st.multiselect(
        "Sources",
        source_options,
        key="source_filter",
        help="Compare only the weather providers you want to keep in view.",
    )
    use_latest_only = st.checkbox(
        "Use latest record per city/source",
        key="use_latest_only",
        help="Good for a current snapshot instead of historical rows.",
    )

with st.sidebar.expander("Cities", expanded=False):
    city_scope = st.radio(
        "City selection",
        ["All cities", "Choose cities"],
        key="city_scope",
        help="Keep all cities visible or manually pick a subset.",
    )

    filtered_city_options = get_city_options(
        weather_df,
        tuple(selected_countries),
        tuple(selected_sources),
    )

    if city_scope == "Choose cities":
        selected_cities = st.multiselect(
            "Cities",
            filtered_city_options,
            default=[
                city for city in st.session_state.city_filter
                if city in filtered_city_options
            ],
            key="city_filter",
            help="Select cities to compare.",
        )
    else:
        selected_cities = filtered_city_options
        st.caption(f"Using all matching cities: {len(filtered_city_options)}")

with st.sidebar.expander("Thresholds & ranking", expanded=False):
    min_comfort = st.slider(
        "Minimum comfort score",
        min_value=0,
        max_value=100,
        key="min_comfort",
    )
    top_n = st.slider(
        "Top cities to display",
        min_value=5,
        max_value=default_top_n,
        key="top_n",
    )
    sort_option = st.selectbox(
        "Ranking sort",
        [
            "Comfort Score (High to Low)",
            "Comfort Score (Low to High)",
            "Temperature (High to Low)",
            "Humidity (Low to High)",
            "City (A-Z)",
        ],
        key="sort_option",
    )

filtered_df = apply_dashboard_filters(
    weather_df,
    tuple(selected_countries),
    tuple(selected_cities),
    tuple(selected_sources),
    use_latest_only,
    min_comfort,
)

if filtered_df.empty:
    st.warning("No records match current filters. Try lowering constraints.")
    st.stop()

ranking_df = build_city_ranking(filtered_df)
ranking_df = apply_sort(ranking_df, sort_option)
insights = get_high_level_insights(ranking_df)
snapshot_df = filter_latest_per_city_source(filtered_df)
snapshot_ranking_df = build_city_ranking(snapshot_df)
source_health_df = build_source_health_summary(filtered_df)
source_health_display_df = prepare_display_df(
    source_health_df,
    [
        "Avg Temp (C)",
        "Avg Humidity (%)",
        "Avg Wind (km/h)",
        "Avg Comfort",
        "Condition Missing (%)",
    ],
)
country_summary_df = build_country_summary(ranking_df)
temp_by_source, wind_by_source, disagreement = build_source_summary(filtered_df)
alerts = build_alerts(ranking_df, disagreement)
all_cities_table = build_all_cities_table(filtered_df, ranking_df)
summary_report_df = build_filtered_summary_report(filtered_df)
condition_analysis_df = filter_condition_analysis_by_scope(
    load_condition_analysis(),
    filtered_df,
)

ml_results_df = st.session_state.get("ml_results", pd.DataFrame())
ml_best_model = st.session_state.get("ml_best_model", "Not yet computed")
ml_importance_df = st.session_state.get("ml_importance", pd.DataFrame())

eda_snapshot_df = build_eda_snapshot(filtered_df, ranking_df)

if ranking_df.empty:
    st.warning("Not enough data to build city ranking.")
    st.stop()

avg_comfort = ranking_df["Comfort Score"].mean() if "Comfort Score" in ranking_df.columns else float("nan")
city_count = ranking_df["City"].nunique() if "City" in ranking_df.columns else 0
record_count = len(filtered_df)
snapshot_city_count = snapshot_ranking_df["City"].nunique() if "City" in snapshot_ranking_df.columns and not snapshot_ranking_df.empty else 0
country_count = ranking_df["Country"].nunique() if "Country" in ranking_df.columns else 0
top_city_options = ranking_df["City"].head(min(top_n, len(ranking_df))).tolist()
compare_default = top_city_options[: min(3, len(top_city_options))]

active_filters = [
    f'<span class="active-filter">{city_count} cities in view</span>',
    f'<span class="active-filter">{len(selected_sources)} sources</span>',
    f'<span class="active-filter">{record_count} records</span>',
    f'<span class="active-filter">Comfort >= {min_comfort}</span>',
]
if use_latest_only:
    active_filters.append('<span class="active-filter">Latest records only</span>')
st.markdown(f'<div class="active-filter-bar">{"".join(active_filters)}</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="large")
with c1:
    metric_tile("Best city now", insights["best_city"], "Highest comfort score")
with c2:
    metric_tile("Current snapshot", str(int(snapshot_city_count)), "Cities with latest-source view")
with c3:
    metric_tile("Average comfort", f"{avg_comfort:.1f}" if pd.notna(avg_comfort) else "N/A", "Across filtered cities")
with c4:
    metric_tile("Coverage", f"{int(city_count)} cities - {int(country_count)} countries", "Across current filters")

overview_tab, explorer_tab, source_tab, planner_tab, ai_tab, analytics_tab, all_cities_tab = st.tabs(
    ["Overview", "City Explorer", "Source Quality", "Trip Planner", "AI Recommender", "Insights Center", "All Cities"]
)

with overview_tab:
    panel_start("Overview", "Use this view as the project command center for current rankings, alerts, and country-level summaries.")

    st.markdown('<div class="section-note">Operational alerts from the current filtered dataset</div>', unsafe_allow_html=True)
    for alert in alerts:
        st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("City Ranking")
        ranking_preview = ranking_df.head(top_n)
        st.dataframe(ranking_preview, width="stretch", hide_index=True)

    with right:
        st.subheader("Travel Insights")

        st.markdown(
            f"""
            <p class="travel-insights">

            <span class="travel-item">• Best city for outdoor comfort: <b>{insights['best_city']}</b></span><br>
            <span class="travel-item">• Lowest-comfort city right now: <b>{insights['worst_city']}</b></span><br>
            <span class="travel-item">• City with highest humidity: <b>{insights['most_humid_city']}</b></span><br>
            <span class="travel-item">• City with highest feels-like temperature: <b>{insights['hottest_feels_like_city']}</b></span><br>
            <span class="travel-item">• Good travel options: <b>{insights['good_cities']}</b></span><br>
            <span class="travel-item">• Cities to avoid: <b>{insights['avoid_cities']}</b></span>

            </p>
            """,
            unsafe_allow_html=True
        )
        csv_bytes = ranking_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download current ranking as CSV",
            data=csv_bytes,
            file_name="city_ranking_filtered.csv",
            mime="text/csv",
            width="stretch",
        )

    st.divider()
    _, mid, _ = st.columns([1, 2, 1])

    with mid:
        st.markdown('<div class="select-wrapper">', unsafe_allow_html=True)

        selected_metric = st.selectbox(
            "Primary ranking metric",
            [c for c in ["Comfort Score", "Avg Temperature_C", "Avg FeelsLike_C",
                        "Avg Humidity_%", "Avg WindSpeed_kmh"]
            if c in ranking_df.columns],
            index=0,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    CHART_HEIGHT = 450

    fig_comfort = px.bar(
        ranking_df.head(top_n).sort_values(selected_metric, ascending=False),
        x="City",
        y=selected_metric,
        color="Travel Recommendation" if "Travel Recommendation" in ranking_df.columns else "Comfort Score",
        title=f"{selected_metric} Across Cities",
        color_discrete_map={
            "Ideal": PALETTE["mint"],
            "Good": PALETTE["blue"],
            "Moderate": PALETTE["yellow"],
            "Avoid": PALETTE["accent"],
            "Unknown": PALETTE["stone"],
        },
        text_auto=".1f",
    )

    fig_comfort.update_layout(
        xaxis_title="City",
        yaxis_title=selected_metric,
        height=CHART_HEIGHT
    )

    style_figure(fig_comfort, height=CHART_HEIGHT)

    st.plotly_chart(fig_comfort, use_container_width=True)
    st.divider()


    compare_cities = st.multiselect(
        "Compare cities",
        options=ranking_df["City"].tolist(),
        default=compare_default,
    )

    top_left, top_right = st.columns([1.05, 1])

    with top_left:
        if not country_summary_df.empty:
            st.subheader("Country Comfort Summary")

            st.dataframe(
                country_summary_df.head(10),
                width="stretch",
                hide_index=True
            )

    with top_right:
        st.subheader("Selected City Comparison")

        if compare_cities:
            compare_df = ranking_df[ranking_df["City"].isin(compare_cities)]

            st.dataframe(
                compare_df,
                width="stretch",
                hide_index=True
            )
        else:
            st.info("Select cities to compare")

    st.divider()

    bottom_left, bottom_right = st.columns([1, 1])

    with bottom_left:
        score_band_df = build_score_band_summary(ranking_df)

        if not score_band_df.empty:
            st.subheader("Comfort Score Distribution")

            fig_band = px.area(
                score_band_df,
                x="Score Band",
                y="City Count",
                title="Comfort score distribution across cities",
            )

            fig_band.update_traces(
                line=dict(color=PALETTE["teal"], width=2),
                fillcolor="rgba(86, 197, 193, 0.28)",
            )

            fig_band.update_layout(height=420, margin=dict(l=22, r=22, t=60, b=24))
            style_figure(fig_band, height=420)
            st.plotly_chart(fig_band, use_container_width=True)
    with bottom_right:
        if not country_summary_df.empty:
            st.subheader("Country Comfort & City Coverage")
            
            df_plot = country_summary_df.head(10).sort_values("Avg Comfort Score", ascending=False).copy()
            df_plot["Cities"] = df_plot["Cities"].astype(int)
            df_plot["# Cities"] = df_plot["Cities"].astype(str)  # ← string = categorical color

            blue_shades = {
                "1": "#AED6F1",
                "2": "#5DADE2",
                "3": "#2E86C1",
                "4": "#1A5276",
                "5": "#0B2545",
            }

            fig_country = px.bar(
                df_plot,
                x="Country",
                y="Avg Comfort Score",
                color="# Cities",                          # ← categorical now
                title="Comfort Score by Country & Cities Covered",
                text_auto=True,
                color_discrete_map=blue_shades,            # ← maps "1" → light blue, etc.
                category_orders={"# Cities": ["1","2","3","4","5"]},  # legend order
            )

            fig_country.update_yaxes(title="Avg Comfort Score")
            fig_country.update_layout(
                xaxis_title="Country",
                legend_title_text="# Cities",
                height=420,
                margin=dict(l=22, r=22, t=60, b=24)
            )
            style_figure(fig_country, height=420)
            st.plotly_chart(fig_country, use_container_width=True)

    panel_end()
with explorer_tab:
    panel_start("City Explorer", "Inspect one city deeply, compare sources, and review the latest raw observations behind its ranking.")
    st.subheader("City-Level Weather Explorer")

    explorer_city_options = ranking_df["City"].tolist() if "City" in ranking_df.columns else []
    if explorer_city_options:
        current_explorer_city = st.session_state.get("explorer_selected_city")
        if current_explorer_city not in explorer_city_options:
            st.session_state.explorer_selected_city = explorer_city_options[0]

    default_city = ranking_df.iloc[0]["City"] if not ranking_df.empty else None
    selected_city = st.selectbox(
        "Select one city to inspect",
        options=explorer_city_options,
        index=0 if default_city is not None else None,
        key="explorer_selected_city",
    )

    city_df = filtered_df[filtered_df["City"] == selected_city].copy()
    city_rank = ranking_df[ranking_df["City"] == selected_city]
    city_latest_df = snapshot_df[snapshot_df["City"] == selected_city].copy()

    if not city_rank.empty:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Comfort Score", f"{city_rank.iloc[0]['Comfort Score']:.1f}")
        if "Avg Temperature_C" in city_rank.columns:
            r2.metric("Avg Temp (C)", f"{city_rank.iloc[0]['Avg Temperature_C']:.1f}")
        if "Avg FeelsLike_C" in city_rank.columns:
            r3.metric("Avg Feels-Like (C)", f"{city_rank.iloc[0]['Avg FeelsLike_C']:.1f}")
        if "Travel Recommendation" in city_rank.columns:
            r4.metric("Recommendation", str(city_rank.iloc[0]["Travel Recommendation"]))

        st.markdown('<div class="packing-tips">', unsafe_allow_html=True)
        st.markdown("**Quick packing tips**")
        for tip in add_quick_trip_tips(city_rank.iloc[0]):
            st.markdown(f"- {tip}")
        st.markdown("</div>", unsafe_allow_html=True)
        

    explorer_metric = st.selectbox(
        "Explorer trend metric",
        options=[c for c in ["Temperature_C", "FeelsLike_C", "Humidity_%", "WindSpeed_kmh", "Comfort Score"] if c in city_df.columns],
        index=0,
    )

    exp_left, exp_right = st.columns(2)

    with exp_left:
        if all(c in city_df.columns for c in ["Date", explorer_metric]):
            trend = city_df.dropna(subset=["Date", explorer_metric]).copy()
            if not trend.empty:
                trend = trend.groupby("Date", as_index=False)[explorer_metric].mean()
                fig_trend = px.line(
                    trend,
                    x="Date",
                    y=explorer_metric,
                    markers=True,
                    title=f"{selected_city}: {explorer_metric} Trend",
                )
                fig_trend.update_layout(xaxis_title="Date", yaxis_title=explorer_metric)
                fig_trend.update_traces(line=dict(color=PALETTE["blue"], width=3))
                style_figure(fig_trend, height=410)
                st.plotly_chart(fig_trend, width="stretch")

    with exp_right:
        if all(c in city_df.columns for c in ["Temperature_C", "FeelsLike_C"]):
            scatter = city_df.dropna(subset=["Temperature_C", "FeelsLike_C"])
            if not scatter.empty:
                fig_scatter = px.scatter(
                    scatter,
                    x="Temperature_C",
                    y="FeelsLike_C",
                    color="SourceWebsite" if "SourceWebsite" in scatter.columns else None,
                    size="Humidity_%" if "Humidity_%" in scatter.columns else None,
                    title=f"{selected_city}: Temperature vs Feels-Like",
                    hover_data=[c for c in ["Date", "Condition", "Humidity_%"] if c in scatter.columns],
                    color_discrete_sequence=[
                        PALETTE["blue"],
                        PALETTE["teal"],
                        PALETTE["accent"],
                        PALETTE["mint"],
                    ],
                )
                style_figure(fig_scatter, height=410)
                st.plotly_chart(fig_scatter, width="stretch")

    if all(c in city_df.columns for c in ["Date", "Temperature_C", "SourceWebsite"]):
        source_split = (
            city_df.dropna(subset=["Date", "Temperature_C"])
            .groupby(["Date", "SourceWebsite"], as_index=False)["Temperature_C"]
            .mean()
        )
        if not source_split.empty:
            fig_sources = px.line(
                source_split,
                x="Date",
                y="Temperature_C",
                color="SourceWebsite",
                title=f"{selected_city}: Temperature by Source",
                color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["accent"]],
            )
            style_figure(fig_sources, height=360)
            st.plotly_chart(fig_sources, width="stretch")

    st.divider()

    st.subheader("Weather Conditions Frequency")
    if "Condition" in city_df.columns and city_df["Condition"].notna().any():
        condition_counts = (
            city_df["Condition"].fillna("Unknown").value_counts().reset_index()
        )
        condition_counts.columns = ["Condition", "Count"]
        fig_condition = px.bar(
            condition_counts,
            x="Condition",
            y="Count",
            title=f"{selected_city}: Weather Condition Frequency",
            color="Condition",
            color_discrete_sequence=[
                PALETTE["mint"],
                PALETTE["blue"],
                PALETTE["yellow"],
                PALETTE["accent"],
                PALETTE["stone"],
                PALETTE["teal"],
            ],
        )
        style_figure(fig_condition, height=410)
        st.plotly_chart(fig_condition, width="stretch")
    if not city_latest_df.empty:
        st.subheader("Latest Weather Conditions by Source")

        latest_cols = [
            c for c in [
                "SourceWebsite",
                "ScrapeDateTime",
                "Temperature_C",
                "FeelsLike_C",
                "Humidity_%",
                "WindSpeed_kmh",
                "Condition",
                "Comfort Score",
            ] if c in city_latest_df.columns
        ]

        display_df = city_latest_df[latest_cols].copy()

        # format datetime nicely
        if "ScrapeDateTime" in display_df.columns:
            display_df["ScrapeDateTime"] = (
                pd.to_datetime(display_df["ScrapeDateTime"])
                .dt.strftime("%Y-%m-%d %H:%M")
            )

        # rename ONLY for display
        display_df = display_df.rename(columns={
            "ScrapeDateTime": "Date"
        })

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        panel_end()

        st.divider()

with source_tab:
    panel_start("Source Quality", "Audit how each source behaves, where coverage is weak, and which cities show disagreement across providers.")
    st.subheader("Statistics of Weather Based on Source")
    if not source_health_display_df.empty:
        st.dataframe(source_health_display_df, width="stretch", hide_index=True)

    s1, s2 = st.columns(2)

    with s1:
        if not temp_by_source.empty:
            fig_temp_source = px.bar(
                temp_by_source,
                x="SourceWebsite",
                y="Avg Temperature_C",
                color="SourceWebsite",
                title="Average Temperature by Source",
                color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["mint"]],
            )
            style_figure(fig_temp_source, height=400)
            st.plotly_chart(fig_temp_source, width="stretch")
        else:
            st.info("Temperature comparison by source is unavailable.")

    with s2:
        if not wind_by_source.empty:
            fig_hum_source = px.bar(
                wind_by_source,
                x="SourceWebsite",
                y="Avg Wind (km/h)",
                color="SourceWebsite",
                title="Average Wind by Source",
                color_discrete_sequence=[PALETTE["yellow"], PALETTE["mint"], PALETTE["stone"]],
            )
            style_figure(fig_hum_source, height=400)
            st.plotly_chart(fig_hum_source, width="stretch")
        else:
            st.info("Wind comparison by source is unavailable.")

    st.divider()

    st.subheader("Temperature Variations by Data Source")
    if disagreement.empty:
        st.info("Need City + SourceWebsite + Temperature_C to compute disagreement.")
    else:
        show_rows = min(20, len(disagreement))
        max_disagreement = float(disagreement["Temp Disagreement (C)"].max())

    if max_disagreement == 0.0:
        threshold = 0.0
    else:
        threshold = st.slider(
            "Minimum Temperature Difference to Display",
            min_value=0.0,
            max_value=max_disagreement,
            value=0.0,
            step=0.1,
        )
        disagreement = disagreement[disagreement["Temp Disagreement (C)"] >= threshold]

        disagreement_display = disagreement.rename(
            columns={"Temp Disagreement (C)": "Temp Difference (°C)"}
        )

        st.dataframe(
            disagreement_display[["City", "Source Count", "Temp Difference (°C)"]].head(show_rows),
            width="stretch",
            hide_index=True,
        )
        disagreement_display = disagreement.rename(
            columns={"Temp Disagreement (C)": "Temp Difference (°C)"}
        )

        fig_disagree = px.bar(
            disagreement_display.head(show_rows),
            x="City",
            y="Temp Difference (°C)",
            color="Temp Difference (°C)",
            color_continuous_scale=[PALETTE["accent_soft"], PALETTE["accent"]],
            title="Cities with Largest Temperature Differences Across Sources",
        )

        fig_disagree.update_yaxes(title="Temp Difference (°C)")

        st.plotly_chart(fig_disagree, use_container_width=True) 

    coverage_matrix = build_source_coverage_matrix(filtered_df)
    if not coverage_matrix.empty:
        st.subheader("Records from Each Source per City")
        st.dataframe(coverage_matrix, width="stretch", hide_index=True)
    panel_end()

    st.divider()

with planner_tab:
    panel_start("Trip Planner", "Build a practical shortlist by recommendation, temperature comfort, humidity, and best known hour.")
    st.subheader("Travel Planner Assistant")

    target_label = st.selectbox(
        "Target recommendation level",
        options=["Ideal", "Good", "Moderate", "Avoid"],
        index=1,
    )

    label_order = {"Ideal": 0, "Good": 1, "Moderate": 2, "Avoid": 3}
    cutoff = label_order[target_label]

    planner_df = ranking_df.copy()
    
    # Filter by Travel Recommendation if column exists
    if "Travel Recommendation" in planner_df.columns:
        planner_df["_rank"] = planner_df["Travel Recommendation"].map(label_order).fillna(999)
        planner_df = planner_df[planner_df["_rank"] <= cutoff].drop(columns=["_rank"])
    else:
        st.warning("⚠️ Travel Recommendation column not found in data.")

    pref_col1, pref_col2 = st.columns(2)
    with pref_col1:
        max_temp = st.slider(
            "Maximum average temperature (C)",
            min_value=0.0,
            max_value=45.0,
            value=35.0,
            step=0.5,
        )
    with pref_col2:
        max_humidity = st.slider(
            "Maximum average humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
        )

    if "Avg Temperature_C" in planner_df.columns:
        planner_df = planner_df[planner_df["Avg Temperature_C"] <= max_temp]
    if "Avg Humidity_%" in planner_df.columns:
        planner_df = planner_df[planner_df["Avg Humidity_%"] <= max_humidity]

    planner_sort = st.radio(
        "Planner sort mode",
        options=["Best comfort", "Cooler weather", "Lower humidity"],
        horizontal=True,
    )
    if planner_sort == "Best comfort" and "Comfort Score" in planner_df.columns:
        planner_df = planner_df.sort_values("Comfort Score", ascending=False)
    elif planner_sort == "Cooler weather" and "Avg Temperature_C" in planner_df.columns:
        planner_df = planner_df.sort_values("Avg Temperature_C", ascending=True)
    elif planner_sort == "Lower humidity" and "Avg Humidity_%" in planner_df.columns:
        planner_df = planner_df.sort_values("Avg Humidity_%", ascending=True)

    if planner_df.empty:
        st.warning("No cities match your planner target with current filters.")
    else:
        st.write(f"**Cities that satisfy your target** _{target_label} or better_: **{len(planner_df)}** cities")
        st.dataframe(planner_df, width="stretch", hide_index=True)

    st.subheader("Best Hour of the Day")
    best_hour_df = best_hour_analysis(filtered_df)
    if best_hour_df.empty:
        st.info("Not enough timestamp richness to estimate best hour per city.")
    else:
        planner_hour_df = best_hour_df
        if not planner_df.empty and "City" in planner_df.columns:
            planner_hour_df = planner_hour_df[planner_hour_df["City"].isin(planner_df["City"])]
        st.dataframe(planner_hour_df, width="stretch", hide_index=True)

    st.divider()

    st.subheader("Temperature Trend Over Time")
    if all(c in filtered_df.columns for c in ["Date", "City", "Temperature_C"]):
        trend_df = (
            filtered_df.dropna(subset=["Date", "Temperature_C"])
            .groupby(["Date", "City"], as_index=False)["Temperature_C"]
            .mean()
        )
        if not trend_df.empty:
            fig_line = px.line(
                trend_df,
                x="Date",
                y="Temperature_C",
                color="City",
                title="Temperature Trend Over Time (Filtered Cities)",
                color_discrete_sequence=[
                    PALETTE["accent"],
                    PALETTE["blue"],
                    PALETTE["teal"],
                    PALETTE["mint"],
                    PALETTE["yellow"],
                    PALETTE["stone"],
                ],
            )
            fig_line.update_layout(xaxis_title="Date", yaxis_title="Avg Temperature (C)")
            style_figure(fig_line, height=440)
            st.plotly_chart(fig_line, width="stretch")
    planner_csv = planner_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download planner results",
        data=planner_csv,
        file_name="planner_results.csv",
        mime="text/csv",
    )
    panel_end()

with ai_tab:
    panel_start("AI Recommender", "Describe your ideal trip weather and get an AI-style shortlist of matching cities from the current data.")
    st.subheader("Natural-Language Travel Recommender")

    best_hour_df = best_hour_analysis(filtered_df)
    ai_request = st.text_area(
        "Describe the kind of trip weather you want",
        value="I want a cool, dry city with good comfort for walking outside.",
        height=100,
        key="ai_trip_request",
        placeholder="Example: I want a warm beach city with low wind and good comfort.",
    )
    if st.button("Search recommendations", key="ai_recommend_submit"):
        st.session_state.ai_submitted_request = ai_request

    submitted_ai_request = st.session_state.get("ai_submitted_request", ai_request)
    ai_rec_df = build_ai_recommendations(ranking_df, best_hour_df, submitted_ai_request, top_k=5)
    st.markdown(build_ai_summary(submitted_ai_request, ai_rec_df))

    if ai_rec_df.empty:
        st.info("No AI recommendation could be generated from the current data and filters.")
    else:
        top_pick = ai_rec_df.iloc[0]
        st.markdown(
            f"**Top pick:** {top_pick['City']} with AI match score **{top_pick['AI Match Score']:.1f}** "
            f"and recommendation **{top_pick['Travel Recommendation']}**."
        )
        ai_display_cols = [
            c for c in [
                "City",
                "Country",
                "Travel Recommendation",
                "AI Match Score",
                "Comfort Score",
                "Avg Temperature_C",
                "Avg Humidity_%",
                "Avg WindSpeed_kmh",
                "Best Hour",
                "Why it matches",
            ] if c in ai_rec_df.columns
        ]
        st.dataframe(ai_rec_df[ai_display_cols], width="stretch", hide_index=True)

        ai_csv = ai_rec_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download AI recommendations as CSV",
            data=ai_csv,
            file_name="ai_recommendations.csv",
            mime="text/csv",
        )
    panel_end()

with analytics_tab:
    panel_start("Insights Center", "Use these views to understand the data, check forecast-prediction quality, and review how weather descriptions are written.")
    eda_subtab, ml_subtab, nlp_subtab = st.tabs(["Data Overview", "Prediction Quality", "Weather Language"])

    with eda_subtab:
        st.subheader("Data Overview")
        if not eda_snapshot_df.empty:
            st.dataframe(eda_snapshot_df, width="stretch", hide_index=True)

        if not summary_report_df.empty:
            st.subheader("Saved Data Summary")
            st.dataframe(summary_report_df, width="stretch", hide_index=True)

        eda_left, eda_right = st.columns(2)
        with eda_left:
            if "Country" in ranking_df.columns and "Comfort Score" in ranking_df.columns and not country_summary_df.empty:
                df_plot = country_summary_df.head(10).sort_values("Avg Comfort Score", ascending=False).copy()
                df_plot["Cities"] = df_plot["Cities"].astype(int)
                df_plot["# Cities"] = df_plot["Cities"].astype(str)

                blue_shades = {
                    "1": "#AED6F1",
                    "2": "#5DADE2",
                    "3": "#2E86C1",
                    "4": "#1A5276",
                    "5": "#0B2545",
                }

                fig_country_eda = px.bar(
                    df_plot,
                    x="Country",
                    y="Avg Comfort Score",
                    color="# Cities",
                    title="Comfort Score by Country & Cities Covered",
                    text_auto=True,
                    color_discrete_map=blue_shades,
                    category_orders={"# Cities": ["1", "2", "3", "4", "5"]},
                )
                style_figure(fig_country_eda, height=380)
                st.plotly_chart(fig_country_eda, width="stretch")
        with eda_right:
            if "SourceWebsite" in filtered_df.columns:
                source_volume = (
                    filtered_df["SourceWebsite"]
                    .value_counts()
                    .rename_axis("SourceWebsite")
                    .reset_index(name="Rows")
                )
                fig_source_volume = px.pie(
                    source_volume,
                    names="SourceWebsite",
                    values="Rows",
                    title="Row Distribution by Source",
                    hole=0.35,
                    color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["accent"], PALETTE["mint"]],
                )
                style_figure(fig_source_volume, height=380)
                st.plotly_chart(fig_source_volume, width="stretch")

    with ml_subtab:
        st.subheader("Prediction Quality")
        with st.spinner(" Training prediction models... (this may take 30-60 seconds on first load)"):
            ml_results_df, ml_best_model, ml_importance_df = run_ml_analysis_dashboard_from_df(filtered_df)
        
        if ml_results_df is None or ml_results_df.empty:
            st.info("Prediction results are unavailable or there is not enough data to train the models.")
        else:
            st.markdown(f"**Best prediction model:** {ml_best_model}")
            st.dataframe(ml_results_df, width="stretch", hide_index=True)

            fig_ml = px.bar(
                ml_results_df,
                x="Model",
                y="RMSE",
                color="Model",
                title="Model RMSE Comparison",
                color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["accent"]],
                text_auto=".3f",
            )
            style_figure(fig_ml, height=360)
            st.plotly_chart(fig_ml, width="stretch")

            if not ml_importance_df.empty:
                st.subheader("What Influences the Prediction Most")
                st.dataframe(ml_importance_df, width="stretch", hide_index=True)
                fig_imp = px.bar(
                    ml_importance_df.head(10).sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Most Important Inputs in the Best Prediction Model",
                    color="Importance",
                    color_continuous_scale=[PALETTE["accent_soft"], PALETTE["accent"]],
                )
                style_figure(fig_imp, height=420)
                st.plotly_chart(fig_imp, width="stretch")

    with nlp_subtab:
        st.subheader("Weather Language")
        if condition_analysis_df.empty:
            st.info("Weather-language analysis output is not available.")
        else:
            nlp_df = condition_analysis_df.copy()

            normalized_counts = (
                nlp_df["NormalizedCondition"]
                .fillna("unknown")
                .value_counts()
                .rename_axis("NormalizedCondition")
                .reset_index(name="Count")
            )
            source_condition_counts = (
                nlp_df.groupby(["SourceWebsite", "NormalizedCondition"])
                .size()
                .reset_index(name="Count")
            ) if all(c in nlp_df.columns for c in ["SourceWebsite", "NormalizedCondition"]) else pd.DataFrame()

            nlp_left, nlp_right = st.columns(2)
            with nlp_left:
                if not normalized_counts.empty:
                    fig_nlp = px.bar(
                        normalized_counts,
                        x="NormalizedCondition",
                        y="Count",
                        color="NormalizedCondition",
                        title="Normalized Weather Conditions",
                        color_discrete_sequence=[PALETTE["mint"], PALETTE["blue"], PALETTE["yellow"], PALETTE["accent"], PALETTE["teal"], PALETTE["stone"]],
                    )
                    style_figure(fig_nlp, height=360)
                    st.plotly_chart(fig_nlp, width="stretch")
            with nlp_right:
                if not source_condition_counts.empty:
                    fig_source_nlp = px.bar(
                        source_condition_counts,
                        x="SourceWebsite",
                        y="Count",
                        color="NormalizedCondition",
                        title="How Each Source Describes Conditions",
                        barmode="stack",
                        color_discrete_sequence=[PALETTE["mint"], PALETTE["blue"], PALETTE["yellow"], PALETTE["accent"], PALETTE["teal"], PALETTE["stone"]],
                    )
                    style_figure(fig_source_nlp, height=360)
                    st.plotly_chart(fig_source_nlp, width="stretch")

            top_raw_conditions = (
                nlp_df["CleanCondition"]
                .fillna("unknown")
                .value_counts()
                .head(15)
                .rename_axis("CleanCondition")
                .reset_index(name="Count")
            ) if "CleanCondition" in nlp_df.columns else pd.DataFrame()
            if not top_raw_conditions.empty:
                st.subheader("Top Raw Condition Phrases")
                st.dataframe(top_raw_conditions, width="stretch", hide_index=True)
    panel_end()

with all_cities_tab:
    panel_start("All Cities", "Use this table as the operational dataset view: search, slice, and export city-level results with source coverage.")
    st.subheader("All Filtered Cities")

    search_col, filter_col1, filter_col2 = st.columns([1.2, 1, 1])
    with search_col:
        city_query = st.text_input("Search cities", placeholder="Type a city name", key="all_cities_search")
    with filter_col1:
        recommendation_filter = st.multiselect(
            "Recommendation filter",
            options=sorted(ranking_df["Travel Recommendation"].dropna().unique().tolist()) if "Travel Recommendation" in ranking_df.columns else [],
        )
    with filter_col2:
        country_filter_quick = st.multiselect(
            "Country filter",
            options=sorted(ranking_df["Country"].dropna().unique().tolist()) if "Country" in ranking_df.columns else [],
        )

    all_cities_df = filter_all_cities_table(
        all_cities_table,
        city_query,
        tuple(recommendation_filter),
        tuple(country_filter_quick),
    )

    if all_cities_df.empty:
        st.info("No cities match the current filters/search.")
    else:
        st.dataframe(all_cities_df, width="stretch", hide_index=True)
        all_cities_csv = all_cities_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download all filtered cities as CSV",
            data=all_cities_csv,
            file_name="all_filtered_cities.csv",
            mime="text/csv",
        )

    panel_end()
