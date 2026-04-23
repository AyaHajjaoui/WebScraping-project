import os
import re
import importlib.util
import hashlib
import pandas as pd
import plotly.express as px
import streamlit as st
import random


COLORS = {
    "Ideal": "#86efac",
    "Good": "#93c5fd",
    "Moderate": "#fef3c7",
    "Avoid": "#fca5a5",
    "Unknown": "#d1d5db"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "weather_data.csv")
SUMMARY_REPORT_PATH = os.path.join(BASE_DIR, "data", "processed", "summary_report.csv")
CONDITION_ANALYSIS_PATH = os.path.join(BASE_DIR, "data", "processed", "condition_analysis.csv")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_and_prepare_data(path: str = DATA_PATH) -> pd.DataFrame:
    return clean_data(load_data(path))


PLACEHOLDER_NA_VALUES = {"", "-", "N/A", "NA", "None", "nan", "null"}


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
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
    safe_df = make_arrow_compatible(df)
    if safe_df.empty or not na_columns:
        return safe_df

    display_df = safe_df.copy()
    for col in na_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype("string").fillna("N/A")
    return display_df


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"

    safe_df = df.copy()
    safe_df = safe_df.reindex(sorted(safe_df.columns), axis=1)
    hashed = pd.util.hash_pandas_object(safe_df, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


@st.cache_resource(
    show_spinner="Training models (first run only)...",
    hash_funcs={pd.DataFrame: dataframe_fingerprint},
)
def train_models(df: pd.DataFrame) -> dict[str, object]:
    df_local = df.copy()
    ml_module = load_analysis_module("analysis_ml_dashboard", os.path.join("analysis", "ml_analysis.py"))
    empty_result = {
        "comparison_df": pd.DataFrame(),
        "baseline_comparison_df": pd.DataFrame(),
        "experiment_flow_df": pd.DataFrame(),
        "preprocessing_summary_df": pd.DataFrame(),
        "tuning_summary_df": pd.DataFrame(),
        "best_model_name": "Unavailable",
        "feature_importance_df": pd.DataFrame(),
        "feature_importance_text": "Classification results are unavailable.",
        "class_distribution_df": pd.DataFrame(),
        "classification_report_df": pd.DataFrame(),
        "class_metric_chart_df": pd.DataFrame(),
        "confusion_matrix_df": pd.DataFrame(),
        "confusion_pairs_df": pd.DataFrame(),
        "confusion_summary": "Classification results are unavailable.",
        "bias_summary": "Classification results are unavailable.",
        "final_model_reasoning": "Classification results are unavailable.",
        "best_pipeline": None,
        "feature_frame": pd.DataFrame(),
    }
    if ml_module is None:
        return empty_result

    prepared = ml_module.prepare_features(df_local)
    if not isinstance(prepared, tuple) or len(prepared) != 4:
        empty_result["best_model_name"] = "Module reload needed"
        empty_result["bias_summary"] = (
            "The ML analysis module returned an outdated result shape. "
            "Please rerun the dashboard so the latest classification module is loaded."
        )
        return empty_result

    X, y, _, class_distribution_df = prepared
    min_rows = int(getattr(ml_module, "MIN_TRAINING_ROWS", 20))
    if len(X) < min_rows or y.nunique() < 2:
        empty_result["best_model_name"] = "Not enough data"
        empty_result["bias_summary"] = "There are not enough rows or classes to train a classification model."
        empty_result["class_distribution_df"] = make_arrow_compatible(class_distribution_df.copy())
        empty_result["feature_frame"] = X.copy()
        return empty_result

    class_counts = y.value_counts()
    if not class_counts.empty and int(class_counts.min()) < 2:
        empty_result["best_model_name"] = "Class imbalance too severe"
        empty_result["bias_summary"] = "At least one class has fewer than two rows, so a stratified train/test split would not be reliable."
        empty_result["class_distribution_df"] = make_arrow_compatible(class_distribution_df.copy())
        empty_result["feature_frame"] = X.copy()
        return empty_result

    X_train, X_test, y_train, y_test = ml_module.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    comparison_df, best_model_name, best_pipeline, best_artifacts = ml_module.train_and_evaluate_models(
        X_train, X_test, y_train, y_test, class_distribution_df=class_distribution_df
    )
    feature_importance_df = ml_module.get_feature_importance(best_pipeline)
    feature_importance_text = ml_module.summarize_feature_importance(feature_importance_df)

    return {
        "comparison_df": make_arrow_compatible(comparison_df.copy().round(4)),
        "baseline_comparison_df": make_arrow_compatible(best_artifacts["baseline_comparison_df"].copy().round(4)),
        "experiment_flow_df": make_arrow_compatible(best_artifacts["experiment_flow_df"].copy().round(4)),
        "preprocessing_summary_df": make_arrow_compatible(best_artifacts["preprocessing_summary_df"].copy()),
        "tuning_summary_df": make_arrow_compatible(best_artifacts["tuning_summary_df"].copy().round(4)),
        "best_model_name": str(best_model_name),
        "feature_importance_df": make_arrow_compatible(feature_importance_df.copy().round(4)),
        "feature_importance_text": str(feature_importance_text),
        "class_distribution_df": make_arrow_compatible(class_distribution_df.copy()),
        "classification_report_df": make_arrow_compatible(best_artifacts["classification_report_df"].copy().round(4)),
        "class_metric_chart_df": make_arrow_compatible(best_artifacts["class_metric_chart_df"].copy().round(4)),
        "confusion_matrix_df": make_arrow_compatible(best_artifacts["confusion_matrix_df"].copy()),
        "confusion_pairs_df": make_arrow_compatible(best_artifacts["confusion_pairs_df"].copy()),
        "confusion_summary": str(best_artifacts["confusion_summary"]),
        "bias_summary": str(best_artifacts["bias_summary"]),
        "final_model_reasoning": str(best_artifacts["final_model_reasoning"]),
        "best_pipeline": best_pipeline,
        "feature_frame": X.copy(),
    }


def load_analysis_module(module_name: str, relative_path: str):
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
    parsed = pd.to_datetime(series, errors="coerce", utc=True)


    failed_mask = series.notna() & parsed.isna()
    if failed_mask.any():
        reparsed = series.loc[failed_mask].apply(
            lambda value: pd.to_datetime(value, errors="coerce", utc=True)
        )
        parsed.loc[failed_mask] = reparsed

    return parsed.dt.tz_localize(None)


def comfort_score(temp, feels_like, humidity) -> float:
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
    tips = []

    temp = city_row.get("Avg Temperature_C")
    humid = city_row.get("Avg Humidity_%")
    wind = city_row.get("Avg WindSpeed_kmh")
    rain = city_row.get("Avg Precipitation_mm") if "Avg Precipitation_mm" in city_row else None
    aqi = city_row.get("Avg AQI") if "Avg AQI" in city_row else None


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


    if pd.notna(humid):
        if humid >= 85:
            tips.append("Very high humidity — expect sticky conditions and reduced comfort.")
            tips.append("Choose ultra-breathable fabrics and avoid heavy meals.")
        elif humid >= 70:
            tips.append("High humidity — breathable clothes recommended.")
        elif humid <= 30:
            tips.append("Dry air — stay hydrated and consider moisturizer for skin protection.")
            tips.append("Dry throat possible — carry water or lozenges.")


    if pd.notna(wind):
        if wind >= 40:
            tips.append("Strong winds expected — avoid loose items and wear wind-resistant jacket.")
        elif wind >= 25:
            tips.append("Windy conditions — carry a light windbreaker.")
        elif wind >= 15:
            tips.append("Light breeze — comfortable outdoor conditions.")


        if "rain" in globals() and pd.notna(rain):
            if rain >= 80:
                tips.append("Heavy rain expected — waterproof jacket and umbrella required.")
            elif rain >= 40:
                tips.append("Possible rain — carry an umbrella or light raincoat.")
            elif rain > 0:
                tips.append("Light rain possible — be prepared for brief showers.")


        if "aqi" in globals() and pd.notna(aqi):
            if aqi >= 150:
                tips.append("Unhealthy air quality — limit outdoor activity and consider a mask.")
            elif aqi >= 100:
                tips.append("Moderate pollution — sensitive individuals should reduce exposure.")
            elif aqi >= 50:
                tips.append("Acceptable air quality — generally safe for outdoor activities.")


        if not tips:
            tips.append("Weather is stable — no special packing precautions needed.")

    return tips


def parse_trip_request(request: str) -> dict:
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
    prefs = parse_trip_request(request)
    pref_bits = prefs["keywords"] if prefs["keywords"] else ["balanced weather"]

    if rec_df.empty:
        return f"I couldn't find a strong match for {', '.join(pref_bits)} in the current filtered data."

    city_list = ", ".join(rec_df["City"].tolist())
    return f"Based on your request for {', '.join(pref_bits)}, the strongest matches right now are {city_list}."


@st.cache_data(show_spinner=False)
def load_summary_report(path: str = SUMMARY_REPORT_PATH) -> pd.DataFrame:
    return make_arrow_compatible(load_data(path))


@st.cache_data(show_spinner=False)
def load_condition_analysis(path: str = CONDITION_ANALYSIS_PATH) -> pd.DataFrame:
    df = load_data(path)
    if "ScrapeDateTime" in df.columns:
        df["ScrapeDateTime"] = parse_datetime(df["ScrapeDateTime"])
    return make_arrow_compatible(df)


@st.cache_data(show_spinner=False)
def run_ml_analysis_dashboard(data_path: str = DATA_PATH) -> dict[str, object]:
    df = load_data(data_path)
    return train_models(df)


def run_ml_analysis_dashboard_from_df(_df: pd.DataFrame) -> dict[str, object]:
    return train_models(_df)


def build_filtered_summary_report(_df: pd.DataFrame) -> pd.DataFrame:
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

    if ranking_df.empty:
        return ["No data available for alerts."]

    pool = []


    if "Avg Temperature_C" in ranking_df.columns:
        hot_cities = ranking_df[ranking_df["Avg Temperature_C"] >= 32]["City"].dropna().head(5).tolist()
        if hot_cities:
            city_list = ", ".join(hot_cities)
            pool.append(f"High heat detected in: {city_list}.")
            pool.append(f"Rising temperatures affecting: {city_list}.")
            pool.append(f"Hot weather conditions reported in: {city_list}.")
            pool.append(f"Extreme warmth expected in: {city_list}.")


    if "Avg Humidity_%" in ranking_df.columns:
        humid_cities = ranking_df[ranking_df["Avg Humidity_%"] >= 80]["City"].dropna().head(5).tolist()
        if humid_cities:
            city_list = ", ".join(humid_cities)
            pool.append(f"Very humid conditions in: {city_list}.")
            pool.append(f"Sticky air conditions affecting: {city_list}.")
            pool.append(f"High moisture levels detected in: {city_list}.")


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


    if not pool:
        return ["No major weather or data issues detected in the current view."]


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
                background: transparent !important;
                color: {PALETTE["text"]} !important;
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(255,255,255,0.95), transparent 32%),
                    linear-gradient(180deg, {PALETTE["paper"]} 0%, #f6f2ee 100%);
                color: {PALETTE["text"]};
            }}

            .block-container {{
                max-width: 1420px;
                padding-top: 5.75rem;
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
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 1.2rem 1.25rem 1.1rem;

                min-height: 132px;
                height: 132px;

                box-shadow: 0 14px 24px rgba(188, 180, 172, 0.10);

                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;

                text-align: center;
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
                margin: 0.75rem 0 1.5rem 0;
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

            [data-testid="stSpinner"] {{
                background: transparent !important;
                border: none !important;
                border-radius: 0;
                box-shadow: none;
                padding: 0 !important;
            }}

            [data-testid="stSpinner"] > div {{
                background: transparent !important;
            }}

            div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stSpinner"]) {{
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
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


def render_title() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Weather Intelligence</div>
            <div class="hero-title">Smart Weather Travel & Comfort Dashboard</div>
            <div class="hero-sub">
                Compare travel comfort, review source quality, and explore the latest processed weather data in one place.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
inject_styles()

weather_df = load_and_prepare_data(DATA_PATH)
if weather_df.empty:
    st.error(
        f"Could not load processed weather data from `{DATA_PATH}`. "
        "Please make sure the file exists and contains data."
    )
    st.stop()

render_title()

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
    st.session_state.ai_submitted_request = "I want a cool, dry city with good comfort for walking outside."

st.sidebar.header("Dashboard Filters")


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
            df_plot["# Cities"] = df_plot["Cities"].astype(str)

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
                color="# Cities",
                title="Comfort Score by Country & Cities Covered",
                text_auto=True,
                color_discrete_map=blue_shades,
                category_orders={"# Cities": ["1","2","3","4","5"]},
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


        if "ScrapeDateTime" in display_df.columns:
            display_df["ScrapeDateTime"] = (
                pd.to_datetime(display_df["ScrapeDateTime"])
                .dt.strftime("%Y-%m-%d %H:%M")
            )


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
    panel_start("Insights Center", "Use these views to understand the data, compare recommendation classes, and review how weather descriptions are written.")
    eda_subtab, ml_subtab, nlp_subtab = st.tabs(
        [
            "Exploratory Data Analysis",
            "Machine Learning",
            "Natural Language Processing",
        ]
    )

    with eda_subtab:
        st.subheader("EDA")
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
        st.subheader("ML")
        ml_results = run_ml_analysis_dashboard_from_df(filtered_df)

        ml_results_df = ml_results["comparison_df"]
        ml_baseline_df = ml_results["baseline_comparison_df"]
        ml_experiment_df = ml_results["experiment_flow_df"]
        ml_preprocessing_df = ml_results["preprocessing_summary_df"]
        ml_tuning_df = ml_results["tuning_summary_df"]
        ml_best_model = ml_results["best_model_name"]
        ml_importance_df = ml_results["feature_importance_df"]
        ml_importance_text = ml_results["feature_importance_text"]
        ml_class_distribution_df = ml_results["class_distribution_df"]
        ml_report_df = ml_results["classification_report_df"]
        ml_class_metric_chart_df = ml_results["class_metric_chart_df"]
        ml_confusion_df = ml_results["confusion_matrix_df"]
        ml_confusion_pairs_df = ml_results["confusion_pairs_df"]
        ml_confusion_summary = ml_results["confusion_summary"]
        ml_bias_summary = ml_results["bias_summary"]
        ml_final_model_reasoning = ml_results["final_model_reasoning"]
        ml_best_pipeline = ml_results["best_pipeline"]
        ml_feature_frame = ml_results["feature_frame"]

        if ml_results_df is None or ml_results_df.empty:
            st.info("Classification results are unavailable or there is not enough data to train the models.")
        else:
            percent_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "Weighted F1", "Macro F1", "Percentage"]
            def format_percent_df(df: pd.DataFrame) -> pd.DataFrame:
                display_df = df.copy()
                for col in percent_cols:
                    if col in display_df.columns:
                        display_df[col] = (pd.to_numeric(display_df[col], errors="coerce") * 100).round(1)
                return display_df

            def add_model_labels(df: pd.DataFrame) -> pd.DataFrame:
                labeled_df = df.copy()
                if "Model" in labeled_df.columns:
                    labeled_df["Model Label"] = (
                        labeled_df["Model"]
                        .replace(
                            {
                                "Logistic Regression": "Logistic",
                                "Decision Tree Classifier": "Decision Tree",
                                "Random Forest Classifier": "Random Forest",
                                "Tuned Random Forest": "Tuned RF",
                                "Tuned Decision Tree": "Tuned Tree",
                            }
                        )
                    )
                return labeled_df

            best_row = ml_results_df.iloc[0]
            baseline_display_df = format_percent_df(add_model_labels(ml_baseline_df))
            experiment_display_df = format_percent_df(add_model_labels(ml_experiment_df))
            tuning_display_df = format_percent_df(add_model_labels(ml_tuning_df))
            class_distribution_display_df = ml_class_distribution_df.copy()
            if "Percentage" in class_distribution_display_df.columns:
                class_distribution_display_df["Percentage"] = pd.to_numeric(class_distribution_display_df["Percentage"], errors="coerce").round(1)
            report_display_df = format_percent_df(ml_report_df)
            class_metric_chart_plot_df = ml_class_metric_chart_df.copy()
            if "Score" in class_metric_chart_plot_df.columns:
                class_metric_chart_plot_df["Score Percent"] = pd.to_numeric(class_metric_chart_plot_df["Score"], errors="coerce") * 100

            st.subheader("Data Preparation Summary")
            prep_left, prep_right = st.columns([1.05, 0.95])
            with prep_left:
                st.dataframe(ml_preprocessing_df, width="stretch", hide_index=True)
            with prep_right:
                st.markdown("**Target classes**: Ideal, Good, Moderate, Avoid")
                st.markdown("**Target creation**: The label is derived from the travel comfort score already used elsewhere in the dashboard.")
                st.markdown("**Split strategy**: Stratified train/test split keeps the recommendation classes proportionally represented.")

            st.subheader("Baseline Model Performance")
            baseline_left, baseline_right = st.columns([0.95, 1.25])
            with baseline_left:
                st.dataframe(baseline_display_df, width="stretch", hide_index=True)
            with baseline_right:
                baseline_plot_df = add_model_labels(ml_baseline_df)
                base_acc = px.bar(
                    baseline_plot_df,
                    x="Model Label",
                    y="Accuracy",
                    color="Model Label",
                    title="Baseline Accuracy",
                    text_auto=".1%",
                    color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["accent"]],
                )
                base_acc.update_yaxes(tickformat=".0%")
                style_figure(base_acc, height=360)
                st.plotly_chart(base_acc, width="stretch")

            st.subheader("Tuned Model Comparison")
            tune_left, tune_right = st.columns([0.95, 1.25])
            with tune_left:
                st.dataframe(tuning_display_df, width="stretch", hide_index=True)
            with tune_right:
                experiment_plot_df = add_model_labels(ml_experiment_df)
                tuned_wf1 = px.bar(
                    experiment_plot_df,
                    x="Model Label",
                    y="Weighted F1",
                    color="Stage",
                    barmode="group",
                    title="Baseline vs Tuned Weighted F1",
                    text_auto=".1%",
                    color_discrete_sequence=[PALETTE["stone"], PALETTE["accent"]],
                )
                tuned_wf1.update_yaxes(tickformat=".0%")
                style_figure(tuned_wf1, height=360)
                st.plotly_chart(tuned_wf1, width="stretch")

            st.subheader("Final Model Selection")
            st.markdown(f"**Selected final model:** {ml_best_model}")
            st.caption("The final model is chosen primarily by weighted F1-score, with accuracy and macro F1 used as supporting evidence.")
            st.markdown(ml_final_model_reasoning)

            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Weighted F1", f"{best_row['Weighted F1'] * 100:.1f}%")
            metric_b.metric("Accuracy", f"{best_row['Accuracy'] * 100:.1f}%")
            metric_c.metric("Macro F1", f"{best_row['F1-Score'] * 100:.1f}%")

            st.subheader("Class-Level Performance Analysis")
            class_perf_left, class_perf_right = st.columns([0.95, 1.25])
            with class_perf_left:
                st.dataframe(report_display_df, width="stretch", hide_index=True)
                st.markdown(f"**Bias / balance interpretation:** {ml_bias_summary}")
            with class_perf_right:
                class_metric_chart = px.bar(
                    class_metric_chart_plot_df,
                    x="Class",
                    y="Score Percent",
                    color="Metric",
                    barmode="group",
                    title="Precision, Recall, and F1 by Class",
                    text_auto=".1f",
                    color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["accent"]],
                    category_orders={"Class": ["Ideal", "Good", "Moderate", "Avoid"]},
                )
                class_metric_chart.update_yaxes(title="Score (%)", range=[0, 100])
                style_figure(class_metric_chart, height=400)
                st.plotly_chart(class_metric_chart, width="stretch")

            st.subheader("Error Analysis from the Confusion Matrix")
            confusion_left, confusion_right = st.columns([1.05, 1.15])
            with confusion_left:
                fig_confusion = px.imshow(
                    ml_confusion_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=["#fffaf0", PALETTE["blue"], PALETTE["accent"]],
                    title="Final Model Confusion Matrix",
                )
                fig_confusion.update_xaxes(side="bottom")
                style_figure(fig_confusion, height=430)
                st.plotly_chart(fig_confusion, width="stretch")
            with confusion_right:
                confusion_display_df = ml_confusion_df.reset_index().rename(columns={"index": "Actual"})
                st.dataframe(confusion_display_df, width="stretch", hide_index=True)
                if not ml_confusion_pairs_df.empty:
                    st.markdown("**Most confused class pairs**")
                    st.dataframe(ml_confusion_pairs_df, width="stretch", hide_index=True)
                st.markdown(ml_confusion_summary)

            st.subheader("Class Distribution")
            dist_col, experiment_col = st.columns([0.9, 1.3])
            with dist_col:
                st.dataframe(class_distribution_display_df, width="stretch", hide_index=True)
                fig_class_dist = px.bar(
                    class_distribution_display_df,
                    x="Class",
                    y="Percentage",
                    color="Class",
                    title="Travel Recommendation Distribution",
                    text_auto=".1f",
                    color_discrete_map=COLORS,
                    category_orders={"Class": ["Ideal", "Good", "Moderate", "Avoid"]},
                )
                fig_class_dist.update_yaxes(title="Rows (%)")
                style_figure(fig_class_dist, height=340)
                st.plotly_chart(fig_class_dist, width="stretch")
            with experiment_col:
                st.dataframe(experiment_display_df, width="stretch", hide_index=True)
                exp_acc = px.bar(
                    add_model_labels(ml_experiment_df),
                    x="Model Label",
                    y="Accuracy",
                    color="Stage",
                    barmode="group",
                    title="Experiment Flow Accuracy",
                    text_auto=".1%",
                    color_discrete_sequence=[PALETTE["stone"], PALETTE["blue"]],
                )
                exp_acc.update_yaxes(tickformat=".0%")
                style_figure(exp_acc, height=340)
                st.plotly_chart(exp_acc, width="stretch")

            if not ml_importance_df.empty:
                st.subheader("What Influences the Travel Recommendation Most")
                st.dataframe(ml_importance_df, width="stretch", hide_index=True)
                fig_imp = px.bar(
                    ml_importance_df.head(10).sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Clean Feature",
                    orientation="h",
                    title="Top 10 Feature Importance",
                    color="Importance",
                    color_continuous_scale=[PALETTE["accent_soft"], PALETTE["accent"]],
                )
                style_figure(fig_imp, height=430)
                st.plotly_chart(fig_imp, width="stretch")
                st.markdown(ml_importance_text)

            if ml_best_pipeline is not None and ml_feature_frame is not None and not ml_feature_frame.empty:
                st.subheader("Interactive Prediction")

                prediction_source_options = sorted(ml_feature_frame["SourceWebsite"].dropna().astype(str).unique().tolist()) if "SourceWebsite" in ml_feature_frame.columns else []
                prediction_city_options = sorted(ml_feature_frame["City"].dropna().astype(str).unique().tolist()) if "City" in ml_feature_frame.columns else []
                prediction_country_options = sorted(ml_feature_frame["Country"].dropna().astype(str).unique().tolist()) if "Country" in ml_feature_frame.columns else []

                defaults = {}
                for numeric_col in ["FeelsLike_C", "Humidity_%", "WindSpeed_kmh", "Hour", "Day", "Month"]:
                    if numeric_col in ml_feature_frame.columns:
                        defaults[numeric_col] = float(pd.to_numeric(ml_feature_frame[numeric_col], errors="coerce").median())

                pred_row1, pred_row2, pred_row3 = st.columns(3)
                with pred_row1:
                    pred_source = st.selectbox(
                        "Source website",
                        options=prediction_source_options,
                        index=0 if prediction_source_options else None,
                        key="ml_predict_source",
                    )
                    pred_feels_like = st.number_input(
                        "Feels like (C)",
                        value=round(defaults.get("FeelsLike_C", 20.0), 1),
                        key="ml_predict_feels_like",
                    )
                    pred_hour = st.slider(
                        "Hour",
                        min_value=0,
                        max_value=23,
                        value=int(round(defaults.get("Hour", 12.0))),
                        key="ml_predict_hour",
                    )
                with pred_row2:
                    pred_city = st.selectbox(
                        "City",
                        options=prediction_city_options,
                        index=0 if prediction_city_options else None,
                        key="ml_predict_city",
                    )
                    pred_humidity = st.slider(
                        "Humidity (%)",
                        min_value=0,
                        max_value=100,
                        value=int(round(defaults.get("Humidity_%", 60.0))),
                        key="ml_predict_humidity",
                    )
                    pred_day = st.slider(
                        "Day of month",
                        min_value=1,
                        max_value=31,
                        value=max(1, min(31, int(round(defaults.get("Day", 15.0))))),
                        key="ml_predict_day",
                    )
                with pred_row3:
                    pred_country = st.selectbox(
                        "Country",
                        options=prediction_country_options,
                        index=0 if prediction_country_options else None,
                        key="ml_predict_country",
                    )
                    pred_wind = st.number_input(
                        "Wind speed (km/h)",
                        min_value=0.0,
                        value=round(defaults.get("WindSpeed_kmh", 10.0), 1),
                        key="ml_predict_wind",
                    )
                    pred_month = st.slider(
                        "Month",
                        min_value=1,
                        max_value=12,
                        value=max(1, min(12, int(round(defaults.get("Month", 6.0))))),
                        key="ml_predict_month",
                    )

                if st.button("Predict class", key="ml_predict_submit"):
                    prediction_input = pd.DataFrame(
                        [
                            {
                                "SourceWebsite": pred_source,
                                "City": pred_city,
                                "Country": pred_country,
                                "FeelsLike_C": pred_feels_like,
                                "Humidity_%": pred_humidity,
                                "WindSpeed_kmh": pred_wind,
                                "Hour": pred_hour,
                                "Day": pred_day,
                                "Month": pred_month,
                            }
                        ]
                    )
                    ml_module = load_analysis_module("analysis_ml_dashboard_prediction", os.path.join("analysis", "ml_analysis.py"))
                    if ml_module is not None:
                        predicted_label, confidence, probability_df = ml_module.predict_with_pipeline(
                            ml_best_pipeline,
                            prediction_input,
                        )
                        prob_label = f"{confidence:.1%}" if confidence is not None else "N/A"
                        result_left, result_right = st.columns([0.7, 1.3])
                        with result_left:
                            st.metric("Predicted class", predicted_label)
                            st.metric("Confidence", prob_label)
                        with result_right:
                            if not probability_df.empty:
                                probability_df = make_arrow_compatible(probability_df.copy().round(4))
                                st.dataframe(probability_df, width="stretch", hide_index=True)
                                fig_prob = px.bar(
                                    probability_df.sort_values("Probability", ascending=True),
                                    x="Probability",
                                    y="Class",
                                    orientation="h",
                                    color="Class",
                                    title="Predicted Class Probabilities",
                                    color_discrete_map=COLORS,
                                    text_auto=".1%",
                                    category_orders={"Class": ["Ideal", "Good", "Moderate", "Avoid"]},
                                )
                                fig_prob.update_xaxes(tickformat=".0%")
                                style_figure(fig_prob, height=320)
                                st.plotly_chart(fig_prob, width="stretch")
                        if not ml_importance_df.empty:
                            prediction_explanation = ml_module.explain_prediction(
                                prediction_input,
                                ml_importance_df,
                                predicted_label,
                            )
                            st.markdown(f"**Decision hint:** {prediction_explanation}")

    with nlp_subtab:
        st.subheader("NLP")
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
            top_raw_conditions = (
                nlp_df["CleanCondition"]
                .fillna("unknown")
                .value_counts()
                .head(15)
                .rename_axis("CleanCondition")
                .reset_index(name="Count")
            ) if "CleanCondition" in nlp_df.columns else pd.DataFrame()

            nlp_before_subtab, nlp_after_subtab = st.tabs(
                [
                    "NLP Before Analysis",
                    "NLP After Analysis",
                ]
            )

            with nlp_before_subtab:
                st.subheader("NLP Before Analysis: Text Preparation and Feature Engineering")
                st.caption("This stage prepares raw weather-condition text before downstream analysis by cleaning the phrases and mapping them into broader categories.")

                prep_summary_df = pd.DataFrame(
                    [
                        {"Step": "Raw text input", "Details": "Start from the original Condition column collected from each weather source."},
                        {"Step": "Lowercasing and trimming", "Details": "Condition text is converted to lowercase and stripped of extra spaces."},
                        {"Step": "Missing / empty filtering", "Details": "Rows with missing or empty condition text are removed before analysis."},
                        {"Step": "Rule-based normalization", "Details": "CleanCondition values are mapped into broader labels such as clear, cloudy, rain, storm, fog, snow, or other."},
                    ]
                )
                st.dataframe(prep_summary_df, width="stretch", hide_index=True)

                prep_metric_1, prep_metric_2, prep_metric_3 = st.columns(3)
                prep_metric_1.metric("Rows prepared", f"{len(nlp_df):,}")
                prep_metric_2.metric(
                    "Unique raw phrases",
                    f"{nlp_df['CleanCondition'].nunique():,}" if "CleanCondition" in nlp_df.columns else "N/A",
                )
                prep_metric_3.metric(
                    "Normalized categories",
                    f"{nlp_df['NormalizedCondition'].nunique():,}" if "NormalizedCondition" in nlp_df.columns else "N/A",
                )

                preview_cols = [
                    c for c in ["SourceWebsite", "Condition", "CleanCondition", "NormalizedCondition"]
                    if c in nlp_df.columns
                ]
                if preview_cols:
                    st.markdown("**Preprocessing preview**")
                    preview_df = nlp_df[preview_cols].drop_duplicates().head(15).copy()
                    st.dataframe(preview_df, width="stretch", hide_index=True)

            with nlp_after_subtab:
                st.subheader("NLP After Analysis: Results, Insights, and Model Findings")
                st.caption("This stage shows the outputs and insights produced after the weather-condition text has been cleaned and normalized.")

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
