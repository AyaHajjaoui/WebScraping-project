# Run with: streamlit run dashboard.py

import os

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "weather_data.csv")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV path with safe fallback."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse datetimes and normalize to timezone-naive UTC for safe comparisons."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and compute comfort metrics."""
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


def filter_latest_per_city_source(df: pd.DataFrame) -> pd.DataFrame:
    """Keep most recent row for each city/source pair."""
    if df.empty or "City" not in df.columns or "SourceWebsite" not in df.columns:
        return df

    temp = df.copy()
    if "ScrapeDateTime" in temp.columns:
        temp = temp.sort_values("ScrapeDateTime")
    elif "Date" in temp.columns:
        temp = temp.sort_values("Date")

    return temp.drop_duplicates(subset=["City", "SourceWebsite"], keep="last")


def build_city_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Build city-level aggregation used by KPI cards and tables."""
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


def build_source_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return temperature by source, humidity by source, and city disagreement table."""
    temp_by_source = pd.DataFrame()
    humid_by_source = pd.DataFrame()
    disagreement = pd.DataFrame()

    if df.empty or "SourceWebsite" not in df.columns:
        return temp_by_source, humid_by_source, disagreement

    if "Temperature_C" in df.columns:
        temp_by_source = (
            df.groupby("SourceWebsite", as_index=False)["Temperature_C"]
            .mean()
            .rename(columns={"Temperature_C": "Avg Temperature_C"})
        )
        temp_by_source["Avg Temperature_C"] = temp_by_source["Avg Temperature_C"].round(1)

    if "Humidity_%" in df.columns:
        humid_by_source = (
            df.groupby("SourceWebsite", as_index=False)["Humidity_%"]
            .mean()
            .rename(columns={"Humidity_%": "Avg Humidity_%"})
        )
        humid_by_source["Avg Humidity_%"] = humid_by_source["Avg Humidity_%"].round(1)

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

    return temp_by_source, humid_by_source, disagreement


def best_hour_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Find best hour by city when timestamp detail is sufficient."""
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


def get_high_level_insights(ranking: pd.DataFrame) -> dict:
    """Create practical quick insights from ranking table."""
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


def get_data_freshness(df: pd.DataFrame) -> str:
    """Human-readable freshness summary for the dataset."""
    if "ScrapeDateTime" in df.columns and df["ScrapeDateTime"].notna().any():
        return str(df["ScrapeDateTime"].max())
    if "Date" in df.columns and df["Date"].notna().any():
        return str(df["Date"].max().date())
    return "Unknown"


def add_quick_trip_tips(city_row: pd.Series) -> list[str]:
    """Generate practical travel tips based on city averages."""
    tips = []

    temp = city_row.get("Avg Temperature_C")
    humid = city_row.get("Avg Humidity_%")
    wind = city_row.get("Avg WindSpeed_kmh")

    if pd.notna(temp):
        if temp >= 30:
            tips.append("Pack light clothes and stay hydrated.")
        elif temp <= 10:
            tips.append("Bring warm layers for cold conditions.")
        else:
            tips.append("Mild weather expected, light layers are enough.")

    if pd.notna(humid):
        if humid >= 75:
            tips.append("High humidity likely, choose breathable clothes.")
        elif humid <= 30:
            tips.append("Dry air expected, keep water and moisturizer handy.")

    if pd.notna(wind) and wind >= 25:
        tips.append("It may feel windy outdoors, carry a windproof jacket.")

    if not tips:
        tips.append("No specific packing risk detected from current averages.")

    return tips


st.set_page_config(
    page_title="Smart Weather Travel & Comfort Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Smart Weather Travel & Comfort Dashboard")
st.caption(
    "A practical city-comparison dashboard for comfort and travel planning using "
    "multi-source scraped weather data."
)

raw_df = load_data(DATA_PATH)
if raw_df.empty:
    st.error(
        f"Could not load processed weather data from `{DATA_PATH}`. "
        "Please make sure the file exists and contains data."
    )
    st.stop()

weather_df = clean_data(raw_df)
if weather_df.empty:
    st.error("Dataset is empty after cleaning. Please inspect your processed CSV.")
    st.stop()

st.sidebar.header("Dashboard Filters")

city_options = sorted(weather_df["City"].dropna().unique().tolist()) if "City" in weather_df.columns else []
source_options = (
    sorted(weather_df["SourceWebsite"].dropna().unique().tolist())
    if "SourceWebsite" in weather_df.columns
    else []
)

selected_cities = st.sidebar.multiselect("Cities", city_options, default=city_options)
selected_sources = st.sidebar.multiselect("Sources", source_options, default=source_options)

use_latest_only = st.sidebar.checkbox("Use latest record per city/source", value=False)
min_comfort = st.sidebar.slider("Minimum comfort score", min_value=0, max_value=100, value=0)

sort_option = st.sidebar.selectbox(
    "Ranking sort",
    [
        "Comfort Score (High to Low)",
        "Comfort Score (Low to High)",
        "Temperature (High to Low)",
        "Humidity (Low to High)",
        "City (A-Z)",
    ],
    index=0,
)

filtered_df = weather_df.copy()

if selected_cities and "City" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["City"].isin(selected_cities)]

if selected_sources and "SourceWebsite" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["SourceWebsite"].isin(selected_sources)]

if "Date" in filtered_df.columns and filtered_df["Date"].notna().any():
    min_date = filtered_df["Date"].min().date()
    max_date = filtered_df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df["Date"] >= start_date) & (filtered_df["Date"] <= end_date)
        ]

if use_latest_only:
    filtered_df = filter_latest_per_city_source(filtered_df)

if "Comfort Score" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Comfort Score"] >= min_comfort]

if filtered_df.empty:
    st.warning("No records match current filters. Try lowering constraints.")
    st.stop()

ranking_df = build_city_ranking(filtered_df)
ranking_df = apply_sort(ranking_df, sort_option)
insights = get_high_level_insights(ranking_df)
last_updated = get_data_freshness(filtered_df)

if ranking_df.empty:
    st.warning("Not enough data to build city ranking.")
    st.stop()

avg_comfort = ranking_df["Comfort Score"].mean() if "Comfort Score" in ranking_df.columns else float("nan")
city_count = ranking_df["City"].nunique() if "City" in ranking_df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best city now", insights["best_city"])
c2.metric("Worst city now", insights["worst_city"])
c3.metric("Average comfort", f"{avg_comfort:.1f}" if pd.notna(avg_comfort) else "N/A")
c4.metric("Cities in view", int(city_count))

st.caption(f"Data freshness: {last_updated}")

overview_tab, explorer_tab, source_tab, planner_tab = st.tabs(
    ["Overview", "City Explorer", "Source Quality", "Trip Planner"]
)

with overview_tab:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("City Ranking")
        st.dataframe(ranking_df, width="stretch", hide_index=True)

        csv_bytes = ranking_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download current ranking as CSV",
            data=csv_bytes,
            file_name="city_ranking_filtered.csv",
            mime="text/csv",
        )

    with right:
        st.subheader("Travel Insights")
        st.markdown(f"- Best city for outdoor comfort: **{insights['best_city']}**")
        st.markdown(f"- City with highest humidity: **{insights['most_humid_city']}**")
        st.markdown(
            f"- City with highest feels-like temperature: **{insights['hottest_feels_like_city']}**"
        )
        st.markdown(f"- Cities good for travel today: **{insights['good_cities']}**")
        st.markdown(f"- Cities to avoid today: **{insights['avoid_cities']}**")

        if "Travel Recommendation" in ranking_df.columns:
            rec_counts = (
                ranking_df["Travel Recommendation"]
                .value_counts()
                .rename_axis("Recommendation")
                .reset_index(name="City Count")
            )
            fig_reco = px.pie(
                rec_counts,
                names="Recommendation",
                values="City Count",
                title="Recommendation Mix",
                hole=0.35,
            )
            st.plotly_chart(fig_reco, width="stretch")

    st.subheader("Comfort Score by City")
    fig_comfort = px.bar(
        ranking_df,
        x="City",
        y="Comfort Score",
        color="Travel Recommendation" if "Travel Recommendation" in ranking_df.columns else "Comfort Score",
        title="Comfort Ranking Across Cities",
    )
    fig_comfort.update_layout(xaxis_title="City", yaxis_title="Comfort Score")
    st.plotly_chart(fig_comfort, width="stretch")

    if "Avg Humidity_%" in ranking_df.columns:
        st.subheader("Average Humidity by City")
        fig_humidity = px.bar(
            ranking_df,
            x="City",
            y="Avg Humidity_%",
            color="Avg Humidity_%",
            color_continuous_scale="Blues",
            title="Humidity Comparison",
        )
        fig_humidity.update_layout(xaxis_title="City", yaxis_title="Humidity (%)")
        st.plotly_chart(fig_humidity, width="stretch")

with explorer_tab:
    st.subheader("City-Level Weather Explorer")

    default_city = ranking_df.iloc[0]["City"] if not ranking_df.empty else None
    selected_city = st.selectbox(
        "Select one city to inspect",
        options=ranking_df["City"].tolist(),
        index=0 if default_city is not None else None,
    )

    city_df = filtered_df[filtered_df["City"] == selected_city].copy()
    city_rank = ranking_df[ranking_df["City"] == selected_city]

    if not city_rank.empty:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Comfort Score", f"{city_rank.iloc[0]['Comfort Score']:.1f}")
        if "Avg Temperature_C" in city_rank.columns:
            r2.metric("Avg Temp (C)", f"{city_rank.iloc[0]['Avg Temperature_C']:.1f}")
        if "Avg FeelsLike_C" in city_rank.columns:
            r3.metric("Avg Feels-Like (C)", f"{city_rank.iloc[0]['Avg FeelsLike_C']:.1f}")
        if "Travel Recommendation" in city_rank.columns:
            r4.metric("Recommendation", str(city_rank.iloc[0]["Travel Recommendation"]))

        st.markdown("**Quick packing tips**")
        for tip in add_quick_trip_tips(city_rank.iloc[0]):
            st.markdown(f"- {tip}")

    if all(c in city_df.columns for c in ["Temperature_C", "FeelsLike_C"]):
        scatter = city_df.dropna(subset=["Temperature_C", "FeelsLike_C"])
        if not scatter.empty:
            fig_scatter = px.scatter(
                scatter,
                x="Temperature_C",
                y="FeelsLike_C",
                color="SourceWebsite" if "SourceWebsite" in scatter.columns else None,
                title=f"{selected_city}: Temperature vs Feels-Like",
                hover_data=[c for c in ["Date", "Condition", "Humidity_%"] if c in scatter.columns],
            )
            st.plotly_chart(fig_scatter, width="stretch")

    if all(c in city_df.columns for c in ["Date", "Temperature_C"]):
        trend = city_df.dropna(subset=["Date", "Temperature_C"]).copy()
        if not trend.empty:
            trend = trend.groupby("Date", as_index=False)["Temperature_C"].mean()
            fig_trend = px.line(
                trend,
                x="Date",
                y="Temperature_C",
                markers=True,
                title=f"{selected_city}: Temperature Trend",
            )
            fig_trend.update_layout(xaxis_title="Date", yaxis_title="Temperature (C)")
            st.plotly_chart(fig_trend, width="stretch")

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
        )
        st.plotly_chart(fig_condition, width="stretch")

with source_tab:
    st.subheader("Multi-Source Reliability View")

    temp_by_source, humid_by_source, disagreement = build_source_summary(filtered_df)

    s1, s2 = st.columns(2)

    with s1:
        if not temp_by_source.empty:
            fig_temp_source = px.bar(
                temp_by_source,
                x="SourceWebsite",
                y="Avg Temperature_C",
                color="SourceWebsite",
                title="Average Temperature by Source",
            )
            st.plotly_chart(fig_temp_source, width="stretch")
        else:
            st.info("Temperature comparison by source is unavailable.")

    with s2:
        if not humid_by_source.empty:
            fig_hum_source = px.bar(
                humid_by_source,
                x="SourceWebsite",
                y="Avg Humidity_%",
                color="SourceWebsite",
                title="Average Humidity by Source",
            )
            st.plotly_chart(fig_hum_source, width="stretch")
        else:
            st.info("Humidity comparison by source is unavailable.")

    st.subheader("Source Disagreement by City")
    if disagreement.empty:
        st.info("Need City + SourceWebsite + Temperature_C to compute disagreement.")
    else:
        show_rows = min(20, len(disagreement))
        st.dataframe(
            disagreement[["City", "Source Count", "Temp Disagreement (C)"]].head(show_rows),
            width="stretch",
            hide_index=True,
        )

        fig_disagree = px.bar(
            disagreement.head(show_rows),
            x="City",
            y="Temp Disagreement (C)",
            color="Temp Disagreement (C)",
            color_continuous_scale="Reds",
            title="Cities with Largest Temperature Disagreement Across Sources",
        )
        st.plotly_chart(fig_disagree, width="stretch")

with planner_tab:
    st.subheader("Travel Planner Assistant")

    target_label = st.selectbox(
        "Target recommendation level",
        options=["Ideal", "Good", "Moderate", "Avoid"],
        index=1,
    )

    label_order = {"Ideal": 0, "Good": 1, "Moderate": 2, "Avoid": 3}
    cutoff = label_order[target_label]

    planner_df = ranking_df.copy()
    planner_df["_rank"] = planner_df["Travel Recommendation"].map(label_order)
    planner_df = planner_df[planner_df["_rank"] <= cutoff].drop(columns=["_rank"])

    if planner_df.empty:
        st.warning("No cities match your planner target with current filters.")
    else:
        st.write("Cities that satisfy your target:")
        st.dataframe(planner_df, width="stretch", hide_index=True)

    st.subheader("Best Time of Day (if hourly data is available)")
    best_hour_df = best_hour_analysis(filtered_df)
    if best_hour_df.empty:
        st.info("Not enough timestamp richness to estimate best hour per city.")
    else:
        st.dataframe(best_hour_df, width="stretch", hide_index=True)

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
            )
            fig_line.update_layout(xaxis_title="Date", yaxis_title="Avg Temperature (C)")
            st.plotly_chart(fig_line, width="stretch")
