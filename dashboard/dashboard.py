import os
import pandas as pd
import plotly.express as px
import streamlit as st


COLORS = {
    "Ideal": "#86efac",      # pastel green
    "Good": "#93c5fd",       # pastel blue
    "Moderate": "#fef3c7",   # pastel yellow
    "Avoid": "#fca5a5",      # pastel red
    "Unknown": "#d1d5db"     # pastel gray
}

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

# 🎨 Custom Theme (matches your screenshot style)
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
    background-color: #fca5a5;
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
    background-color: #fecaca !important;
}

/* Tabs - light red */
.stTabs [data-baseweb="tab"] {
    color: #000000;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    color: #000000 !important;
    border-bottom: 2px solid #fca5a5;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #000000;
    background-color: #f0f9ff;
    border-radius: 8px;
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

/* Spacing between sections */
.stDivider {
    margin: 32px 0 !important;
}

/* Add spacing to subheaders */
h2, h3 {
    margin-top: 32px !important;
    margin-bottom: 20px !important;
}

/* Spacing for markdown content */
.stMarkdown {
    margin: 12px 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Smart Weather Travel & Comfort Dashboard")

# Welcome section
st.markdown("""
<div style="background-color: #dbeafe; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
    <h3 style="color: #000000; margin-top: 0;">Welcome!</h3>
    <p style="color: #000000; margin-bottom: 0;">
        This dashboard helps you find the perfect destination based on weather and comfort. 
        <br><br>
        <strong>Quick Start:</strong> Select your cities and sources from the sidebar, set your comfort requirements, 
        then explore visualizations and recommendations below.
    </p>
</div>
""", unsafe_allow_html=True)

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

st.sidebar.header("Filter Options")

with st.sidebar.expander("Program Filters", expanded=True):
    city_options = sorted(weather_df["City"].dropna().unique().tolist())
    source_options = sorted(weather_df["SourceWebsite"].dropna().unique().tolist())

    selected_city = st.selectbox(
        "Select City",
        options=["All"] + city_options,
        key="city_filter"
    )

    selected_source = st.selectbox(
        "Select Source",
        options=["All"] + source_options,
        key="source_filter"
    )

with st.sidebar.expander("Financial Options", expanded=False):
    min_comfort = st.slider("Minimum comfort score", 0, 100, 0, key="comfort_filter")

with st.sidebar.expander("Record Requirements", expanded=False):
    use_latest_only = st.checkbox("Use latest record per city/source")
    sort_option = st.selectbox(
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

if selected_city != "All" and "City" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["City"] == selected_city]

if selected_source != "All" and "SourceWebsite" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["SourceWebsite"] == selected_source]

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
col_clear = st.sidebar.columns(1)[0]
with col_clear:
    if st.button("Clear Filters", use_container_width=True):
        st.session_state.clear()
        st.rerun()

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


col1, col2, col3, col4 = st.columns(4, gap="medium")
st.markdown(f"<p style='color: #000000; font-size: 16px; font-weight: 500;'>Found <strong>{city_count} cities</strong> matching your criteria</p>", unsafe_allow_html=True)


with col1:
    st.markdown(
        f"""<div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;">
        <p style="color: #94a3b8; margin: 0; font-size: 12px; text-transform: uppercase;">Best City</p>
        <p style="color: #000000; margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">{insights['best_city']}</p>
        </div>""",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""<div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;">
        <p style="color: #94a3b8; margin: 0; font-size: 12px; text-transform: uppercase;">Worst City</p>
        <p style="color: #000000; margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">{insights['worst_city']}</p>
        </div>""",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""<div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;">
        <p style="color: #94a3b8; margin: 0; font-size: 12px; text-transform: uppercase;">Avg Comfort</p>
        <p style="color: #000000; margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">{f'{avg_comfort:.1f}' if pd.notna(avg_comfort) else 'N/A'}</p>
        </div>""",
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""<div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;">
        <p style="color: #94a3b8; margin: 0; font-size: 12px; text-transform: uppercase;">Cities</p>
        <p style="color: #000000; margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">{int(city_count)}</p>
        </div>""",
        unsafe_allow_html=True
    )

overview_tab, explorer_tab, source_tab, planner_tab, all_cities_tab = st.tabs(
    ["Overview", "City Explorer", "Source Quality", "Trip Planner", "All Cities"]
)

with overview_tab:
    st.subheader("City Ranking")
    st.dataframe(ranking_df, width="stretch", hide_index=True)

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    csv_bytes = ranking_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current ranking as CSV",
        data=csv_bytes,
        file_name="city_ranking_filtered.csv",
        mime="text/csv",
        use_container_width=False
    )

    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)

    st.subheader("Travel Insights")
    st.markdown(f"- Best city for outdoor comfort: **{insights['best_city']}**")
    st.markdown(f"- City with highest humidity: **{insights['most_humid_city']}**")
    st.markdown(
        f"- City with highest feels-like temperature: **{insights['hottest_feels_like_city']}**"
    )
    st.markdown(f"- Cities good for travel today: **{insights['good_cities']}**")
    st.markdown(f"- Cities to avoid today: **{insights['avoid_cities']}**")

    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)

    if "Travel Recommendation" in ranking_df.columns:
        st.subheader("Recommendation Distribution")
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
            color="Recommendation",
            color_discrete_map=COLORS,
            title="Recommendation Mix",
            hole=0.35,
        )
        fig_reco.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(color="#000000", size=12),
            title_font=dict(color="#000000", size=14),
            showlegend=True,
            legend=dict(font=dict(color="#000000"))
        )
        st.plotly_chart(fig_reco, width="stretch")

    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)

    st.subheader("Comfort Score by City")
    fig_comfort = px.bar(
    ranking_df,
    x="City",
    y="Comfort Score",
    color="Travel Recommendation",
    color_discrete_map=COLORS,
    title="Comfort Ranking Across Cities",
)
    fig_comfort.update_layout(
        xaxis_title="City", 
        yaxis_title="Comfort Score", 
        paper_bgcolor="#ffffff", 
        plot_bgcolor="#f8fafc",
        font=dict(color="#000000", size=12),
        title_font=dict(color="#000000", size=14),
        xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
        yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
    )
    st.plotly_chart(fig_comfort, width="stretch")

    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)

    if "Avg Humidity_%" in ranking_df.columns:
        st.subheader("Average Humidity by City")
        fig_humidity = px.bar(
            ranking_df,
            x="City",
            y="Avg Humidity_%",
            color="Avg Humidity_%",
            color_continuous_scale=["#dbeafe", "#bfdbfe", "#93c5fd"],
            title="Humidity Comparison",
        )
        fig_humidity.update_layout(
            xaxis_title="City", 
            yaxis_title="Humidity (%)", 
            paper_bgcolor="#ffffff", 
            plot_bgcolor="#f8fafc",
            font=dict(color="#000000", size=12),
            title_font=dict(color="#000000", size=14),
            xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
            yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
        )
        st.plotly_chart(fig_humidity, width="stretch")

    st.divider()

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

    st.divider()

    st.subheader("Temperature vs Feels-Like Analysis")
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
            fig_scatter.update_layout(
                paper_bgcolor="#ffffff", 
                plot_bgcolor="#f8fafc",
                font=dict(color="#000000", size=12),
                title_font=dict(color="#000000", size=14),
                xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
                yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
                hovermode='closest'
            )
            st.plotly_chart(fig_scatter, width="stretch")

    st.divider()

    st.subheader("Temperature Trend Analysis")
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
            fig_trend.update_layout(
                xaxis_title="Date", 
                yaxis_title="Temperature (C)", 
                paper_bgcolor="#ffffff", 
                plot_bgcolor="#f8fafc",
                font=dict(color="#000000", size=12),
                title_font=dict(color="#000000", size=14),
                xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
                yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
            )
            st.plotly_chart(fig_trend, width="stretch")

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
        )
        fig_condition.update_layout(
            paper_bgcolor="#ffffff", 
            plot_bgcolor="#f8fafc",
            font=dict(color="#000000", size=12),
            title_font=dict(color="#000000", size=14),
            xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
            yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
        )
        st.plotly_chart(fig_condition, width="stretch")

    st.divider()

with source_tab:
    st.subheader("Multi-Source Reliability View")

    temp_by_source, humid_by_source, disagreement = build_source_summary(filtered_df)

    st.divider()

    st.subheader("Average Temperature by Source")
    if not temp_by_source.empty:
        fig_temp_source = px.bar(
            temp_by_source,
            x="SourceWebsite",
            y="Avg Temperature_C",
            color="Avg Temperature_C",
            color_continuous_scale=["#dbeafe", "#bfdbfe", "#93c5fd"],
            title="Average Temperature by Source",
        )
        fig_temp_source.update_layout(
            paper_bgcolor="#ffffff", 
            plot_bgcolor="#f8fafc",
            font=dict(color="#000000", size=12),
            title_font=dict(color="#000000", size=14),
            xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
            yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
        )
        st.plotly_chart(fig_temp_source, width="stretch")
    else:
        st.info("Temperature comparison by source is unavailable.")

    st.divider()

    st.subheader("Average Humidity by Source")
    if not humid_by_source.empty:
        fig_hum_source = px.bar(
            humid_by_source,
            x="SourceWebsite",
            y="Avg Humidity_%",
            color="Avg Humidity_%",
            color_continuous_scale=["#dbeafe", "#bfdbfe", "#93c5fd"],
            title="Average Humidity by Source",
        )
        fig_hum_source.update_layout(
            paper_bgcolor="#ffffff", 
            plot_bgcolor="#f8fafc",
            font=dict(color="#000000", size=12),
            title_font=dict(color="#000000", size=14),
            xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
            yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
        )
        st.plotly_chart(fig_hum_source, width="stretch")
    else:
        st.info("Humidity comparison by source is unavailable.")

    st.divider()

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
            color_continuous_scale=["#fee2e2", "#fecaca", "#fca5a5"],
            title="Cities with Largest Temperature Disagreement Across Sources",
        )
        fig_disagree.update_layout(
            paper_bgcolor="#ffffff", 
            plot_bgcolor="#f8fafc",
            font=dict(color="#000000", size=12),
            title_font=dict(color="#000000", size=14),
            xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
            yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
        )
        st.plotly_chart(fig_disagree, width="stretch")

    st.divider()

with planner_tab:
    st.subheader("Smart Travel Planner")
    
    # Clean form container
    st.markdown("""
    <div style="background-color: #f3f4f6; padding: 20px; border-radius: 10px; border-left: 4px solid #fca5a5; margin-bottom: 24px;">
    <p style="color: #000000; margin: 0; font-size: 14px;"><strong>Select your preferences and click Search to find your ideal destination</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Form inputs in organized columns
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        st.markdown("<p style='color: #000000; font-weight: 600; margin-bottom: 8px; font-size: 13px;'>Temperature</p>", unsafe_allow_html=True)
        temp_pref = st.selectbox(
            "Temperature preference:",
            options=[15, 18, 20, 22, 25, 28, 30, 35],
            format_func=lambda x: f"{x}°C",
            label_visibility="collapsed",
            key="temp_select"
        )
    
    with col2:
        st.markdown("<p style='color: #000000; font-weight: 600; margin-bottom: 8px; font-size: 13px;'>Humidity</p>", unsafe_allow_html=True)
        humidity_pref = st.selectbox(
            "Humidity preference:",
            options=[20, 30, 40, 50, 60, 70, 80],
            format_func=lambda x: f"{x}%",
            label_visibility="collapsed",
            key="humidity_select"
        )
    
    with col3:
        st.markdown("<p style='color: #000000; font-weight: 600; margin-bottom: 8px; font-size: 13px;'>Country</p>", unsafe_allow_html=True)
        country_options = sorted(ranking_df["Country"].dropna().unique().tolist()) if "Country" in ranking_df.columns else []
        country_filter = st.selectbox(
            "Country preference:",
            options=["All Countries"] + country_options,
            label_visibility="collapsed",
            key="country_select_new"
        )
        country_filter = None if country_filter == "All Countries" else [country_filter]
    
    with col4:
        st.markdown("<p style='color: #000000; font-weight: 600; margin-bottom: 8px; font-size: 13px;'>Comfort Level</p>", unsafe_allow_html=True)
        recommendation_level = st.selectbox(
            "Minimum comfort:",
            options=["Ideal", "Good", "Moderate", "Avoid"],
            label_visibility="collapsed",
            key="comfort_select_new"
        )
    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
    with col_btn_center:
        search_clicked = st.button("Search Destinations", use_container_width=True, type="primary")
    
    if search_clicked:
        smart_df = ranking_df.copy()
        
        if country_filter and "Country" in smart_df.columns:
            smart_df = smart_df[smart_df["Country"].isin(country_filter)]
        
        if "Avg Temperature_C" in smart_df.columns and "Avg Humidity_%" in smart_df.columns:
            smart_df["Preference Score"] = smart_df.apply(
                lambda row: 100 - (abs(row["Avg Temperature_C"] - temp_pref) * 2 + abs(row["Avg Humidity_%"] - humidity_pref) * 0.5),
                axis=1
            )
        
        label_order = {"Ideal": 0, "Good": 1, "Moderate": 2, "Avoid": 3}
        cutoff = label_order[recommendation_level]
        smart_df["_rank"] = smart_df["Travel Recommendation"].map(label_order)
        smart_df = smart_df[smart_df["_rank"] <= cutoff].drop(columns=["_rank"])
        
        if "Preference Score" in smart_df.columns:
            smart_df = smart_df.sort_values("Preference Score", ascending=False)
        else:
            smart_df = smart_df.sort_values("Comfort Score", ascending=False)
        
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        
        if smart_df.empty:
            st.warning("No cities match your preferences. Try adjusting your filters.")
        else:
            st.success(f"Found {len(smart_df)} cities matching your preferences!")
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            display_cols = [c for c in ["City", "Country", "Avg Temperature_C", "Avg Humidity_%", "Comfort Score", "Preference Score"] if c in smart_df.columns]
            st.dataframe(smart_df[display_cols], width="stretch", hide_index=True, use_container_width=True)
            
            if not smart_df.empty:
                st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
                best_match = smart_df.iloc[0]
                st.markdown(
                    f"""<div style='background-color: #fecaca; padding: 16px; border-radius: 10px; margin-top: 16px;'>
                    <h4 style='color: #000000; margin-top: 0; margin-bottom: 12px;'>Top Recommendation: {best_match['City']}</h4>
                    <p style='color: #000000; margin: 8px 0;'><strong>Comfort Score:</strong> {best_match['Comfort Score']:.1f}</p>"""
                    + (f"<p style='color: #000000; margin: 8px 0;'><strong>Match Score:</strong> {best_match['Preference Score']:.1f}</p>" if "Preference Score" in smart_df.columns else "")
                    + (f"<p style='color: #000000; margin: 8px 0;'><strong>Temperature:</strong> {best_match['Avg Temperature_C']:.1f}°C</p>" if "Avg Temperature_C" in smart_df.columns else "")
                    + (f"<p style='color: #000000; margin: 8px 0;'><strong>Humidity:</strong> {best_match['Avg Humidity_%']:.1f}%</p>" if "Avg Humidity_%" in smart_df.columns else "")
                    + "</div>", 
                    unsafe_allow_html=True
                )
    
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Best Time of Day (if hourly data is available)")
    best_hour_df = best_hour_analysis(filtered_df)
    if best_hour_df.empty:
        st.info("Not enough timestamp richness to estimate best hour per city.")
    else:
        st.dataframe(best_hour_df, width="stretch", hide_index=True)

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
            )
            fig_line.update_layout(
                xaxis_title="Date", 
                yaxis_title="Avg Temperature (C)", 
                paper_bgcolor="#ffffff", 
                plot_bgcolor="#f8fafc",
                font=dict(color="#000000", size=12),
                title_font=dict(color="#000000", size=14),
                xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
                yaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
            )
            st.plotly_chart(fig_line, width="stretch")

with all_cities_tab:
    st.subheader("All Cities Directory")
    
    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
    
    # Create comprehensive city view
    all_cities_df = build_city_ranking(weather_df)
    
    # Real-time search input
    search_term = st.text_input(
        "Search cities by name or country", 
        placeholder="e.g., Cairo, Egypt",
        help="Results update as you type"
    )
    
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    
    # Filter based on search
    filtered_all_cities = all_cities_df.copy()
    if search_term:
        search_lower = search_term.lower()
        mask = (filtered_all_cities["City"].str.lower().str.contains(search_lower, na=False)) | \
               (filtered_all_cities["Country"].str.lower().str.contains(search_lower, na=False))
        filtered_all_cities = filtered_all_cities[mask]
    
    st.markdown(f"**Total: {len(filtered_all_cities)} cities**")
    st.markdown(f"*Showing all {len(all_cities_df)} available cities*")
    
    # Display with styling
    st.dataframe(filtered_all_cities, width="stretch", hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Download all cities
    csv_all = all_cities_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download All Cities as CSV",
        data=csv_all,
        file_name="all_cities_directory.csv",
        mime="text/csv",
        use_container_width=True
    )
