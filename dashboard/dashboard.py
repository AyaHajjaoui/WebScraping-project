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


def format_timestamp_label(value: str) -> str:
    """Format dataset freshness into a compact readable label."""
    try:
        dt = pd.to_datetime(value)
    except Exception:
        return str(value)

    if pd.isna(dt):
        return "Unknown"
    return dt.strftime("%d %b %Y, %H:%M")


def build_score_band_summary(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Bucket comfort scores into simple bands for overview distribution."""
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
                border: 1px solid {PALETTE["border"]};
                border-radius: 22px;
                padding: 1.2rem 1.25rem 1.1rem 1.25rem;
                min-height: 132px;
                box-shadow: 0 14px 24px rgba(188, 180, 172, 0.10);
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
                margin-bottom: 0.75rem;
            }}

            [data-testid="stTabs"] [data-baseweb="tab-list"] {{
                gap: 0.8rem;
                margin-top: 0.7rem;
                margin-bottom: 1.2rem;
            }}

            [data-testid="stTabs"] [data-baseweb="tab"] {{
                background: rgba(255, 250, 247, 0.92);
                border: 1px solid {PALETTE["border"]};
                border-radius: 999px;
                padding: 0.62rem 1.15rem;
                color: {PALETTE["muted"]};
            }}

            [data-testid="stTabs"] [aria-selected="true"] {{
                border-color: rgba(240, 90, 90, 0.45) !important;
                color: {PALETTE["accent"]} !important;
                background: {PALETTE["accent_soft"]} !important;
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
            }}

            .active-filter {{
                background: rgba(255,255,255,0.78);
                border: 1px solid {PALETTE["border"]};
                border-radius: 999px;
                color: {PALETTE["muted"]};
                font-size: 0.8rem;
                padding: 0.45rem 0.9rem;
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

inject_styles()

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-kicker">Weather Intelligence</div>
        <div class="hero-title">Smart Weather Travel & Comfort Dashboard</div>
        <div class="hero-sub">
            A softer editorial dashboard theme inspired by the reference layout:
            pale navigation, coral filter accents, airy panels, and cleaner analytics surfaces.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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
st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <div class="sidebar-title">Control Panel</div>
        <div class="sidebar-copy">
            Search for cities, narrow the data window, and compare only the parts of the dataset you care about.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

city_options = sorted(weather_df["City"].dropna().unique().tolist()) if "City" in weather_df.columns else []
source_options = (
    sorted(weather_df["SourceWebsite"].dropna().unique().tolist())
    if "SourceWebsite" in weather_df.columns
    else []
)

city_search = st.sidebar.text_input("Search city", placeholder="Type a city name")
visible_city_options = [
    city for city in city_options if city_search.lower() in city.lower()
] if city_search else city_options
default_city_selection = visible_city_options if visible_city_options else city_options

selected_cities = st.sidebar.multiselect("Cities", visible_city_options, default=default_city_selection)
selected_sources = st.sidebar.multiselect("Sources", source_options, default=source_options)

use_latest_only = st.sidebar.checkbox("Use latest record per city/source", value=False)
min_comfort = st.sidebar.slider("Minimum comfort score", min_value=0, max_value=100, value=0)
top_n = st.sidebar.slider(
    "Top cities to display",
    min_value=5,
    max_value=max(5, min(25, len(city_options))) if city_options else 5,
    value=min(10, max(5, len(city_options))) if city_options else 5,
)

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
last_updated_label = format_timestamp_label(last_updated)
top_city_options = ranking_df["City"].head(min(top_n, len(ranking_df))).tolist()
compare_default = top_city_options[: min(3, len(top_city_options))]

rec_labels = []
if "Travel Recommendation" in ranking_df.columns:
    rec_counts = ranking_df["Travel Recommendation"].value_counts()
    for label in ["Ideal", "Good", "Moderate", "Avoid"]:
        if label in rec_counts:
            rec_labels.append(f'<span class="pill">{label} {int(rec_counts[label])}</span>')
if rec_labels:
    st.markdown(f'<div class="pill-row">{"".join(rec_labels)}</div>', unsafe_allow_html=True)

active_filters = [
    f'<span class="active-filter">{city_count} cities in view</span>',
    f'<span class="active-filter">{len(selected_sources)} sources</span>',
    f'<span class="active-filter">Comfort >= {min_comfort}</span>',
    f'<span class="active-filter">Updated {last_updated_label}</span>',
]
if use_latest_only:
    active_filters.append('<span class="active-filter">Latest records only</span>')
st.markdown(f'<div class="active-filter-bar">{"".join(active_filters)}</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="large")
with c1:
    metric_tile("Best city now", insights["best_city"], "Highest comfort score")
with c2:
    metric_tile("Worst city now", insights["worst_city"], "Lowest comfort score")
with c3:
    metric_tile("Average comfort", f"{avg_comfort:.1f}" if pd.notna(avg_comfort) else "N/A", "Across filtered cities")
with c4:
    metric_tile("Cities in view", str(int(city_count)), f"Freshness: {last_updated_label}")

overview_tab, explorer_tab, source_tab, planner_tab = st.tabs(
    ["Overview", "City Explorer", "Source Quality", "Trip Planner"]
)

with overview_tab:
    panel_start("Overview", "A softer, lighter canvas for ranking, recommendation mix, and city-level comparison.")
    chart_controls_col, compare_controls_col = st.columns([1.1, 1.3])
    with chart_controls_col:
        selected_metric = st.selectbox(
            "Primary ranking metric",
            [c for c in ["Comfort Score", "Avg Temperature_C", "Avg FeelsLike_C", "Avg Humidity_%", "Avg WindSpeed_kmh"] if c in ranking_df.columns],
            index=0,
        )
    with compare_controls_col:
        compare_cities = st.multiselect(
            "Compare cities",
            options=ranking_df["City"].tolist(),
            default=compare_default,
        )

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("City Ranking")
        ranking_preview = ranking_df.head(top_n)
        st.dataframe(ranking_preview, use_container_width=True, hide_index=True)

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
                color="Recommendation",
                color_discrete_map={
                    "Ideal": PALETTE["mint"],
                    "Good": PALETTE["blue"],
                    "Moderate": PALETTE["yellow"],
                    "Avoid": PALETTE["accent"],
                    "Unknown": PALETTE["stone"],
                },
            )
            style_figure(fig_reco, height=420)
            st.plotly_chart(fig_reco, use_container_width=True)

    st.subheader(f"{selected_metric} by City")
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
    fig_comfort.update_layout(xaxis_title="City", yaxis_title=selected_metric)
    style_figure(fig_comfort, height=430)
    st.plotly_chart(fig_comfort, use_container_width=True)

    lower_left, lower_right = st.columns([1.1, 1])
    with lower_left:
        if compare_cities:
            compare_df = ranking_df[ranking_df["City"].isin(compare_cities)]
            if not compare_df.empty:
                st.subheader("Selected City Comparison")
                st.dataframe(compare_df, use_container_width=True, hide_index=True)
    with lower_right:
        score_band_df = build_score_band_summary(ranking_df)
        if not score_band_df.empty:
            st.subheader("Comfort Score Distribution")
            fig_band = px.area(
                score_band_df,
                x="Score Band",
                y="City Count",
                title="How many cities fall into each comfort band",
            )
            fig_band.update_traces(
                line=dict(color=PALETTE["teal"], width=2),
                fillcolor="rgba(86, 197, 193, 0.28)",
            )
            style_figure(fig_band, height=320)
            st.plotly_chart(fig_band, use_container_width=True)
    panel_end()

with explorer_tab:
    panel_start("City Explorer", "Inspect one city at a time with the lighter dashboard theme carried through each chart.")
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
                st.plotly_chart(fig_trend, use_container_width=True)

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
                st.plotly_chart(fig_scatter, use_container_width=True)

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
            st.plotly_chart(fig_sources, use_container_width=True)

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
        st.plotly_chart(fig_condition, use_container_width=True)
    panel_end()

with source_tab:
    panel_start("Source Quality", "Compare how each source behaves using the same muted palette and softer panels.")
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
                color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["mint"]],
            )
            style_figure(fig_temp_source, height=400)
            st.plotly_chart(fig_temp_source, use_container_width=True)
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
                color_discrete_sequence=[PALETTE["yellow"], PALETTE["mint"], PALETTE["stone"]],
            )
            style_figure(fig_hum_source, height=400)
            st.plotly_chart(fig_hum_source, use_container_width=True)
        else:
            st.info("Humidity comparison by source is unavailable.")

    st.subheader("Source Disagreement by City")
    if disagreement.empty:
        st.info("Need City + SourceWebsite + Temperature_C to compute disagreement.")
    else:
        show_rows = min(20, len(disagreement))
        max_disagreement = float(disagreement["Temp Disagreement (C)"].max())
        threshold = st.slider(
            "Minimum disagreement to display",
            min_value=0.0,
            max_value=max_disagreement,
            value=0.0,
            step=0.1,
        )
        disagreement = disagreement[disagreement["Temp Disagreement (C)"] >= threshold]
        st.dataframe(
            disagreement[["City", "Source Count", "Temp Disagreement (C)"]].head(show_rows),
            use_container_width=True,
            hide_index=True,
        )

        fig_disagree = px.bar(
            disagreement.head(show_rows),
            x="City",
            y="Temp Disagreement (C)",
            color="Temp Disagreement (C)",
            color_continuous_scale=[PALETTE["accent_soft"], PALETTE["accent"]],
            title="Cities with Largest Temperature Disagreement Across Sources",
        )
        style_figure(fig_disagree, height=420)
        st.plotly_chart(fig_disagree, use_container_width=True)
    panel_end()

with planner_tab:
    panel_start("Trip Planner", "Keep the planning workflow intact, but present it in the lighter report-style dashboard treatment.")
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
        st.write("Cities that satisfy your target:")
        st.dataframe(planner_df, use_container_width=True, hide_index=True)

    st.subheader("Best Time of Day (if hourly data is available)")
    best_hour_df = best_hour_analysis(filtered_df)
    if best_hour_df.empty:
        st.info("Not enough timestamp richness to estimate best hour per city.")
    else:
        st.dataframe(best_hour_df, use_container_width=True, hide_index=True)

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
            st.plotly_chart(fig_line, use_container_width=True)
    planner_csv = planner_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download planner results",
        data=planner_csv,
        file_name="planner_results.csv",
        mime="text/csv",
    )
    panel_end()
