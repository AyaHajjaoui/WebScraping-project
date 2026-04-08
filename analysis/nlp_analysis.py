from __future__ import annotations

import re
from collections import Counter

import pandas as pd

DATA_PATH = "data/processed/weather_data.csv"
OUTPUT_PATH = "data/processed/condition_analysis.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load weather dataset from CSV."""
    return pd.read_csv(path)


def normalize_condition(text: str) -> str:
    """
    Normalize raw weather condition text to broader weather categories.
    Uses simple keyword checks (student-friendly approach).
    """
    value = text.lower().strip()

    mapping = {
        "clear": ["clear sky", "sunny", "clear"],
        "cloudy": ["partly cloudy", "mostly cloudy", "overcast", "cloudy"],
        "rain": ["light rain", "rain shower", "drizzle", "rain"],
        "storm": ["thunderstorm", "storm"],
        "fog": ["fog", "mist", "haze"],
        "snow": ["snowfall", "snow"],
    }

    for category, keywords in mapping.items():
        for keyword in keywords:
            if keyword in value:
                return category

    return "other"


def analyze_conditions(df: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """Overall normalized condition counts + raw word frequency."""
    overall_counts = (
        df["NormalizedCondition"]
        .value_counts()
        .rename_axis("NormalizedCondition")
        .reset_index(name="Count")
    )

    words = []
    for phrase in df["CleanCondition"]:
        words.extend(re.findall(r"[a-z]+", phrase))
    word_counts = Counter(words)

    return overall_counts, word_counts


def analyze_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Most common normalized conditions by source website."""
    counts = (
        df.groupby(["SourceWebsite", "NormalizedCondition"])
        .size()
        .reset_index(name="Count")
        .sort_values(["SourceWebsite", "Count"], ascending=[True, False])
    )
    return counts


def analyze_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Most common normalized conditions by city."""
    counts = (
        df.groupby(["City", "NormalizedCondition"])
        .size()
        .reset_index(name="Count")
        .sort_values(["City", "Count"], ascending=[True, False])
    )
    return counts


def print_top_words(word_counts: Counter, top_n: int = 15) -> None:
    """Print top words used in raw condition phrases."""
    top_words_df = pd.DataFrame(word_counts.most_common(top_n), columns=["Word", "Count"])
    print(f"\nTop {top_n} Most Frequent Words in Raw Condition Text:")
    if top_words_df.empty:
        print("No words found.")
    else:
        print(top_words_df.to_string(index=False))


def main() -> None:
    df = load_data()

    if "Condition" not in df.columns:
        print("Column 'Condition' not found in dataset.")
        return

    # Clean condition text: lowercase, trim spaces, drop missing.
    condition_df = df.copy()
    condition_df["CleanCondition"] = (
        condition_df["Condition"]
        .astype("string")
        .str.lower()
        .str.strip()
    )
    condition_df = condition_df.dropna(subset=["CleanCondition"])
    condition_df = condition_df[condition_df["CleanCondition"] != ""].copy()

    if condition_df.empty:
        print("No valid condition text found after cleaning.")
        return

    # Normalize into broader categories.
    condition_df["NormalizedCondition"] = condition_df["CleanCondition"].apply(normalize_condition)

    # Optional output file that can be reused later.
    save_cols = [
        "SourceWebsite",
        "City",
        "Country",
        "ScrapeDateTime",
        "Condition",
        "CleanCondition",
        "NormalizedCondition",
    ]
    condition_df[save_cols].to_csv(OUTPUT_PATH, index=False)

    # 1) Most common normalized conditions overall
    overall_counts, word_counts = analyze_conditions(condition_df)
    print("\nMost Common Normalized Conditions (Overall):")
    print(overall_counts.to_string(index=False))

    # 2) Most common conditions by source
    source_counts = analyze_by_source(condition_df)
    print("\nMost Common Normalized Conditions by Source:")
    print(source_counts.to_string(index=False))

    # 3) Most common conditions by city
    city_counts = analyze_by_city(condition_df)
    print("\nMost Common Normalized Conditions by City:")
    print(city_counts.to_string(index=False))

    # 4) Word frequency in raw condition text
    print_top_words(word_counts, top_n=15)

    # 5) Optional: top 10 raw condition phrases
    top_phrases = (
        condition_df["CleanCondition"]
        .value_counts()
        .head(10)
        .rename_axis("RawCondition")
        .reset_index(name="Count")
    )
    print("\nTop 10 Raw Condition Phrases:")
    print(top_phrases.to_string(index=False))

    # Optional comparison: how websites describe similar weather
    comparison = pd.crosstab(
        condition_df["SourceWebsite"],
        condition_df["NormalizedCondition"],
    )
    print("\nWebsite vs Normalized Condition (Counts):")
    print(comparison.to_string())

    print("\nSummary:")
    print(f"- Total rows analyzed: {len(condition_df)}")
    print(f"- Unique raw condition phrases: {condition_df['CleanCondition'].nunique()}")
    print(f"- Unique normalized categories: {condition_df['NormalizedCondition'].nunique()}")
    print(f"- Saved normalized condition data to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
