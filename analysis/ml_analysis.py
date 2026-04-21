# from __future__ import annotations

# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.tree import DecisionTreeRegressor

# DATA_PATH = "data/processed/weather_data.csv"


# def load_data(path: str = DATA_PATH) -> pd.DataFrame:
#     """Load weather dataset from CSV."""
#     return pd.read_csv(path)


# def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
#     """Clean and prepare feature matrix X and target y."""
#     data = df.copy()

#     # Convert datetime safely and create time-based features.
#     try:
#         data["ScrapeDateTime"] = pd.to_datetime(
#             data["ScrapeDateTime"], errors="coerce", utc=True, format="mixed"
#         )
#     except TypeError:
#         data["ScrapeDateTime"] = pd.to_datetime(
#             data["ScrapeDateTime"], errors="coerce", utc=True
#         )

#     data["Hour"] = data["ScrapeDateTime"].dt.hour
#     data["Day"] = data["ScrapeDateTime"].dt.day
#     data["Month"] = data["ScrapeDateTime"].dt.month

#     # Convert numeric columns safely.
#     numeric_cols = ["Temperature_C", "FeelsLike_C", "Humidity_%", "WindSpeed_kmh"]
#     for col in numeric_cols:
#         data[col] = pd.to_numeric(data[col], errors="coerce")

#     # Drop rows with missing target as requested.
#     data = data.dropna(subset=["Temperature_C"])

#     feature_cols = [
#         "SourceWebsite",
#         "City",
#         "Country",
#         "FeelsLike_C",
#         "Humidity_%",
#         "WindSpeed_kmh",
#         "Hour",
#         "Day",
#         "Month",
#     ]

#     # Keep only rows where these columns exist, then split X/y.
#     data = data[feature_cols + ["Temperature_C"]].copy()
#     y = data["Temperature_C"]
#     X = data[feature_cols]
#     return X, y


# def train_and_evaluate_models(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_train: pd.Series,
#     y_test: pd.Series,
# ) -> tuple[pd.DataFrame, str, Pipeline]:
#     """Train multiple models and return comparison table + best model."""
#     categorical_features = ["SourceWebsite", "City", "Country"]
#     numeric_features = ["FeelsLike_C", "Humidity_%", "WindSpeed_kmh", "Hour", "Day", "Month"]

#     preprocessor = ColumnTransformer(
#         transformers=[
#             (
#                 "cat",
#                 Pipeline(
#                     steps=[
#                         ("imputer", SimpleImputer(strategy="most_frequent")),
#                         ("onehot", OneHotEncoder(handle_unknown="ignore")),
#                     ]
#                 ),
#                 categorical_features,
#             ),
#             (
#                 "num",
#                 Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
#                 numeric_features,
#             ),
#         ]
#     )

#     models = {
#         "Linear Regression": LinearRegression(),
#         "Decision Tree": DecisionTreeRegressor(random_state=42),
#         "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
#     }

#     results: list[dict[str, float | str]] = []
#     fitted_pipelines: dict[str, Pipeline] = {}

#     for model_name, model in models.items():
#         pipeline = Pipeline(
#             steps=[
#                 ("preprocessor", preprocessor),
#                 ("model", model),
#             ]
#         )
#         pipeline.fit(X_train, y_train)
#         preds = pipeline.predict(X_test)

#         mae = mean_absolute_error(y_test, preds)
#         rmse = np.sqrt(mean_squared_error(y_test, preds))
#         r2 = r2_score(y_test, preds)

#         results.append(
#             {
#                 "Model": model_name,
#                 "MAE": mae,
#                 "RMSE": rmse,
#                 "R2": r2,
#             }
#         )
#         fitted_pipelines[model_name] = pipeline

#     results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
#     best_model_name = str(results_df.loc[0, "Model"])
#     best_pipeline = fitted_pipelines[best_model_name]
#     return results_df, best_model_name, best_pipeline


# def print_feature_importance_if_available(best_model_name: str, best_pipeline: Pipeline) -> None:
#     """Print top random forest feature importances when available."""
#     if best_model_name != "Random Forest":
#         return

#     preprocessor = best_pipeline.named_steps["preprocessor"]
#     model = best_pipeline.named_steps["model"]

#     if not hasattr(model, "feature_importances_"):
#         return

#     feature_names = preprocessor.get_feature_names_out()
#     importances = model.feature_importances_

#     importance_df = (
#         pd.DataFrame({"Feature": feature_names, "Importance": importances})
#         .sort_values("Importance", ascending=False)
#         .head(10)
#         .reset_index(drop=True)
#     )

#     print("\nTop 10 Feature Importances (Random Forest):")
#     print(importance_df.to_string(index=False))


# def main() -> None:
#     df = load_data()
#     X, y = prepare_features(df)

#     if len(X) < 10:
#         print("Not enough rows after preprocessing. Add more data and try again.")
#         return

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     results_df, best_model_name, best_pipeline = train_and_evaluate_models(
#         X_train, X_test, y_train, y_test
#     )

#     print("\nModel Comparison Results:")
#     print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

#     print(f"\nBest Model (lowest RMSE): {best_model_name}")

#     print_feature_importance_if_available(best_model_name, best_pipeline)


# if __name__ == "__main__":
#     main()

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/processed/weather_data.csv"
TARGET_COLUMN = "Travel Recommendation"
TARGET_ORDER = ["Ideal", "Good", "Moderate", "Avoid"]
MIN_TRAINING_ROWS = 20


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def comfort_score(temp: object, feels_like: object, humidity: object) -> float:
    """
    Compute a simple travel-comfort score from weather inputs.

    This score already exists in the dashboard logic, so using it here keeps the
    classification target easy to explain in a report and consistent across the app.
    """
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


def travel_recommendation(score: object) -> str:
    """Convert comfort score into clear user-facing travel classes."""
    if pd.isna(score):
        return "Unknown"
    if float(score) >= 80:
        return "Ideal"
    if float(score) >= 60:
        return "Good"
    if float(score) >= 40:
        return "Moderate"
    return "Avoid"


def add_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the classification target from weather conditions.

    We predict travel recommendation classes instead of a raw temperature value.
    This is easier to present because the output is actionable for dashboard users:
    Ideal, Good, Moderate, or Avoid.
    """
    data = df.copy()

    try:
        data["ScrapeDateTime"] = pd.to_datetime(
            data["ScrapeDateTime"], errors="coerce", utc=True, format="mixed"
        )
    except TypeError:
        data["ScrapeDateTime"] = pd.to_datetime(
            data["ScrapeDateTime"], errors="coerce", utc=True
        )

    numeric_cols = ["Temperature_C", "FeelsLike_C", "Humidity_%", "WindSpeed_kmh"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data.get(col), errors="coerce")

    data["Comfort Score"] = data.apply(
        lambda row: comfort_score(
            row.get("Temperature_C"),
            row.get("FeelsLike_C"),
            row.get("Humidity_%"),
        ),
        axis=1,
    )
    data[TARGET_COLUMN] = data["Comfort Score"].apply(travel_recommendation)
    data = data[data[TARGET_COLUMN] != "Unknown"].copy()
    data[TARGET_COLUMN] = pd.Categorical(
        data[TARGET_COLUMN],
        categories=TARGET_ORDER,
        ordered=True,
    )
    return data


def get_preprocessing_summary() -> pd.DataFrame:
    """Describe the preprocessing steps that actually exist in the pipeline."""
    return pd.DataFrame(
        [
            {"Step": "Target creation", "Details": "Travel Recommendation derived from Comfort Score into Ideal, Good, Moderate, Avoid."},
            {"Step": "Missing-value handling", "Details": "Categorical values use most-frequent imputation; numeric values use median imputation."},
            {"Step": "Feature engineering", "Details": "Hour, Day, and Month are created from ScrapeDateTime."},
            {"Step": "Categorical encoding", "Details": "SourceWebsite, City, and Country are one-hot encoded."},
            {"Step": "Numeric scaling", "Details": "Numeric features are standardized after imputation."},
            {"Step": "Data split", "Details": "Train/test split uses stratification to preserve class proportions."},
        ]
    )


def get_class_distribution(y: pd.Series) -> pd.DataFrame:
    """Return class counts and percentages in a display-friendly table."""
    counts = y.value_counts(dropna=False).reindex(TARGET_ORDER, fill_value=0)
    distribution_df = pd.DataFrame(
        {
            "Class": counts.index,
            "Count": counts.values.astype(int),
            "Percentage": (counts.values / max(len(y), 1) * 100).round(2),
        }
    )
    return distribution_df


def print_class_distribution(y: pd.Series) -> pd.DataFrame:
    """Print class counts so imbalance is visible in CLI runs and notebooks."""
    distribution_df = get_class_distribution(y)
    print("\nClass Distribution:")
    print(distribution_df.to_string(index=False))
    return distribution_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Create the feature matrix, target labels, prepared dataset, and class counts."""
    data = add_classification_target(df)

    data["Hour"] = data["ScrapeDateTime"].dt.hour
    data["Day"] = data["ScrapeDateTime"].dt.day
    data["Month"] = data["ScrapeDateTime"].dt.month

    feature_cols = [
        "SourceWebsite",
        "City",
        "Country",
        "FeelsLike_C",
        "Humidity_%",
        "WindSpeed_kmh",
        "Hour",
        "Day",
        "Month",
    ]

    data = data[feature_cols + [TARGET_COLUMN, "Comfort Score"]].copy()
    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    X = data[feature_cols]
    y = data[TARGET_COLUMN].astype(str)
    class_distribution_df = print_class_distribution(y)
    return X, y, data, class_distribution_df


def build_preprocessor() -> ColumnTransformer:
    """Build a reusable preprocessing pipeline for classification models."""
    categorical_features = ["SourceWebsite", "City", "Country"]
    numeric_features = ["FeelsLike_C", "Humidity_%", "WindSpeed_kmh", "Hour", "Day", "Month"]

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )


def build_models() -> dict[str, object]:
    """
    Return classification models to compare.

    `class_weight="balanced"` is enough here because the existing class
    distribution is already reasonably healthy.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=5,
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=1,
        ),
    }


def clean_feature_name(name: str) -> str:
    """Make pipeline feature names easier to read in the dashboard."""
    cleaned = name
    cleaned = cleaned.replace("cat__", "").replace("num__", "")
    cleaned = cleaned.replace("onehot__", "").replace("imputer__", "")
    cleaned = cleaned.replace("SourceWebsite_", "Source: ")
    cleaned = cleaned.replace("City_", "City: ")
    cleaned = cleaned.replace("Country_", "Country: ")
    cleaned = cleaned.replace("Humidity_%", "Humidity (%)")
    cleaned = cleaned.replace("WindSpeed_kmh", "Wind Speed (km/h)")
    cleaned = cleaned.replace("FeelsLike_C", "Feels Like (C)")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_classification_report_df(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Convert sklearn classification report into a tidy dataframe."""
    report = classification_report(
        y_true,
        y_pred,
        labels=TARGET_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df.rename(
        columns={
            "index": "Label",
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
            "support": "Support",
        }
    )
    if "accuracy" in report_df["Label"].values:
        report_df.loc[report_df["Label"] == "accuracy", ["Recall", "F1-Score", "Support"]] = np.nan
    report_df["Support"] = pd.to_numeric(report_df["Support"], errors="coerce").round(0)
    return report_df


def build_confusion_matrix_df(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Build a labeled confusion matrix dataframe for the best model."""
    matrix = confusion_matrix(y_true, y_pred, labels=TARGET_ORDER)
    return pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in TARGET_ORDER],
        columns=[f"Predicted: {label}" for label in TARGET_ORDER],
    )


def build_class_metric_chart_df(report_df: pd.DataFrame) -> pd.DataFrame:
    """Build a chart-ready per-class metrics table."""
    class_rows = report_df[report_df["Label"].isin(TARGET_ORDER)].copy()
    chart_df = class_rows.melt(
        id_vars="Label",
        value_vars=["Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )
    chart_df = chart_df.rename(columns={"Label": "Class"})
    return chart_df


def analyze_confusion_pairs(confusion_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Identify the most common class confusions and summarize them."""
    pairs: list[dict[str, object]] = []
    labels = TARGET_ORDER
    matrix = confusion_df.to_numpy()

    for actual_index, actual_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            if actual_index == predicted_index:
                continue
            count = int(matrix[actual_index, predicted_index])
            if count > 0:
                pairs.append(
                    {
                        "Actual Class": actual_label,
                        "Predicted Class": predicted_label,
                        "Misclassifications": count,
                    }
                )

    pairs_df = pd.DataFrame(pairs).sort_values("Misclassifications", ascending=False).reset_index(drop=True)
    if pairs_df.empty:
        return pairs_df, "The confusion matrix shows no meaningful class mix-ups on the test split."

    top_pair = pairs_df.iloc[0]
    summary = (
        f"The most common confusion is {top_pair['Actual Class']} predicted as {top_pair['Predicted Class']} "
        f"({int(top_pair['Misclassifications'])} cases), which suggests those two recommendation levels share overlapping weather patterns."
    )
    return pairs_df.head(5), summary


def summarize_bias(
    comparison_df: pd.DataFrame,
    report_df: pd.DataFrame,
    class_distribution_df: pd.DataFrame,
    confusion_pairs_df: pd.DataFrame,
) -> str:
    """Produce a stronger interpretation of balance, weak classes, and metric gaps."""
    class_rows = report_df[report_df["Label"].isin(TARGET_ORDER)].copy()
    if class_rows.empty or comparison_df.empty:
        return "Bias summary unavailable."

    class_rows["Recall"] = pd.to_numeric(class_rows["Recall"], errors="coerce")
    class_rows["F1-Score"] = pd.to_numeric(class_rows["F1-Score"], errors="coerce")
    weakest_recall_row = class_rows.sort_values("Recall", ascending=True).iloc[0]
    weakest_f1_row = class_rows.sort_values("F1-Score", ascending=True).iloc[0]
    best_row = comparison_df.iloc[0]
    class_spread = float(class_distribution_df["Percentage"].max() - class_distribution_df["Percentage"].min())
    accuracy_gap = float(best_row["Accuracy"] - best_row["Weighted F1"])

    notes: list[str] = []
    if class_spread <= 15:
        notes.append("The class distribution looks balanced enough for a clean classification study, with no class dominating the dataset.")
    else:
        notes.append("The class distribution is somewhat uneven, so weighted metrics are more trustworthy than accuracy alone.")

    if accuracy_gap >= 0.08:
        notes.append("Accuracy is noticeably higher than weighted F1, which suggests the headline score may hide uneven class performance.")
    else:
        notes.append("Accuracy and weighted F1 are close, so overall performance does not look artificially inflated by class imbalance.")

    notes.append(
        f"The weakest class is {weakest_f1_row['Label']} with F1-score {float(weakest_f1_row['F1-Score']):.3f}, "
        f"while the weakest recall also appears in {weakest_recall_row['Label']} at {float(weakest_recall_row['Recall']):.3f}."
    )

    if not confusion_pairs_df.empty:
        top_pair = confusion_pairs_df.iloc[0]
        notes.append(
            f"The main confusion pattern is {top_pair['Actual Class']} vs {top_pair['Predicted Class']}, "
            "which is typical when neighboring recommendation bands have similar weather conditions."
        )

    return " ".join(notes)


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    stage: str,
    notes: str,
) -> dict[str, object]:
    """Evaluate one fitted pipeline and return metrics plus artifacts."""
    preds = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)

    comparison_row = {
        "Stage": stage,
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_macro,
        "Weighted F1": f1_weighted,
        "Notes": notes,
    }

    report_df = build_classification_report_df(y_test, preds)
    confusion_df = build_confusion_matrix_df(y_test, preds)
    confusion_pairs_df, confusion_summary = analyze_confusion_pairs(confusion_df)

    return {
        "comparison_row": comparison_row,
        "classification_report_df": report_df,
        "confusion_matrix_df": confusion_df,
        "class_metric_chart_df": build_class_metric_chart_df(report_df),
        "confusion_pairs_df": confusion_pairs_df,
        "confusion_summary": confusion_summary,
        "predictions": preds,
    }


def fit_baseline_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, dict[str, object]]]:
    """Fit and evaluate baseline models."""
    preprocessor = build_preprocessor()
    models = build_models()

    rows: list[dict[str, object]] = []
    fitted_pipelines: dict[str, Pipeline] = {}
    artifacts: dict[str, dict[str, object]] = {}

    baseline_notes = {
        "Logistic Regression": "Simple linear baseline for class separation.",
        "Decision Tree Classifier": "Single-tree baseline with strong interpretability.",
        "Random Forest Classifier": "Ensemble baseline for stronger stability and robustness.",
    }

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        evaluation = evaluate_pipeline(
            pipeline,
            X_test,
            y_test,
            model_name=model_name,
            stage="Baseline",
            notes=baseline_notes.get(model_name, "Baseline model."),
        )
        rows.append(evaluation["comparison_row"])
        fitted_pipelines[model_name] = pipeline
        artifacts[model_name] = evaluation

    baseline_df = pd.DataFrame(rows).sort_values(
        by=["Weighted F1", "Accuracy", "F1-Score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return baseline_df, fitted_pipelines, artifacts


def tune_tree_model(
    strongest_tree_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, dict[str, object], str]:
    """Tune the strongest tree-based candidate with a light grid search."""
    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    if strongest_tree_name == "Decision Tree Classifier":
        estimator = DecisionTreeClassifier(random_state=42, class_weight="balanced")
        param_grid = {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [8, 14],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 3],
        }
        tuned_name = "Tuned Decision Tree"
    else:
        estimator = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1)
        param_grid = {
            "model__n_estimators": [150, 250],
            "model__max_depth": [None, 12],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }
        tuned_name = "Tuned Random Forest"

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, tuned_name


def build_tuning_summary_df(
    strongest_tree_name: str,
    baseline_df: pd.DataFrame,
    tuned_row: dict[str, object],
    best_params: dict[str, object],
) -> pd.DataFrame:
    """Summarize baseline vs tuned performance for the tuned model family."""
    baseline_row = baseline_df[baseline_df["Model"] == strongest_tree_name].iloc[0]
    tuned_weighted_f1 = float(tuned_row["Weighted F1"])
    baseline_weighted_f1 = float(baseline_row["Weighted F1"])
    tuned_accuracy = float(tuned_row["Accuracy"])
    baseline_accuracy = float(baseline_row["Accuracy"])

    rows = [
        {
            "Stage": "Baseline",
            "Model": strongest_tree_name,
            "Accuracy": baseline_accuracy,
            "Weighted F1": baseline_weighted_f1,
            "Macro F1": float(baseline_row["F1-Score"]),
            "Notes": "Initial tree-based candidate before hyperparameter tuning.",
        },
        {
            "Stage": "Tuned",
            "Model": str(tuned_row["Model"]),
            "Accuracy": tuned_accuracy,
            "Weighted F1": tuned_weighted_f1,
            "Macro F1": float(tuned_row["F1-Score"]),
            "Notes": f"Best parameters: {best_params}",
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_df["Weighted F1 Improvement"] = [
        0.0,
        round(tuned_weighted_f1 - baseline_weighted_f1, 4),
    ]
    return summary_df


def explain_final_model_choice(
    final_model_name: str,
    experiment_df: pd.DataFrame,
    strongest_tree_name: str,
) -> str:
    """Generate an explainable reason for the final selected model."""
    final_row = experiment_df.iloc[0]
    baseline_tree_row = experiment_df[
        (experiment_df["Stage"] == "Baseline") & (experiment_df["Model"] == strongest_tree_name)
    ].iloc[0]
    gain = float(final_row["Weighted F1"] - baseline_tree_row["Weighted F1"])

    if "Decision Tree" in final_model_name:
        return (
            f"{final_model_name} was selected because it achieved the strongest weighted F1 in the experiment flow. "
            f"It is also easier to explain in a student project because the decision process is more interpretable than a large ensemble. "
            f"The tradeoff is that single trees can overfit more easily, so the confusion matrix and class-level metrics should still be checked carefully."
        )

    if "Random Forest" in final_model_name:
        gain_text = "a small but useful" if gain >= 0 else "a competitive"
        return (
            f"{final_model_name} was selected because it delivered {gain_text} weighted F1 score while remaining more stable than a single decision tree. "
            "Random forests are usually a safer final choice when class boundaries overlap, because they reduce overfitting risk and combine many decision paths."
        )

    return (
        f"{final_model_name} was selected because it achieved the strongest weighted F1-score while keeping the overall experiment simple and explainable."
    )


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    class_distribution_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, Pipeline, dict[str, object]]:
    """Train baseline models, tune the strongest tree-based model, and return full experiment artifacts."""
    baseline_df, fitted_pipelines, baseline_artifacts = fit_baseline_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    strongest_tree_name = baseline_df[
        baseline_df["Model"].isin(["Decision Tree Classifier", "Random Forest Classifier"])
    ].iloc[0]["Model"]
    tuned_pipeline, best_params, tuned_name = tune_tree_model(
        str(strongest_tree_name),
        X_train,
        y_train,
    )
    tuned_evaluation = evaluate_pipeline(
        tuned_pipeline,
        X_test,
        y_test,
        model_name=tuned_name,
        stage="Tuned",
        notes=f"Best tree-based model after GridSearchCV tuning.",
    )

    experiment_df = pd.concat(
        [
            baseline_df,
            pd.DataFrame([tuned_evaluation["comparison_row"]]),
        ],
        ignore_index=True,
    ).sort_values(
        by=["Weighted F1", "Accuracy", "F1-Score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    final_model_name = str(experiment_df.loc[0, "Model"])
    final_model_lookup: dict[str, tuple[Pipeline, dict[str, object]]] = {
        **{name: (pipeline, baseline_artifacts[name]) for name, pipeline in fitted_pipelines.items()},
        tuned_name: (tuned_pipeline, tuned_evaluation),
    }
    final_pipeline, final_artifacts = final_model_lookup[final_model_name]

    if class_distribution_df is None:
        class_distribution_df = pd.DataFrame()

    final_artifacts["bias_summary"] = summarize_bias(
        experiment_df,
        final_artifacts["classification_report_df"],
        class_distribution_df,
        final_artifacts["confusion_pairs_df"],
    )
    final_artifacts["baseline_comparison_df"] = baseline_df.copy()
    final_artifacts["experiment_flow_df"] = experiment_df.copy()
    final_artifacts["tuning_summary_df"] = build_tuning_summary_df(
        str(strongest_tree_name),
        baseline_df,
        tuned_evaluation["comparison_row"],
        best_params,
    )
    final_artifacts["best_params"] = best_params
    final_artifacts["strongest_tree_name"] = str(strongest_tree_name)
    final_artifacts["final_model_reasoning"] = explain_final_model_choice(
        final_model_name,
        experiment_df,
        str(strongest_tree_name),
    )
    final_artifacts["preprocessing_summary_df"] = get_preprocessing_summary()

    return experiment_df, final_model_name, final_pipeline, final_artifacts


def get_feature_importance(best_pipeline: Pipeline) -> pd.DataFrame:
    """Return feature importance for tree models or coefficient magnitude for logistic regression."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef_values = np.asarray(model.coef_)
        importance_values = np.mean(np.abs(coef_values), axis=0)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance_values})
        .sort_values("Importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    importance_df["Clean Feature"] = importance_df["Feature"].apply(clean_feature_name)
    return importance_df[["Clean Feature", "Importance"]]


def summarize_feature_importance(importance_df: pd.DataFrame) -> str:
    """Summarize feature importance in plain language."""
    if importance_df.empty:
        return "Feature importance is unavailable for the selected model."

    top_feature = importance_df.iloc[0]
    top_share = float(top_feature["Importance"])
    second_share = float(importance_df.iloc[1]["Importance"]) if len(importance_df) > 1 else 0.0

    if top_share >= second_share * 1.8 and second_share > 0:
        dominance_text = "The model relies quite heavily on its top feature."
    else:
        dominance_text = "The model is not driven by just one variable; several features contribute meaningfully."

    return (
        f"The strongest feature is {top_feature['Clean Feature']} with importance {top_share:.3f}. "
        f"{dominance_text} Secondary signals such as {', '.join(importance_df['Clean Feature'].head(4).tolist()[1:4])} still add useful context to the final recommendation."
    )


def predict_with_pipeline(best_pipeline: Pipeline, input_df: pd.DataFrame) -> tuple[str, float | None, pd.DataFrame]:
    """Predict a class label and optional class probabilities for new rows."""
    prediction = best_pipeline.predict(input_df)[0]

    probability_df = pd.DataFrame(columns=["Class", "Probability"])
    confidence = None

    if hasattr(best_pipeline, "predict_proba"):
        probabilities = best_pipeline.predict_proba(input_df)[0]
        classes = best_pipeline.classes_
        probability_df = (
            pd.DataFrame({"Class": classes, "Probability": probabilities})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        confidence = float(probability_df.loc[0, "Probability"])

    return str(prediction), confidence, probability_df


def explain_prediction(
    input_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    predicted_label: str,
) -> str:
    """Create a light heuristic explanation for the prediction result."""
    if input_df.empty or importance_df.empty:
        return f"The model predicts {predicted_label} based on the overall feature pattern."

    row = input_df.iloc[0].to_dict()
    top_features = importance_df["Clean Feature"].head(3).tolist()
    details: list[str] = []

    for feature in top_features:
        if feature == "Feels Like (C)" and "FeelsLike_C" in row:
            details.append(f"feels-like temperature of {float(row['FeelsLike_C']):.1f}C")
        elif feature == "Humidity (%)" and "Humidity_%" in row:
            details.append(f"humidity of {float(row['Humidity_%']):.0f}%")
        elif feature == "Wind Speed (km/h)" and "WindSpeed_kmh" in row:
            details.append(f"wind speed of {float(row['WindSpeed_kmh']):.1f} km/h")
        elif feature == "Hour" and "Hour" in row:
            details.append(f"time-of-day signal around hour {int(row['Hour'])}")
        elif feature.startswith("Source:") and "SourceWebsite" in row:
            details.append(f"source website {row['SourceWebsite']}")
        elif feature.startswith("City:") and "City" in row:
            details.append(f"city context for {row['City']}")

    if not details:
        return f"The model predicts {predicted_label} from the combined weather pattern across the available inputs."

    return f"The model predicts {predicted_label} mainly because of the {', '.join(details)}."


def main() -> None:
    df = load_data()
    X, y, _, class_distribution_df = prepare_features(df)

    if len(X) < MIN_TRAINING_ROWS or y.nunique() < 2:
        print("Not enough valid rows or classes after preprocessing. Add more data and try again.")
        return

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_class_count < 2:
        print("A class has fewer than 2 rows, so stratified classification training is not safe yet.")
        return

    # ⚡ speed control
    if len(X) > 2000:
        X = X.sample(2000, random_state=42)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    experiment_df, best_model_name, best_pipeline, best_artifacts = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        class_distribution_df=class_distribution_df,
    )

    print("\nPreprocessing Summary:")
    print(best_artifacts["preprocessing_summary_df"].to_string(index=False))

    print("\nClass Distribution:")
    print(class_distribution_df.to_string(index=False))

    print("\nExperiment Flow:")
    print(experiment_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print(f"\nFinal Model: {best_model_name}")
    print(best_artifacts["final_model_reasoning"])

    print("\nClassification Report:")
    print(best_artifacts["classification_report_df"].to_string(index=False))

    print("\nConfusion Matrix:")
    print(best_artifacts["confusion_matrix_df"].to_string())

    print("\nConfusion Analysis:")
    print(best_artifacts["confusion_summary"])

    print("\nBias Check:")
    print(best_artifacts["bias_summary"])

    importance_df = get_feature_importance(best_pipeline)
    if not importance_df.empty:
        print("\nTop Feature Importance:")
        print(importance_df.to_string(index=False))
        print("\nFeature Interpretation:")
        print(summarize_feature_importance(importance_df))


if __name__ == "__main__":
    main()