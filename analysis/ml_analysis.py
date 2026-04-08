from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = "data/processed/weather_data.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load weather dataset from CSV."""
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and prepare feature matrix X and target y."""
    data = df.copy()

    # Convert datetime safely and create time-based features.
    try:
        data["ScrapeDateTime"] = pd.to_datetime(
            data["ScrapeDateTime"], errors="coerce", utc=True, format="mixed"
        )
    except TypeError:
        data["ScrapeDateTime"] = pd.to_datetime(
            data["ScrapeDateTime"], errors="coerce", utc=True
        )

    data["Hour"] = data["ScrapeDateTime"].dt.hour
    data["Day"] = data["ScrapeDateTime"].dt.day
    data["Month"] = data["ScrapeDateTime"].dt.month

    # Convert numeric columns safely.
    numeric_cols = ["Temperature_C", "FeelsLike_C", "Humidity_%", "WindSpeed_kmh"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows with missing target as requested.
    data = data.dropna(subset=["Temperature_C"])

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

    # Keep only rows where these columns exist, then split X/y.
    data = data[feature_cols + ["Temperature_C"]].copy()
    y = data["Temperature_C"]
    X = data[feature_cols]
    return X, y


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, str, Pipeline]:
    """Train multiple models and return comparison table + best model."""
    categorical_features = ["SourceWebsite", "City", "Country"]
    numeric_features = ["FeelsLike_C", "Humidity_%", "WindSpeed_kmh", "Hour", "Day", "Month"]

    preprocessor = ColumnTransformer(
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
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results: list[dict[str, float | str]] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append(
            {
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )
        fitted_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
    best_model_name = str(results_df.loc[0, "Model"])
    best_pipeline = fitted_pipelines[best_model_name]
    return results_df, best_model_name, best_pipeline


def print_feature_importance_if_available(best_model_name: str, best_pipeline: Pipeline) -> None:
    """Print top random forest feature importances when available."""
    if best_model_name != "Random Forest":
        return

    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    print("\nTop 10 Feature Importances (Random Forest):")
    print(importance_df.to_string(index=False))


def main() -> None:
    df = load_data()
    X, y = prepare_features(df)

    if len(X) < 10:
        print("Not enough rows after preprocessing. Add more data and try again.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results_df, best_model_name, best_pipeline = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nBest Model (lowest RMSE): {best_model_name}")

    print_feature_importance_if_available(best_model_name, best_pipeline)


if __name__ == "__main__":
    main()
