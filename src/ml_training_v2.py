import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


DATA_PATH = "data/processed/ml_match_dataset_advanced.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "output/ml"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def prepare_features(df):
    """
    Keep only information that can reasonably be known
    before the match.

    Difference features make the model easier to interpret:
    positive value = Team A has the larger historical value.
    """

    data = df.copy()

    data["recent_win_rate_diff"] = (
        data["team_a_recent_win_rate"]
        - data["team_b_recent_win_rate"]
    )

    data["h2h_win_diff"] = (
        data["team_a_h2h_wins"]
        - data["team_b_h2h_wins"]
    )

    data["runs_per_match_diff"] = (
        data["team_a_runs_per_match"]
        - data["team_b_runs_per_match"]
    )

    data["runs_conceded_diff"] = (
        data["team_a_runs_conceded_per_match"]
        - data["team_b_runs_conceded_per_match"]
    )

    data["wickets_per_match_diff"] = (
        data["team_a_wickets_per_match"]
        - data["team_b_wickets_per_match"]
    )

    data["economy_diff"] = (
        data["team_a_bowling_economy"]
        - data["team_b_bowling_economy"]
    )

    data["career_win_rate_diff"] = (
        data["team_a_win_rate"]
        - data["team_b_win_rate"]
    )

    data["experience_diff"] = (
        data["team_a_matches"]
        - data["team_b_matches"]
    )

    data["recent_wins_diff"] = (
        data["team_a_recent_wins"]
        - data["team_b_recent_wins"]
    )

    # Target
    y = data["team_a_won"].astype(int)

    # Numeric historical/context features
    numeric_features = [
        

        "team_a_matches",
        "team_b_matches",

        "team_a_win_rate",
        "team_b_win_rate",

        "team_a_runs_per_match",
        "team_b_runs_per_match",

        "team_a_runs_conceded_per_match",
        "team_b_runs_conceded_per_match",

        "team_a_wickets_per_match",
        "team_b_wickets_per_match",

        "team_a_bowling_economy",
        "team_b_bowling_economy",

        "team_a_recent_matches",
        "team_b_recent_matches",

        "team_a_recent_wins",
        "team_b_recent_wins",

        "team_a_recent_win_rate",
        "team_b_recent_win_rate",

        "h2h_matches_before",
        "team_a_h2h_wins",
        "team_b_h2h_wins",
        "team_a_h2h_win_rate",

        "venue_matches_before",
        "venue_avg_first_innings",
        "venue_chasing_win_rate",

        "toss_team_a_won",
        "toss_team_b_won",

        "toss_decision_field",
        "toss_decision_bat",

        # Engineered differences
        "recent_win_rate_diff",
        "h2h_win_diff",
        "runs_per_match_diff",
        "runs_conceded_diff",
        "wickets_per_match_diff",
        "economy_diff",
        "career_win_rate_diff",
        "experience_diff",
        "recent_wins_diff",
    ]

    categorical_features = [
        "season",
        "team_a",
        "team_b",
        "venue",
    ]

    features = numeric_features + categorical_features

    X = data[features].copy()

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = None

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    if probabilities is not None:
        roc_auc = roc_auc_score(y_test, probabilities)
    else:
        roc_auc = 0.0

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    model_path = os.path.join(
        MODEL_DIR,
        name.lower().replace(" ", "_") + "_v2.joblib"
    )

    joblib.dump(model, model_path)

    print()
    print(f"Saved model: {model_path}")

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def main():
    print("=" * 70)
    print("IPL ML MATCH PREDICTION - V2")
    print("=" * 70)

    df = load_data()

    print()
    print(f"Total matches: {len(df)}")
    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    X, y, numeric_features, categorical_features = prepare_features(df)

    print()
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Total features before encoding: {X.shape[1]}")

    # Chronological 80/20 split
    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    print()
    print(f"Training matches: {len(X_train)}")
    print(f"Testing matches: {len(X_test)}")

    print()
    print("Training period:")
    print(df.iloc[:split_index]["date"].min().date())
    print("to")
    print(df.iloc[:split_index]["date"].max().date())

    print()
    print("Testing period:")
    print(df.iloc[split_index:]["date"].min().date())
    print("to")
    print(df.iloc[split_index:]["date"].max().date())

    print()
    print("Training target distribution:")
    print(y_train.value_counts().sort_index())

    print()
    print("Testing target distribution:")
    print(y_test.value_counts().sort_index())

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    # ---------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------

    print()
    print("Training Logistic Regression V2...")

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=0.5,
                    random_state=42
                )
            ),
        ]
    )

    logistic_result = evaluate_model(
        "Logistic Regression V2",
        logistic_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    print()
    print("Training Random Forest V2...")

    rf_preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", rf_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            ),
        ]
    )

    rf_result = evaluate_model(
        "Random Forest V2",
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------

    print()
    print("Training XGBoost V2...")

    xgb_preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    xgb_model = Pipeline(
        steps=[
            ("preprocessor", xgb_preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=4
                )
            ),
        ]
    )

    xgb_result = evaluate_model(
        "XGBoost V2",
        xgb_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    results = [
        logistic_result,
        rf_result,
        xgb_result,
    ]

    comparison = pd.DataFrame(
        [
            {
                "model": result["model"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
            }
            for result in results
        ]
    )

    comparison = comparison.sort_values(
        "roc_auc",
        ascending=False
    )

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "model_comparison_v2.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    # Save test predictions from every model
    prediction_output = df.iloc[split_index:][
        [
            "match_id",
            "date",
            "season",
            "team_a",
            "team_b",
            "venue",
        ]
    ].copy()

    prediction_output["actual_team_a_won"] = y_test.values

    prediction_output["logistic_probability"] = (
        logistic_result["probabilities"]
    )

    prediction_output["rf_probability"] = (
        rf_result["probabilities"]
    )

    prediction_output["xgb_probability"] = (
        xgb_result["probabilities"]
    )

    prediction_output["logistic_prediction"] = (
        logistic_result["predictions"]
    )

    prediction_output["rf_prediction"] = (
        rf_result["predictions"]
    )

    prediction_output["xgb_prediction"] = (
        xgb_result["predictions"]
    )

    predictions_path = os.path.join(
        OUTPUT_DIR,
        "test_predictions_v2.csv"
    )

    prediction_output.to_csv(
        predictions_path,
        index=False
    )

    print()
    print("=" * 70)
    print("V2 MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved predictions: {predictions_path}")

    print()
    print("=" * 70)
    print("ML V2 TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()