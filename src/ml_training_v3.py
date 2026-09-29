import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


DATA_PATH = "data/processed/ml_match_dataset_advanced.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "output/ml"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    return df.sort_values("date").reset_index(drop=True)


def build_features(df):
    data = df.copy()

    # Historical difference features.
    data["career_win_rate_diff"] = (
        data["team_a_win_rate"]
        - data["team_b_win_rate"]
    )

    data["recent_win_rate_diff"] = (
        data["team_a_recent_win_rate"]
        - data["team_b_recent_win_rate"]
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

    # Lower bowling economy is better, so reverse the difference.
    data["economy_advantage"] = (
        data["team_b_bowling_economy"]
        - data["team_a_bowling_economy"]
    )

    data["recent_wins_diff"] = (
        data["team_a_recent_wins"]
        - data["team_b_recent_wins"]
    )

    data["h2h_wins_diff"] = (
        data["team_a_h2h_wins"]
        - data["team_b_h2h_wins"]
    )

    data["h2h_win_rate"] = data["team_a_h2h_win_rate"]

    data["experience_diff"] = (
        data["team_a_matches"]
        - data["team_b_matches"]
    )

    # Venue information known before the match.
    data["venue_chasing_win_rate"] = (
        data["venue_chasing_win_rate"]
    )

    data["venue_avg_first_innings"] = (
        data["venue_avg_first_innings"]
    )

    # Toss information.
    data["toss_team_a_won"] = data["toss_team_a_won"]
    data["toss_decision_field"] = data["toss_decision_field"]
    data["toss_decision_bat"] = data["toss_decision_bat"]

    features = [
        "career_win_rate_diff",
        "recent_win_rate_diff",
        "runs_per_match_diff",
        "runs_conceded_diff",
        "wickets_per_match_diff",
        "economy_advantage",
        "recent_wins_diff",
        "h2h_wins_diff",
        "h2h_win_rate",
        "experience_diff",
        "h2h_matches_before",
        "venue_matches_before",
        "venue_chasing_win_rate",
        "venue_avg_first_innings",
        "toss_team_a_won",
        "toss_decision_field",
        "toss_decision_bat",
    ]

    X = data[features].copy()
    y = data["team_a_won"].astype(int)

    return data, X, y, features


def main():

    print("=" * 70)
    print("IPL ML MATCH PREDICTION - V3 NUMERIC SIGNAL MODEL")
    print("=" * 70)

    df = load_data()

    print()
    print("Total matches:", len(df))
    print(
        "Date range:",
        df["date"].min().date(),
        "to",
        df["date"].max().date(),
    )

    data, X, y, features = build_features(df)

    split_index = int(len(data) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print()
    print("Features:", len(features))

    for i, feature in enumerate(features, start=1):
        print(f"{i:02d}. {feature}")

    print()
    print("Training matches:", len(X_train))
    print("Testing matches:", len(X_test))

    print()
    print("Training period:")
    print(data.iloc[:split_index]["date"].min().date())
    print("to")
    print(data.iloc[:split_index]["date"].max().date())

    print()
    print("Testing period:")
    print(data.iloc[split_index:]["date"].min().date())
    print("to")
    print(data.iloc[split_index:]["date"].max().date())

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    C=0.5,
                    random_state=42,
                ),
            ),
        ]
    )

    print()
    print("Training V3 Logistic Regression...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print()
    print("=" * 70)
    print("V3 RESULTS")
    print("=" * 70)

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
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    model_path = os.path.join(
        MODEL_DIR,
        "logistic_regression_v3.joblib",
    )

    joblib.dump(model, model_path)

    prediction_output = data.iloc[split_index:][
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
    prediction_output["prediction"] = predictions
    prediction_output["team_a_win_probability"] = probabilities

    prediction_path = os.path.join(
        OUTPUT_DIR,
        "test_predictions_v3.csv",
    )

    prediction_output.to_csv(
        prediction_path,
        index=False,
    )

    print()
    print("Saved model:", model_path)
    print("Saved predictions:", prediction_path)

    print()
    print("=" * 70)
    print("V3 TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()