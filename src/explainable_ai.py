from pathlib import Path

import joblib
import pandas as pd
import shap


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_match_dataset_advanced.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "logistic_regression_v2_v2.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "explainability"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "team_a_won"


def load_model_and_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset rows: {len(df)}")
    print(f"Dataset columns: {len(df.columns)}")

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")
    print(f"Model type: {type(model)}")

    return df, model

def prepare_features(df, model):
    """
    Prepare the dataset using exactly the feature names
    expected by the trained ML pipeline.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Create engineered difference features used during V2
    # training.
    # ---------------------------------------------------------

    difference_features = {
        "recent_win_rate_diff": (
            "team_a_recent_win_rate",
            "team_b_recent_win_rate"
        ),
        "h2h_win_diff": (
            "team_a_h2h_wins",
            "team_b_h2h_wins"
        ),
        "runs_per_match_diff": (
            "team_a_runs_per_match",
            "team_b_runs_per_match"
        ),
        "runs_conceded_diff": (
            "team_a_runs_conceded_per_match",
            "team_b_runs_conceded_per_match"
        ),
        "wickets_per_match_diff": (
            "team_a_wickets_per_match",
            "team_b_wickets_per_match"
        ),
        "economy_diff": (
            "team_a_bowling_economy",
            "team_b_bowling_economy"
        ),
        "career_win_rate_diff": (
            "team_a_win_rate",
            "team_b_win_rate"
        ),
        "experience_diff": (
            "team_a_matches",
            "team_b_matches"
        ),
        "recent_wins_diff": (
            "team_a_recent_wins",
            "team_b_recent_wins"
        ),
    }

    for new_column, (team_a_column, team_b_column) in difference_features.items():
        if new_column not in df.columns:
            df[new_column] = (
                pd.to_numeric(df[team_a_column], errors="coerce").fillna(0)
                - pd.to_numeric(df[team_b_column], errors="coerce").fillna(0)
            )

    # ---------------------------------------------------------
    # Get the exact feature schema stored in the trained model.
    # ---------------------------------------------------------

    expected_features = list(
        getattr(model, "feature_names_in_", [])
    )

    if not expected_features:
        raise ValueError(
            "Could not read feature names from the trained model."
        )

    # ---------------------------------------------------------
    # Check that every required feature now exists.
    # ---------------------------------------------------------

    missing_features = [
        feature
        for feature in expected_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Dataset is still missing model features: {missing_features}"
        )

    # ---------------------------------------------------------
    # Build X using exactly the same column order as training.
    # ---------------------------------------------------------

    X = df[expected_features].copy()

    # Target is kept separately.
    y = None

    if "team_a_won" in df.columns:
        y = df["team_a_won"].copy()

    return X, y, expected_features

   
def explain():
    print("=" * 70)
    print("IPL EXPLAINABLE AI")
    print("=" * 70)

    df, model = load_model_and_data()

    X, y, expected_features = prepare_features(
        df,
        model
    )

    # ---------------------------------------------------------
    # Split the saved pipeline into preprocessing + estimator
    # ---------------------------------------------------------

    preprocessing = model[:-1]
    estimator = model[-1]

    print("\nTransforming features...")

    X_transformed = preprocessing.transform(X)

    # SHAP works more reliably with a dense matrix here.
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    try:
        transformed_names = (
            preprocessing.get_feature_names_out()
        )
    except Exception:
        transformed_names = [
            f"feature_{i}"
            for i in range(X_transformed.shape[1])
        ]

    print(
        f"Transformed feature count: "
        f"{X_transformed.shape[1]}"
    )

    # ---------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------

    print("\nCreating SHAP LinearExplainer...")

    explainer = shap.LinearExplainer(
        estimator,
        X_transformed
    )

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X_transformed
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = pd.DataFrame(
        shap_values,
        columns=transformed_names
    )

    print("SHAP calculation complete.")

    # ---------------------------------------------------------
    # Global importance
    # ---------------------------------------------------------

    importance = (
        shap_values.abs()
        .mean()
        .sort_values(ascending=False)
    )

    importance_df = pd.DataFrame({
        "feature": importance.index,
        "mean_absolute_shap": importance.values
    })

    importance_path = (
        OUTPUT_DIR
        / "global_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("TOP 15 SHAP FEATURES")
    print("=" * 70)

    print(
        importance_df
        .head(15)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Direction of influence
    # ---------------------------------------------------------

    direction = (
        shap_values
        .mean()
        .sort_values(ascending=False)
    )

    direction_df = pd.DataFrame({
        "feature": direction.index,
        "mean_shap": direction.values
    })

    direction_path = (
        OUTPUT_DIR
        / "feature_direction.csv"
    )

    direction_df.to_csv(
        direction_path,
        index=False
    )

    # ---------------------------------------------------------
    # Explain latest historical match
    # ---------------------------------------------------------

    sample_index = len(df) - 1

    sample = X.iloc[[sample_index]]

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(
        sample
    )[0]

    sample_shap = shap_values.iloc[
        sample_index
    ]

    sample_explanation = pd.DataFrame({
        "feature": sample_shap.index,
        "shap_value": sample_shap.values,
    })

    sample_explanation[
        "absolute_shap"
    ] = sample_explanation[
        "shap_value"
    ].abs()

    sample_explanation = (
        sample_explanation
        .sort_values(
            "absolute_shap",
            ascending=False
        )
    )

    sample_path = (
        OUTPUT_DIR
        / "sample_match_explanation.csv"
    )

    sample_explanation.to_csv(
        sample_path,
        index=False
    )

    # ---------------------------------------------------------
    # Console explanation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE MATCH EXPLANATION")
    print("=" * 70)

    print(
        f"\nDate: {df.iloc[sample_index]['date']}"
    )

    print(
        f"Team A: {df.iloc[sample_index]['team_a']}"
    )

    print(
        f"Team B: {df.iloc[sample_index]['team_b']}"
    )

    print(
        f"Venue: {df.iloc[sample_index]['venue']}"
    )

    print(
        f"\nActual result: "
        f"{'Team A won' if y.iloc[sample_index] == 1 else 'Team B won'}"
    )

    print(
        f"Model prediction: "
        f"{'Team A' if prediction == 1 else 'Team B'}"
    )

    print(
        f"Team A probability: {probabilities[1]:.4f}"
    )

    print(
        f"Team B probability: {probabilities[0]:.4f}"
    )

    print("\nTop factors increasing Team-A prediction:")
    print("-" * 55)

    positive = (
        sample_explanation[
            sample_explanation["shap_value"] > 0
        ]
        .head(10)
    )

    for _, row in positive.iterrows():
        print(
            f"{row['feature']}: "
            f"{row['shap_value']:+.5f}"
        )

    print("\nTop factors decreasing Team-A prediction:")
    print("-" * 55)

    negative = (
        sample_explanation[
            sample_explanation["shap_value"] < 0
        ]
        .sort_values(
            "shap_value"
        )
        .head(10)
    )

    for _, row in negative.iterrows():
        print(
            f"{row['feature']}: "
            f"{row['shap_value']:+.5f}"
        )

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        importance_path
    )

    print(
        direction_path
    )

    print(
        sample_path
    )

    print("\n" + "=" * 70)
    print("EXPLAINABLE AI COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    explain()