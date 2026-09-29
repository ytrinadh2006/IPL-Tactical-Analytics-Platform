from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ml_match_dataset_advanced.csv"
)

MODEL_DIR = (
    ROOT
    / "models"
)

OUTPUT_DIR = (
    ROOT
    / "output"
    / "ml"
)


# ---------------------------------------------------------
# Create output folders
# ---------------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_dataset():
    """Load the validated advanced ML dataset."""

    df = pd.read_csv(
        DATA_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(
        drop=True
    )

    return df


# ---------------------------------------------------------
# Prepare features
# ---------------------------------------------------------

def prepare_features(df):
    """
    Separate ML features from the target.

    This is a post-toss prediction model because toss
    information is available before the match begins.
    """

    target = "team_a_won"

    # Columns that identify a match but should not be
    # directly used as model features.
    drop_columns = [
        target,
        "match_id",
        "date",
        "toss_winner",
        "toss_decision",
    ]

    X = df.drop(
        columns=[
            column
            for column in drop_columns
            if column in df.columns
        ]
    )

    y = df[target].astype(int)

    return X, y


# ---------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------

def chronological_split(
    X,
    y,
    df,
    train_ratio=0.80,
):
    """
    Split historical matches chronologically.

    Earlier matches -> training
    Later matches   -> testing
    """

    split_index = int(
        len(df) * train_ratio
    )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        split_index,
    )


# ---------------------------------------------------------
# Build preprocessing pipeline
# ---------------------------------------------------------

def build_preprocessor(
    X_train,
):
    """Create preprocessing for numerical and categorical data."""

    categorical_columns = (
        X_train
        .select_dtypes(
            include=["object"]
        )
        .columns
        .tolist()
    )

    numerical_columns = (
        X_train
        .select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# Build models
# ---------------------------------------------------------

def build_models():

    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
    }

    if XGBOOST_AVAILABLE:

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    return models


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate_model(
    name,
    model,
    X_test,
    y_test,
):
    """Calculate classification metrics."""

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

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

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        f"\n{'=' * 70}"
    )

    print(name)

    print(
        f"{'=' * 70}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(matrix)

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "predictions": predictions,
        "probabilities": probabilities,
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------

def save_confusion_matrix(
    name,
    matrix,
):
    """Save a confusion matrix image."""

    safe_name = (
        name
        .lower()
        .replace(" ", "_")
    )

    plt.figure(
        figsize=(5, 4)
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
    )

    plt.title(
        f"{name} - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        ["Team B", "Team A"],
    )

    plt.yticks(
        [0, 1],
        ["Team B", "Team A"],
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{safe_name}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def train_models():

    print(
        "=" * 70
    )

    print(
        "IPL ML MATCH PREDICTION"
    )

    print(
        "=" * 70
    )

    if not XGBOOST_AVAILABLE:

        print(
            "\nWARNING: XGBoost is not installed."
        )

        print(
            "Logistic Regression and Random Forest "
            "will still be trained."
        )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    df = load_dataset()

    print(
        f"\nTotal matches: {len(df)}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    # -----------------------------------------------------
    # Features and target
    # -----------------------------------------------------

    X, y = prepare_features(
        df
    )

    print(
        f"Total ML features before encoding: "
        f"{X.shape[1]}"
    )

    # -----------------------------------------------------
    # Chronological split
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        split_index,
    ) = chronological_split(
        X,
        y,
        df,
    )

    print(
        f"\nTraining matches: "
        f"{len(X_train)}"
    )

    print(
        f"Testing matches: "
        f"{len(X_test)}"
    )

    print(
        "\nTraining period:"
    )

    print(
        f"{df.iloc[0]['date'].date()} "
        f"to "
        f"{df.iloc[split_index - 1]['date'].date()}"
    )

    print(
        "\nTesting period:"
    )

    print(
        f"{df.iloc[split_index]['date'].date()} "
        f"to "
        f"{df.iloc[-1]['date'].date()}"
    )

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train.value_counts()
        .sort_index()
    )

    print(
        "\nTesting target distribution:"
    )

    print(
        y_test.value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    preprocessor = build_preprocessor(
        X_train
    )

    models = build_models()

    results = []

    trained_models = {}

    # -----------------------------------------------------
    # Train each model
    # -----------------------------------------------------

    for name, classifier in models.items():

        print(
            f"\nTraining {name}..."
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        result = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        results.append(
            {
                "model": name,
                "accuracy":
                    result["accuracy"],
                "precision":
                    result["precision"],
                "recall":
                    result["recall"],
                "f1":
                    result["f1"],
                "roc_auc":
                    result["roc_auc"],
            }
        )

        trained_models[
            name
        ] = pipeline

        save_confusion_matrix(
            name,
            result["confusion_matrix"],
        )

        safe_name = (
            name
            .lower()
            .replace(" ", "_")
        )

        model_path = (
            MODEL_DIR
            / f"{safe_name}.joblib"
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        print(
            f"Saved model: {model_path}"
        )

    # -----------------------------------------------------
    # Model comparison
    # -----------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        "roc_auc",
        ascending=False,
    )

    comparison_path = (
        OUTPUT_DIR
        / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\nSaved comparison to:"
        f"\n{comparison_path}"
    )

    # -----------------------------------------------------
    # Save test predictions
    # -----------------------------------------------------

    prediction_output = df.iloc[
        split_index:
    ][
        [
            "match_id",
            "date",
            "team_a",
            "team_b",
            "venue",
            "team_a_won",
        ]
    ].copy()

    for name, pipeline in trained_models.items():

        safe_name = (
            name
            .lower()
            .replace(" ", "_")
        )

        probability = pipeline.predict_proba(
            X_test
        )[:, 1]

        prediction_output[
            f"{safe_name}_team_a_probability"
        ] = probability

        prediction_output[
            f"{safe_name}_prediction"
        ] = (
            probability >= 0.5
        ).astype(int)

    prediction_path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    prediction_output.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"\nSaved test predictions to:"
        f"\n{prediction_path}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ML TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    return comparison


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":

    train_models()