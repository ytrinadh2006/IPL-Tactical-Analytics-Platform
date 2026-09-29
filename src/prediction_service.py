from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "processed" / "ml_match_dataset_advanced.csv"
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_v2_v2.joblib"


class PredictionService:
    """Load the trained IPL model and provide match predictions."""

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.dataset = pd.read_csv(DATASET_PATH)

        self.dataset["date"] = pd.to_datetime(
            self.dataset["date"],
            errors="coerce"
        )

        self.dataset = self.dataset.sort_values("date").reset_index(drop=True)

        self.expected_features = list(
            getattr(self.model, "feature_names_in_", [])
        )

        if not self.expected_features:
            raise ValueError(
                "The trained model does not contain feature names."
            )

    def _find_reference_match(self, team_a, team_b, venue=None):
        """
        Find the most recent historical match that can provide
        pre-match feature values for the requested teams.
        """

        team_a = str(team_a).strip()
        team_b = str(team_b).strip()

        matches = self.dataset[
            (self.dataset["team_a"] == team_a)
            & (self.dataset["team_b"] == team_b)
        ].copy()

        # Try the requested venue first.
        if venue:
            venue_matches = matches[
                matches["venue"].str.lower() == str(venue).strip().lower()
            ]

            if not venue_matches.empty:
                matches = venue_matches

        # If the exact order does not exist, try reversed teams.
        reversed_match = False

        if matches.empty:
            matches = self.dataset[
                (self.dataset["team_a"] == team_b)
                & (self.dataset["team_b"] == team_a)
            ].copy()

            reversed_match = True

        if matches.empty:
            raise ValueError(
                f"No historical match context found for "
                f"{team_a} vs {team_b}."
            )

        row = matches.sort_values("date").iloc[-1].copy()

        # If we found a reversed historical matchup,
        # swap the team-specific values.
        if reversed_match:
            row = self._swap_team_features(row)

        return row

    @staticmethod
    def _swap_team_features(row):
        """Swap Team A and Team B feature values."""

        row = row.copy()

        columns = list(row.index)

        for column in columns:
            if column.startswith("team_a_"):
                team_b_column = column.replace(
                    "team_a_", "team_b_", 1
                )

                if team_b_column in row.index:
                    value = row[column]
                    row[column] = row[team_b_column]
                    row[team_b_column] = value

        # Swap difference features.
        difference_columns = [
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

        for column in difference_columns:
            if column in row.index:
                row[column] = -row[column]

        return row

    def _prepare_input(self, row, team_a, team_b, venue=None, season=None):
        """Prepare one prediction row using the trained model schema."""

        row = row.copy()

        row["team_a"] = team_a
        row["team_b"] = team_b

        if venue:
            row["venue"] = venue

        if season:
            row["season"] = season

        # Make sure engineered difference features exist.
        differences = {
            "recent_win_rate_diff": (
                "team_a_recent_win_rate",
                "team_b_recent_win_rate",
            ),
            "h2h_win_diff": (
                "team_a_h2h_wins",
                "team_b_h2h_wins",
            ),
            "runs_per_match_diff": (
                "team_a_runs_per_match",
                "team_b_runs_per_match",
            ),
            "runs_conceded_diff": (
                "team_a_runs_conceded_per_match",
                "team_b_runs_conceded_per_match",
            ),
            "wickets_per_match_diff": (
                "team_a_wickets_per_match",
                "team_b_wickets_per_match",
            ),
            "economy_diff": (
                "team_a_bowling_economy",
                "team_b_bowling_economy",
            ),
            "career_win_rate_diff": (
                "team_a_win_rate",
                "team_b_win_rate",
            ),
            "experience_diff": (
                "team_a_matches",
                "team_b_matches",
            ),
            "recent_wins_diff": (
                "team_a_recent_wins",
                "team_b_recent_wins",
            ),
        }

        for new_column, (column_a, column_b) in differences.items():
            if new_column not in row.index:
                row[new_column] = (
                    float(row[column_a])
                    - float(row[column_b])
                )

        # The trained pipeline expects exactly these columns.
        missing = [
            column
            for column in self.expected_features
            if column not in row.index
        ]

        if missing:
            raise ValueError(
                f"Missing model features: {missing}"
            )

        X = pd.DataFrame(
            [[row[column] for column in self.expected_features]],
            columns=self.expected_features,
        )

        return X

    def predict(
        self,
        team_a,
        team_b,
        venue=None,
        season=None,
    ):
        """Return a probability-based match prediction."""

        reference_row = self._find_reference_match(
            team_a,
            team_b,
            venue,
        )

        X = self._prepare_input(
            reference_row,
            team_a,
            team_b,
            venue,
            season,
        )

        prediction = int(self.model.predict(X)[0])

        probabilities = self.model.predict_proba(X)[0]

        team_a_probability = float(probabilities[1])
        team_b_probability = float(probabilities[0])

        predicted_team = (
            team_a if prediction == 1 else team_b
        )

        return {
            "team_a": team_a,
            "team_b": team_b,
            "venue": (
                venue
                if venue
                else str(reference_row["venue"])
            ),
            "reference_match_date": str(
                reference_row["date"].date()
            ),
            "predicted_team": predicted_team,
            "team_a_probability": round(
                team_a_probability, 4
            ),
            "team_b_probability": round(
                team_b_probability, 4
            ),
            "model": "Logistic Regression V2",
        }


if __name__ == "__main__":

    print("=" * 70)
    print("IPL PREDICTION SERVICE")
    print("=" * 70)

    service = PredictionService()

    print("\nModel loaded successfully.")
    print("Expected features:", len(service.expected_features))

    result = service.predict(
        team_a="Sunrisers Hyderabad",
        team_b="Kolkata Knight Riders",
        venue="MA Chidambaram Stadium, Chepauk, Chennai",
    )

    print("\nSAMPLE PREDICTION")
    print("-" * 70)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 70)
    print("PREDICTION SERVICE COMPLETE")
    print("=" * 70)