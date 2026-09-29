from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------
# Team-name normalization
# ---------------------------------------------------------

TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Rising Pune Supergiants": "Rising Pune Supergiants",
}


# ---------------------------------------------------------
# Load match data
# ---------------------------------------------------------

def load_matches():
    """Load and clean historical IPL match data."""

    matches = pd.read_csv(
        RAW / "matches.csv"
    )

    for column in [
        "team1",
        "team2",
        "winner",
        "toss_winner",
    ]:
        if column in matches.columns:
            matches[column] = (
                matches[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace(TEAM_NAME_MAP)
            )

    matches["date"] = pd.to_datetime(
        matches["date"],
        errors="coerce",
    )

    matches = matches.sort_values(
        ["date", "id"]
    ).reset_index(drop=True)

    return matches


# ---------------------------------------------------------
# Historical team state
# ---------------------------------------------------------

def empty_team_record():
    """Create an empty record for a team."""

    return {
        "matches": 0,
        "wins": 0,
        "runs": 0,
        "runs_conceded": 0,
        "wickets": 0,
        "balls_bowled": 0,
    }


def calculate_team_features(
    record,
    prefix,
):
    """
    Convert historical team information into
    pre-match numerical features.
    """

    matches = record["matches"]

    if matches == 0:
        return {
            f"{prefix}_matches": 0,
            f"{prefix}_win_rate": 0.5,
            f"{prefix}_runs_per_match": 0.0,
            f"{prefix}_runs_conceded_per_match": 0.0,
            f"{prefix}_wickets_per_match": 0.0,
            f"{prefix}_bowling_economy": 0.0,
        }

    balls = record["balls_bowled"]

    economy = (
        record["runs_conceded"]
        / (balls / 6)
        if balls > 0
        else 0.0
    )

    return {
        f"{prefix}_matches": matches,

        f"{prefix}_win_rate": (
            record["wins"] / matches
        ),

        f"{prefix}_runs_per_match": (
            record["runs"] / matches
        ),

        f"{prefix}_runs_conceded_per_match": (
            record["runs_conceded"] / matches
        ),

        f"{prefix}_wickets_per_match": (
            record["wickets"] / matches
        ),

        f"{prefix}_bowling_economy": economy,
    }


# ---------------------------------------------------------
# Match delivery statistics
# ---------------------------------------------------------

def load_deliveries():
    """Load the delivery-level IPL dataset."""

    deliveries = pd.read_csv(
        RAW / "deliveries.csv"
    )

    deliveries["total_runs"] = pd.to_numeric(
        deliveries["total_runs"],
        errors="coerce",
    ).fillna(0)

    deliveries["batsman_runs"] = pd.to_numeric(
        deliveries["batsman_runs"],
        errors="coerce",
    ).fillna(0)

    deliveries["is_wicket"] = pd.to_numeric(
        deliveries["is_wicket"],
        errors="coerce",
    ).fillna(0)

    for column in [
        "batting_team",
        "bowling_team",
    ]:
        deliveries[column] = (
            deliveries[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace(TEAM_NAME_MAP)
        )

    return deliveries


# ---------------------------------------------------------
# Calculate one team's performance in one match
# ---------------------------------------------------------

def calculate_match_team_stats(
    match_deliveries,
    team,
):
    """
    Calculate batting and bowling statistics for one team
    in one completed match.
    """

    batting = match_deliveries[
        match_deliveries["batting_team"] == team
    ]

    bowling = match_deliveries[
        match_deliveries["bowling_team"] == team
    ]

    runs = batting["total_runs"].sum()

    # Legal bowling balls.
    bowling_legal = bowling[
        ~bowling["extras_type"]
        .fillna("")
        .isin(
            [
                "wides",
                "noballs",
            ]
        )
    ]

    balls_bowled = len(
        bowling_legal
    )

    # Runs charged to the bowler.
    bowling_extras = bowling[
        bowling["extras_type"]
        .fillna("")
        .isin(
            [
                "byes",
                "legbyes",
                "penalty",
            ]
        )
    ]

    runs_conceded = (
        bowling["total_runs"].sum()
        - bowling_extras["extra_runs"].sum()
    )

    # Only bowler-creditable wickets.
    wickets = bowling[
        bowling["dismissal_kind"]
        .fillna("")
        .isin(
            [
                "caught",
                "bowled",
                "lbw",
                "caught and bowled",
                "stumped",
                "hit wicket",
            ]
        )
    ]["is_wicket"].sum()

    return {
        "runs": float(runs),
        "runs_conceded": float(
            runs_conceded
        ),
        "wickets": int(wickets),
        "balls_bowled": int(
            balls_bowled
        ),
    }


# ---------------------------------------------------------
# Update historical team state
# ---------------------------------------------------------

def update_team_record(
    record,
    match_stats,
    won,
):
    """Add one completed match to a team's historical record."""

    record["matches"] += 1

    if won:
        record["wins"] += 1

    record["runs"] += match_stats[
        "runs"
    ]

    record["runs_conceded"] += (
        match_stats[
            "runs_conceded"
        ]
    )

    record["wickets"] += (
        match_stats["wickets"]
    )

    record["balls_bowled"] += (
        match_stats["balls_bowled"]
    )


# ---------------------------------------------------------
# Build leakage-safe dataset
# ---------------------------------------------------------

def build_ml_dataset():
    """
    Build a chronological pre-match dataset.

    Every feature for a match is calculated using only
    matches that occurred BEFORE that match.

    Target:
        team_a_won = 1 if Team A won
                     0 if Team B won
    """

    matches = load_matches()
    deliveries = load_deliveries()

    team_history = {}

    rows = []

    for _, match in matches.iterrows():

        team_a = match["team1"]
        team_b = match["team2"]

        winner = match["winner"]

        # Skip matches where the winner is unavailable.
        if winner not in [
            team_a,
            team_b,
        ]:
            continue

        # Initialize team records.
        if team_a not in team_history:
            team_history[
                team_a
            ] = empty_team_record()

        if team_b not in team_history:
            team_history[
                team_b
            ] = empty_team_record()

        record_a = team_history[
            team_a
        ]

        record_b = team_history[
            team_b
        ]

        # -------------------------------------------------
        # IMPORTANT:
        # Features are created BEFORE updating history
        # with the current match.
        # -------------------------------------------------

        features_a = calculate_team_features(
            record_a,
            "team_a",
        )

        features_b = calculate_team_features(
            record_b,
            "team_b",
        )

        row = {
            "match_id": match["id"],
            "date": match["date"],
            "season": match.get(
                "season",
                np.nan,
            ),
            "team_a": team_a,
            "team_b": team_b,
            "venue": match.get(
                "venue",
                "Unknown",
            ),
            "toss_winner": match.get(
                "toss_winner",
                "Unknown",
            ),
            "toss_decision": match.get(
                "toss_decision",
                "Unknown",
            ),
        }

        row.update(features_a)
        row.update(features_b)

        # -------------------------------------------------
        # Difference features
        # -------------------------------------------------

        row[
            "win_rate_difference"
        ] = (
            row["team_a_win_rate"]
            - row["team_b_win_rate"]
        )

        row[
            "runs_per_match_difference"
        ] = (
            row[
                "team_a_runs_per_match"
            ]
            - row[
                "team_b_runs_per_match"
            ]
        )

        row[
            "runs_conceded_difference"
        ] = (
            row[
                "team_a_runs_conceded_per_match"
            ]
            - row[
                "team_b_runs_conceded_per_match"
            ]
        )

        row[
            "wickets_per_match_difference"
        ] = (
            row[
                "team_a_wickets_per_match"
            ]
            - row[
                "team_b_wickets_per_match"
            ]
        )

        row[
            "economy_difference"
        ] = (
            row[
                "team_a_bowling_economy"
            ]
            - row[
                "team_b_bowling_economy"
            ]
        )

        # -------------------------------------------------
        # Target
        # -------------------------------------------------

        row["team_a_won"] = int(
            winner == team_a
        )

        rows.append(row)

        # -------------------------------------------------
        # NOW update historical records.
        #
        # The current match will therefore affect only
        # future matches, never the current row.
        # -------------------------------------------------

        match_id = match["id"]

        match_deliveries = deliveries[
            deliveries["match_id"] == match_id
        ]

        stats_a = calculate_match_team_stats(
            match_deliveries,
            team_a,
        )

        stats_b = calculate_match_team_stats(
            match_deliveries,
            team_b,
        )

        update_team_record(
            record_a,
            stats_a,
            winner == team_a,
        )

        update_team_record(
            record_b,
            stats_b,
            winner == team_b,
        )

    dataset = pd.DataFrame(rows)

    # Replace infinite values.
    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Numerical columns that need a default value
    # for the first matches of a team.
    numerical_columns = [
        column
        for column in dataset.columns
        if column not in [
            "match_id",
            "date",
            "season",
            "team_a",
            "team_b",
            "venue",
            "toss_winner",
            "toss_decision",
        ]
    ]

    dataset[
        numerical_columns
    ] = dataset[
        numerical_columns
    ].fillna(0)

    return dataset


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

def save_ml_dataset():
    """Build and save the leakage-safe ML dataset."""

    dataset = build_ml_dataset()

    output_path = (
        PROCESSED
        / "ml_match_dataset.csv"
    )

    dataset.to_csv(
        output_path,
        index=False,
    )

    print("=" * 70)
    print("ML FEATURE DATASET")
    print("=" * 70)

    print(
        f"Rows: {len(dataset)}"
    )

    print(
        f"Columns: {len(dataset.columns)}"
    )

    print(
        f"Saved to: {output_path}"
    )

    print(
        f"Team A wins: "
        f"{dataset['team_a_won'].sum()}"
    )

    print(
        f"Team B wins: "
        f"{(dataset['team_a_won'] == 0).sum()}"
    )

    print("\nFirst 5 rows:")

    print(
        dataset.head().to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)

    return dataset


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    save_ml_dataset()