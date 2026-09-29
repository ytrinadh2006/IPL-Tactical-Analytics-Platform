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
    "Royal Challengers Bangalore":
        "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru":
        "Royal Challengers Bengaluru",
    "Rising Pune Supergiant":
        "Rising Pune Supergiants",
    "Rising Pune Supergiants":
        "Rising Pune Supergiants",
}


# ---------------------------------------------------------
# Load matches
# ---------------------------------------------------------

def load_matches():

    matches = pd.read_csv(
        RAW / "matches.csv"
    )

    for column in [
        "team1",
        "team2",
        "winner",
        "toss_winner",
    ]:

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
# Load deliveries
# ---------------------------------------------------------

def load_deliveries():

    deliveries = pd.read_csv(
        RAW / "deliveries.csv"
    )

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

    deliveries["total_runs"] = pd.to_numeric(
        deliveries["total_runs"],
        errors="coerce",
    ).fillna(0)

    return deliveries


# ---------------------------------------------------------
# Team recent form
# ---------------------------------------------------------

def recent_form(
    team_history,
    last_n=5,
):
    """
    Return win rate from the team's previous
    completed matches only.
    """

    if not team_history:

        return {
            "matches": 0,
            "wins": 0,
            "win_rate": 0.5,
        }

    recent = team_history[-last_n:]

    wins = sum(
        result == 1
        for result in recent
    )

    return {
        "matches": len(recent),
        "wins": wins,
        "win_rate": (
            wins / len(recent)
            if recent
            else 0.5
        ),
    }


# ---------------------------------------------------------
# Toss features
# ---------------------------------------------------------

def build_toss_features(
    match,
    team_a,
    team_b,
):
    """
    Convert toss information into numerical
    features for ML.
    """

    toss_winner = match[
        "toss_winner"
    ]

    toss_decision = match[
        "toss_decision"
    ]

    return {
        "toss_team_a_won": int(
            toss_winner == team_a
        ),

        "toss_team_b_won": int(
            toss_winner == team_b
        ),

        "toss_decision_field": int(
            toss_decision == "field"
        ),

        "toss_decision_bat": int(
            toss_decision == "bat"
        ),
    }


# ---------------------------------------------------------
# Venue historical state
# ---------------------------------------------------------

def venue_features(
    venue_history,
):
    """
    Calculate venue statistics using only
    previous matches at that venue.
    """

    if not venue_history:

        return {
            "venue_matches_before": 0,
            "venue_avg_first_innings": 0.0,
            "venue_chasing_win_rate": 0.5,
        }

    first_innings_scores = [
        item["first_innings_score"]
        for item in venue_history
        if item["first_innings_score"]
        is not None
    ]

    chasing_results = [
        item["chasing_won"]
        for item in venue_history
        if item["chasing_won"]
        is not None
    ]

    avg_score = (
        np.mean(first_innings_scores)
        if first_innings_scores
        else 0.0
    )

    chasing_rate = (
        np.mean(chasing_results)
        if chasing_results
        else 0.5
    )

    return {
        "venue_matches_before":
            len(venue_history),

        "venue_avg_first_innings":
            float(avg_score),

        "venue_chasing_win_rate":
            float(chasing_rate),
    }


# ---------------------------------------------------------
# Match result for a chasing team
# ---------------------------------------------------------

def determine_chasing_team(
    match,
):
    """
    Determine the team that batted second
    using toss decision.
    """

    toss_winner = match[
        "toss_winner"
    ]

    toss_decision = match[
        "toss_decision"
    ]

    team1 = match["team1"]
    team2 = match["team2"]

    if toss_decision == "field":

        return toss_winner

    if toss_decision == "bat":

        if toss_winner == team1:
            return team2

        return team1

    return None


# ---------------------------------------------------------
# Calculate first innings score
# ---------------------------------------------------------

def get_first_innings_score(
    match_deliveries,
):
    """
    Calculate the score of innings 1.
    """

    innings_one = match_deliveries[
        match_deliveries["inning"] == 1
    ]

    if innings_one.empty:
        return None

    return float(
        innings_one["total_runs"].sum()
    )


# ---------------------------------------------------------
# Build advanced dataset
# ---------------------------------------------------------

def build_advanced_dataset():

    matches = load_matches()
    deliveries = load_deliveries()

    # Load the already validated dataset.
    base = pd.read_csv(
        PROCESSED /
        "ml_match_dataset.csv"
    )

    base["date"] = pd.to_datetime(
        base["date"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Historical state
    # -----------------------------------------------------

    team_history = {}

    h2h_history = {}

    venue_history = {}

    advanced_rows = []

    # -----------------------------------------------------
    # Process matches chronologically.
    # -----------------------------------------------------

    for _, match in matches.iterrows():

        team_a = match["team1"]
        team_b = match["team2"]

        winner = match["winner"]

        # Skip matches without a valid result.
        if winner not in [
            team_a,
            team_b,
        ]:
            continue

        # Initialize team histories.
        if team_a not in team_history:
            team_history[team_a] = []

        if team_b not in team_history:
            team_history[team_b] = []

        # -------------------------------------------------
        # Recent form BEFORE current match
        # -------------------------------------------------

        form_a = recent_form(
            team_history[team_a]
        )

        form_b = recent_form(
            team_history[team_b]
        )

        # -------------------------------------------------
        # H2H BEFORE current match
        # -------------------------------------------------

        h2h_key = tuple(
            sorted(
                [team_a, team_b]
            )
        )

        previous_h2h = h2h_history.get(
            h2h_key,
            [],
        )

        h2h_matches = len(
            previous_h2h
        )

        team_a_h2h_wins = sum(
            result == team_a
            for result in previous_h2h
        )

        team_b_h2h_wins = sum(
            result == team_b
            for result in previous_h2h
        )

        team_a_h2h_rate = (
            team_a_h2h_wins
            / h2h_matches
            if h2h_matches
            else 0.5
        )

        # -------------------------------------------------
        # Venue BEFORE current match
        # -------------------------------------------------

        venue = match["venue"]

        previous_venue = venue_history.get(
            venue,
            [],
        )

        venue_data = venue_features(
            previous_venue
        )

        # -------------------------------------------------
        # Toss
        # -------------------------------------------------

        toss_data = build_toss_features(
            match,
            team_a,
            team_b,
        )

        # -------------------------------------------------
        # Build row
        # -------------------------------------------------

        row = {
            "match_id":
                match["id"],

            "team_a_recent_matches":
                form_a["matches"],

            "team_a_recent_wins":
                form_a["wins"],

            "team_a_recent_win_rate":
                form_a["win_rate"],

            "team_b_recent_matches":
                form_b["matches"],

            "team_b_recent_wins":
                form_b["wins"],

            "team_b_recent_win_rate":
                form_b["win_rate"],

            "recent_win_rate_difference":
                (
                    form_a["win_rate"]
                    - form_b["win_rate"]
                ),

            "h2h_matches_before":
                h2h_matches,

            "team_a_h2h_wins":
                team_a_h2h_wins,

            "team_b_h2h_wins":
                team_b_h2h_wins,

            "team_a_h2h_win_rate":
                team_a_h2h_rate,

            "venue_matches_before":
                venue_data[
                    "venue_matches_before"
                ],

            "venue_avg_first_innings":
                venue_data[
                    "venue_avg_first_innings"
                ],

            "venue_chasing_win_rate":
                venue_data[
                    "venue_chasing_win_rate"
                ],
        }

        row.update(toss_data)

        advanced_rows.append(row)

        # -------------------------------------------------
        # AFTER creating the row, update histories.
        # -------------------------------------------------

        team_history[
            team_a
        ].append(
            int(winner == team_a)
        )

        team_history[
            team_b
        ].append(
            int(winner == team_b)
        )

        # H2H history.
        if h2h_key not in h2h_history:
            h2h_history[
                h2h_key
            ] = []

        h2h_history[
            h2h_key
        ].append(winner)

        # -------------------------------------------------
        # Venue history.
        # -------------------------------------------------

        if venue not in venue_history:
            venue_history[
                venue
            ] = []

        match_id = match["id"]

        match_deliveries = deliveries[
            deliveries["match_id"]
            == match_id
        ]

        first_innings_score = (
            get_first_innings_score(
                match_deliveries
            )
        )

        chasing_team = (
            determine_chasing_team(
                match
            )
        )

        if chasing_team is not None:

            chasing_won = int(
                winner == chasing_team
            )

        else:

            chasing_won = None

        venue_history[
            venue
        ].append(
            {
                "first_innings_score":
                    first_innings_score,

                "chasing_won":
                    chasing_won,
            }
        )

    advanced = pd.DataFrame(
        advanced_rows
    )

    # -----------------------------------------------------
    # Merge with base dataset.
    # -----------------------------------------------------

    advanced = advanced[
        advanced["match_id"].notna()
    ]

    base = base.merge(
        advanced,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # Fill numerical missing values.
    # -----------------------------------------------------

    numeric_columns = (
        base.select_dtypes(
            include=np.number
        ).columns
    )

    base[numeric_columns] = (
        base[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    return base


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

def save_advanced_dataset():

    dataset = build_advanced_dataset()

    output_path = (
        PROCESSED /
        "ml_match_dataset_advanced.csv"
    )

    dataset.to_csv(
        output_path,
        index=False,
    )

    print("=" * 70)
    print("ADVANCED ML FEATURE DATASET")
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

    print("\nNew advanced features:")

    new_features = [
        "team_a_recent_win_rate",
        "team_b_recent_win_rate",
        "recent_win_rate_difference",
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
    ]

    for feature in new_features:

        print(
            f"{feature}: "
            f"{feature in dataset.columns}"
        )

    print("\nValidation:")

    print(
        "Dates sorted:",
        dataset["date"].is_monotonic_increasing,
    )

    print(
        "Duplicate match IDs:",
        dataset["match_id"].duplicated().sum(),
    )

    print(
        "Missing values:",
        dataset.isna().sum().sum(),
    )

    print(
        "First match recent win rate:",
        dataset.iloc[0][
            "team_a_recent_win_rate"
        ],
        dataset.iloc[0][
            "team_b_recent_win_rate"
        ],
    )

    print("=" * 70)

    return dataset


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":

    save_advanced_dataset()