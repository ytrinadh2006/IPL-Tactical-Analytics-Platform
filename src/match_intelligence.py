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
# Team-name cleanup
# ---------------------------------------------------------

TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Rising Pune Supergiants": "Rising Pune Supergiants",
}


def normalize_team_name(name):
    """Return the project's standard team name."""
    return TEAM_NAME_MAP.get(name, name)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():
    """Load and clean the raw IPL match and delivery data."""

    matches = pd.read_csv(RAW / "matches.csv")
    deliveries = pd.read_csv(RAW / "deliveries.csv")

    matches = matches.copy()
    deliveries = deliveries.copy()

    # Standardize team names in match data.
    for column in ["team1", "team2", "winner", "toss_winner"]:
        if column in matches.columns:
            matches[column] = (
                matches[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace(TEAM_NAME_MAP)
            )

    # Standardize team names in delivery data.
    for column in ["batting_team", "bowling_team"]:
        if column in deliveries.columns:
            deliveries[column] = (
                deliveries[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace(TEAM_NAME_MAP)
            )

    matches["date"] = pd.to_datetime(
        matches["date"],
        errors="coerce"
    )

    # Numeric columns.
    for column in [
        "total_runs",
        "batsman_runs",
        "is_wicket",
        "over",
    ]:
        if column in deliveries.columns:
            deliveries[column] = pd.to_numeric(
                deliveries[column],
                errors="coerce"
            ).fillna(0)

    return matches, deliveries


# ---------------------------------------------------------
# Phase classification
# ---------------------------------------------------------

def add_phase(deliveries):
    """
    Classify IPL deliveries into:
    Powerplay: overs 0-5
    Middle overs: overs 6-14
    Death overs: overs 15-19
    """

    deliveries = deliveries.copy()

    deliveries["phase"] = pd.cut(
        deliveries["over"],
        bins=[-1, 5, 14, 19],
        labels=[
            "Powerplay",
            "Middle overs",
            "Death overs",
        ],
    )

    return deliveries


# ---------------------------------------------------------
# Team form
# ---------------------------------------------------------

def team_form(matches, team, last_n=5):
    """Return the team's results from its most recent matches."""

    team_matches = matches[
        (matches["team1"] == team)
        | (matches["team2"] == team)
    ].copy()

    team_matches = team_matches.sort_values("date")

    recent = team_matches.tail(last_n)

    results = []

    for _, row in recent.iterrows():

        if row["winner"] == team:
            results.append("W")
        elif row["winner"] in [row["team1"], row["team2"]]:
            results.append("L")
        else:
            results.append("NR")

    wins = results.count("W")
    losses = results.count("L")

    return {
        "matches_considered": len(recent),
        "results": results,
        "wins": wins,
        "losses": losses,
        "win_pct": round(
            wins / len(recent) * 100,
            2
        ) if recent.empty is False else np.nan,
    }


# ---------------------------------------------------------
# Team overall statistics
# ---------------------------------------------------------

def team_statistics(matches, deliveries, team):
    """Calculate the main historical team statistics."""

    team_matches = matches[
        (matches["team1"] == team)
        | (matches["team2"] == team)
    ].copy()

    match_ids = set(team_matches["id"])

    team_deliveries = deliveries[
        deliveries["match_id"].isin(match_ids)
    ].copy()

    matches_played = len(team_matches)

    wins = int(
        (team_matches["winner"] == team).sum()
    )

    losses = int(
        (
            team_matches["winner"].isin(
                [team]
            )
            == False
        ).sum()
    )

    # Only count matches with a recorded winner when calculating
    # wins/losses.
    completed_matches = team_matches[
        team_matches["winner"].isin(
            [team]
        )
        | team_matches["winner"].isin(
            team_matches["team1"]
        )
    ]

    # More reliable loss calculation.
    losses = 0

    for _, row in team_matches.iterrows():

        if row["winner"] in [row["team1"], row["team2"]]:

            if row["winner"] != team:
                losses += 1

    # Batting runs.
    batting = team_deliveries[
        team_deliveries["batting_team"] == team
    ]

    total_runs = batting["total_runs"].sum()

    # Bowling data.
    bowling = team_deliveries[
        team_deliveries["bowling_team"] == team
    ]

    legal_balls = bowling[
        ~bowling["extras_type"]
        .fillna("")
        .isin(["wides", "noballs"])
    ]

    bowling_runs = (
        bowling["total_runs"].sum()
        - bowling.loc[
            bowling["extras_type"]
            .fillna("")
            .isin(["byes", "legbyes", "penalty"]),
            "extra_runs",
        ].sum()
    )

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

    bowling_balls = len(legal_balls)

    economy = (
        bowling_runs / (bowling_balls / 6)
        if bowling_balls > 0
        else np.nan
    )

        # Chasing and defending.
    chasing_matches = 0
    chasing_wins = 0

    defending_matches = 0
    defending_wins = 0

    for _, row in team_matches.iterrows():

        if row["winner"] not in [row["team1"], row["team2"]]:
            continue

        toss_winner = row.get("toss_winner", "")
        toss_decision = row.get("toss_decision", "")

        # Determine which team chased.
        if toss_decision == "field":
            chasing_team = toss_winner

        elif toss_decision == "bat":
            chasing_team = (
                row["team2"]
                if toss_winner == row["team1"]
                else row["team1"]
            )

        else:
            continue

        if chasing_team == team:
            chasing_matches += 1

            if row["winner"] == team:
                chasing_wins += 1

        else:
            defending_matches += 1

            if row["winner"] == team:
                defending_wins += 1

    return {
        "team": team,
        "matches": matches_played,
        "wins": wins,
        "losses": losses,
        "win_pct": round(
            wins / matches_played * 100,
            2
        ) if matches_played else np.nan,
        "total_runs": int(total_runs),
        "runs_per_match": round(
            total_runs / matches_played,
            2
        ) if matches_played else np.nan,
        "wickets": int(wickets),
        "wickets_per_match": round(
            wickets / matches_played,
            2
        ) if matches_played else np.nan,
        "bowling_economy": round(
            economy,
            2
        ) if pd.notna(economy) else np.nan,
        "chasing_matches": chasing_matches,
        "chasing_wins": chasing_wins,
        "chasing_win_pct": round(
            chasing_wins / chasing_matches * 100,
            2
        ) if chasing_matches else np.nan,
        "defending_matches": defending_matches,
        "defending_wins": defending_wins,
        "defending_win_pct": round(
            defending_wins / defending_matches * 100,
            2
        ) if defending_matches else np.nan,
    }


# ---------------------------------------------------------
# Phase performance
# ---------------------------------------------------------

def phase_performance(deliveries, team):
    """
    Calculate batting and bowling performance by phase.

    Batting:
        runs
        balls
        strike rate
        dot-ball percentage

    Bowling:
        runs conceded
        legal balls
        wickets
        economy
        dot-ball percentage
    """

    data = deliveries[
        (
            deliveries["batting_team"] == team
        )
        | (
            deliveries["bowling_team"] == team
        )
    ].copy()

    data = add_phase(data)

    result = {}

    for phase in [
        "Powerplay",
        "Middle overs",
        "Death overs",
    ]:

        phase_data = data[
            data["phase"] == phase
        ]

        # -----------------------------
        # Batting
        # -----------------------------

        batting = phase_data[
            phase_data["batting_team"] == team
        ]

        batting_runs = batting["batsman_runs"].sum()

        batting_legal = batting[
            ~batting["extras_type"]
            .fillna("")
            .isin(["wides", "noballs"])
        ]

        batting_balls = len(batting_legal)

        batting_dots = (
            batting_legal["batsman_runs"] == 0
        ).sum()

        strike_rate = (
            batting_runs / batting_balls * 100
            if batting_balls
            else np.nan
        )

        dot_ball_pct = (
            batting_dots / batting_balls * 100
            if batting_balls
            else np.nan
        )

        # -----------------------------
        # Bowling
        # -----------------------------

        bowling = phase_data[
            phase_data["bowling_team"] == team
        ]

        bowling_legal = bowling[
            ~bowling["extras_type"]
            .fillna("")
            .isin(["wides", "noballs"])
        ]

        bowling_balls = len(bowling_legal)

        bowling_runs = (
            bowling["total_runs"].sum()
            - bowling.loc[
                bowling["extras_type"]
                .fillna("")
                .isin(
                    [
                        "byes",
                        "legbyes",
                        "penalty",
                    ]
                ),
                "extra_runs",
            ].sum()
        )

        bowling_dots = (
            bowling_legal["total_runs"] == 0
        ).sum()

        bowling_wickets = bowling[
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

        economy = (
            bowling_runs / (bowling_balls / 6)
            if bowling_balls
            else np.nan
        )

        bowling_dot_pct = (
            bowling_dots / bowling_balls * 100
            if bowling_balls
            else np.nan
        )

        result[phase] = {
            "batting_runs": int(batting_runs),
            "batting_balls": int(batting_balls),
            "batting_strike_rate": round(
                strike_rate,
                2
            ) if pd.notna(strike_rate) else np.nan,
            "batting_dot_ball_pct": round(
                dot_ball_pct,
                2
            ) if pd.notna(dot_ball_pct) else np.nan,
            "bowling_runs_conceded": int(
                bowling_runs
            ),
            "bowling_balls": int(
                bowling_balls
            ),
            "bowling_wickets": int(
                bowling_wickets
            ),
            "bowling_economy": round(
                economy,
                2
            ) if pd.notna(economy) else np.nan,
            "bowling_dot_ball_pct": round(
                bowling_dot_pct,
                2
            ) if pd.notna(bowling_dot_pct) else np.nan,
        }

    return result


# ---------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------

def head_to_head(matches, team_a, team_b):
    """Return historical head-to-head results."""

    h2h = matches[
        (
            (
                matches["team1"] == team_a
            )
            & (
                matches["team2"] == team_b
            )
        )
        |
        (
            (
                matches["team1"] == team_b
            )
            & (
                matches["team2"] == team_a
            )
        )
    ]

    a_wins = int(
        (h2h["winner"] == team_a).sum()
    )

    b_wins = int(
        (h2h["winner"] == team_b).sum()
    )

    return {
        "matches": len(h2h),
        f"{team_a}_wins": a_wins,
        f"{team_b}_wins": b_wins,
        "no_result_or_unknown": int(
            len(h2h) - a_wins - b_wins
        ),
    }


# ---------------------------------------------------------
# Venue intelligence
# ---------------------------------------------------------

def venue_intelligence(
    matches,
    deliveries,
    venue,
):
    """Calculate historical behavior of a venue."""

    venue_matches = matches[
        matches["venue"] == venue
    ].copy()

    match_ids = set(venue_matches["id"])

    venue_deliveries = deliveries[
        deliveries["match_id"].isin(match_ids)
    ].copy()

    # First-innings scores.
    innings_scores = (
        venue_deliveries
        .groupby(
            ["match_id", "inning"],
            observed=True
        )["total_runs"]
        .sum()
        .reset_index()
    )

    first_innings = innings_scores[
        innings_scores["inning"] == 1
    ]

    avg_first_innings_score = (
        first_innings["total_runs"].mean()
        if not first_innings.empty
        else np.nan
    )

    # Chasing results.
    chasing_matches = 0
    chasing_wins = 0

    for _, row in venue_matches.iterrows():

        if row["winner"] not in [
            row["team1"],
            row["team2"],
        ]:
            continue

        toss_winner = row.get(
            "toss_winner",
            ""
        )

        toss_decision = row.get(
            "toss_decision",
            ""
        )

        # Determine which team chased.
        if toss_decision == "field":

            chasing_team = toss_winner

        elif toss_decision == "bat":

            if toss_winner == row["team1"]:
                chasing_team = row["team2"]
            else:
                chasing_team = row["team1"]

        else:
            continue

        chasing_matches += 1

        if row["winner"] == chasing_team:
            chasing_wins += 1

    chasing_win_pct = (
        chasing_wins
        / chasing_matches
        * 100
        if chasing_matches
        else np.nan
    )

    return {
        "venue": venue,
        "matches": len(venue_matches),
        "avg_first_innings_score": round(
            avg_first_innings_score,
            2
        ) if pd.notna(
            avg_first_innings_score
        ) else np.nan,
        "chasing_matches": chasing_matches,
        "chasing_wins": chasing_wins,
        "chasing_win_pct": round(
            chasing_win_pct,
            2
        ) if pd.notna(
            chasing_win_pct
        ) else np.nan,
    }

# ---------------------------------------------------------
# Toss context
# ---------------------------------------------------------

def toss_context(matches, team_a, team_b):
    """Summarize toss history for the two teams."""

    relevant = matches[
        (
            matches["team1"].isin(
                [team_a, team_b]
            )
        )
        |
        (
            matches["team2"].isin(
                [team_a, team_b]
            )
        )
    ]

    result = {}

    for team in [team_a, team_b]:

        toss_matches = relevant[
            relevant["toss_winner"] == team
        ]

        toss_wins = len(toss_matches)

        match_wins_after_toss = int(
            (
                toss_matches["winner"] == team
            ).sum()
        )

        result[team] = {
            "toss_wins": toss_wins,
            "wins_after_winning_toss":
                match_wins_after_toss,
            "toss_to_win_pct": round(
                match_wins_after_toss
                / toss_wins
                * 100,
                2,
            ) if toss_wins else np.nan,
        }

    return result


# ---------------------------------------------------------
# Complete match intelligence
# ---------------------------------------------------------

def build_match_intelligence(
    team_a,
    team_b,
    venue=None,
):
    """
    Build a complete analytical profile for a hypothetical
    matchup between two teams.

    This function does NOT predict the winner.
    """

    matches, deliveries = load_data()

    team_a = normalize_team_name(team_a)
    team_b = normalize_team_name(team_b)

    available_teams = sorted(
        set(matches["team1"])
        | set(matches["team2"])
    )

    if team_a not in available_teams:
        raise ValueError(
            f"Unknown team: {team_a}"
        )

    if team_b not in available_teams:
        raise ValueError(
            f"Unknown team: {team_b}"
        )

    if team_a == team_b:
        raise ValueError(
            "Team A and Team B must be different."
        )

    if venue is not None:

        available_venues = set(
            matches["venue"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        if venue not in available_venues:
            raise ValueError(
                f"Unknown venue: {venue}"
            )

    intelligence = {
        "team_a": team_a,
        "team_b": team_b,
        "venue": venue,
        "team_a_statistics":
            team_statistics(
                matches,
                deliveries,
                team_a,
            ),
        "team_b_statistics":
            team_statistics(
                matches,
                deliveries,
                team_b,
            ),
        "team_a_recent_form":
            team_form(
                matches,
                team_a,
            ),
        "team_b_recent_form":
            team_form(
                matches,
                team_b,
            ),
        "team_a_phase_performance":
            phase_performance(
                deliveries,
                team_a,
            ),
        "team_b_phase_performance":
            phase_performance(
                deliveries,
                team_b,
            ),
        "head_to_head":
            head_to_head(
                matches,
                team_a,
                team_b,
            ),
        "toss_context":
            toss_context(
                matches,
                team_a,
                team_b,
            ),
    }

    if venue is not None:

        intelligence["venue_intelligence"] = (
            venue_intelligence(
                matches,
                deliveries,
                venue,
            )
        )

    return intelligence


# ---------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------

def print_report(report):
    """Print the match intelligence report."""

    print("\n" + "=" * 70)
    print("IPL MATCH INTELLIGENCE")
    print("=" * 70)

    print(f"\nTeam A: {report['team_a']}")
    print(f"Team B: {report['team_b']}")

    if report["venue"]:
        print(f"Venue: {report['venue']}")

    print("\n" + "-" * 70)
    print("TEAM PERFORMANCE")
    print("-" * 70)

    for key in [
        "team_a_statistics",
        "team_b_statistics",
    ]:

        stats = report[key]

        print(f"\n{stats['team']}")

        print(
            f"Matches: {stats['matches']}"
        )
        print(
            f"Wins: {stats['wins']}"
        )
        print(
            f"Losses: {stats['losses']}"
        )
        print(
            f"Win %: {stats['win_pct']}"
        )
        print(
            f"Runs/match: {stats['runs_per_match']}"
        )
        print(
            f"Wickets/match: {stats['wickets_per_match']}"
        )
        print(
            f"Bowling economy: {stats['bowling_economy']}"
        )
        print(
            f"Chasing win %: {stats['chasing_win_pct']}"
        )
        print(
            f"Defending win %: {stats['defending_win_pct']}"
        )

    print("\n" + "-" * 70)
    print("RECENT FORM")
    print("-" * 70)

    print(
        report["team_a"],
        ":",
        report["team_a_recent_form"]["results"],
    )

    print(
        report["team_b"],
        ":",
        report["team_b_recent_form"]["results"],
    )

    print("\n" + "-" * 70)
    print("HEAD-TO-HEAD")
    print("-" * 70)

    for key, value in report["head_to_head"].items():
        print(f"{key}: {value}")

    print("\n" + "-" * 70)
    print("PHASE PERFORMANCE")
    print("-" * 70)

    for team_key in [
        "team_a_phase_performance",
        "team_b_phase_performance",
    ]:

        team_name = (
            report["team_a"]
            if team_key == "team_a_phase_performance"
            else report["team_b"]
        )

        print(f"\n{team_name}")

        phases = report[team_key]

        for phase, values in phases.items():

            print(f"\n  {phase}")

            print(
                f"    Batting runs: "
                f"{values['batting_runs']}"
            )

            print(
                f"    Batting SR: "
                f"{values['batting_strike_rate']}"
            )

            print(
                f"    Batting dot-ball %: "
                f"{values['batting_dot_ball_pct']}"
            )

            print(
                f"    Bowling wickets: "
                f"{values['bowling_wickets']}"
            )

            print(
                f"    Bowling economy: "
                f"{values['bowling_economy']}"
            )

            print(
                f"    Bowling dot-ball %: "
                f"{values['bowling_dot_ball_pct']}"
            )

    print("\n" + "-" * 70)
    print("TOSS CONTEXT")
    print("-" * 70)

    for team, values in report[
        "toss_context"
    ].items():

        print(f"\n{team}")

        for key, value in values.items():
            print(
                f"  {key}: {value}"
            )

    if "venue_intelligence" in report:

        print("\n" + "-" * 70)
        print("VENUE INTELLIGENCE")
        print("-" * 70)

        venue_data = report[
            "venue_intelligence"
        ]

        for key, value in venue_data.items():

            if key != "venue":
                print(
                    f"{key}: {value}"
                )

    print("\n" + "=" * 70)
    print(
        "NOTE: This module provides historical "
        "analytical context only."
    )
    print(
        "It does not predict the match winner."
    )
    print("=" * 70)


# ---------------------------------------------------------
# Example execution
# ---------------------------------------------------------

if __name__ == "__main__":

    report = build_match_intelligence(
        "Mumbai Indians",
        "Chennai Super Kings",
        "Wankhede Stadium",
    )

    print_report(report)