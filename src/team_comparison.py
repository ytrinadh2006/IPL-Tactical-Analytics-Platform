from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Rising Pune Supergiants": "Rising Pune Supergiants",
}


def normalize_team(value):
    return TEAM_NAME_MAP.get(value, value)


def load_data():
    matches = pd.read_csv(RAW / "matches.csv")
    deliveries = pd.read_csv(RAW / "deliveries.csv")

    for column in ["team1", "team2", "winner", "toss_winner"]:
        if column in matches.columns:
            matches[column] = matches[column].replace(TEAM_NAME_MAP)

    deliveries["match_id"] = deliveries["match_id"]

    if "batting_team" in deliveries.columns:
        deliveries["batting_team"] = deliveries["batting_team"].replace(
            TEAM_NAME_MAP
        )

    if "bowling_team" in deliveries.columns:
        deliveries["bowling_team"] = deliveries["bowling_team"].replace(
            TEAM_NAME_MAP
        )

    return matches, deliveries


def team_summary(team, matches, deliveries):
    team_matches = matches[
        (matches["team1"] == team) |
        (matches["team2"] == team)
    ].copy()

    wins = int((team_matches["winner"] == team).sum())
    matches_played = len(team_matches)
    losses = matches_played - wins

    win_pct = (
        round((wins / matches_played) * 100, 2)
        if matches_played
        else 0
    )

    team_deliveries = deliveries[
        (deliveries["batting_team"] == team) |
        (deliveries["bowling_team"] == team)
    ].copy()

    batting = deliveries[
        deliveries["batting_team"] == team
    ].copy()

    bowling = deliveries[
        deliveries["bowling_team"] == team
    ].copy()

    total_runs = int(batting["total_runs"].sum())

    runs_per_match = (
        round(total_runs / matches_played, 2)
        if matches_played
        else 0
    )

    wickets = 0

    if "is_wicket" in bowling.columns:
        wickets = int(
            bowling["is_wicket"]
            .fillna(0)
            .astype(int)
            .sum()
        )

    wickets_per_match = (
        round(wickets / matches_played, 2)
        if matches_played
        else 0
    )

    bowling_runs = int(bowling["total_runs"].sum())

    legal_balls = len(
        bowling[
            ~bowling["extras_type"]
            .fillna("")
            .str.contains("wides|noballs", case=False, regex=True)
        ]
    )

    overs = legal_balls / 6

    economy = (
        round(bowling_runs / overs, 2)
        if overs
        else 0
    )

    # Phase-wise batting runs
    powerplay = batting[batting["over"] <= 5]["total_runs"].sum()
    middle = batting[
        (batting["over"] >= 6) &
        (batting["over"] <= 14)
    ]["total_runs"].sum()
    death = batting[batting["over"] >= 15]["total_runs"].sum()

    # Toss impact
    toss_wins = int(
        (team_matches["toss_winner"] == team).sum()
    )

    toss_wins_and_match_wins = int(
        (
            (team_matches["toss_winner"] == team) &
            (team_matches["winner"] == team)
        ).sum()
    )

    toss_conversion = (
        round(
            toss_wins_and_match_wins / toss_wins * 100,
            2
        )
        if toss_wins
        else 0
    )

    # Chasing / defending
    chasing_matches = team_matches[
        team_matches["team2"] == team
    ]

    batting_first_matches = team_matches[
        team_matches["team1"] == team
    ]

    chasing_wins = int(
        (chasing_matches["winner"] == team).sum()
    )

    defending_wins = int(
        (batting_first_matches["winner"] == team).sum()
    )

    return {
        "team": team,
        "matches": matches_played,
        "wins": wins,
        "losses": losses,
        "win_pct": win_pct,
        "total_runs": total_runs,
        "runs_per_match": runs_per_match,
        "wickets": wickets,
        "wickets_per_match": wickets_per_match,
        "bowling_economy": economy,
        "powerplay_runs": int(powerplay),
        "middle_over_runs": int(middle),
        "death_over_runs": int(death),
        "toss_wins": toss_wins,
        "toss_to_win_conversion_pct": toss_conversion,
        "chasing_matches": len(chasing_matches),
        "chasing_wins": chasing_wins,
        "defending_matches": len(batting_first_matches),
        "defending_wins": defending_wins,
    }


def head_to_head(team_a, team_b, matches):
    h2h = matches[
        (
            ((matches["team1"] == team_a) &
             (matches["team2"] == team_b))
            |
            ((matches["team1"] == team_b) &
             (matches["team2"] == team_a))
        )
    ].copy()

    team_a_wins = int((h2h["winner"] == team_a).sum())
    team_b_wins = int((h2h["winner"] == team_b).sum())

    return {
        "matches": len(h2h),
        f"{team_a}_wins": team_a_wins,
        f"{team_b}_wins": team_b_wins,
    }


def compare_teams(team_a, team_b):
    matches, deliveries = load_data()

    team_a = normalize_team(team_a)
    team_b = normalize_team(team_b)

    available_teams = set(matches["team1"]) | set(matches["team2"])

    if team_a not in available_teams:
        raise ValueError(f"Unknown team: {team_a}")

    if team_b not in available_teams:
        raise ValueError(f"Unknown team: {team_b}")

    if team_a == team_b:
        raise ValueError("Please select two different teams.")

    comparison = {
        "team_a": team_summary(
            team_a,
            matches,
            deliveries
        ),
        "team_b": team_summary(
            team_b,
            matches,
            deliveries
        ),
        "head_to_head": head_to_head(
            team_a,
            team_b,
            matches
        ),
    }

    return comparison


if __name__ == "__main__":
    result = compare_teams(
        "Mumbai Indians",
        "Chennai Super Kings"
    )

    print("\nTEAM COMPARISON")
    print("=" * 60)

    print("\nTeam A")
    for key, value in result["team_a"].items():
        print(f"{key}: {value}")

    print("\nTeam B")
    for key, value in result["team_b"].items():
        print(f"{key}: {value}")

    print("\nHead-to-Head")
    for key, value in result["head_to_head"].items():
        print(f"{key}: {value}")