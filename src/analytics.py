from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# IPL INTELLIGENCE & TACTICAL ANALYTICS PLATFORM
# Core Analytics Builder
# ============================================================
#
# Metric definitions used in this module:
#
# 1. Legal ball:
#    A delivery is legal unless it is a wide or no-ball.
#
# 2. Bowling runs:
#    Bowler is charged for normal runs, wides and no-ball runs.
#    Byes, leg-byes and penalty runs are excluded.
#
# 3. Bowling wickets:
#    Only wickets credited to the bowler are counted:
#    caught, bowled, lbw, caught and bowled, stumped,
#    and hit wicket.
#
# 4. Batting average:
#    Runs / actual batting dismissals.
#    Retired hurt is not counted as a dismissal.
#
# 5. Phases:
#    Powerplay   = overs 0-5
#    Middle      = overs 6-14
#    Death       = overs 15-19
#
# 6. Fielding:
#    Only directly observable fielding events are counted.
#    We do NOT invent dropped catches, misfields or runs saved
#    because this dataset does not reliably contain those events.
#
# 7. Fielding Impact:
#    A transparent heuristic composite:
#    catch = 1.0
#    run-out = 1.5
#    stumping = 1.2
#
#    These weights are project-defined and are NOT presented as
#    official cricket values.
#
# 8. Toss impact:
#    We report association between winning the toss and winning
#    the match. This is descriptive and does NOT establish causality.
#
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Historical team-name normalization
# ------------------------------------------------------------

TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Rising Pune Supergiants": "Rising Pune Supergiants",
}


# ------------------------------------------------------------
# IPL innings phases
# ------------------------------------------------------------

PHASE_LABELS = [
    "Powerplay",
    "Middle overs",
    "Death overs",
]

PHASE_BINS = [-1, 5, 14, 19]


# ------------------------------------------------------------
# Wickets credited to the bowler
# ------------------------------------------------------------

BOWLER_WICKET_TYPES = {
    "caught",
    "bowled",
    "lbw",
    "caught and bowled",
    "stumped",
    "hit wicket",
}


# ------------------------------------------------------------
# Retired hurt is not a batting dismissal
# ------------------------------------------------------------

NON_DISMISSAL_TYPES = {
    "retired hurt"
}


# ------------------------------------------------------------
# Fielding-impact weights
# ------------------------------------------------------------

FIELDING_WEIGHT_CATCH = 1.0
FIELDING_WEIGHT_RUN_OUT = 1.5
FIELDING_WEIGHT_STUMPING = 1.2


# ============================================================
# DATA LOADING
# ============================================================

def normalize_team_names(df):
    """
    Normalize historical IPL team names.

    The raw CSV files are not modified.
    """

    df = df.copy()

    columns = [
        "team1",
        "team2",
        "winner",
        "toss_winner",
    ]

    for column in columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .replace(TEAM_NAME_MAP)
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    return df


def load_data():
    """
    Load and perform basic type cleaning on the raw datasets.
    """

    matches = pd.read_csv(
        RAW / "matches.csv"
    )

    deliveries = pd.read_csv(
        RAW / "deliveries.csv"
    )

    # Normalize historical team names.
    matches = normalize_team_names(matches)

    # Convert date.
    if "date" in matches.columns:

        matches["date"] = pd.to_datetime(
            matches["date"],
            errors="coerce"
        )

    # Clean match text fields.
    string_columns = [
        "team1",
        "team2",
        "winner",
        "toss_winner",
        "toss_decision",
        "venue",
        "city",
    ]

    for column in string_columns:

        if column in matches.columns:

            matches[column] = (
                matches[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    # Numeric delivery fields.
    numeric_columns = [
        "batsman_runs",
        "total_runs",
        "extra_runs",
        "is_wicket",
        "over",
        "ball",
    ]

    for column in numeric_columns:

        if column in deliveries.columns:

            deliveries[column] = pd.to_numeric(
                deliveries[column],
                errors="coerce"
            ).fillna(0)

    deliveries["over"] = (
        deliveries["over"]
        .astype(int)
    )

    deliveries["is_wicket"] = (
        deliveries["is_wicket"]
        .astype(int)
    )

    # Clean delivery text fields.
    for column in [
        "extras_type",
        "dismissal_kind",
        "player_dismissed",
        "fielder",
    ]:

        if column in deliveries.columns:

            deliveries[column] = (
                deliveries[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    return matches, deliveries


# ============================================================
# DELIVERY-LEVEL CRICKET RULES
# ============================================================

def add_delivery_rules(deliveries):
    """
    Add consistent cricket-aware delivery flags.

    These definitions are reused throughout the project.
    """

    df = deliveries.copy()

    extras = (
        df["extras_type"]
        .fillna("")
        .str.lower()
    )

    dismissal = (
        df["dismissal_kind"]
        .fillna("")
        .str.lower()
    )

    # --------------------------------------------------------
    # LEGAL BALL
    # --------------------------------------------------------

    # Wide and no-ball do not count as legal deliveries.
    df["is_legal_ball"] = ~extras.isin({
        "wides",
        "noballs",
    })

    # --------------------------------------------------------
    # BOWLER RUNS
    # --------------------------------------------------------

    # Start with total runs.
    df["bowler_runs"] = df["total_runs"]

    # Byes, leg-byes and penalty runs are not charged to bowler.
    non_bowler_extras = extras.isin({
        "byes",
        "legbyes",
        "penalty",
    })

    df.loc[
        non_bowler_extras,
        "bowler_runs"
    ] = (
        df.loc[
            non_bowler_extras,
            "total_runs"
        ]
        -
        df.loc[
            non_bowler_extras,
            "extra_runs"
        ]
    )

    # Safety against malformed negative values.
    df["bowler_runs"] = (
        pd.to_numeric(
            df["bowler_runs"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # BOWLER-CREDITED WICKET
    # --------------------------------------------------------

    df["bowler_wicket"] = (
        dismissal
        .isin(BOWLER_WICKET_TYPES)
        .astype(int)
    )

    # --------------------------------------------------------
    # BATTING DOT BALL
    # --------------------------------------------------------

    # A batting dot requires:
    # - legal delivery
    # - batsman scored 0
    # - no total runs occurred
    df["batting_dot"] = (
        df["is_legal_ball"]
        &
        (df["batsman_runs"] == 0)
        &
        (df["total_runs"] == 0)
    )

    # --------------------------------------------------------
    # BOWLING DOT BALL
    # --------------------------------------------------------

    df["bowling_dot"] = (
        df["is_legal_ball"]
        &
        (df["total_runs"] == 0)
    )

    # --------------------------------------------------------
    # ACTUAL BATTING DISMISSAL
    # --------------------------------------------------------

    df["batting_dismissal"] = (
        (df["is_wicket"] == 1)
        &
        df["player_dismissed"].ne("")
        &
        ~dismissal.isin(NON_DISMISSAL_TYPES)
    )

    return df


# ============================================================
# PHASE ASSIGNMENT
# ============================================================

def add_phase(df):
    """
    Assign every delivery to an IPL innings phase.
    """

    df = df.copy()

    df["phase"] = pd.cut(
        df["over"],
        bins=PHASE_BINS,
        labels=PHASE_LABELS,
    )

    return df


# ============================================================
# BATTING PHASE STATISTICS
# ============================================================

def batting_phase_table(
    df,
    key="batter"
):
    """
    Batting statistics by phase.
    """

    out = (
        df
        .groupby(
            [key, "phase"],
            observed=False
        )
        .agg(
            runs=(
                "batsman_runs",
                "sum"
            ),

            balls=(
                "is_legal_ball",
                "sum"
            ),

            fours=(
                "batsman_runs",
                lambda x: (x == 4).sum()
            ),

            sixes=(
                "batsman_runs",
                lambda x: (x == 6).sum()
            ),

            dot_balls=(
                "batting_dot",
                "sum"
            ),
        )
        .reset_index()
    )

    out["strike_rate"] = np.where(
        out["balls"] > 0,
        out["runs"]
        / out["balls"]
        * 100,
        0,
    )

    out["dot_ball_pct"] = np.where(
        out["balls"] > 0,
        out["dot_balls"]
        / out["balls"]
        * 100,
        0,
    )

    return out


# ============================================================
# BOWLING PHASE STATISTICS
# ============================================================

def bowling_phase_table(
    df,
    key="bowler"
):
    """
    Bowling statistics by phase.
    """

    out = (
        df
        .groupby(
            [key, "phase"],
            observed=False
        )
        .agg(
            runs=(
                "bowler_runs",
                "sum"
            ),

            balls=(
                "is_legal_ball",
                "sum"
            ),

            wickets=(
                "bowler_wicket",
                "sum"
            ),

            dot_balls=(
                "bowling_dot",
                "sum"
            ),
        )
        .reset_index()
    )

    out["overs"] = (
        out["balls"] / 6
    )

    out["economy"] = np.where(
        out["balls"] > 0,
        out["runs"]
        / (out["balls"] / 6),
        0,
    )

    out["dot_ball_pct"] = np.where(
        out["balls"] > 0,
        out["dot_balls"]
        / out["balls"]
        * 100,
        0,
    )

    return out


# ============================================================
# PLAYER BATTING STATISTICS
# ============================================================

def build_batting_stats(df):
    """
    Build player-level batting statistics.
    """

    bat = (
        df
        .groupby("batter")
        .agg(
            runs=(
                "batsman_runs",
                "sum"
            ),

            balls=(
                "is_legal_ball",
                "sum"
            ),

            fours=(
                "batsman_runs",
                lambda x: (x == 4).sum()
            ),

            sixes=(
                "batsman_runs",
                lambda x: (x == 6).sum()
            ),

            matches=(
                "match_id",
                "nunique"
            ),
        )
        .reset_index()
        .rename(
            columns={
                "batter": "player"
            }
        )
    )

    # Actual dismissals.
    dismissals = (
        df.loc[
            df["batting_dismissal"]
        ]
        .groupby("player_dismissed")
        .size()
        .rename("dismissals")
    )

    bat = bat.merge(
        dismissals,
        left_on="player",
        right_index=True,
        how="left",
    )

    bat["dismissals"] = (
        bat["dismissals"]
        .fillna(0)
    )

    # Strike rate.
    bat["strike_rate"] = np.where(
        bat["balls"] > 0,
        bat["runs"]
        / bat["balls"]
        * 100,
        0,
    )

    # Batting average.
    bat["average"] = np.where(
        bat["dismissals"] > 0,
        bat["runs"]
        / bat["dismissals"],
        np.nan,
    )

    # Percentage of runs from boundaries.
    bat["boundary_runs_pct"] = np.where(
        bat["runs"] > 0,
        (
            bat["fours"] * 4
            +
            bat["sixes"] * 6
        )
        / bat["runs"]
        * 100,
        0,
    )

    # Dot balls.
    dots = (
        df.loc[
            df["batting_dot"]
        ]
        .groupby("batter")
        .size()
    )

    bat["dot_balls"] = (
        dots
        .reindex(bat["player"])
        .fillna(0)
        .to_numpy()
    )

    bat["dot_ball_pct"] = np.where(
        bat["balls"] > 0,
        bat["dot_balls"]
        / bat["balls"]
        * 100,
        0,
    )

    return bat


# ============================================================
# PLAYER BOWLING STATISTICS
# ============================================================

def build_bowling_stats(df):
    """
    Build player-level bowling statistics.
    """

    bowl = (
        df
        .groupby("bowler")
        .agg(
            balls=(
                "is_legal_ball",
                "sum"
            ),

            runs_conceded=(
                "bowler_runs",
                "sum"
            ),

            wickets=(
                "bowler_wicket",
                "sum"
            ),

            matches=(
                "match_id",
                "nunique"
            ),

            dot_balls=(
                "bowling_dot",
                "sum"
            ),
        )
        .reset_index()
        .rename(
            columns={
                "bowler": "player"
            }
        )
    )

    bowl["overs"] = (
        bowl["balls"] / 6
    )

    # Economy = bowler runs / overs.
    bowl["economy"] = np.where(
        bowl["balls"] > 0,
        bowl["runs_conceded"]
        / (bowl["balls"] / 6),
        0,
    )

    # Bowling average = runs conceded / wickets.
    bowl["bowling_average"] = np.where(
        bowl["wickets"] > 0,
        bowl["runs_conceded"]
        / bowl["wickets"],
        np.nan,
    )

    # Bowling strike rate = balls / wickets.
    bowl["strike_rate"] = np.where(
        bowl["wickets"] > 0,
        bowl["balls"]
        / bowl["wickets"],
        np.nan,
    )

    bowl["dot_ball_pct"] = np.where(
        bowl["balls"] > 0,
        bowl["dot_balls"]
        / bowl["balls"]
        * 100,
        0,
    )

    return bowl


# ============================================================
# BATSMAN VS BOWLER MATCHUPS
# ============================================================

def build_matchups(df):
    """
    Build batsman-vs-bowler matchup statistics.

    Matchup balls use legal deliveries.
    Matchup wickets distinguish actual batter dismissal
    from wickets credited to the bowler.
    """

    mu = (
        df
        .groupby(
            ["batter", "bowler"]
        )
        .agg(
            balls=(
                "is_legal_ball",
                "sum"
            ),

            runs=(
                "batsman_runs",
                "sum"
            ),

            dismissals=(
                "batting_dismissal",
                "sum"
            ),

            bowler_wickets=(
                "bowler_wicket",
                "sum"
            ),

            fours=(
                "batsman_runs",
                lambda x: (x == 4).sum()
            ),

            sixes=(
                "batsman_runs",
                lambda x: (x == 6).sum()
            ),

            dot_balls=(
                "batting_dot",
                "sum"
            ),
        )
        .reset_index()
    )

    mu["strike_rate"] = np.where(
        mu["balls"] > 0,
        mu["runs"]
        / mu["balls"]
        * 100,
        0,
    )

    mu["boundary_runs_pct"] = np.where(
        mu["runs"] > 0,
        (
            mu["fours"] * 4
            +
            mu["sixes"] * 6
        )
        / mu["runs"]
        * 100,
        0,
    )

    mu["dot_ball_pct"] = np.where(
        mu["balls"] > 0,
        mu["dot_balls"]
        / mu["balls"]
        * 100,
        0,
    )

    return mu


# ============================================================
# MOST RUNS AGAINST
# ============================================================

def build_most_runs_against(matchups):
    """
    For each batter, show runs scored against each bowler.
    """

    out = matchups[
        [
            "batter",
            "bowler",
            "balls",
            "runs",
            "strike_rate",
            "dismissals",
        ]
    ].copy()

    out = out.sort_values(
        [
            "batter",
            "runs",
            "balls",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    return out


# ============================================================
# VENUE STATISTICS
# ============================================================

def build_venue_stats(
    df,
    matches
):
    """
    Build historical venue behavior metrics.

    These are historical statistical proxies, not claims about
    physical pitch composition.
    """

    # First innings score.
    first_innings = (
        df
        .groupby(
            [
                "match_id",
                "inning",
            ],
            as_index=False
        )["total_runs"]
        .sum()
        .rename(
            columns={
                "total_runs": "innings_score"
            }
        )
    )

    first_innings = first_innings[
        first_innings["inning"] == 1
    ]

    first_innings = first_innings.merge(
        matches[
            [
                "id",
                "venue",
            ]
        ],
        left_on="match_id",
        right_on="id",
        how="left",
    )

    first_score = (
        first_innings
        .groupby("venue")["innings_score"]
        .mean()
        .rename(
            "avg_first_innings_score"
        )
    )

    # Main venue statistics.
    venue = (
        df
        .groupby("venue")
        .agg(
            matches=(
                "match_id",
                "nunique"
            ),

            total_runs=(
                "total_runs",
                "sum"
            ),

            legal_balls=(
                "is_legal_ball",
                "sum"
            ),

            wickets=(
                "bowler_wicket",
                "sum"
            ),

            fours=(
                "batsman_runs",
                lambda x: (x == 4).sum()
            ),

            sixes=(
                "batsman_runs",
                lambda x: (x == 6).sum()
            ),

            dot_balls=(
                "bowling_dot",
                "sum"
            ),
        )
        .reset_index()
    )

    venue["avg_runs_per_match"] = np.where(
        venue["matches"] > 0,
        venue["total_runs"]
        / venue["matches"],
        0,
    )

    venue["run_rate"] = np.where(
        venue["legal_balls"] > 0,
        venue["total_runs"]
        / (venue["legal_balls"] / 6),
        0,
    )

    venue["wickets_per_match"] = np.where(
        venue["matches"] > 0,
        venue["wickets"]
        / venue["matches"],
        0,
    )

    venue["boundary_rate_per_legal_ball"] = np.where(
        venue["legal_balls"] > 0,
        (
            venue["fours"]
            +
            venue["sixes"]
        )
        / venue["legal_balls"]
        * 100,
        0,
    )

    venue["six_rate_per_legal_ball"] = np.where(
        venue["legal_balls"] > 0,
        venue["sixes"]
        / venue["legal_balls"]
        * 100,
        0,
    )

    venue["dot_ball_pct"] = np.where(
        venue["legal_balls"] > 0,
        venue["dot_balls"]
        / venue["legal_balls"]
        * 100,
        0,
    )

    # --------------------------------------------------------
    # CHASING ANALYSIS
    # --------------------------------------------------------

    m = matches[
        matches["winner"].ne("Unknown")
    ].copy()

    # If toss winner chose to field, toss winner chased.
    #
    # If toss winner chose to bat, the other team chased.
    m["chasing_team"] = np.where(
        m["toss_decision"]
        .str.lower()
        .eq("field"),

        m["toss_winner"],

        np.where(
            m["toss_decision"]
            .str.lower()
            .eq("bat"),

            m["team2"].where(
                m["team1"].eq(
                    m["toss_winner"]
                ),
                m["team1"]
            ),

            "Unknown",
        ),
    )

    m["chasing_win"] = (
        m["winner"]
        == m["chasing_team"]
    )

    chasing = (
        m[
            m["chasing_team"].ne("Unknown")
        ]
        .groupby("venue")
        .agg(
            chasing_matches=(
                "id",
                "count"
            ),

            chasing_wins=(
                "chasing_win",
                "sum"
            ),
        )
        .reset_index()
    )

    chasing["chasing_win_pct"] = np.where(
        chasing["chasing_matches"] > 0,
        chasing["chasing_wins"]
        / chasing["chasing_matches"]
        * 100,
        np.nan,
    )

    venue = (
        venue
        .merge(
            first_score,
            on="venue",
            how="left"
        )
        .merge(
            chasing,
            on="venue",
            how="left"
        )
    )

    return venue


# ============================================================
# FIELDING STATISTICS
# ============================================================

def build_fielding_stats(df):
    """
    Build directly observable fielding statistics.

    Included:
    - catches
    - caught and bowled
    - run-outs with recorded fielder
    - stumpings

    We do NOT invent:
    - dropped catches
    - misfields
    - runs saved

    because the raw dataset does not reliably contain those events.
    """

    kind = (
        df["dismissal_kind"]
        .fillna("")
        .str.lower()
    )

    fielder = (
        df["fielder"]
        .fillna("")
        .str.strip()
    )

    # Normal catches.
    normal_catches = (
        df.loc[
            (kind == "caught")
            &
            fielder.ne("")
        ]
        .groupby("fielder")
        .size()
        .rename("catches")
    )

    # Caught and bowled:
    # the bowler is also the fielder.
    caught_and_bowled = (
        df.loc[
            kind == "caught and bowled"
        ]
        .groupby("bowler")
        .size()
        .rename("caught_and_bowled")
    )

    # Run-outs only where the dataset identifies the fielder.
    run_outs = (
        df.loc[
            (kind == "run out")
            &
            fielder.ne("")
        ]
        .groupby("fielder")
        .size()
        .rename("run_outs")
    )

    # Stumpings.
    stumpings = (
        df.loc[
            (kind == "stumped")
            &
            fielder.ne("")
        ]
        .groupby("fielder")
        .size()
        .rename("stumpings")
    )

    field = pd.concat(
        [
            normal_catches,
            caught_and_bowled,
            run_outs,
            stumpings,
        ],
        axis=1,
    ).fillna(0).reset_index()

    field = field.rename(
        columns={
            "index": "player",
            "fielder": "player",
            "bowler": "player",
        }
    )

    for column in [
        "catches",
        "caught_and_bowled",
        "run_outs",
        "stumpings",
    ]:

        field[column] = (
            field[column]
            .astype(int)
        )

    # Caught-and-bowled is a catch credited to the bowler.
    field["catches"] = (
        field["catches"]
        +
        field["caught_and_bowled"]
    )

    field["dismissals"] = (
        field["catches"]
        +
        field["run_outs"]
        +
        field["stumpings"]
    )

    # Transparent project-defined heuristic.
    field["fielding_impact"] = (
        field["catches"]
        * FIELDING_WEIGHT_CATCH

        +

        field["run_outs"]
        * FIELDING_WEIGHT_RUN_OUT

        +

        field["stumpings"]
        * FIELDING_WEIGHT_STUMPING

    ).round(2)

    return field


# ============================================================
# TEAM STATISTICS
# ============================================================

def build_team_stats(
    df,
    matches
):
    """
    Build overall team performance statistics.
    """

    teams = sorted(
        set(matches["team1"])
        |
        set(matches["team2"])
    )

    teams = [
        team
        for team in teams
        if team != "Unknown"
    ]

    rows = []

    for team in teams:

        team_matches = matches[
            (matches["team1"] == team)
            |
            (matches["team2"] == team)
        ]

        valid_matches = team_matches[
            team_matches["winner"].ne("Unknown")
        ]

        wins = int(
            (valid_matches["winner"] == team)
            .sum()
        )

        losses = int(
            len(valid_matches)
            -
            wins
        )

        toss_wins = int(
            (team_matches["toss_winner"] == team)
            .sum()
        )

        wins_after_toss = int(
            (
                (team_matches["toss_winner"] == team)
                &
                (team_matches["winner"] == team)
            )
            .sum()
        )

        batting = df[
            df["batting_team"] == team
        ]

        bowling = df[
            df["bowling_team"] == team
        ]

        total_legal_balls = int(
            bowling["is_legal_ball"].sum()
        )

        rows.append({

            "team": team,

            "matches": len(team_matches),

            "wins": wins,

            "losses": losses,

            "win_pct": (
                wins
                / len(valid_matches)
                * 100
                if len(valid_matches)
                else 0
            ),

            # Team score includes extras.
            "total_runs": int(
                batting["total_runs"].sum()
            ),

            "runs_per_match": (
                batting["total_runs"].sum()
                / len(team_matches)
                if len(team_matches)
                else 0
            ),

            # Only bowler-credited wickets.
            "wickets": int(
                bowling["bowler_wicket"].sum()
            ),

            "wickets_per_match": (
                bowling["bowler_wicket"].sum()
                / len(team_matches)
                if len(team_matches)
                else 0
            ),

            "bowling_economy": (
                bowling["bowler_runs"].sum()
                /
                (total_legal_balls / 6)
                if total_legal_balls
                else 0
            ),

            # Team phase scoring uses total team runs,
            # including extras.
            "powerplay_runs": int(
                batting.loc[
                    batting["phase"] == "Powerplay",
                    "total_runs"
                ].sum()
            ),

            "middle_over_runs": int(
                batting.loc[
                    batting["phase"] == "Middle overs",
                    "total_runs"
                ].sum()
            ),

            "death_over_runs": int(
                batting.loc[
                    batting["phase"] == "Death overs",
                    "total_runs"
                ].sum()
            ),

            "toss_wins": toss_wins,

            "wins_after_toss": wins_after_toss,

            "toss_to_win_conversion_pct": (
                wins_after_toss
                / toss_wins
                * 100
                if toss_wins
                else np.nan
            ),

            "toss_win_rate_pct": (
                toss_wins
                / len(team_matches)
                * 100
                if len(team_matches)
                else 0
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# TOSS IMPACT
# ============================================================

def build_toss_impact(matches):
    """
    Build descriptive toss-impact statistics.

    Important:
    These numbers show association, not causation.
    """

    rows = []

    teams = sorted(
        set(matches["team1"])
        |
        set(matches["team2"])
    )

    for team in teams:

        if team == "Unknown":
            continue

        team_matches = matches[
            (matches["team1"] == team)
            |
            (matches["team2"] == team)
        ]

        # Exclude matches without a recorded winner.
        valid = team_matches[
            team_matches["winner"].ne("Unknown")
        ]

        toss_wins = valid[
            valid["toss_winner"] == team
        ]

        toss_losses = valid[
            valid["toss_winner"] != team
        ]

        wins_after_toss = int(
            (toss_wins["winner"] == team)
            .sum()
        )

        wins_without_toss = int(
            (toss_losses["winner"] == team)
            .sum()
        )

        toss_win_pct = (
            wins_after_toss
            / len(toss_wins)
            * 100
            if len(toss_wins)
            else np.nan
        )

        no_toss_win_pct = (
            wins_without_toss
            / len(toss_losses)
            * 100
            if len(toss_losses)
            else np.nan
        )

        rows.append({

            "team": team,

            "matches": len(valid),

            "toss_wins": len(toss_wins),

            "wins_after_toss": wins_after_toss,

            "toss_to_win_conversion_pct": toss_win_pct,

            "matches_without_toss_win": len(toss_losses),

            "wins_without_toss": wins_without_toss,

            "win_pct_without_toss": no_toss_win_pct,

            "toss_win_rate_pct": (
                len(toss_wins)
                / len(valid)
                * 100
                if len(valid)
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# TEAM SEASON STATISTICS
# ============================================================

def build_team_season_stats(matches):
    """
    Build team performance by season.
    """

    rows = []

    for season, group in matches.groupby("season"):

        teams = sorted(
            set(group["team1"])
            |
            set(group["team2"])
        )

        for team in teams:

            if team == "Unknown":
                continue

            team_matches = group[
                (group["team1"] == team)
                |
                (group["team2"] == team)
            ]

            valid_matches = team_matches[
                team_matches["winner"].ne("Unknown")
            ]

            wins = int(
                (valid_matches["winner"] == team)
                .sum()
            )

            rows.append({

                "season": season,

                "team": team,

                "matches": len(team_matches),

                "wins": wins,

                "win_pct": (
                    wins
                    / len(valid_matches)
                    * 100
                    if len(valid_matches)
                    else 0
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# NORMALIZATION FOR PLAYER IMPACT
# ============================================================

def _scale_0_100(series):
    """
    Robust percentile-based scaling.

    Uses the 5th and 95th percentiles to reduce the effect
    of extreme outliers.
    """

    s = pd.to_numeric(
        series,
        errors="coerce"
    )

    lo = s.quantile(0.05)
    hi = s.quantile(0.95)

    if (
        pd.isna(lo)
        or pd.isna(hi)
        or hi <= lo
    ):

        return pd.Series(
            50.0,
            index=s.index
        )

    return (
        (s - lo)
        /
        (hi - lo)
        * 100
    ).clip(
        0,
        100
    )


# ============================================================
# PLAYER IMPACT SCORE
# ============================================================

def build_player_impact(
    batting,
    bowling,
    fielding
):
    """
    Build a transparent composite player-impact score.

    This is a project-defined analytical score.
    It is not an official cricket ranking.
    """

    players = pd.DataFrame({
        "player": sorted(
            set(batting["player"])
            |
            set(bowling["player"])
            |
            set(fielding["player"])
        )
    })

    players = (
        players
        .merge(
            batting,
            on="player",
            how="left"
        )
        .merge(
            bowling.add_suffix("_bowl"),
            left_on="player",
            right_on="player_bowl",
            how="left"
        )
        .drop(
            columns=[
                "player_bowl"
            ]
        )
        .merge(
            fielding,
            on="player",
            how="left"
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # BATTING SCORE
    # --------------------------------------------------------

    batting_raw = (
        players["runs"] * 0.45
        +
        players["strike_rate"] * 0.35
        +
        players["fours"] * 0.05
        +
        players["sixes"] * 0.15
    )

    players["batting_score"] = _scale_0_100(
        batting_raw
    )

    # --------------------------------------------------------
    # BOWLING SCORE
    # --------------------------------------------------------

    economy = (
        players["economy_bowl"]
        .replace(0, np.nan)
    )

    economy_reference = (
        economy.median()
    )

    if pd.isna(
        economy_reference
    ):

        economy_reference = 8.0

    # Lower economy is better.
    economy_component = (
        economy_reference
        -
        economy
    ).fillna(0)

    bowling_raw = (
        players["wickets_bowl"] * 0.55
        +
        players["dot_ball_pct_bowl"] * 0.20
        +
        economy_component * 0.25
    )

    players["bowling_score"] = _scale_0_100(
        bowling_raw
    )

    # --------------------------------------------------------
    # FIELDING SCORE
    # --------------------------------------------------------

    players["fielding_score"] = _scale_0_100(
        players["fielding_impact"]
    )

    # --------------------------------------------------------
    # FINAL IMPACT SCORE
    # --------------------------------------------------------

    players["impact_score"] = (
        players["batting_score"] * 0.45
        +
        players["bowling_score"] * 0.40
        +
        players["fielding_score"] * 0.15
    ).round(2)

    return players[
        [
            "player",
            "impact_score",
            "batting_score",
            "bowling_score",
            "fielding_score",
        ]
    ]


# ============================================================
# BUILD EVERYTHING
# ============================================================

def build_all():

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    matches, deliveries = load_data()

    # --------------------------------------------------------
    # Apply common cricket rules
    # --------------------------------------------------------

    deliveries = add_delivery_rules(
        deliveries
    )

    # --------------------------------------------------------
    # Match context
    # --------------------------------------------------------

    match_context = matches[
        [
            "id",
            "season",
            "date",
            "venue",
            "winner",
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
        ]
    ].copy()

    # --------------------------------------------------------
    # Merge delivery data with match context
    # --------------------------------------------------------

    b = deliveries.merge(
        match_context,
        left_on="match_id",
        right_on="id",
        how="left",
        suffixes=(
            "",
            "_match"
        ),
    )

    # --------------------------------------------------------
    # Add innings phases
    # --------------------------------------------------------

    b = add_phase(b)

    # --------------------------------------------------------
    # Build analytical tables
    # --------------------------------------------------------

    batting = build_batting_stats(b)

    bowling = build_bowling_stats(b)

    matchups = build_matchups(b)

    most_runs_against = (
        build_most_runs_against(
            matchups
        )
    )

    batting_phase = (
        batting_phase_table(
            b,
            "batter"
        )
    )

    bowling_phase = (
        bowling_phase_table(
            b,
            "bowler"
        )
    )

    venue = build_venue_stats(
        b,
        matches
    )

    fielding = build_fielding_stats(
        b
    )

    team = build_team_stats(
        b,
        matches
    )

    toss_impact = build_toss_impact(
        matches
    )

    team_season = (
        build_team_season_stats(
            matches
        )
    )

    player_impact = build_player_impact(
        batting,
        bowling,
        fielding
    )

    # --------------------------------------------------------
    # Validation summary
    # --------------------------------------------------------

    summary = {

        "matches": int(
            len(matches)
        ),

        "deliveries": int(
            len(deliveries)
        ),

        "players_batting": int(
            batting["player"].nunique()
        ),

        "players_bowling": int(
            bowling["player"].nunique()
        ),

        "teams": int(
            team["team"].nunique()
        ),

        "venues": int(
            venue["venue"].nunique()
        ),

        "legal_balls": int(
            b["is_legal_ball"].sum()
        ),

        "bowler_wickets": int(
            b["bowler_wicket"].sum()
        ),

        "batting_dismissals": int(
            b["batting_dismissal"].sum()
        ),
    }

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    outputs = {

        "batting_stats.csv":
            batting,

        "bowling_stats.csv":
            bowling,

        "batting_phase_stats.csv":
            batting_phase,

        "bowling_phase_stats.csv":
            bowling_phase,

        "matchup_stats.csv":
            matchups,

        "most_runs_against.csv":
            most_runs_against,

        "venue_stats.csv":
            venue,

        "fielding_stats.csv":
            fielding,

        "player_impact.csv":
            player_impact,

        "team_stats.csv":
            team,

        "team_season_stats.csv":
            team_season,

        "toss_impact.csv":
            toss_impact,

        "deliveries_enriched.csv":
            b,
    }

    for name, frame in outputs.items():

        frame.to_csv(
            OUT / name,
            index=False
        )

    # Summary JSON.
    pd.Series(
        summary
    ).to_json(
        OUT / "summary.json",
        indent=2
    )

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = build_all()

    print(
        "analytics tables rebuilt"
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )