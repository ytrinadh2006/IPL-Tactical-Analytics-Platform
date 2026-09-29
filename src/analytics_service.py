import os
import pandas as pd


class AnalyticsService:

    TEAM_MAP = {
        "Delhi Daredevils": "Delhi Capitals",
        "Delhi Capitals": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Punjab Kings": "Punjab Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
        "Rising Pune Supergiant": "Rising Pune Supergiants",
        "Rising Pune Supergiants": "Rising Pune Supergiants",
    }

    def __init__(self):

        # Project structure:
        #
        # IPL_Intelligence_Tactical_Analytics_Platform/
        # ├── data/
        # │   ├── raw/
        # │   │   ├── matches.csv
        # │   │   └── deliveries.csv
        # │   └── processed/
        # │       ├── batting_stats.csv
        # │       ├── bowling_stats.csv
        # │       └── ...
        # └── src/
        #     └── analytics_service.py

        self.base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.data_dir = os.path.join(
            self.base_dir,
            "data"
        )

        self.raw_dir = os.path.join(
            self.data_dir,
            "raw"
        )

        self.processed_dir = os.path.join(
            self.data_dir,
            "processed"
        )

        # Raw data
        self.matches = self.load_raw("matches.csv")
        self.deliveries = self.load_raw("deliveries.csv")

        # Processed analytics
        self.batting = self.load_processed(
            "batting_stats.csv"
        )

        self.bowling = self.load_processed(
            "bowling_stats.csv"
        )

        self.batting_phase = self.load_processed(
            "batting_phase_stats.csv"
        )

        self.bowling_phase = self.load_processed(
            "bowling_phase_stats.csv"
        )

        self.matchups = self.load_processed(
            "matchup_stats.csv"
        )

        self.runs_against = self.load_processed(
            "most_runs_against.csv"
        )

        self.venues = self.load_processed(
            "venue_stats.csv"
        )

        self.fielding = self.load_processed(
            "fielding_stats.csv"
        )

        self.impact = self.load_processed(
            "player_impact.csv"
        )

        self.teams = self.load_processed(
            "team_stats.csv"
        )

        self.team_seasons = self.load_processed(
            "team_season_stats.csv"
        )

        self.toss_impact = self.load_processed(
            "toss_impact.csv"
        )

    # =========================================================
    # DATA LOADING
    # =========================================================

    def load_raw(self, filename):

        path = os.path.join(
            self.raw_dir,
            filename
        )

        if not os.path.exists(path):
            return pd.DataFrame()

        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def load_processed(self, filename):

        path = os.path.join(
            self.processed_dir,
            filename
        )

        if not os.path.exists(path):
            return pd.DataFrame()

        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    # =========================================================
    # HELPERS
    # =========================================================

    def normalize_team(self, team):

        if team is None:
            return team

        team = str(team).strip()

        return self.TEAM_MAP.get(
            team,
            team
        )

    def records(self, df, limit=None):

        if df is None or df.empty:
            return []

        result = df.copy()

        if limit is not None:
            result = result.head(limit)

        result = result.where(
            pd.notnull(result),
            None
        )

        return result.to_dict(
            orient="records"
        )

    def column(self, df, names):

        if df is None or df.empty:
            return None

        lookup = {
            str(col).lower().strip(): col
            for col in df.columns
        }

        for name in names:

            key = str(name).lower().strip()

            if key in lookup:
                return lookup[key]

        return None

    def player_filter(self, df, player):

        if df is None or df.empty:
            return pd.DataFrame()

        col = self.column(
            df,
            [
                "player",
                "player_name",
                "batter",
                "bowler",
                "fielder"
            ]
        )

        if not col:
            return pd.DataFrame()

        return df[
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            str(player)
            .strip()
            .lower()
        ].copy()

    # =========================================================
    # SUMMARY
    # =========================================================

    def get_summary(self):

        matches = len(self.matches)

        deliveries = len(self.deliveries)

        seasons = 0

        if not self.matches.empty:

            season_col = self.column(
                self.matches,
                ["season"]
            )

            if season_col:

                seasons = (
                    self.matches[season_col]
                    .dropna()
                    .nunique()
                )

        teams = self.get_teams()["count"]

        venues = self.get_venues()["count"]

        players = set()

        # Count players from raw deliveries
        if not self.deliveries.empty:

            for name in [
                "batter",
                "bowler"
            ]:

                col = self.column(
                    self.deliveries,
                    [name]
                )

                if col:

                    values = (
                        self.deliveries[col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    players.update(
                        x for x in values
                        if x and x.lower() != "nan"
                    )

        # Fallback to processed data
        if not players:

            for df in [
                self.batting,
                self.bowling,
                self.impact
            ]:

                col = self.column(
                    df,
                    [
                        "player",
                        "player_name",
                        "batter",
                        "bowler"
                    ]
                )

                if col:

                    values = (
                        df[col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    players.update(
                        x for x in values
                        if x and x.lower() != "nan"
                    )

        return {
            "success": True,
            "matches": int(matches),
            "deliveries": int(deliveries),
            "seasons": int(seasons),
            "teams": int(teams),
            "venues": int(venues),
            "players": int(len(players))
        }

    # =========================================================
    # SEASONS
    # =========================================================

    def get_seasons(self):

        if self.matches.empty:

            return {
                "success": True,
                "count": 0,
                "seasons": []
            }

        col = self.column(
            self.matches,
            ["season"]
        )

        if not col:

            return {
                "success": True,
                "count": 0,
                "seasons": []
            }

        seasons = (
            pd.to_numeric(
                self.matches[col],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        seasons.sort()

        return {
            "success": True,
            "count": len(seasons),
            "seasons": seasons
        }

    # =========================================================
    # TEAMS
    # =========================================================

    def get_teams(self):

        teams = set()

        # Processed team statistics
        if not self.teams.empty:

            col = self.column(
                self.teams,
                [
                    "team",
                    "team_name"
                ]
            )

            if col:

                teams.update(
                    self.normalize_team(x)
                    for x in
                    self.teams[col]
                    .dropna()
                    .astype(str)
                )

        # Raw matches
        if not self.matches.empty:

            for name in [
                "team1",
                "team2",
                "winner"
            ]:

                col = self.column(
                    self.matches,
                    [name]
                )

                if col:

                    teams.update(
                        self.normalize_team(x)
                        for x in
                        self.matches[col]
                        .dropna()
                        .astype(str)
                    )

        teams = sorted(
            x for x in teams
            if x
            and str(x).lower() != "nan"
        )

        return {
            "success": True,
            "count": len(teams),
            "teams": teams
        }

    def get_team(self, team_name):

        team_name = self.normalize_team(
            team_name
        )

        if self.teams.empty:
            return None

        col = self.column(
            self.teams,
            [
                "team",
                "team_name"
            ]
        )

        if not col:
            return None

        df = self.teams.copy()

        df["_normalized_team"] = (
            df[col]
            .astype(str)
            .map(self.normalize_team)
        )

        rows = df[
            df["_normalized_team"]
            .str.lower()
            ==
            team_name.lower()
        ]

        if rows.empty:
            return None

        result = rows.iloc[0].to_dict()

        result.pop(
            "_normalized_team",
            None
        )

        return result

    def get_team_seasons(self, team_name):

        team_name = self.normalize_team(
            team_name
        )

        if self.team_seasons.empty:

            return {
                "success": True,
                "team": team_name,
                "count": 0,
                "seasons": []
            }

        col = self.column(
            self.team_seasons,
            [
                "team",
                "team_name"
            ]
        )

        if not col:

            return {
                "success": True,
                "team": team_name,
                "count": 0,
                "seasons": []
            }

        df = self.team_seasons.copy()

        df["_normalized_team"] = (
            df[col]
            .astype(str)
            .map(self.normalize_team)
        )

        rows = df[
            df["_normalized_team"]
            .str.lower()
            ==
            team_name.lower()
        ].copy()

        rows.drop(
            columns=["_normalized_team"],
            inplace=True
        )

        return {
            "success": True,
            "team": team_name,
            "count": len(rows),
            "seasons": self.records(rows)
        }

    # =========================================================
    # TEAM COMPARISON
    # =========================================================

    def compare_teams(
        self,
        team_a,
        team_b
    ):

        team_a = self.normalize_team(
            team_a
        )

        team_b = self.normalize_team(
            team_b
        )

        data_a = self.get_team(
            team_a
        )

        data_b = self.get_team(
            team_b
        )

        if data_a is None or data_b is None:
            return None

        h2h_matches = 0

        team_a_wins = 0

        team_b_wins = 0

        if not self.matches.empty:

            team1_col = self.column(
                self.matches,
                ["team1"]
            )

            team2_col = self.column(
                self.matches,
                ["team2"]
            )

            winner_col = self.column(
                self.matches,
                ["winner"]
            )

            if team1_col and team2_col:

                df = self.matches.copy()

                df["_team1"] = (
                    df[team1_col]
                    .astype(str)
                    .map(self.normalize_team)
                )

                df["_team2"] = (
                    df[team2_col]
                    .astype(str)
                    .map(self.normalize_team)
                )

                rows = df[
                    (
                        (df["_team1"] == team_a)
                        &
                        (df["_team2"] == team_b)
                    )
                    |
                    (
                        (df["_team1"] == team_b)
                        &
                        (df["_team2"] == team_a)
                    )
                ]

                h2h_matches = len(rows)

                if winner_col:

                    winners = (
                        rows[winner_col]
                        .astype(str)
                        .map(self.normalize_team)
                    )

                    team_a_wins = int(
                        (winners == team_a).sum()
                    )

                    team_b_wins = int(
                        (winners == team_b).sum()
                    )

        return {
            "team_a": data_a,
            "team_b": data_b,
            "head_to_head": {
                "matches": h2h_matches,
                "team_a_wins": team_a_wins,
                "team_b_wins": team_b_wins
            }
        }

    # =========================================================
    # PLAYERS
    # =========================================================

    def get_players(
        self,
        limit=100
    ):

        limit = max(
            1,
            min(int(limit), 500)
        )

        if self.impact.empty:

            return {
                "success": True,
                "count": 0,
                "players": []
            }

        df = self.impact.copy()

        player_col = self.column(
            df,
            [
                "player",
                "player_name"
            ]
        )

        impact_col = self.column(
            df,
            [
                "impact_score",
                "impact",
                "player_impact"
            ]
        )

        if not player_col:

            return {
                "success": True,
                "count": 0,
                "players": []
            }

        if impact_col:

            df[impact_col] = pd.to_numeric(
                df[impact_col],
                errors="coerce"
            ).fillna(0)

            df = df.sort_values(
                impact_col,
                ascending=False
            )

        players = []

        for _, row in df.head(limit).iterrows():

            score = 0

            if impact_col:

                try:
                    score = round(
                        float(row[impact_col]),
                        2
                    )
                except Exception:
                    score = 0

            players.append({
                "player": str(
                    row[player_col]
                ),
                "impact_score": score
            })

        return {
            "success": True,
            "count": len(players),
            "players": players
        }

    def get_player(
        self,
        player_name
    ):

        result = {
            "success": True,
            "player": player_name,
            "impact": None,
            "batting": [],
            "bowling": [],
            "fielding": []
        }

        batting_rows = self.player_filter(
            self.batting,
            player_name
        )

        bowling_rows = self.player_filter(
            self.bowling,
            player_name
        )

        fielding_rows = self.player_filter(
            self.fielding,
            player_name
        )

        impact_rows = self.player_filter(
            self.impact,
            player_name
        )

        result["batting"] = self.records(
            batting_rows
        )

        result["bowling"] = self.records(
            bowling_rows
        )

        result["fielding"] = self.records(
            fielding_rows
        )

        if not impact_rows.empty:

            result["impact"] = self.records(
                impact_rows,
                1
            )[0]

        return result

    # =========================================================
    # VENUES
    # =========================================================

    def get_venues(self):

        if self.venues.empty:

            return {
                "success": True,
                "count": 0,
                "venues": []
            }

        col = self.column(
            self.venues,
            [
                "venue",
                "venue_name"
            ]
        )

        if not col:

            return {
                "success": True,
                "count": 0,
                "venues": []
            }

        venues = sorted(
            x.strip()
            for x in
            self.venues[col]
            .dropna()
            .astype(str)
            .unique()
            if x.strip()
        )

        return {
            "success": True,
            "count": len(venues),
            "venues": venues
        }

    def get_venue(
        self,
        venue_name
    ):

        if self.venues.empty:
            return None

        col = self.column(
            self.venues,
            [
                "venue",
                "venue_name"
            ]
        )

        if not col:
            return None

        rows = self.venues[
            self.venues[col]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            str(venue_name)
            .strip()
            .lower()
        ]

        if rows.empty:
            return None

        return self.records(
            rows,
            1
        )[0]

    # =========================================================
    # MATCHUPS
    # =========================================================

    def get_matchup(
        self,
        batter,
        bowler=None,
        limit=100
    ):

        if self.matchups.empty:

            return {
                "success": True,
                "count": 0,
                "matchups": []
            }

        df = self.matchups.copy()

        batter_col = self.column(
            df,
            [
                "batter",
                "batsman",
                "striker",
                "player"
            ]
        )

        bowler_col = self.column(
            df,
            ["bowler"]
        )

        if batter_col:

            df = df[
                df[batter_col]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                str(batter)
                .strip()
                .lower()
            ]

        if bowler and bowler_col:

            df = df[
                df[bowler_col]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                str(bowler)
                .strip()
                .lower()
            ]

        return {
            "success": True,
            "count": min(
                len(df),
                limit
            ),
            "matchups": self.records(
                df,
                limit
            )
        }

    def search_matchups(
        self,
        query="",
        limit=100
    ):

        if self.matchups.empty:

            return {
                "success": True,
                "count": 0,
                "matchups": []
            }

        df = self.matchups.copy()

        if query:

            mask = pd.Series(
                False,
                index=df.index
            )

            for name in [
                "batter",
                "batsman",
                "striker",
                "bowler",
                "player"
            ]:

                col = self.column(
                    df,
                    [name]
                )

                if col:

                    mask |= (
                        df[col]
                        .astype(str)
                        .str.contains(
                            str(query),
                            case=False,
                            na=False
                        )
                    )

            df = df[mask]

        return {
            "success": True,
            "count": min(
                len(df),
                limit
            ),
            "matchups": self.records(
                df,
                limit
            )
        }

    # =========================================================
    # FIELDING
    # =========================================================

    def get_fielding(
        self,
        limit=100
    ):

        return {
            "success": True,
            "count": min(
                len(self.fielding),
                limit
            ),
            "fielding": self.records(
                self.fielding,
                limit
            )
        }

    # =========================================================
    # MOST RUNS AGAINST
    # =========================================================

    def get_most_runs_against(
        self,
        limit=100
    ):

        return {
            "success": True,
            "count": min(
                len(self.runs_against),
                limit
            ),
            "records": self.records(
                self.runs_against,
                limit
            )
        }

    # =========================================================
    # TOP BATTERS
    # =========================================================

    def get_top_batters(
        self,
        limit=10
    ):

        if self.batting.empty:

            return {
                "success": True,
                "count": 0,
                "batters": []
            }

        df = self.batting.copy()

        runs_col = self.column(
            df,
            [
                "runs",
                "total_runs"
            ]
        )

        if runs_col:

            df[runs_col] = pd.to_numeric(
                df[runs_col],
                errors="coerce"
            ).fillna(0)

            df = df.sort_values(
                runs_col,
                ascending=False
            )

        return {
            "success": True,
            "count": min(
                len(df),
                limit
            ),
            "batters": self.records(
                df,
                limit
            )
        }

    # =========================================================
    # TOP BOWLERS
    # =========================================================

    def get_top_bowlers(
        self,
        limit=10
    ):

        if self.bowling.empty:

            return {
                "success": True,
                "count": 0,
                "bowlers": []
            }

        df = self.bowling.copy()

        wickets_col = self.column(
            df,
            [
                "wickets",
                "total_wickets"
            ]
        )

        if wickets_col:

            df[wickets_col] = pd.to_numeric(
                df[wickets_col],
                errors="coerce"
            ).fillna(0)

            df = df.sort_values(
                wickets_col,
                ascending=False
            )

        return {
            "success": True,
            "count": min(
                len(df),
                limit
            ),
            "bowlers": self.records(
                df,
                limit
            )
        }

    # =========================================================
    # PLAYER IMPACT
    # =========================================================

    def get_player_impact(
        self,
        player_name=None,
        limit=100
    ):

        if self.impact.empty:

            return {
                "success": True,
                "count": 0,
                "players": []
            }

        df = self.impact.copy()

        if player_name:

            df = self.player_filter(
                df,
                player_name
            )

        impact_col = self.column(
            df,
            [
                "impact_score",
                "impact",
                "player_impact"
            ]
        )

        if impact_col:

            df[impact_col] = pd.to_numeric(
                df[impact_col],
                errors="coerce"
            ).fillna(0)

            df = df.sort_values(
                impact_col,
                ascending=False
            )

        return {
            "success": True,
            "count": min(
                len(df),
                limit
            ),
            "players": self.records(
                df,
                limit
            )
        }


# =============================================================
# SINGLE SERVICE INSTANCE
# =============================================================

analytics_service = AnalyticsService()