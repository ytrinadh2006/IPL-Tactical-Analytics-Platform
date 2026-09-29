from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
matches = pd.read_csv(ROOT / "data/raw/matches.csv")
deliveries = pd.read_csv(ROOT / "data/raw/deliveries.csv")

required_matches = {"id","season","date","venue","team1","team2","winner","toss_winner","toss_decision"}
required_deliveries = {"match_id","inning","over","ball","batter","bowler","batsman_runs","total_runs","is_wicket"}

missing_m = required_matches - set(matches.columns)
missing_d = required_deliveries - set(deliveries.columns)

print(f"matches: {len(matches):,} rows")
print(f"deliveries: {len(deliveries):,} rows")
print("missing match columns:", sorted(missing_m))
print("missing delivery columns:", sorted(missing_d))
print("duplicate match ids:", int(matches.id.duplicated().sum()))
print("null match dates:", int(matches.date.isna().sum()))
