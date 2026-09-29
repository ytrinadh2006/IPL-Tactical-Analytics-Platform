from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Rising Pune Supergiants": "Rising Pune Supergiants",
}

match = pd.read_csv(ROOT / "data/raw/matches.csv")
delivery = pd.read_csv(ROOT / "data/raw/deliveries.csv")

for column in ["team1", "team2", "winner", "toss_winner"]:
    if column in match.columns:
        match[column] = match[column].replace(TEAM_NAME_MAP)

match["date"] = pd.to_datetime(match["date"], errors="coerce")
for col in ["team1","team2","winner","toss_winner","venue"]:
    match[col] = match[col].fillna("Unknown").astype(str).str.strip()
for col in ["batsman_runs","total_runs","is_wicket"]:
    delivery[col] = pd.to_numeric(delivery[col], errors="coerce").fillna(0)

joined = delivery.merge(match[["id","season","date","venue","winner","team1","team2","toss_winner","toss_decision"]], left_on="match_id", right_on="id", how="left")
joined["phase"] = pd.cut(joined["over"], [-1,5,14,19], labels=["Powerplay","Middle overs","Death overs"])
out = ROOT / "data/processed"
out.mkdir(parents=True, exist_ok=True)
joined.to_csv(out / "deliveries_enriched.csv", index=False)
print(f"saved {len(joined):,} enriched delivery rows")
