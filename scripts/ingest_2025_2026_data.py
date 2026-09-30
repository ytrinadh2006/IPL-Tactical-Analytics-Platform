"""
Ingest IPL 2025 and 2026 match & delivery data from Cricsheet official dataset into data/raw/matches.csv and data/raw/deliveries.csv.
"""
from pathlib import Path
import json
import shutil
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MATCHES_CSV = RAW_DIR / "matches.csv"
DELIVERIES_CSV = RAW_DIR / "deliveries.csv"
BACKUP_DIR = RAW_DIR / "backup_2024"

ZIP_PATH = Path(r"C:\Users\trinadh18\.gemini\antigravity\scratch\cricsheet_ipl\ipl_json.zip")

def backup_existing():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not (BACKUP_DIR / "matches.csv").exists() and MATCHES_CSV.exists():
        shutil.copy(MATCHES_CSV, BACKUP_DIR / "matches.csv")
    if not (BACKUP_DIR / "deliveries.csv").exists() and DELIVERIES_CSV.exists():
        shutil.copy(DELIVERIES_CSV, BACKUP_DIR / "deliveries.csv")
    print("Backup verified.")

def ingest():
    backup_existing()
    
    matches_df = pd.read_csv(MATCHES_CSV)
    deliveries_df = pd.read_csv(DELIVERIES_CSV)
    existing_match_ids = set(matches_df["id"])
    print(f"Existing matches: {len(existing_match_ids):,}")
    print(f"Existing deliveries: {len(deliveries_df):,}")

    new_matches = []
    new_deliveries = []

    with zipfile.ZipFile(ZIP_PATH) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            mid = int(name.replace(".json", ""))
            if mid in existing_match_ids:
                continue

            d = json.loads(z.read(name).decode("utf-8"))
            info = d.get("info", {})
            season = str(info.get("season"))
            if season not in ["2025", "2026"]:
                continue

            event = info.get("event", {})
            stage = event.get("stage")
            match_type = stage if stage else "League"

            pom = info.get("player_of_match", [])
            player_of_match = pom[0] if pom else None

            t1, t2 = info["teams"][0], info["teams"][1]
            toss = info.get("toss", {})
            toss_winner = toss.get("winner")
            toss_decision = toss.get("decision")

            outcome = info.get("outcome", {})
            winner = outcome.get("winner")

            result = None
            result_margin = None
            if "by" in outcome:
                if "runs" in outcome["by"]:
                    result = "runs"
                    result_margin = outcome["by"]["runs"]
                elif "wickets" in outcome["by"]:
                    result = "wickets"
                    result_margin = outcome["by"]["wickets"]
            elif "result" in outcome:
                result = outcome["result"]
                result_margin = None

            target_runs = None
            target_overs = None
            for inn in d.get("innings", []):
                if "target" in inn:
                    target_runs = inn["target"].get("runs")
                    target_overs = inn["target"].get("overs")
                    break

            is_super = "Y" if ("eliminator" in outcome or any(inn.get("super_over") for inn in d.get("innings", []))) else "N"
            method = "D/L" if outcome.get("method") in ["D/L", "Duckworth Lewis"] else None

            umpires = info.get("officials", {}).get("umpires", [])
            u1 = umpires[0] if len(umpires) > 0 else None
            u2 = umpires[1] if len(umpires) > 1 else None

            new_matches.append({
                "id": mid,
                "season": season,
                "city": info.get("city"),
                "date": info.get("dates", [""])[0],
                "match_type": match_type,
                "player_of_match": player_of_match,
                "venue": info.get("venue"),
                "team1": t1,
                "team2": t2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                "result": result,
                "result_margin": result_margin,
                "target_runs": target_runs,
                "target_overs": target_overs,
                "super_over": is_super,
                "method": method,
                "umpire1": u1,
                "umpire2": u2
            })

            teams = info["teams"]
            for inn_idx, inn in enumerate(d.get("innings", []), 1):
                bat_team = inn.get("team")
                bowl_team = [t for t in teams if t != bat_team][0] if len(teams) == 2 else ""
                for over_obj in inn.get("overs", []):
                    over_num = over_obj["over"]
                    for ball_idx, deliv in enumerate(over_obj["deliveries"], 1):
                        batter = deliv.get("batter")
                        bowler = deliv.get("bowler")
                        non_striker = deliv.get("non_striker")

                        runs = deliv.get("runs", {})
                        batsman_runs = runs.get("batter", 0)
                        extra_runs = runs.get("extras", 0)
                        total_runs = runs.get("total", 0)

                        extras = deliv.get("extras", {})
                        extras_type = list(extras.keys())[0] if extras else None

                        wickets = deliv.get("wickets", [])
                        if wickets:
                            w = wickets[0]
                            is_wicket = 1
                            player_dismissed = w.get("player_out")
                            dismissal_kind = w.get("kind")
                            fielders = w.get("fielders", [])
                            fielder = fielders[0].get("name") if fielders else None
                        else:
                            is_wicket = 0
                            player_dismissed = None
                            dismissal_kind = None
                            fielder = None

                        new_deliveries.append({
                            "match_id": mid,
                            "inning": inn_idx,
                            "batting_team": bat_team,
                            "bowling_team": bowl_team,
                            "over": over_num,
                            "ball": ball_idx,
                            "batter": batter,
                            "bowler": bowler,
                            "non_striker": non_striker,
                            "batsman_runs": batsman_runs,
                            "extra_runs": extra_runs,
                            "total_runs": total_runs,
                            "extras_type": extras_type,
                            "is_wicket": is_wicket,
                            "player_dismissed": player_dismissed,
                            "dismissal_kind": dismissal_kind,
                            "fielder": fielder
                        })

    new_m_df = pd.DataFrame(new_matches)
    new_d_df = pd.DataFrame(new_deliveries)

    combined_m_df = pd.concat([matches_df, new_m_df], ignore_index=True)
    combined_m_df["date"] = pd.to_datetime(combined_m_df["date"], errors="coerce")
    combined_m_df = combined_m_df.sort_values(["date", "id"]).reset_index(drop=True)
    combined_m_df["date"] = combined_m_df["date"].dt.strftime("%Y-%m-%d")

    combined_d_df = pd.concat([deliveries_df, new_d_df], ignore_index=True)

    combined_m_df.to_csv(MATCHES_CSV, index=False)
    combined_d_df.to_csv(DELIVERIES_CSV, index=False)

    print(f"Updated matches: {len(combined_m_df):,} (added {len(new_m_df)} matches)")
    print(f"Updated deliveries: {len(combined_d_df):,} (added {len(new_d_df)} deliveries)")
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
