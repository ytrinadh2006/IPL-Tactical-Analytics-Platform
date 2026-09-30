"""
Generate the 5 key analysis snapshot charts for the IPL Analytics Platform.
Saves high-res charts to both output/graphs/ and frontend/public/.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT_DIRS = [ROOT / "output" / "graphs", ROOT / "frontend" / "public"]

TEAM_COLORS = {
    "Mumbai Indians": "#004BA0",
    "Chennai Super Kings": "#F9CD05",
    "Kolkata Knight Riders": "#3A225D",
    "Royal Challengers Bengaluru": "#EC1C24",
    "Delhi Capitals": "#0078BC",
    "Punjab Kings": "#DD1F2D",
    "Rajasthan Royals": "#EA1A85",
    "Sunrisers Hyderabad": "#F26522",
    "Gujarat Titans": "#1B2133",
    "Lucknow Super Giants": "#0057E7",
}

def save_to_all(fig, filename):
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / filename
        fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")

def plot_all():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
    })

    # 1. Team Wins
    team_df = pd.read_csv(DATA / "team_stats.csv").sort_values("wins", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [TEAM_COLORS.get(t, "#4a5568") for t in team_df["team"]]
    bars = ax.barh(team_df["team"][::-1], team_df["wins"][::-1], color=colors[::-1], edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_title("Top IPL Teams by Total Wins (2008 - 2026)", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Total Match Wins", fontsize=11, weight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", weight="bold", fontsize=10)
    save_to_all(fig, "team_wins.png")

    # 2. Season Matches
    matches_raw = pd.read_csv(ROOT / "data" / "raw" / "matches.csv")
    season_counts = matches_raw.groupby("season")["id"].count().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(season_counts["season"], season_counts["id"], color="#2b6cb0", edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_title("IPL Matches by Season (2008 - 2026)", fontsize=14, weight="bold", pad=12)
    ax.set_ylabel("Matches Played", fontsize=11, weight="bold")
    ax.set_xticks(range(len(season_counts)))
    ax.set_xticklabels(season_counts["season"], rotation=40, ha="right", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{int(h)}", ha="center", weight="bold", fontsize=9)
    save_to_all(fig, "season_matches.png")

    # 3. Top Run Scorers
    bat_df = pd.read_csv(DATA / "batting_stats.csv").sort_values("runs", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(bat_df["player"], bat_df["runs"], color="#d69e2e", edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_title("All-Time Top Run Scorers (2008 - 2026)", fontsize=14, weight="bold", pad=12)
    ax.set_ylabel("Total Runs", fontsize=11, weight="bold")
    ax.set_xticks(range(len(bat_df)))
    ax.set_xticklabels(bat_df["player"], rotation=30, ha="right", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 80, f"{int(h):,}", ha="center", weight="bold", fontsize=8.5)
    save_to_all(fig, "top_run_scorers.png")

    # 4. Top Wicket Takers
    bowl_df = pd.read_csv(DATA / "bowling_stats.csv").sort_values("wickets", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(bowl_df["player"], bowl_df["wickets"], color="#805ad5", edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_title("All-Time Top Wicket Takers (2008 - 2026)", fontsize=14, weight="bold", pad=12)
    ax.set_ylabel("Total Wickets", fontsize=11, weight="bold")
    ax.set_xticks(range(len(bowl_df)))
    ax.set_xticklabels(bowl_df["player"], rotation=30, ha="right", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{int(h)}", ha="center", weight="bold", fontsize=9)
    save_to_all(fig, "top_wicket_takers.png")

    # 5. Player Impact
    impact_df = pd.read_csv(DATA / "player_impact.csv").sort_values("impact_score", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(impact_df["player"], impact_df["impact_score"], color="#319795", edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_title("Top 10 All-Time Player Impact Scores (2008 - 2026)", fontsize=14, weight="bold", pad=12)
    ax.set_ylabel("Composite Impact Score", fontsize=11, weight="bold")
    ax.set_xticks(range(len(impact_df)))
    ax.set_xticklabels(impact_df["player"], rotation=30, ha="right", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", weight="bold", fontsize=9)
    save_to_all(fig, "player_impact.png")

    print("All 5 snapshot plots successfully regenerated.")

if __name__ == "__main__":
    plot_all()
