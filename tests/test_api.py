from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_summary_2026():
    r = client.get("/api/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["matches"] == 1243
    assert data["deliveries"] == 295732
    assert data["venues"] >= 60
    assert data["teams"] >= 10


def test_teams():
    r = client.get("/api/teams")
    assert r.status_code == 200
    teams = r.json()
    assert len(teams) >= 10
    names = [t.get("team") for t in teams]
    assert "Mumbai Indians" in names
    assert "Chennai Super Kings" in names
    assert "Kolkata Knight Riders" in names


def test_team_detail_and_seasons():
    r = client.get("/api/teams/Mumbai%20Indians")
    assert r.status_code == 200
    data = r.json()
    assert "profile" in data
    assert "season_history" in data
    seasons = [s["season"] for s in data["season_history"]]
    assert "2025" in seasons
    assert "2026" in seasons


def test_team_analytics():
    r = client.get("/api/team-analytics")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_team_comparison():
    r = client.get("/api/teams/compare/Mumbai%20Indians/Chennai%20Super%20Kings")
    assert r.status_code == 200
    data = r.json()
    assert "team_a" in data
    assert "team_b" in data
    assert data["team_a"]["overall"]["team"] == "Mumbai Indians"
    assert data["team_b"]["overall"]["team"] == "Chennai Super Kings"


def test_players_and_impact():
    r = client.get("/api/players?limit=10")
    assert r.status_code == 200
    players = r.json()
    assert len(players) == 10
    assert "impact_score" in players[0]


def test_player_detail_batting_bowling_fielding():
    r = client.get("/api/players/V%20Kohli")
    assert r.status_code == 200
    data = r.json()
    assert "batting" in data
    assert "bowling" in data
    assert "fielding" in data
    assert "impact" in data
    assert data["batting"].get("runs", 0) > 8000


def test_venues():
    r = client.get("/api/venues")
    assert r.status_code == 200
    assert len(r.json()) >= 50


def test_matchup():
    r = client.get("/api/matchups/V%20Kohli/JJ%20Bumrah")
    assert r.status_code == 200
    data = r.json()
    assert data["batter"] == "V Kohli"
    assert data["bowler"] == "JJ Bumrah"
    assert data["balls"] > 0


def test_batting_bowling_fielding_endpoints():
    r_bat = client.get("/api/batting?limit=5")
    assert r_bat.status_code == 200
    assert len(r_bat.json()) == 5

    r_bowl = client.get("/api/bowling?limit=5")
    assert r_bowl.status_code == 200
    assert len(r_bowl.json()) == 5

    r_fld = client.get("/api/fielding?limit=5")
    assert r_fld.status_code == 200
    assert len(r_fld.json()) == 5


def test_prediction_features():
    r = client.post("/api/predict", json={
        "team1_win_rate": 0.55,
        "team2_win_rate": 0.45,
        "venue_chase_rate": 0.52,
        "toss_team1": 1,
        "field_first": 1
    })
    assert r.status_code == 200
    data = r.json()
    assert "team1_win_probability" in data
    assert "team2_win_probability" in data
    assert data["team1_win_probability"] > 0


def test_prediction_teams():
    r = client.post("/api/predict", json={
        "team_a": "Chennai Super Kings",
        "team_b": "Mumbai Indians"
    })
    assert r.status_code == 200
    data = r.json()
    assert "team1_win_probability" in data
    assert "team2_win_probability" in data
    assert data["team1_win_probability"] > 0
