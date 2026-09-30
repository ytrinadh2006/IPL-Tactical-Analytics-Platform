from fastapi.testclient import TestClient
from backend.app.main import app as backend_app
from src.main import app as src_app

client_backend = TestClient(backend_app)
client_src = TestClient(src_app)

def test_root():
    r = client_backend.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"

def test_summary_2026():
    r = client_backend.get("/api/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["matches"] == 1243
    assert data["deliveries"] == 295732
    assert data["venues"] >= 60
    assert data["teams"] >= 10

def test_teams():
    r = client_backend.get("/api/teams")
    assert r.status_code == 200
    teams = r.json()
    assert len(teams) >= 10
    team_names = [t.get("team") for t in teams]
    assert "Mumbai Indians" in team_names
    assert "Chennai Super Kings" in team_names
    assert "Kolkata Knight Riders" in team_names

def test_team_analytics():
    r = client_backend.get("/api/team-analytics")
    assert r.status_code == 200
    assert len(r.json()) > 0

def test_team_detail_and_seasons():
    r = client_backend.get("/api/teams/Mumbai%20Indians")
    assert r.status_code == 200
    data = r.json()
    assert "profile" in data
    assert "season_history" in data
    seasons = [s["season"] for s in data["season_history"]]
    assert "2025" in seasons
    assert "2026" in seasons

def test_team_comparison():
    r = client_backend.get("/api/teams/compare/Mumbai%20Indians/Chennai%20Super%20Kings")
    assert r.status_code == 200
    data = r.json()
    assert "team_a" in data
    assert "team_b" in data
    assert data["team_a"]["overall"]["team"] == "Mumbai Indians"

def test_players_and_impact():
    r = client_backend.get("/api/players?limit=10")
    assert r.status_code == 200
    players = r.json()
    assert len(players) == 10
    assert "impact_score" in players[0]

def test_player_detail_batting_bowling_fielding():
    r = client_backend.get("/api/players/V%20Kohli")
    assert r.status_code == 200
    data = r.json()
    assert "batting" in data
    assert "bowling" in data
    assert "fielding" in data
    assert "impact" in data
    assert data["batting"].get("runs", 0) > 8000

def test_venues():
    r = client_backend.get("/api/venues")
    assert r.status_code == 200
    venues = r.json()
    assert len(venues) >= 50

def test_matchup():
    r = client_backend.get("/api/matchups/V%20Kohli/JJ%20Bumrah")
    assert r.status_code == 200
    data = r.json()
    assert data["batter"] == "V Kohli"
    assert data["bowler"] == "JJ Bumrah"
    assert data["balls"] > 0

def test_backend_prediction():
    r = client_backend.post("/api/predict", json={
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

def test_src_api_predict():
    r = client_src.post("/api/predict", json={
        "team_a": "Chennai Super Kings",
        "team_b": "Mumbai Indians"
    })
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "prediction" in data
    assert data["prediction"]["team_a_probability"] > 0
