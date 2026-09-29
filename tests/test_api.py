from fastapi.testclient import TestClient
from backend.app.main import app

client=TestClient(app)

def test_root():
    r=client.get("/")
    assert r.status_code==200
    assert r.json()["status"]=="running"

def test_summary():
    r=client.get("/api/summary")
    assert r.status_code==200
    assert r.json()["matches"]>0

def test_teams():
    r=client.get("/api/teams")
    assert r.status_code==200
    assert len(r.json())>0
