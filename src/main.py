from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.prediction_service import PredictionService
from src.analytics_service import analytics_service


app = FastAPI(
    title="IPL Intelligence & Tactical Analytics API",
    description="Backend API for IPL analytics and ML predictions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the prediction service once when the API starts.
try:
    prediction_service = PredictionService()
except Exception as error:
    prediction_service = None
    startup_error = str(error)


class PredictionRequest(BaseModel):
    team_a: str
    team_b: str
    venue: Optional[str] = None
    season: Optional[str] = None


@app.get("/")
def root():
    return {
        "message": "IPL Intelligence & Tactical Analytics API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "prediction_service": (
            "ready"
            if prediction_service is not None
            else "unavailable"
        ),
    }


@app.post("/api/predict")
def predict_match(request: PredictionRequest):

    if prediction_service is None:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction service failed: {startup_error}",
        )

    try:
        result = prediction_service.predict(
            team_a=request.team_a,
            team_b=request.team_b,
            venue=request.venue,
            season=request.season,
        )

        return {
            "success": True,
            "prediction": result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
# ============================================================
# ANALYTICS ENDPOINTS
# ============================================================

@app.get("/api/teams")
def get_teams():
    try:
        return analytics_service.get_teams()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.get("/api/teams/{team_name}")
def get_team(team_name: str):
    try:
        data = analytics_service.get_team(team_name)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Team not found: {team_name}",
            )

        return {
            "success": True,
            "team": team_name,
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/api/teams/{team_name}/seasons")
def get_team_seasons(team_name: str):
    try:
        data = analytics_service.get_team_seasons(team_name)

        return {
            "success": True,
            "team": team_name,
            "count": len(data),
            "data": data,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

@app.get("/api/teams/compare/{team_a}/{team_b}")
def compare_teams(team_a: str, team_b: str):
    try:
        data = analytics_service.compare_teams(team_a, team_b)

        if not data:
            raise HTTPException(
                status_code=404,
                detail="Unable to compare the selected teams.",
            )

        return {
            "success": True,
            "team_a": team_a,
            "team_b": team_b,
            "data": data,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

@app.get("/api/players")
def get_players(limit: int = 100):
    try:
        limit = max(1, min(limit, 500))

        players = analytics_service.get_players(limit)

        return players

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

@app.get("/api/players/{player_name}")
def get_player(player_name: str):
    try:
        data = analytics_service.get_player(player_name)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Player not found: {player_name}",
            )

        return {
            "success": True,
            "player": player_name,
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

@app.get("/api/venues")
def get_venues():
    try:
        return analytics_service.get_venues()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.get("/api/venues/{venue_name}")
def get_venue(venue_name: str):
    try:
        data = analytics_service.get_venue(venue_name)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Venue not found: {venue_name}",
            )

        return {
            "success": True,
            "venue": venue_name,
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/api/matchups/{batsman}/{bowler}")
def get_matchup(batsman: str, bowler: str):
    try:
        data = analytics_service.get_matchup(
            batsman,
            bowler,
        )

        if not data:
            raise HTTPException(
                status_code=404,
                detail="Matchup not found",
            )

        return {
            "success": True,
            "batsman": batsman,
            "bowler": bowler,
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/api/analytics/top-batters")
def get_top_batters(limit: int = 10):
    try:
        limit = max(1, min(limit, 100))

        data = analytics_service.get_top_batters(limit)

        return {
            "success": True,
            "count": len(data),
            "data": data,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/api/analytics/top-bowlers")
def get_top_bowlers(limit: int = 10):
    try:
        limit = max(1, min(limit, 100))

        data = analytics_service.get_top_bowlers(limit)

        return {
            "success": True,
            "count": len(data),
            "data": data,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/api/analytics/fielding")
def get_fielding(player: Optional[str] = None, limit: int = 100):
    try:
        limit = max(1, min(limit, 500))

        data = analytics_service.get_fielding(
            
            limit=limit,
        )

        return {
            "success": True,
            "count": len(data),
            "data": data,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
@app.get("/api/summary")
def get_summary():
    try:
        return analytics_service.get_summary()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
