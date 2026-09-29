from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd, numpy as np, joblib

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/"data/processed"
router=APIRouter()

def read(name):
    return pd.read_csv(DATA/name)

def clean(obj):
    if isinstance(obj, dict): return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj, list): return [clean(v) for v in obj]
    if pd.isna(obj): return None
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return float(obj)
    return obj

@router.get("/summary")
def summary():
    import json
    return json.loads((DATA/"summary.json").read_text())

@router.get("/teams")
def teams(): return clean(read("team_stats.csv").to_dict("records"))

@router.get("/team-analytics")
def team_analytics():
    d=pd.read_csv(ROOT/"data/raw/deliveries.csv")
    batting=d.groupby("batting_team").agg(runs=("batsman_runs","sum"),balls=("match_id","size"),boundaries=("batsman_runs",lambda x:((x==4)|(x==6)).sum())).reset_index().rename(columns={"batting_team":"team"})
    batting["strike_rate"]=np.where(batting.balls,batting.runs/batting.balls*100,0)
    bowling=d.groupby("bowling_team").agg(runs_conceded=("total_runs","sum"),balls=("match_id","size"),wickets=("is_wicket","sum"),dot_balls=("total_runs",lambda x:(x==0).sum())).reset_index().rename(columns={"bowling_team":"team"})
    bowling["economy"]=np.where(bowling.balls,bowling.runs_conceded/(bowling.balls/6),0); bowling["dot_ball_pct"]=np.where(bowling.balls,bowling.dot_balls/bowling.balls*100,0)
    out=batting.merge(bowling,on="team",how="outer").fillna(0); return clean(out.to_dict("records"))

@router.get("/teams/{team}")
def team(team:str):
    df=read("team_stats.csv"); row=df[df.team.str.lower()==team.lower()]
    if row.empty: raise HTTPException(404,"Team not found")
    season=read("team_season_stats.csv"); season=season[season.team.str.lower()==team.lower()]
    return clean({"profile":row.iloc[0].to_dict(),"season_history":season.to_dict("records")})

@router.get("/teams/compare/{team_a}/{team_b}")
def compare(team_a:str,team_b:str):
    df=read("team_stats.csv"); a=df[df.team.str.lower()==team_a.lower()]; b=df[df.team.str.lower()==team_b.lower()]
    if a.empty or b.empty: raise HTTPException(404,"One or both teams not found")
    d=pd.read_csv(ROOT/"data/raw/deliveries.csv")
    ba=d.groupby("batting_team").agg(runs=("batsman_runs","sum"),balls=("match_id","size"),fours=("batsman_runs",lambda x:(x==4).sum()),sixes=("batsman_runs",lambda x:(x==6).sum())).reset_index().rename(columns={"batting_team":"team"})
    ba["strike_rate"]=np.where(ba.balls,ba.runs/ba.balls*100,0)
    bo=d.groupby("bowling_team").agg(runs_conceded=("total_runs","sum"),balls=("match_id","size"),wickets=("is_wicket","sum"),dot_balls=("total_runs",lambda x:(x==0).sum())).reset_index().rename(columns={"bowling_team":"team"})
    bo["economy"]=np.where(bo.balls,bo.runs_conceded/(bo.balls/6),0); bo["dot_ball_pct"]=np.where(bo.balls,bo.dot_balls/bo.balls*100,0)
    def details(name):
        row=df[df.team.str.lower()==name.lower()].iloc[0].to_dict(); x=ba[ba.team.str.lower()==name.lower()]; y=bo[bo.team.str.lower()==name.lower()]
        return {"overall":row,"batting":x.iloc[0].to_dict() if not x.empty else {},"bowling":y.iloc[0].to_dict() if not y.empty else {}}
    return clean({"team_a":details(team_a),"team_b":details(team_b)})

@router.get("/players")
def players(limit:int=100):
    df=read("player_impact.csv").sort_values("impact_score",ascending=False).head(limit)
    return clean(df.to_dict("records"))

@router.get("/players/{player}")
def player(player:str):
    bat=read("batting_stats.csv"); bowl=read("bowling_stats.csv"); fld=read("fielding_stats.csv"); imp=read("player_impact.csv")
    def one(df,col):
        x=df[df[col].astype(str).str.lower()==player.lower()]
        return x.iloc[0].to_dict() if not x.empty else {}
    return clean({"batting":one(bat,"player"),"bowling":one(bowl,"player"),"fielding":one(fld,"player"),"impact":one(imp,"player")})

@router.get("/matchups/{batter}/{bowler}")
def matchup(batter:str,bowler:str):
    df=read("matchup_stats.csv"); x=df[(df.batter.str.lower()==batter.lower())&(df.bowler.str.lower()==bowler.lower())]
    if x.empty: raise HTTPException(404,"Matchup not found")
    return clean(x.iloc[0].to_dict())

@router.get("/venues")
def venues(): return clean(read("venue_stats.csv").sort_values("matches",ascending=False).to_dict("records"))

@router.get("/venues/{venue}")
def venue(venue:str):
    df=read("venue_stats.csv"); x=df[df.venue.str.lower()==venue.lower()]
    if x.empty: raise HTTPException(404,"Venue not found")
    return clean(x.iloc[0].to_dict())

class PredictionRequest(BaseModel):
    team1_win_rate: float
    team2_win_rate: float
    venue_chase_rate: float=0.5
    toss_team1: int=1
    field_first: int=1

@router.post("/predict")
def predict(req:PredictionRequest):
    pack=joblib.load(ROOT/"models/match_winner_model.joblib")
    X=pd.DataFrame([[req.team1_win_rate,req.team2_win_rate,req.venue_chase_rate,req.toss_team1,req.field_first]],columns=pack["features"])
    p=float(pack["model"].predict_proba(X)[0,1])
    return {"team1_win_probability":round(p*100,2),"team2_win_probability":round((1-p)*100,2),"note":"Historical model estimate, not a guarantee."}

@router.get("/model/metrics")
def model_metrics():
    import json
    return json.loads((DATA/"model_metrics.json").read_text())

@router.get("/player-recommendations")
def recommendations(role:str="all", venue:str="", opponent:str="", limit:int=5):
    imp=read("player_impact.csv"); bat=read("batting_stats.csv"); bowl=read("bowling_stats.csv")
    out=imp.copy()
    if role.lower()=="batsman": out=out.merge(bat[["player","runs","strike_rate","average"]],on="player",how="left")
    elif role.lower()=="bowler": out=out.merge(bowl[["player","wickets","economy","dot_ball_pct"]],on="player",how="left")
    out=out.sort_values("impact_score",ascending=False).head(limit)
    return clean(out.to_dict("records"))

@router.get("/strengths/{player}")
def strengths(player:str):
    mu=read("matchup_stats.csv"); x=mu[mu.batter.str.lower()==player.lower()].sort_values("strike_rate",ascending=False).head(5); y=mu[mu.bowler.str.lower()==player.lower()].sort_values("economy",ascending=True).head(5) if "economy" in mu.columns else pd.DataFrame()
    return clean({"batting_matchups":x.to_dict("records"),"bowling_matchups":y.to_dict("records")})

@router.post("/explain")
def explain(req:PredictionRequest):
    pack=joblib.load(ROOT/"models/match_winner_model.joblib")
    X=pd.DataFrame([[req.team1_win_rate,req.team2_win_rate,req.venue_chase_rate,req.toss_team1,req.field_first]],columns=pack["features"])
    try:
        import shap
        explainer=shap.TreeExplainer(pack["model"])
        values=explainer.shap_values(X)
        vals=values[1][0] if isinstance(values,list) else values[0]
        return {"features":[{"name":n,"impact":round(float(v),5)} for n,v in zip(pack["features"],vals)]}
    except Exception:
        importance=pack["model"].feature_importances_
        return {"features":[{"name":n,"impact":round(float(v),5)} for n,v in zip(pack["features"],importance)],"note":"Feature importance fallback used when SHAP is unavailable."}
