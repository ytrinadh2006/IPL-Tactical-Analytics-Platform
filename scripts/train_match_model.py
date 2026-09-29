from pathlib import Path
import pandas as pd, numpy as np, joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
m=pd.read_csv(ROOT/"data/raw/matches.csv"); m["date"]=pd.to_datetime(m.date,errors="coerce")
teams=sorted(set(m.team1)|set(m.team2)); wins={t:0 for t in teams}; games={t:0 for t in teams}; vgames={}; vchase={}; rows=[]
for _,r in m.sort_values(["date","id"]).iterrows():
    t1,t2,v=r.team1,r.team2,r.venue
    f1=wins.get(t1,0)/max(games.get(t1,0),1); f2=wins.get(t2,0)/max(games.get(t2,0),1); vr=vchase.get(v,0)/max(vgames.get(v,0),1) if v in vgames else .5
    rows.append([f1,f2,vr,int(r.toss_winner==t1),int(r.toss_decision=="field"),int(r.winner==t1)])
    games[t1]=games.get(t1,0)+1; games[t2]=games.get(t2,0)+1
    if r.winner in wins: wins[r.winner]+=1
    chasing=r.toss_winner if r.toss_decision=="field" else (t2 if r.winner==t1 else t1)
    vgames[v]=vgames.get(v,0)+1; vchase[v]=vchase.get(v,0)+int(r.winner==chasing)
df=pd.DataFrame(rows,columns=["team1_win_rate","team2_win_rate","venue_chase_rate","toss_team1","field_first","target"])
X=df.drop(columns="target"); y=df.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=250,max_depth=7,min_samples_leaf=4,random_state=42,class_weight="balanced"); model.fit(Xtr,ytr)
p=model.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
metrics={"accuracy":round(accuracy_score(yte,pred),4),"roc_auc":round(roc_auc_score(yte,p),4),"features":list(X.columns)}
joblib.dump({"model":model,"features":list(X.columns),"metrics":metrics},ROOT/"models/match_winner_model.joblib")
(ROOT/"data/processed/model_metrics.json").write_text(json.dumps(metrics,indent=2)); print(metrics)
