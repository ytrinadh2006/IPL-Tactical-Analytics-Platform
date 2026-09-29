from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
import os
ROOT=Path(__file__).resolve().parents[1]
url=os.getenv("DATABASE_URL","postgresql+psycopg2://ipl_user:ipl_password@localhost:5432/ipl_intelligence")
engine=create_engine(url)
with engine.begin() as con:
    con.execute(text((ROOT/"sql/schema.sql").read_text()))
m=pd.read_csv(ROOT/"data/raw/matches.csv"); d=pd.read_csv(ROOT/"data/raw/deliveries.csv")
m.to_sql("matches",engine,if_exists="append",index=False,method="multi",chunksize=500)
d.to_sql("deliveries",engine,if_exists="append",index=False,method="multi",chunksize=2000)
print("database seeded")
