from sqlalchemy import create_engine
import os
engine=create_engine(os.getenv("DATABASE_URL","postgresql+psycopg2://ipl_user:ipl_password@localhost:5432/ipl_intelligence"), pool_pre_ping=True)
