from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    database_url:str="postgresql+psycopg2://ipl_user:ipl_password@localhost:5432/ipl_intelligence"
    class Config: env_file=".env"
settings=Settings()
