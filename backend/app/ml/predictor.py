from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parents[3]
MODEL=joblib.load(ROOT/"models/match_winner_model.joblib")
