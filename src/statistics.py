from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
ROOT=Path(__file__).resolve().parents[1]
m=pd.read_csv(ROOT/"data/raw/matches.csv")
valid=m[m.winner.notna()].copy()
# Toss relationship
ct=pd.crosstab(valid.toss_winner==valid.winner, valid.toss_decision)
chi2,p,_,_=chi2_contingency(ct) if ct.shape[0]>1 and ct.shape[1]>1 else (0,1,0,0)
# First innings scores by match outcome is not used as a causal claim; this compares distributions only.
result={"toss_winner_equals_match_winner_chi_square":round(float(chi2),4),"p_value":round(float(p),6),"note":"Association is not the same as causation."}
(ROOT/"data/processed/statistical_summary.json").write_text(__import__('json').dumps(result,indent=2))
print(result)
