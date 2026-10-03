# Project 1 - Credit Risk & Loan Default Prediction

**Question:** Which loan applicants should the bank approve, and where should the score cut-off be?

## Data
"Give Me Some Credit" (Kaggle): https://www.kaggle.com/c/GiveMeSomeCredit/data
Download `cs-training.csv` and put it in `data/`. (150,000 customers, ~6.7% default rate.)
German Credit or Lending Club also work, but you must rename columns / adapt `clean()`.

## Run
```bash
pip install -r requirements.txt
python credit_risk.py
```
Outputs (in `outputs/`): `model_comparison.csv`, `scorecard_points.csv`, `cutoff_table.csv`, `roc_ks.png`, `cutoff_tradeoff.png`.

## What the script does (and why)
| Step | What | Why a bank cares |
|---|---|---|
| 1. Load | Read data, show class imbalance | Defaults are rare, so accuracy is a misleading metric |
| 2. Clean | Median-fill income + missing flag, cap outliers, fix 96/98 codes | Real bank data is dirty; missingness can itself signal risk |
| 3. WOE / IV | Bin each variable, replace bins with Weight of Evidence, rank variables by Information Value | Standard scorecard practice: interpretable, robust to outliers, monotonic |
| 4. Logistic scorecard | Logistic regression on WOE features, converted to points (600 = 50:1 odds, PDO 20) | Regulators and credit officers need explainable decisions |
| 5. XGBoost | Gradient-boosted trees on raw features | Challenger model: how much accuracy do we lose by staying simple? |
| 6. Compare | ROC-AUC and KS on a held-out test set | AUC = ranking power; KS = max separation of good vs bad |
| 7. Cut-off | For each score: approval rate, defaults prevented, good customers rejected, expected profit | Turns a model into a decision |
| 8. Recommendation | Profit-maximising cut-off printed in plain words | This is what you tell the credit committee |
| 9. PSI (bonus) | Population Stability Index | Shows you know models are monitored after launch |

## Key concepts to be able to explain
* **WOE** = ln(%good / %bad) in a bin. **IV** = sum over bins of (%good - %bad) x WOE. IV < 0.02 useless, 0.1-0.3 medium, > 0.3 strong.
* **ROC-AUC**: chance a random bad customer gets a riskier score than a random good one. 0.5 = random.
* **KS**: largest gap between cumulative bad and cumulative good curves. 0.3+ acceptable, 0.4+ good for retail.
* **Cut-off trade-off**: raising the cut-off prevents more defaults but rejects more good customers. The right point depends on the **economics** (`PROFIT_GOOD`, `LOSS_BAD` in section 7). Change them and see the cut-off move.
* **Why a scorecard still wins in practice**: similar AUC, but fully explainable (adverse-action reasons, regulation).

## Typical results on the real dataset (for sanity checking)
Scorecard AUC ~0.85-0.86, XGBoost ~0.86-0.87, KS ~0.55-0.60. If yours is far off, check the cleaning step.

## Likely interview questions
1. Why WOE instead of raw features? 2. What is KS and what is a good value? 3. How did you choose the cut-off?
4. Why not use accuracy? 5. How would the cut-off change if the bank's loss per default doubled?
6. What is data leakage / PSI / reject inference? (Be honest: this project uses only accepted-loan outcomes - reject inference is a known limitation.)

## Ideas to extend (optional)
Monotonic binning with `optbinning`, SHAP values for XGBoost, probability calibration, a time-based train/test split.
