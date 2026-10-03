# %% [markdown]
# # Project 1: Credit Risk & Loan Default Prediction
# Logistic-regression **scorecard** (WOE) vs **XGBoost**, evaluated with ROC-AUC and KS,
# then a **cut-off recommendation** showing defaults prevented vs good customers rejected.
#
# Dataset: "Give Me Some Credit" (Kaggle) -> save `cs-training.csv` into `data/`.
# If the file is missing, a synthetic dataset with the same columns is generated
# so the whole pipeline still runs end to end.
#
# In VS Code: install the Python + Jupyter extensions, then click "Run Cell"
# above any `# %%` block, or just run `python credit_risk.py`.

# %%
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # keeps the script runnable even without xgboost
    HAS_XGB = False

warnings.filterwarnings("ignore")
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # plain `python credit_risk.py`: save PNGs, no blocking windows

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_FILE = BASE / "data" / "cs-training.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
TARGET = "SeriousDlqin2yrs"

# %% [markdown]
# ## 1. Load data
# Target `SeriousDlqin2yrs` = 1 if the customer became 90+ days delinquent within 2 years.

# %%
def make_synthetic(n=30000, seed=SEED):
    """Synthetic stand-in with the same schema as Give Me Some Credit."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 85, n)
    util = np.clip(rng.beta(1.2, 3, n) * rng.choice([1, 1, 1, 1.5], n), 0, 3)
    income = np.exp(rng.normal(8.6, 0.6, n))
    debt_ratio = np.clip(rng.lognormal(-1, 1, n), 0, 5000)
    d30 = rng.poisson(0.4, n)
    d60 = rng.poisson(0.15, n)
    d90 = rng.poisson(0.15, n)
    lines = rng.poisson(8, n)
    re = rng.poisson(1, n)
    deps = rng.poisson(0.8, n).astype(float)
    logit = (-4.0 + 2.2 * util + 0.7 * d30 + 0.9 * d60 + 1.3 * d90
             - 0.03 * (age - 45) + 0.25 * np.minimum(debt_ratio, 3)
             - 0.4 * np.log(income / 5000))
    p = 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        TARGET: (rng.random(n) < p).astype(int),
        "RevolvingUtilizationOfUnsecuredLines": util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": d30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": lines,
        "NumberOfTimes90DaysLate": d90,
        "NumberRealEstateLoansOrLines": re,
        "NumberOfTime60-89DaysPastDueNotWorse": d60,
        "NumberOfDependents": deps,
    })
    df.loc[rng.random(n) < 0.2, "MonthlyIncome"] = np.nan   # missing income
    df.loc[rng.random(n) < 0.03, "NumberOfDependents"] = np.nan
    return df


if DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE, index_col=0)
    print("Loaded real data:", df.shape)
else:
    df = make_synthetic()
    print(f"[!] {DATA_FILE} not found -> using SYNTHETIC data {df.shape}. "
          "Download the real file for your portfolio.")

print(df[TARGET].value_counts(normalize=True).rename("share"))
df.head()

# %% [markdown]
# ## 2. Clean the data
# Typical banking-data issues handled here:
# * Missing income / dependents -> median fill **plus** a missing-flag (missingness is itself informative)
# * Delinquency counts contain placeholder codes 96/98 -> cap counts at 5
# * Utilisation and debt ratio have absurd outliers -> cap at the 99th percentile

# %%
def clean(d):
    d = d.copy()
    d["income_missing"] = d["MonthlyIncome"].isna().astype(int)
    d["MonthlyIncome"] = d["MonthlyIncome"].fillna(d["MonthlyIncome"].median())
    d["NumberOfDependents"] = d["NumberOfDependents"].fillna(0)
    for c in ["NumberOfTime30-59DaysPastDueNotWorse",
              "NumberOfTime60-89DaysPastDueNotWorse",
              "NumberOfTimes90DaysLate"]:
        d[c] = d[c].clip(upper=5)           # 96 / 98 are data-entry codes
    for c in ["NumberOfDependents", "NumberRealEstateLoansOrLines"]:
        d[c] = d[c].clip(upper=5)
    for c in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
        d[c] = d[c].clip(upper=d[c].quantile(0.99))
    return d


df = clean(df)
FEATURES = [c for c in df.columns if c != TARGET]

# Stratified split keeps the (rare) default rate identical in train and test
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.3, stratify=df[TARGET], random_state=SEED)
print("Train:", X_train.shape, " Test:", X_test.shape,
      " Default rate train/test: %.3f / %.3f" % (y_train.mean(), y_test.mean()))

# %% [markdown]
# ## 3. Weight of Evidence (WOE) binning
# A scorecard does not feed raw numbers into the model. Each variable is cut into bins and every bin
# is replaced by its **WOE** = ln(%good / %bad). Benefits: handles outliers, makes the relationship
# monotonic/linear in log-odds, and gives interpretable points.
# **Information Value (IV)** tells us how predictive each variable is.

# %%
N_BINS = 6
EPS = 0.5  # smoothing so empty bins don't give infinite WOE


def make_edges(x, n_bins=N_BINS):
    """Bin edges for one feature.
    * Few distinct values (counts like 0,1,2,3...) -> one bin per value.
      (Quantile binning would collapse these skewed columns into ONE bin and
      throw away our strongest predictors - the late-payment counts.)
    * Otherwise -> quantile bins with equal population."""
    uniq = np.sort(x.unique())
    if len(uniq) <= 8:
        edges = np.concatenate([[-np.inf], (uniq[:-1] + uniq[1:]) / 2, [np.inf]])
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    return edges


def fit_woe(x, y):
    """Return bin edges and a WOE / IV table for one feature."""
    edges = make_edges(x)
    bins = pd.cut(x, edges)
    t = pd.DataFrame({"bin": bins, "bad": y}).groupby("bin", observed=False)["bad"].agg(["sum", "count"])
    t["bad"] = t["sum"]
    t["good"] = t["count"] - t["bad"]
    t["woe"] = np.log(((t["good"] + EPS) / (t["good"].sum() + EPS)) /
                      ((t["bad"] + EPS) / (t["bad"].sum() + EPS)))
    t["iv"] = ((t["good"] / t["good"].sum()) - (t["bad"] / t["bad"].sum())) * t["woe"]
    return edges, t


woe_maps, iv_rows = {}, []
for col in FEATURES:
    edges, table = fit_woe(X_train[col], y_train)
    woe_maps[col] = (edges, table["woe"].values)
    iv_rows.append((col, table["iv"].sum()))

iv_df = pd.DataFrame(iv_rows, columns=["feature", "IV"]).sort_values("IV", ascending=False)
print(iv_df.to_string(index=False))
print("\nRule of thumb: IV <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium | >0.3 strong")


def apply_woe(X):
    out = pd.DataFrame(index=X.index)
    for col, (edges, woes) in woe_maps.items():
        idx = pd.cut(X[col], edges, labels=False).astype(int)
        out[col] = woes[idx]
    return out


# Keep variables with IV >= 0.02 (drop the useless ones)
KEEP = iv_df.loc[iv_df["IV"] >= 0.02, "feature"].tolist()
W_train, W_test = apply_woe(X_train)[KEEP], apply_woe(X_test)[KEEP]
print("\nVariables kept:", KEEP)

# %% [markdown]
# ## 4. Logistic regression scorecard

# %%
logit = LogisticRegression(max_iter=1000, class_weight=None)
logit.fit(W_train, y_train)
p_lr = logit.predict_proba(W_test)[:, 1]

coef = pd.Series(logit.coef_[0], index=KEEP).sort_values()
print("Coefficients (all should be NEGATIVE: higher WOE = safer = lower default prob):")
print(coef.round(3))

# %% [markdown]
# ### Convert probabilities to a credit score
# Industry convention: **600 points = 50:1 good:bad odds, +20 points = double the odds (PDO=20)**.
# Score = Offset + Factor * ln(odds_good), with Factor = PDO / ln 2.

# %%
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def prob_to_score(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return OFFSET + FACTOR * np.log((1 - p) / p)


score_lr = prob_to_score(p_lr)
print("Score range on test set: %.0f - %.0f (mean %.0f)" % (score_lr.min(), score_lr.max(), score_lr.mean()))

# Points per variable bin (what a credit officer actually reads)
n_vars = len(KEEP)
rows = []
for i, col in enumerate(KEEP):
    edges, woes = woe_maps[col]
    for j, w in enumerate(woes):
        pts = -(w * logit.coef_[0][i] + logit.intercept_[0] / n_vars) * FACTOR + OFFSET / n_vars
        rows.append((col, f"({edges[j]:.2f}, {edges[j + 1]:.2f}]", round(pts, 1)))
points_table = pd.DataFrame(rows, columns=["variable", "bin", "points"])
points_table.to_csv(OUT / "scorecard_points.csv", index=False)
print(points_table.head(12).to_string(index=False))

# %% [markdown]
# ## 5. Challenger model: XGBoost
# Trained on the raw (cleaned) features. Tree models capture non-linearities and interactions
# automatically, so no WOE step is needed.

# %%
if HAS_XGB:
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    gbm = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="auc", random_state=SEED,
                        scale_pos_weight=pos_weight)
    MODEL_NAME = "XGBoost"
else:
    print("[!] xgboost not installed -> using sklearn HistGradientBoosting (same idea). "
          "Run `pip install xgboost` to use the real thing.")
    gbm = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                         random_state=SEED)
    MODEL_NAME = "GradientBoosting"

gbm.fit(X_train, y_train)
p_gb = gbm.predict_proba(X_test)[:, 1]

# %% [markdown]
# ## 6. Compare: ROC-AUC and KS
# * **ROC-AUC**: probability a random bad is ranked riskier than a random good (0.5 = coin flip).
# * **KS**: max gap between cumulative-bad and cumulative-good curves. Banks use it daily;
#   >0.3 is acceptable, >0.4 good for retail credit.

# %%
def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


results = pd.DataFrame({
    "model": ["Logistic Scorecard", MODEL_NAME],
    "ROC-AUC": [roc_auc_score(y_test, p_lr), roc_auc_score(y_test, p_gb)],
    "KS": [ks_stat(y_test, p_lr), ks_stat(y_test, p_gb)],
}).round(4)
print(results.to_string(index=False))
results.to_csv(OUT / "model_comparison.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, p in [("Logistic Scorecard", p_lr), (MODEL_NAME, p_gb)]:
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", alpha=.4)
ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
ax[0].legend()

# KS plot for the scorecard
order = np.argsort(score_lr)
ys = y_test.values[order]
cum_bad = np.cumsum(ys) / ys.sum()
cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
ax[1].plot(cum_bad, label="Cumulative bad")
ax[1].plot(cum_good, label="Cumulative good")
k = np.argmax(cum_bad - cum_good)
ax[1].vlines(k, cum_good[k], cum_bad[k], color="red", label=f"KS = {cum_bad[k] - cum_good[k]:.3f}")
ax[1].set(title="KS plot (scorecard, low score -> high score)", xlabel="Applicants sorted by score")
ax[1].legend()
plt.tight_layout()
plt.savefig(OUT / "roc_ks.png", dpi=130)
plt.show()

# %% [markdown]
# ## 7. Choose an approval cut-off
# Applicants **at or above** the cut-off score are approved. For each cut-off we compute:
# * **Defaults prevented** = bad customers we reject (they would have defaulted)
# * **Good customers rejected** = lost business
# * **Expected profit** using simple, explicit economics (change them to match your bank):
#   a good loan earns `PROFIT_GOOD`, a default loses `LOSS_BAD` (loss given default x exposure).

# %%
PROFIT_GOOD = 1.0   # interest margin earned on a repaid loan (units)
LOSS_BAD = 8.0      # loss on a defaulted loan -> a default costs 8 good loans' profit


def cutoff_table(score, y, cutoffs):
    y = np.asarray(y)
    n_bad, n_good = y.sum(), (1 - y).sum()
    rows = []
    for c in cutoffs:
        approve = score >= c
        rej = ~approve
        bad_rej, good_rej = int((y[rej] == 1).sum()), int((y[rej] == 0).sum())
        bad_app, good_app = int((y[approve] == 1).sum()), int((y[approve] == 0).sum())
        rows.append({
            "cutoff": c,
            "approval_rate": approve.mean(),
            "bad_rate_approved": bad_app / max(approve.sum(), 1),
            "defaults_prevented": bad_rej,
            "pct_defaults_prevented": bad_rej / n_bad,
            "good_rejected": good_rej,
            "pct_good_rejected": good_rej / n_good,
            "profit": good_app * PROFIT_GOOD - bad_app * LOSS_BAD,
        })
    return pd.DataFrame(rows)


cutoffs = np.arange(np.floor(score_lr.min() / 10) * 10, np.ceil(score_lr.max() / 10) * 10, 10)
ct = cutoff_table(score_lr, y_test, cutoffs)
best = ct.loc[ct["profit"].idxmax()]

# Baseline = approve everyone
base_profit = (y_test == 0).sum() * PROFIT_GOOD - (y_test == 1).sum() * LOSS_BAD

pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
print(ct.iloc[::2].to_string(index=False))
ct.to_csv(OUT / "cutoff_table.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ct["cutoff"], ct["pct_defaults_prevented"], label="% defaults prevented")
ax[0].plot(ct["cutoff"], ct["pct_good_rejected"], label="% good customers rejected")
ax[0].axvline(best["cutoff"], color="red", ls="--", label=f"Recommended = {best['cutoff']:.0f}")
ax[0].set(title="Trade-off by cut-off", xlabel="Score cut-off", ylabel="Share")
ax[0].legend()
ax[1].plot(ct["cutoff"], ct["profit"], color="green")
ax[1].axhline(base_profit, color="grey", ls=":", label="Approve everyone")
ax[1].axvline(best["cutoff"], color="red", ls="--")
ax[1].set(title="Expected profit by cut-off", xlabel="Score cut-off", ylabel="Profit (units)")
ax[1].legend()
plt.tight_layout()
plt.savefig(OUT / "cutoff_tradeoff.png", dpi=130)
plt.show()

# %% [markdown]
# ## 8. Business recommendation

# %%
n_bad = int(y_test.sum())
n_good = int((1 - y_test).sum())
print("=" * 70)
print(f"RECOMMENDED CUT-OFF SCORE: {best['cutoff']:.0f}  (approve if score >= {best['cutoff']:.0f})")
print("=" * 70)
print(f"Approval rate:               {best['approval_rate']:.1%}")
print(f"Bad rate among approved:     {best['bad_rate_approved']:.2%}  "
      f"(vs {y_test.mean():.2%} if we approve everyone)")
print(f"Defaults prevented:          {best['defaults_prevented']:.0f} of {n_bad} "
      f"({best['pct_defaults_prevented']:.1%})")
print(f"Good customers rejected:     {best['good_rejected']:.0f} of {n_good} "
      f"({best['pct_good_rejected']:.1%})")
print(f"Profit vs approve-all:       {best['profit']:.0f} vs {base_profit:.0f} "
      f"(uplift {best['profit'] - base_profit:+.0f})")
print("\nInterpretation: for every default avoided we turn away about "
      f"{best['good_rejected'] / max(best['defaults_prevented'], 1):.1f} good customers.")
print("The cut-off moves with the economics: if a default costs more (LOSS_BAD up) the bank "
      "should be stricter; if growth matters more, relax it.")

# Model choice commentary
auc_gap = results.loc[1, "ROC-AUC"] - results.loc[0, "ROC-AUC"]
verdict = (f"{MODEL_NAME} is better by {auc_gap:.4f} AUC" if auc_gap > 0.002
           else "the two models perform about the same")
print(f"\nModel choice: {verdict}. In regulated lending the scorecard is usually deployed "
      "anyway because every decision can be explained to the customer and the regulator; "
      "the boosted model is kept as a challenger / benchmark. If the gap were large "
      "(>0.02-0.03 AUC) that would justify exploring non-linear effects in the scorecard.")

# %% [markdown]
# ## 9. Stability check (PSI) - bonus
# Banks monitor whether the score distribution drifts between development and new data.
# PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 recalibrate.

# %%
def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


score_train = prob_to_score(logit.predict_proba(W_train)[:, 1])
print(f"PSI train vs test score distribution: {psi(score_train, score_lr):.4f}")
