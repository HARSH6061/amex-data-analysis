"""American Express-style Transaction Data Analysis (synthetic data).
Pipeline: generate -> SQL load -> clean -> EDA -> segmentation -> anomalies -> churn -> forecast -> dashboard."""
import json, os, sqlite3
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/charts", exist_ok=True)
rng = np.random.default_rng(42)
sns.set_theme(style="whitegrid", palette=["#006FCF", "#00A3E0", "#7A5AF8", "#12B76A", "#F79009", "#F04438"])

# 1. DATA COLLECTION (synthetic; swap for pd.read_csv on real data) ---------------------
START, DAYS, N = pd.Timestamp("2025-10-01"), 365, 1500
CATS = {"Travel": 220, "Dining": 55, "Groceries": 70, "Shopping": 110,
        "Entertainment": 45, "Fuel": 50, "Utilities": 90, "Health": 80}
REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
cust = pd.DataFrame({
    "customer_id": np.arange(1, N + 1),
    "region": rng.choice(REGIONS, N, p=[.38, .27, .22, .08, .05]),
    "card": rng.choice(["Green", "Gold", "Platinum", "Centurion"], N, p=[.4, .35, .22, .03]),
    "spend_mult": rng.lognormal(0, .55, N), "freq": rng.uniform(.5, 1.6, N),
})
cust["stop_day"] = np.where(rng.random(N) < .22, rng.integers(215, 320, N), DAYS)
rows = []
for c in cust.itertuples():
    n = rng.poisson(28 * c.freq)
    d = rng.integers(0, c.stop_day, n)
    cat = rng.choice(list(CATS), n, p=np.array([.08, .2, .2, .17, .1, .1, .08, .07]))
    amt = rng.lognormal(0, .6, n) * np.array([CATS[k] for k in cat]) * c.spend_mult
    rows.append(pd.DataFrame({"customer_id": c.customer_id, "day": d, "category": cat, "amount": amt}))
tx = pd.concat(rows, ignore_index=True)
tx["date"] = START + pd.to_timedelta(tx.day, unit="D")
hp = np.r_[np.full(7, .01), np.full(15, .0614), np.full(2, .0079)]
tx["hour"] = rng.choice(24, len(tx), p=hp / hp.sum())
fraud = rng.random(len(tx)) < .012
tx.loc[fraud, "amount"] *= rng.uniform(8, 20, fraud.sum())
tx.loc[fraud, "hour"] = rng.integers(1, 5, fraud.sum())
tx["is_fraud"] = fraud.astype(int)
tx = tx.drop(columns="day").merge(cust[["customer_id", "region", "card"]], on="customer_id")
tx["txn_id"] = np.arange(1, len(tx) + 1)
tx.loc[rng.random(len(tx)) < .003, "amount"] = np.nan          # dirty data to clean

# 2. SQL --------------------------------------------------------------------------------
con = sqlite3.connect(":memory:")
tx.to_sql("transactions", con, index=False)
top_cat = pd.read_sql("SELECT category, ROUND(SUM(amount),0) spend, COUNT(*) n FROM transactions "
                      "WHERE amount IS NOT NULL GROUP BY category ORDER BY spend DESC", con)
region_sql = pd.read_sql("SELECT region, ROUND(SUM(amount),0) spend FROM transactions "
                         "GROUP BY region ORDER BY spend DESC", con)

# 3. CLEANING ---------------------------------------------------------------------------
tx = tx.drop_duplicates("txn_id").dropna(subset=["amount"]).copy()
tx["date"] = pd.to_datetime(tx["date"])
tx["month"] = tx["date"].dt.to_period("M").astype(str)

# 4. EDA --------------------------------------------------------------------------------
monthly = tx.groupby("month").amount.sum()
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
monthly.plot(marker="o", ax=ax[0, 0], title="Monthly spend"); ax[0, 0].tick_params(axis="x", rotation=45)
sns.barplot(data=top_cat, x="spend", y="category", ax=ax[0, 1]).set_title("Spend by category")
sns.heatmap(tx.pivot_table(index="region", columns="category", values="amount", aggfunc="mean"),
            annot=True, fmt=".0f", cmap="Blues", ax=ax[1, 0]).set_title("Avg transaction: region x category")
sns.histplot(np.log10(tx.amount), bins=40, ax=ax[1, 1]).set_title("Transaction amount (log10)")
plt.tight_layout(); plt.savefig(f"{HERE}/charts/eda_overview.png", dpi=130); plt.close()

# 5. FEATURE ENGINEERING + SEGMENTATION -------------------------------------------------
g = tx.groupby("customer_id")
feat = pd.DataFrame({"total_spend": g.amount.sum(), "avg_txn": g.amount.mean(),
                     "txn_count": g.size(), "n_categories": g.category.nunique()})
X = StandardScaler().fit_transform(np.log1p(feat[["total_spend", "avg_txn", "txn_count", "n_categories"]]))
feat["cluster"] = KMeans(4, n_init=10, random_state=42).fit_predict(X)
order = feat.groupby("cluster").total_spend.mean().sort_values().index
names = dict(zip(order, ["Budget Savers", "Steady Spenders", "Premium Explorers", "Elite Big Spenders"]))
feat["segment"] = feat.cluster.map(names)
seg = feat.groupby("segment").agg(customers=("total_spend", "size"), avg_spend=("total_spend", "mean"),
                                  avg_txns=("txn_count", "mean")).reindex(names.values())
plt.figure(figsize=(8, 5))
sns.scatterplot(data=feat, x="txn_count", y="total_spend", hue="segment", hue_order=list(names.values()), s=25)
plt.yscale("log"); plt.title("Customer segments"); plt.tight_layout()
plt.savefig(f"{HERE}/charts/segments.png", dpi=130); plt.close()

# 6. FRAUD / ANOMALY DETECTION ----------------------------------------------------------
z = tx.groupby("customer_id").amount.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
A = pd.DataFrame({"log_amt": np.log1p(tx.amount), "z": z, "hour": tx.hour,
                  "cat": tx.category.astype("category").cat.codes})
iso = IsolationForest(contamination=.015, random_state=42).fit(A)
tx["flag"] = (iso.predict(A) == -1).astype(int)
a_prec, a_rec = precision_score(tx.is_fraud, tx.flag), recall_score(tx.is_fraud, tx.flag)
anom = tx[tx.flag == 1].nlargest(8, "amount")

# 7. CHURN MODEL (features before cutoff, label = silent for last 90 days) --------------
cut = START + pd.Timedelta(days=DAYS - 90)
pre, post = tx[tx.date < cut], tx[tx.date >= cut]
gp = pre.groupby("customer_id")
F = pd.DataFrame({"total_spend": gp.amount.sum(), "avg_txn": gp.amount.mean(), "txn_count": gp.size(),
                  "n_categories": gp.category.nunique(),
                  "last_60d_txns": pre[pre.date >= cut - pd.Timedelta(days=60)].groupby("customer_id").size(),
                  "tenure_days": gp.date.agg(lambda s: (s.max() - s.min()).days)}).fillna(0)
y = (~F.index.isin(post.customer_id.unique())).astype(int)
Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=.25, random_state=42, stratify=y)
rf = RandomForestClassifier(300, random_state=42, class_weight="balanced").fit(Xtr, ytr)
p = rf.predict(Xte)
imp = pd.Series(rf.feature_importances_, F.columns).sort_values(ascending=False)

# 8. SPEND FORECAST ---------------------------------------------------------------------
m = monthly.iloc[:12]; t = np.arange(len(m)).reshape(-1, 1)
lr = LinearRegression().fit(t, m.values)
fut = pd.period_range(pd.Period(m.index[-1]) + 1, periods=3, freq="M").astype(str)
fc = lr.predict(np.arange(len(m), len(m) + 3).reshape(-1, 1))

# 9. EXPORT DASHBOARD -------------------------------------------------------------------
R = lambda v, d=0: round(float(v), d)
data = {
    "kpi": {"spend": R(tx.amount.sum()), "txns": len(tx), "customers": int(tx.customer_id.nunique()),
            "avg_txn": R(tx.amount.mean(), 2), "flagged": int(tx.flag.sum())},
    "monthly": {"labels": list(m.index), "actual": [R(v) for v in m.values], "f_labels": list(fut),
                "forecast": [R(v) for v in fc]},
    "categories": {"labels": list(top_cat.category), "values": [R(v) for v in top_cat.spend]},
    "regions": {"labels": list(region_sql.region), "values": [R(v) for v in region_sql.spend]},
    "segments": [{"name": n, "customers": int(r.customers), "avg_spend": R(r.avg_spend), "avg_txns": R(r.avg_txns, 1)}
                 for n, r in seg.iterrows()],
    "anomalies": [{"id": int(r.txn_id), "cust": int(r.customer_id), "cat": r.category, "amt": R(r.amount),
                   "hour": int(r.hour), "date": r.date.strftime("%d %b %Y"), "fraud": int(r.is_fraud)}
                  for r in anom.itertuples()],
    "metrics": {"acc": R(accuracy_score(yte, p), 3), "prec": R(precision_score(yte, p), 3),
                "rec": R(recall_score(yte, p), 3), "f1": R(f1_score(yte, p), 3),
                "a_prec": R(a_prec, 3), "a_rec": R(a_rec, 3), "churn_rate": R(y.mean(), 3)},
    "importance": [{"f": k, "v": R(v, 3)} for k, v in imp.head(5).items()],
}
json.dump(data, open(f"{HERE}/data.json", "w"), indent=1)
html = open(f"{HERE}/dashboard_template.html").read().replace("/*__DATA__*/null", json.dumps(data))
open(f"{HERE}/dashboard.html", "w").write(html)
print(json.dumps(data["kpi"]), json.dumps(data["metrics"]))
