# American Express Data Analysis

Transaction analytics: spending trends, customer segmentation, fraud/anomaly detection, churn prediction and spend forecasting, with an interactive dashboard.

**Stack:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SQL (SQLite), Jupyter/VS Code, Git.

## Run
```bash
pip install -r requirements.txt
python analysis.py      # writes charts/, data.json, dashboard.html
```
Open `dashboard.html` in a browser.

## Pipeline
1. Collect: synthetic transactions (1,500 customers, ~44k rows); replace with `pd.read_csv(...)` for real data. SQL aggregation via SQLite.
2. Clean: dedupe, drop missing amounts, parse dates.
3. EDA: monthly trend, category, region x category heatmap, amount distribution (`charts/`).
4. Features + K-Means: 4 segments (Budget Savers to Elite Big Spenders).
5. Isolation Forest: flags anomalous transactions, scored against injected fraud.
6. Random Forest: predicts churn (silent for 90 days) from pre-cutoff behavior; accuracy, precision, recall, F1.
7. Linear regression: 3-month spend forecast.

Data is randomly generated, not real Amex data.
