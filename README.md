# 💳 American Express Data Analysis

An end-to-end data analytics and machine learning project that turns raw card-transaction data into business insight: spending trends, customer segments, fraud alerts, churn risk and spend forecasts, all shown in an interactive dashboard.

## 🎯 Problem
Card issuers process huge volumes of transactions, but finding useful patterns in them is hard. This project shows how to identify spending behaviour, segment customers and catch suspicious activity with data analysis and machine learning.

## ✨ Features
- **Data cleaning and SQL:** deduplication, missing-value handling and SQL aggregations
- **EDA:** trends by month, category and region, plus a region × category heatmap
- **Customer segmentation:** K-Means groups customers into 4 spending tiers
- **Fraud detection:** Isolation Forest flags anomalous transactions (86% recall)
- **Churn prediction:** Random Forest classifier (89% accuracy)
- **Spend forecasting:** 3-month forecast using linear regression
- **Interactive dashboard:** KPIs, charts, segment table and model metrics

## 🛠️ Tech Stack
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · SQL (SQLite) · Chart.js · Git

## 🚀 How to Run
```bash
   pip install -r requirements.txt
   python analysis.py
```
Then open `dashboard.html` in your browser.

## 📁 Project Structure
- `analysis.py`: full analysis pipeline
- `dashboard.html`: interactive dashboard
- `charts/`: Matplotlib and Seaborn visualizations
- `data.json`: model results

## ⚠️ Note
The dataset is randomly generated to simulate card transactions. It is not real American Express data.
