
# 📰 + ⛅ News & Weather Add‑On

- **News**: Google News RSS (no key). Headlines → sentiment via **VADER** → daily average `news_sentiment`.
- **Weather**: Open‑Meteo archive + forecast (no key). Daily `temp_avg`, `precip_avg` across Karnal (IN), Bangkok (TH), Can Tho (VN).
- **Exogenous SARIMAX**: Optionally include these features to forecast prices.

## Files
- `news_weather.py` — fetch headlines, sentiment, and weather; build `exog` features.
- `model_exog.py` — SARIMAX with `exog` (past + future features).
- `news_tab.py` — Streamlit UI for latest headlines & feature building.
- `requirements.txt` — adds `feedparser`, `vaderSentiment`.

## How to integrate
1. Copy these files into your repo.
2. In `streamlit_app.py`, add a **News & Weather** tab:

```python
from news_tab import news_tab
tab1, tab2, tab3 = st.tabs(["Rice Benchmarks", "Company Stocks", "News & Weather"])
with tab3:
    news_tab()
```

3. To use exogenous features in rice forecasts:

```python
from news_weather import assemble_exog
from model_exog import forecast_with_exog

past, future = assemble_exog(days_back=120, days_forward=16)
df = pd.read_csv("data/rough_rice_yahoo.csv")  # or world bank csv
exog_fore_30 = forecast_with_exog(df, past, future, horizon_days=30)
```

The app will still work if these services are temporarily unreachable; it falls back to price‑only SARIMAX.
