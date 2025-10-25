
# 🌾 International Rice & Basmati Company Forecasts

**No API keys required.** Sources:
- **Yahoo Finance** — Rough Rice futures (`ZR=F`) and company stocks
- **World Bank Pink Sheet** — Thai 5% broken rice (monthly Excel)

## Forecast horizons
- **1 Week (7d)**
- **1 Month (30d)**
- **6 Months (180d)**
- **1 Year (365d)**

## Deploy
1. Create a repo and upload these files.
2. Streamlit Cloud → New app  
   - Main file: `streamlit_app.py`  
   - Python: 3.11
3. In the app, press **Fetch** to download data, then forecasts appear with download buttons.

## Automatic updates
GitHub Action `.github/workflows/daily.yml` runs daily and commits fresh CSVs to `data/`.

## Edit company tickers
Open `config.json` and update arrays. Examples included:
- Global traders: `["ADM","BG"]`
- India basmati (edit): `["KRBL.NS","DAAWAT.NS","WIL.NS"]`  
If a ticker has no data on Yahoo, it's skipped automatically.

## Files
- `fetchers.py` — downloads rice benchmarks and stocks
- `model.py` — multi-horizon forecaster (SARIMAX)
- `config.json` — default groups/tickers
- `streamlit_app.py` — UI with two tabs
- `.github/workflows/daily.yml` — scheduler
- `requirements.txt`
