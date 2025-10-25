
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

HORIZONS = [7, 30, 180, 365]

def _fit(series):
    # Simple SARIMAX; robust defaults
    m = SARIMAX(series, order=(1,1,1), seasonal_order=(0,1,1,7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return m

def multi_forecast(date_price_df: pd.DataFrame, horizons=HORIZONS):
    df = date_price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    s = df["Price"].astype(float).asfreq("D").ffill()
    if len(s) < 20:
        # naive forecasts if not enough history
        last = s.iloc[-1] if len(s) else 0.0
        out = {}
        for h in horizons:
            idx = pd.date_range(s.index.max() if len(s) else pd.Timestamp.today(), periods=h, freq="D")
            out[h] = pd.DataFrame({"date": idx, "forecast": last})
        return out
    m = _fit(s)
    out = {}
    max_h = max(horizons)
    fore = m.get_forecast(steps=max_h)
    mean = fore.predicted_mean
    idx_all = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=max_h, freq="D")
    mean.index = idx_all
    for h in horizons:
        idx = idx_all[:h]
        out[h] = pd.DataFrame({"date": idx, "forecast": mean.loc[idx].values})
    return out
