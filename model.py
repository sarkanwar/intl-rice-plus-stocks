
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

HORIZONS = [7, 30, 180, 365]

PRICE_CANDIDATES = ["price","close","adj close","adj_close","settle","value","last","rate"]
DATE_CANDIDATES = ["date","timestamp","time"]

def _find_col(cols, candidates):
    cl = [c.lower() for c in cols]
    for cand in candidates:
        if cand in cl:
            return cols[cl.index(cand)]
    # fallback: first numeric for price, first datetime-like for date handled later
    return None

def _prepare_series(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Try to identify columns
    price_col = _find_col(df.columns, PRICE_CANDIDATES) or df.columns[-1]
    date_col  = _find_col(df.columns, DATE_CANDIDATES)  or df.columns[0]

    # Coerce date & price
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s[price_col] = pd.to_numeric(s[price_col], errors="coerce")

    s = s.dropna(subset=[date_col, price_col])
    if s.empty:
        return pd.Series(dtype=float)
    s = s.sort_values(date_col).set_index(date_col)[price_col].astype(float)

    # Make daily, forward-fill gaps
    s = s.asfreq("D").ffill()
    return s

def _fit(series):
    m = SARIMAX(series, order=(1,1,1), seasonal_order=(0,1,1,7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return m

def multi_forecast(date_price_df: pd.DataFrame, horizons=HORIZONS):
    s = _prepare_series(date_price_df)
    out = {}
    if s.empty:
        # Return empty frames for consistent UI
        for h in horizons:
            out[h] = pd.DataFrame(columns=["date","forecast"])
        return out

    if len(s) < 20:
        last = float(s.iloc[-1])
        for h in horizons:
            idx = pd.date_range(s.index.max(), periods=h, freq="D")
            out[h] = pd.DataFrame({"date": idx, "forecast": last})
        return out

    max_h = max(horizons)
    m = _fit(s)
    f = m.get_forecast(steps=max_h)
    mean = f.predicted_mean
    idx_all = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=max_h, freq="D")
    mean.index = idx_all

    for h in horizons:
        idx = idx_all[:h]
        out[h] = pd.DataFrame({"date": idx, "forecast": mean.loc[idx].values})
    return out
