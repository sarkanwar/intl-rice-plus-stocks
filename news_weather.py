
import datetime as dt
from datetime import date, timedelta
import pandas as pd, requests, feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ANALYZER = SentimentIntensityAnalyzer()

# ---- NEWS ----

def fetch_rice_news(days=7, max_items=40, query="rice price OR basmati price OR rough rice OR FAO rice"):
    """Fetch latest rice headlines from Google News RSS (no API key). Returns list of dicts."""
    # time filter with when:Xd helps get recent items
    q = requests.utils.quote(f'{query} when:{days}d')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        # Try to parse date
        pub = None
        for k in ("published", "updated", "pubDate"):
            if k in e:
                try:
                    pub = pd.to_datetime(getattr(e, k))
                except Exception:
                    pub = None
        items.append({
            "title": e.title,
            "link": e.link,
            "published": pub,
            "source": getattr(getattr(e, "source", None), "title", ""),
            "summary": getattr(e, "summary", ""),
        })
    return items

def build_news_sentiment(days_back=120, query="rice price OR basmati price OR rough rice OR FAO rice"):
    """Daily sentiment series (compound mean) from Google News RSS headlines/summaries."""
    items = fetch_rice_news(days=min(days_back, 30), max_items=100, query=query)  # Google News returns recent days
    if not items:
        return pd.DataFrame(columns=["Date","news_sentiment"])
    df = pd.DataFrame(items)
    # Sentiment from title+summary
    def comp(row):
        text = f"{row.get('title','')} {row.get('summary','')}"
        return ANALYZER.polarity_scores(text)["compound"]
    df["sent"] = df.apply(comp, axis=1)
    df["Date"] = pd.to_datetime(df["published"]).dt.date
    df = df.dropna(subset=["Date"])
    agg = df.groupby("Date")["sent"].mean().rename("news_sentiment").reset_index()
    return agg

# ---- WEATHER ----

def fetch_weather_daily(lat, lon, start_date, end_date):
    """Open-Meteo ERA5 archive daily temp/precip."""
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": ["temperature_2m_mean","precipitation_sum"],
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    j = r.json()
    if "daily" not in j or not j["daily"].get("time"):
        return pd.DataFrame(columns=["Date","temp","precip"])
    d = pd.DataFrame({
        "Date": pd.to_datetime(j["daily"]["time"]).dt.date,
        "temp": j["daily"]["temperature_2m_mean"],
        "precip": j["daily"]["precipitation_sum"],
    })
    return d

def fetch_weather_forecast(lat, lon, days_forward=16):
    """Open-Meteo forecast up to 16 days."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": ["temperature_2m_mean","precipitation_sum"],
        "forecast_days": days_forward,
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    j = r.json()
    if "daily" not in j or not j["daily"].get("time"):
        return pd.DataFrame(columns=["Date","temp","precip"])
    d = pd.DataFrame({
        "Date": pd.to_datetime(j["daily"]["time"]).dt.date,
        "temp": j["daily"]["temperature_2m_mean"],
        "precip": j["daily"]["precipitation_sum"],
    })
    return d

RICE_REGIONS = {
    # name: (lat, lon) — tune as you like
    "Karnal, India": (29.6857, 76.9905),
    "Bangkok, Thailand": (13.7563, 100.5018),
    "Can Tho, Vietnam": (10.0452, 105.7469),
}

def build_weather_features(days_back=120, days_forward=16, regions=RICE_REGIONS):
    """Return two dataframes:
       - past features (Date + region columns)
       - future features for forecast horizon (Date + region columns)
       Each region contributes temp & precip; we also compute averages across regions.
    """
    today = date.today()
    start = today - timedelta(days=days_back)
    past_frames = []
    future_frames = []
    for name, (lat, lon) in regions.items():
        p = fetch_weather_daily(lat, lon, start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        if not p.empty:
            p = p.rename(columns={"temp": f"temp_{name}", "precip": f"precip_{name}"})
            past_frames.append(p)
        fut = fetch_weather_forecast(lat, lon, days_forward=days_forward)
        if not fut.empty:
            fut = fut.rename(columns={"temp": f"temp_{name}", "precip": f"precip_{name}"})
            future_frames.append(fut)

    if past_frames:
        past = past_frames[0]
        for f in past_frames[1:]:
            past = pd.merge(past, f, on="Date", how="outer")
        past = past.sort_values("Date")
        # Averages across regions
        temp_cols = [c for c in past.columns if c.startswith("temp_")]
        pr_cols = [c for c in past.columns if c.startswith("precip_")]
        past["temp_avg"] = past[temp_cols].mean(axis=1)
        past["precip_avg"] = past[pr_cols].mean(axis=1)
    else:
        past = pd.DataFrame(columns=["Date","temp_avg","precip_avg"])

    if future_frames:
        fut = future_frames[0]
        for f in future_frames[1:]:
            fut = pd.merge(fut, f, on="Date", how="outer")
        fut = fut.sort_values("Date")
        temp_cols = [c for c in fut.columns if c.startswith("temp_")]
        pr_cols = [c for c in fut.columns if c.startswith("precip_")]
        fut["temp_avg"] = fut[temp_cols].mean(axis=1)
        fut["precip_avg"] = fut[pr_cols].mean(axis=1)
    else:
        fut = pd.DataFrame(columns=["Date","temp_avg","precip_avg"])

    return past, fut

def assemble_exog(days_back=120, days_forward=16):
    """Join news sentiment + weather (avg) and return past & future exogenous frames with same columns."""
    news = build_news_sentiment(days_back=days_back)
    w_past, w_future = build_weather_features(days_back=days_back, days_forward=days_forward)

    # Merge on Date
    past = pd.merge(w_past, news, on="Date", how="outer")
    # Fill missing with forward/backward mean
    past = past.sort_values("Date")
    for col in [c for c in past.columns if c != "Date"]:
        past[col] = pd.to_numeric(past[col], errors="coerce")
    past = past.fillna(method="ffill").fillna(method="bfill")

    future = w_future.copy()
    if "news_sentiment" in past.columns:
        # For future sentiment, neutral (0) by default
        future["news_sentiment"] = 0.0
    else:
        future["news_sentiment"] = 0.0

    # Reorder
    cols = ["Date"] + [c for c in past.columns if c != "Date"]
    past = past[cols]
    future = future[cols]
    return past, future
