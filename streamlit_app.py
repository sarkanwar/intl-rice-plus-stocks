
import os, json
import streamlit as st, pandas as pd
from fetchers import fetch_yahoo_rough_rice, fetch_worldbank_pinksheet_rice, fetch_stocks_to_csv
from model import multi_forecast, HORIZONS

st.set_page_config(page_title="International Rice & Stocks Forecasts", page_icon="🌾", layout="wide")
st.title("🌾 International Rice & Basmati Company Forecasts")

# load config
cfg = {}
if os.path.exists("config.json"):
    cfg = json.load(open("config.json"))
yahoo_symbol = cfg.get("rice_benchmarks",{}).get("yahoo_symbol","ZR=F")
groups = cfg.get("company_groups",{})

tab1, tab2 = st.tabs(["Rice Benchmarks", "Company Stocks"])

with tab1:
    st.subheader("Sources")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Fetch: Yahoo Rough Rice (daily)"):
            p = fetch_yahoo_rough_rice("data/rough_rice_yahoo.csv")
            st.success(f"Saved {p}")
    with colB:
        if st.button("Fetch: World Bank Thai 5% (monthly)"):
            p = fetch_worldbank_pinksheet_rice("data/rice_wb_thai5.csv")
            st.success(f"Saved {p}")

    st.divider()
    choice = st.radio("Choose dataset", ["Yahoo (ZR=F)", "World Bank (Thai 5%)"], horizontal=True)
    path = "data/rough_rice_yahoo.csv" if choice.startswith("Yahoo") else "data/rice_wb_thai5.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df.tail(30), use_container_width=True)
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=os.path.basename(path), mime="text/csv")
        st.markdown("### Forecasts")
        outs = multi_forecast(df, horizons=HORIZONS)
        cols = st.columns(4)
        labels = {7:"1 Week", 30:"1 Month", 180:"6 Months", 365:"1 Year"}
        for i, h in enumerate(HORIZONS):
            with cols[i]:
                st.markdown(f"**{labels[h]}**")
                st.dataframe(outs[h].head(), use_container_width=True)
                st.download_button(f"Download {labels[h]}", data=outs[h].to_csv(index=False).encode("utf-8"),
                                   file_name=f"rice_forecast_{h}d.csv", mime="text/csv", key=f"dlf_rice_{h}")
    else:
        st.info("Click a fetch button above to download data.")


with tab2:
    st.subheader("Ticker groups")
    # show default groups with editable textarea
    all_groups = list(groups.keys()) if groups else []
    preset = st.selectbox("Preset groups", options=["(none)"] + all_groups)
    default_list = ",".join(groups.get(preset, [])) if preset in groups else "ADM,BG,KRBL.NS,DAAWAT.NS"
    tickers = st.text_input("Tickers (comma-separated, Yahoo symbols)", value=default_list, help="Edit list then click Fetch")
    if st.button("Fetch stock data"):
        tk = [t.strip() for t in tickers.split(",") if t.strip()]
        result = fetch_stocks_to_csv(tk, out_dir="data/stocks")
        if not result:
            st.error("No tickers returned data. Check symbols.")
        else:
            st.success(f"Saved: {len(result)} files")
    st.divider()

    # list files found and show forecasts
    import glob
    files = sorted(glob.glob("data/stocks/*.csv"))
    if not files:
        st.info("No stock files yet. Enter tickers and click 'Fetch stock data'.")
    else:
        for path in files:
            name = os.path.basename(path).replace(".csv","")
            st.markdown(f"### {name}")
            df = pd.read_csv(path)
            st.dataframe(df.tail(15), use_container_width=True)
            outs = multi_forecast(df, horizons=HORIZONS)
            cols = st.columns(4)
            labels = {7:"1 Week", 30:"1 Month", 180:"6 Months", 365:"1 Year"}
            for i, h in enumerate(HORIZONS):
                with cols[i]:
                    st.markdown(f"**{labels[h]}**")
                    st.dataframe(outs[h].head(), use_container_width=True)
                    st.download_button(f"Download {labels[h]}", data=outs[h].to_csv(index=False).encode("utf-8"),
                                       file_name=f"{name}_forecast_{h}d.csv", mime="text/csv", key=f"dlf_{name}_{h}")
