
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from statistics import NormalDist

st.set_page_config(page_title="INDF Stock Analytics", page_icon="📈", layout="wide")

BLUE="#005BAC"; DARK="#003B73"; LIGHT="#EAF4FF"; BG="#F7F9FC"
GREEN="#16A34A"; RED="#DC2626"; TEXT="#172033"; MUTED="#64748B"

st.markdown(f"""
<style>
.stApp {{background:{BG}; color:{TEXT};}}
.block-container {{padding-top:1.2rem; padding-bottom:2rem; max-width:1500px;}}
.hero {{
 background:linear-gradient(120deg,{DARK},{BLUE}); color:white;
 padding:24px 28px; border-radius:22px; margin-bottom:18px;
 box-shadow:0 10px 28px rgba(0,91,172,.18);
}}
.hero h1 {{margin:0; font-size:2.05rem;}}
.hero p {{margin:.35rem 0 0; color:#EAF4FF; font-size:1.02rem;}}
.badge {{display:inline-block; margin-top:12px; padding:5px 10px; border-radius:999px;
 background:rgba(255,255,255,.16); color:white; font-weight:700;}}
div[data-testid="stMetric"] {{
 background:white !important; border:1px solid #DDE7F2 !important;
 border-top:4px solid {BLUE} !important; border-radius:16px !important;
 padding:14px 15px !important; box-shadow:0 3px 12px rgba(15,23,42,.06) !important;
 min-height:122px;
}}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {{
 color:{MUTED} !important; opacity:1 !important; font-weight:650 !important;
}}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{
 color:{DARK} !important; opacity:1 !important; font-weight:800 !important;
 font-size:1.55rem !important;
}}
.section-title {{font-size:1.45rem; font-weight:800; color:{DARK}; margin:16px 0 8px;}}
.insight {{
 background:white; border-left:5px solid {BLUE}; border-radius:14px;
 padding:16px 18px; box-shadow:0 3px 12px rgba(15,23,42,.05);
}}
.risk-note {{
 background:#FFF7ED; border:1px solid #FED7AA; border-radius:12px;
 padding:12px 14px; color:#7C2D12;
}}
</style>
""", unsafe_allow_html=True)

DATA_FILE = Path(__file__).parent / "Data Indofood.json"

@st.cache_data
def load_data():
    df = pd.read_json(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").drop_duplicates("Date", keep="last")
    # File menyimpan harga seperti 7,85; tampilkan sebagai Rp7.850 bila skala median < 100.
    scale = 1000 if df["Close"].median() < 100 else 1
    df["Price"] = df["Close"] * scale
    df["Return"] = np.log(df["Price"] / df["Price"].shift(1))
    df["MA20"] = df["Price"].rolling(20).mean()
    df["MA60"] = df["Price"].rolling(60).mean()
    df["Vol20"] = df["Return"].rolling(20).std() * np.sqrt(252) * 100
    peak = df["Price"].cummax()
    df["Drawdown"] = (df["Price"] / peak - 1) * 100
    return df.reset_index(drop=True)

df = load_data()

st.markdown("""
<div class="hero">
  <h1>INDF Stock Analytics & Risk Dashboard</h1>
  <p>Historical Performance • Forecasting • Return • Volatility • Value at Risk</p>
  <span class="badge">INDF • Indofood Sukses Makmur Tbk</span>
</div>
""", unsafe_allow_html=True)

# ---------------- FILTER ----------------
min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
f1,f2,f3 = st.columns([2,1,1])
with f1:
    dr = st.date_input("Rentang tanggal", value=(min_d,max_d), min_value=min_d, max_value=max_d)
with f2:
    horizon = st.selectbox("Horizon prediksi", [7,14,30], index=2)
with f3:
    investment = st.number_input("Nilai investasi (Rp)", min_value=1_000_000, value=100_000_000, step=5_000_000)

if isinstance(dr, (tuple,list)) and len(dr)==2:
    d0,d1 = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
else:
    d0,d1 = pd.Timestamp(min_d), pd.Timestamp(max_d)

x = df[(df["Date"]>=d0)&(df["Date"]<=d1)].copy()
if len(x) < 30:
    st.warning("Rentang terpilih terlalu pendek. Pilih sedikitnya sekitar 30 observasi agar analisis lebih stabil.")
    x = df.copy()

rets = x["Return"].dropna()
last = float(x["Price"].iloc[-1]); first = float(x["Price"].iloc[0])
period_ret = (last/first-1)*100
hi = float(x["Price"].max()); lo=float(x["Price"].min())
vol = float(rets.std()*np.sqrt(252)*100) if len(rets)>1 else np.nan
maxdd = float(x["Drawdown"].min())

# ---------------- FORECAST ARIMA ----------------
@st.cache_data(show_spinner=False)
def forecast_arima(prices_tuple, horizon):
    prices = pd.Series(prices_tuple, dtype=float)
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(prices, order=(1,1,1)).fit()
        fc = model.get_forecast(steps=horizon)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=.05), dtype=float)
        return mean, ci[:,0], ci[:,1], "ARIMA(1,1,1)"
    except Exception:
        # fallback random-walk drift
        lr = np.log(prices/prices.shift(1)).dropna()
        mu, sig = lr.mean(), lr.std()
        h = np.arange(1,horizon+1)
        mean = prices.iloc[-1]*np.exp(mu*h)
        lo = prices.iloc[-1]*np.exp(mu*h-1.96*sig*np.sqrt(h))
        hi = prices.iloc[-1]*np.exp(mu*h+1.96*sig*np.sqrt(h))
        return np.asarray(mean),np.asarray(lo),np.asarray(hi),"Random Walk + Drift"

mean_fc, low_fc, high_fc, model_name = forecast_arima(tuple(x["Price"].values), horizon)
last_date=x["Date"].max()
future_dates=pd.bdate_range(last_date+pd.Timedelta(days=1), periods=horizon)
fc=pd.DataFrame({"Date":future_dates,"Forecast":mean_fc,"Lower":low_fc,"Upper":high_fc})
pred=float(mean_fc[-1]); pred_change=(pred/last-1)*100

# ---------------- RISK ----------------
q05=float(rets.quantile(.05)) if len(rets) else np.nan
q01=float(rets.quantile(.01)) if len(rets) else np.nan
var95_pct=max(0,-q05*100); var99_pct=max(0,-q01*100)
var95_rp=investment*var95_pct/100; var99_rp=investment*var99_pct/100

# ---------------- KPI ----------------
st.markdown('<div class="section-title">Ringkasan Pasar</div>', unsafe_allow_html=True)
k1,k2,k3,k4,k5,k6=st.columns(6)
k1.metric("Harga terakhir", f"Rp{last:,.0f}".replace(",","."))
k2.metric("Return periode", f"{period_ret:+.2f}%")
k3.metric("Harga tertinggi", f"Rp{hi:,.0f}".replace(",","."))
k4.metric("Harga terendah", f"Rp{lo:,.0f}".replace(",","."))
k5.metric(f"Prediksi H+{horizon}", f"Rp{pred:,.0f}".replace(",","."), f"{pred_change:+.2f}%")
k6.metric("Volatilitas tahunan", f"{vol:.2f}%")

# ---------------- PRICE CHART ----------------
st.markdown('<div class="section-title">Pergerakan Harga & Moving Average</div>', unsafe_allow_html=True)
plot=x[["Date","Price","MA20","MA60"]].melt("Date",var_name="Series",value_name="Value").dropna()
chart=alt.Chart(plot).mark_line(strokeWidth=2.3).encode(
    x=alt.X("Date:T",title=None),
    y=alt.Y("Value:Q",title="Harga (Rp)",scale=alt.Scale(zero=False)),
    color=alt.Color("Series:N",
        scale=alt.Scale(domain=["Price","MA20","MA60"],range=[BLUE,"#49A6E9","#94A3B8"]),
        legend=alt.Legend(title=None,orient="top")),
    tooltip=[alt.Tooltip("Date:T",title="Tanggal"),alt.Tooltip("Series:N"),alt.Tooltip("Value:Q",format=",.0f")]
).properties(height=390).interactive()
st.altair_chart(chart,use_container_width=True)

# ---------------- RETURN + VOL ----------------
c1,c2=st.columns(2)
with c1:
    st.markdown('<div class="section-title">Return Harian</div>',unsafe_allow_html=True)
    rr=x.dropna(subset=["Return"]).copy(); rr["ReturnPct"]=rr["Return"]*100
    rchart=alt.Chart(rr).mark_bar().encode(
        x=alt.X("Date:T",title=None),
        y=alt.Y("ReturnPct:Q",title="Log return (%)"),
        color=alt.condition("datum.ReturnPct >= 0",alt.value(GREEN),alt.value(RED)),
        tooltip=[alt.Tooltip("Date:T"),alt.Tooltip("ReturnPct:Q",format=".2f",title="Return (%)")]
    ).properties(height=300)
    st.altair_chart(rchart,use_container_width=True)
with c2:
    st.markdown('<div class="section-title">Rolling Volatility 20 Hari</div>',unsafe_allow_html=True)
    vv=x.dropna(subset=["Vol20"])
    vchart=alt.Chart(vv).mark_area(line={"color":BLUE},color=LIGHT,opacity=.8).encode(
        x=alt.X("Date:T",title=None),y=alt.Y("Vol20:Q",title="Volatilitas tahunan (%)"),
        tooltip=[alt.Tooltip("Date:T"),alt.Tooltip("Vol20:Q",format=".2f")]
    ).properties(height=300)
    st.altair_chart(vchart,use_container_width=True)

# ---------------- FORECAST ----------------
st.markdown('<div class="section-title">Prediksi Harga</div>',unsafe_allow_html=True)
hist=x[["Date","Price"]].tail(120)
base=alt.Chart(hist).mark_line(color=BLUE,strokeWidth=2.5).encode(
    x=alt.X("Date:T",title=None),y=alt.Y("Price:Q",title="Harga (Rp)",scale=alt.Scale(zero=False)),
    tooltip=[alt.Tooltip("Date:T"),alt.Tooltip("Price:Q",format=",.0f")]
)
band=alt.Chart(fc).mark_area(color="#93C5FD",opacity=.28).encode(
    x="Date:T",y=alt.Y("Lower:Q",scale=alt.Scale(zero=False)),y2="Upper:Q"
)
fline=alt.Chart(fc).mark_line(color="#0EA5E9",strokeDash=[7,4],strokeWidth=3).encode(
    x="Date:T",y="Forecast:Q",
    tooltip=[alt.Tooltip("Date:T"),alt.Tooltip("Forecast:Q",format=",.0f")]
)
st.altair_chart((band+base+fline).properties(height=390),use_container_width=True)
st.caption(f"Model dashboard: {model_name}. Area biru muda menunjukkan interval prediksi 95%.")

# ---------------- RISK ----------------
st.markdown('<div class="section-title">Analisis Risiko</div>',unsafe_allow_html=True)
r1,r2,r3,r4=st.columns(4)
r1.metric("Historical VaR 95%",f"{var95_pct:.2f}%",f"Rp{var95_rp:,.0f}".replace(",","."))
r2.metric("Historical VaR 99%",f"{var99_pct:.2f}%",f"Rp{var99_rp:,.0f}".replace(",","."))
r3.metric("Maximum Drawdown",f"{maxdd:.2f}%")
r4.metric("Return terburuk",f"{rets.min()*100:.2f}%")

dd=x[["Date","Drawdown"]].dropna()
ddchart=alt.Chart(dd).mark_area(color=RED,opacity=.22,line={"color":RED}).encode(
    x=alt.X("Date:T",title=None),y=alt.Y("Drawdown:Q",title="Drawdown (%)"),
    tooltip=[alt.Tooltip("Date:T"),alt.Tooltip("Drawdown:Q",format=".2f")]
).properties(height=280)
st.altair_chart(ddchart,use_container_width=True)

# ---------------- INSIGHT ----------------
trend="meningkat" if period_ret>=0 else "menurun"
risk_dir="naik" if pred_change>=0 else "turun"
hirow=x.loc[x["Price"].idxmax()]; lorow=x.loc[x["Price"].idxmin()]
st.markdown('<div class="section-title">Executive Insight</div>',unsafe_allow_html=True)
st.markdown(f"""
<div class="insight">
Selama periode terpilih, harga INDF {trend} sebesar <b>{abs(period_ret):.2f}%</b>.
Harga tertinggi tercatat <b>Rp{hi:,.0f}</b> pada {hirow["Date"].strftime("%d %b %Y")},
sedangkan harga terendah <b>Rp{lo:,.0f}</b> pada {lorow["Date"].strftime("%d %b %Y")}.
Model {model_name} memperkirakan harga H+{horizon} sekitar <b>Rp{pred:,.0f}</b>
atau {risk_dir} <b>{abs(pred_change):.2f}%</b> dari harga terakhir.
Historical VaR 95% sebesar <b>{var95_pct:.2f}%</b>, setara sekitar
<b>Rp{var95_rp:,.0f}</b> untuk nilai investasi Rp{investment:,.0f}.
Maximum drawdown pada rentang terpilih adalah <b>{maxdd:.2f}%</b>.
</div>
""".replace(",", "."),unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
st.markdown("""
<div class="risk-note">
<b>Catatan:</b> Prediksi dan VaR merupakan estimasi statistik berdasarkan data historis.
Interval prediksi dan VaR tidak menjamin harga atau kerugian aktual akan berada dalam batas tersebut.
Dashboard ini ditujukan untuk analisis data, bukan rekomendasi membeli atau menjual saham.
</div>
""",unsafe_allow_html=True)

# ---------------- DETAIL ----------------
st.markdown('<div class="section-title">Data Detail</div>',unsafe_allow_html=True)
detail=x[["Date","Price","Return","MA20","MA60","Vol20","Drawdown"]].copy()
detail["Return (%)"]=detail["Return"]*100
detail=detail.drop(columns="Return")
st.dataframe(detail.sort_values("Date",ascending=False),use_container_width=True,hide_index=True)
st.download_button("Download data terfilter",detail.to_csv(index=False).encode("utf-8"),
                   "INDF_dashboard_data.csv","text/csv")
