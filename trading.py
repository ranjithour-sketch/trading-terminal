"""
trading.py  —  Indian Intraday & Options Terminal
==================================================
Run:  streamlit run trading.py

Tabs:
  1. 📋 Watchlist          — live prices for all sectors
  2. 🎯 Trade Setup        — 11-factor checklist + entry/exit
  3. 🏦 Smart Money        — institutional activity detector
  4. 🧮 P&L Calculator     — options profit/loss before trade
  5. 📰 News & Events      — market news + daily checklist
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm
import math, time, pytz, requests
import xml.etree.ElementTree as ET
from ml_engine import train_model, predict_next_move, approximate_realtime


# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intraday & Options Terminal",
    layout="wide", page_icon="🎯",
    initial_sidebar_state="expanded",
)
IST = pytz.timezone("Asia/Kolkata")

# ── URL-based tab routing ──────────────────────────────────
# Each tab has a URL: ?tab=watchlist, ?tab=setup, etc.
# This allows opening any tab in a separate browser window
TAB_ROUTES = {
    "watchlist": 0,
    "setup":     1,
    "scanner":   2,
    "ml":        3,
    "smart":     4,
    "calc":      5,
    "news":      6,
    "paper":     7,
    "journal":   8,
}
TAB_NAMES = [
    "📋 Watchlist",
    "🎯 Trade Setup",
    "🔍 Auto Scanner",
    "🤖 ML Prediction",
    "🏦 Smart Money",
    "🧮 P&L Calculator",
    "📰 News & Events",
    "📝 Paper Trading",
    "📓 Trade Journal",
]
TAB_ICONS = ["📋","🎯","🔍","🤖","🏦","🧮","📰","📝","📓"]
TAB_KEYS  = list(TAB_ROUTES.keys())

# Read current tab from URL
_qp = st.query_params
_tab_key = _qp.get("tab", "watchlist")
_default_tab = TAB_ROUTES.get(_tab_key, 0)

# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ═══════════════════════════════════════════════
   BASE
═══════════════════════════════════════════════ */
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', sans-serif !important;
    background-color: #f4f6f9 !important;
    color: #1e293b !important;
}
.block-container {
    padding-top: 1rem !important;
    max-width: 1400px !important;
}
/* Hide Streamlit default top bar decoration */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}
/* Ensure main content not hidden under header */
.main .block-container {
    padding-top: 1rem !important;
}

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    padding: 12px 8px !important;
}
section[data-testid="stSidebar"] * {
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #374151 !important;
    margin: 12px 0 6px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
section[data-testid="stSidebar"] p {
    font-size: 12px !important;
    color: #64748b !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #334155 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    width: 100% !important;
    text-align: left !important;
    margin: 1px 0 !important;
    transition: all 0.15s ease !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #eff6ff !important;
    border-color: #3b82f6 !important;
    color: #1d4ed8 !important;
}
/* Hide expander arrow corruption completely */
section[data-testid="stSidebar"] .streamlit-expanderHeader {
    display: none !important;
}
section[data-testid="stSidebar"] .streamlit-expanderContent {
    border: none !important;
    padding: 0 !important;
}

/* ═══════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 5px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 14px !important;
    border: none !important;
    white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #f1f5f9 !important;
    color: #334155 !important;
}
.stTabs [aria-selected="true"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 4px rgba(59,130,246,0.3) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem !important;
    background: transparent !important;
}

/* ═══════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════ */
.stButton button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
    border: 1px solid #e2e8f0 !important;
    color: #374151 !important;
    background: #ffffff !important;
}
.stButton button[kind="primary"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(59,130,246,0.35) !important;
}
.stButton button[kind="primary"]:hover {
    background: #2563eb !important;
    box-shadow: 0 4px 10px rgba(59,130,246,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton button:hover {
    border-color: #94a3b8 !important;
    background: #f8fafc !important;
}

/* ═══════════════════════════════════════════════
   INPUTS & SELECTS
═══════════════════════════════════════════════ */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
    font-size: 13px !important;
}
[data-baseweb="popover"] [data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-baseweb="option"] {
    background: #ffffff !important;
    color: #1e293b !important;
    font-size: 13px !important;
}
[data-baseweb="option"]:hover {
    background: #eff6ff !important;
}

/* ═══════════════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════════════ */
div[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1e293b !important;
    font-weight: 700 !important;
    font-size: 22px !important;
}

/* ═══════════════════════════════════════════════
   ALERTS
═══════════════════════════════════════════════ */
div[data-testid="stSuccess"] {
    background: #f0fdf4 !important;
    border: 1px solid #86efac !important;
    border-radius: 10px !important;
    color: #166534 !important;
}
div[data-testid="stWarning"] {
    background: #fffbeb !important;
    border: 1px solid #fcd34d !important;
    border-radius: 10px !important;
    color: #92400e !important;
}
div[data-testid="stError"] {
    background: #fef2f2 !important;
    border: 1px solid #fca5a5 !important;
    border-radius: 10px !important;
    color: #991b1b !important;
}
div[data-testid="stInfo"] {
    background: #eff6ff !important;
    border: 1px solid #93c5fd !important;
    border-radius: 10px !important;
    color: #1e40af !important;
}
div[data-testid="stSuccess"] p,
div[data-testid="stWarning"] p,
div[data-testid="stError"] p,
div[data-testid="stInfo"] p {
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* ═══════════════════════════════════════════════
   EXPANDER — fix arrow corruption
═══════════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
}
.streamlit-expanderHeader svg {
    display: none !important;
}
.streamlit-expanderHeader p,
.streamlit-expanderHeader span {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
/* Remove the ::before arrow that causes corruption */
.streamlit-expanderHeader::before {
    content: none !important;
}
details summary::marker,
details summary::-webkit-details-marker {
    display: none !important;
    content: '' !important;
}
.streamlit-expanderContent {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 12px !important;
}

/* ═══════════════════════════════════════════════
   DATAFRAME / TABLE
═══════════════════════════════════════════════ */
.stDataFrame {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.stDataFrame table {
    color: #1e293b !important;
    font-size: 13px !important;
}
.stDataFrame thead th {
    background: #f8fafc !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.3px !important;
    border-bottom: 1px solid #e2e8f0 !important;
}
.stDataFrame tbody tr:hover {
    background: #f0f9ff !important;
}

/* ═══════════════════════════════════════════════
   SCANNER CARDS — ensure stock name is visible
═══════════════════════════════════════════════ */
.scanner-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    color: #1e293b !important;
}
.scanner-card .stock-name {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}

/* ═══════════════════════════════════════════════
   MARKDOWN & HEADINGS
═══════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: #1e293b !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}
.stMarkdown p {
    color: #374151 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
.stMarkdown strong { color: #1e293b !important; }
caption, .stCaption {
    color: #64748b !important;
    font-size: 12px !important;
}

/* ═══════════════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════════════ */
.stProgress > div > div {
    background: #3b82f6 !important;
    border-radius: 4px !important;
}

/* ═══════════════════════════════════════════════
   CHECKBOX & TOGGLE
═══════════════════════════════════════════════ */
.stCheckbox label,
.stToggle label { color: #374151 !important; font-size: 13px !important; }

/* ═══════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px }
::-webkit-scrollbar-track { background: #f1f5f9 }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px }
::-webkit-scrollbar-thumb:hover { background: #94a3b8 }

/* ═══════════════════════════════════════════════
   PLOTLY CHARTS — light background
═══════════════════════════════════════════════ */
.js-plotly-plot .plotly .main-svg {
    background: #ffffff !important;
    border-radius: 10px !important;
}

/* ═══════════════════════════════════════════════
   SLIDER
═══════════════════════════════════════════════ */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# STOCK & SECTOR DATABASE
# ══════════════════════════════════════════════════════════
STOCKS = {
    # Indices
    "NIFTY 50":       "^NSEI",
    "BANK NIFTY":     "^NSEBANK",
    "SENSEX":         "^BSESN",
    "NIFTY IT":       "^CNXIT",
    "NIFTY AUTO":     "^CNXAUTO",
    "NIFTY PHARMA":   "^CNXPHARMA",
    "NIFTY METAL":    "^CNXMETAL",
    "NIFTY FMCG":     "^CNXFMCG",
    # Banking
    "HDFC Bank":      "HDFCBANK.NS",
    "ICICI Bank":     "ICICIBANK.NS",
    "SBI":            "SBIN.NS",
    "Kotak Bank":     "KOTAKBANK.NS",
    "Axis Bank":      "AXISBANK.NS",
    "IndusInd Bank":  "INDUSINDBK.NS",
    "Yes Bank":       "YESBANK.NS",
    "PNB":            "PNB.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Canara Bank":    "CANBK.NS",
    "Federal Bank":   "FEDERALBNK.NS",
    "IDFC First":     "IDFCFIRSTB.NS",
    "Bajaj Finance":  "BAJFINANCE.NS",
    "Bajaj Finserv":  "BAJAJFINSV.NS",
    # IT
    "TCS":            "TCS.NS",
    "Infosys":        "INFY.NS",
    "Wipro":          "WIPRO.NS",
    "HCL Tech":       "HCLTECH.NS",
    "Tech Mahindra":  "TECHM.NS",
    "Persistent":     "PERSISTENT.NS",
    "Coforge":        "COFORGE.NS",
    "LTIMindtree":    "LTIM.NS",
    # Energy
    "Reliance":       "RELIANCE.NS",
    "ONGC":           "ONGC.NS",
    "Indian Oil":     "IOC.NS",
    "BPCL":           "BPCL.NS",
    "NTPC":           "NTPC.NS",
    "Power Grid":     "POWERGRID.NS",
    "Adani Green":    "ADANIGREEN.NS",
    "Tata Power":     "TATAPOWER.NS",
    "Gail":           "GAIL.NS",
    # Auto
    "Ashok Leyland":     "ASHOKLEY.NS",
    "Maruti":         "MARUTI.NS",
    "M&M":            "M&M.BO",
    "Hero MotoCorp":  "HEROMOTOCO.NS",
    "Bajaj Auto":     "BAJAJ-AUTO.BO",
    "TVS Motor":      "TVSMOTOR.NS",
    "Eicher Motors":  "EICHERMOT.NS",
    "Ashok Leyland":  "ASHOKLEY.NS",
    # Pharma
    "Sun Pharma":     "SUNPHARMA.NS",
    "Dr Reddy":       "DRREDDY.NS",
    "Cipla":          "CIPLA.NS",
    "Divi's Lab":     "DIVISLAB.NS",
    "Lupin":          "LUPIN.NS",
    "Apollo Hosp":    "APOLLOHOSP.NS",
    # FMCG
    "HUL":            "HINDUNILVR.NS",
    "ITC":            "ITC.NS",
    "Nestle":         "NESTLEIND.NS",
    "Britannia":      "BRITANNIA.NS",
    "Dabur":          "DABUR.NS",
    "Tata Consumer":  "TATACONSUM.NS",
    # Metals
    "Tata Steel":     "TATASTEEL.NS",
    "JSW Steel":      "JSWSTEEL.NS",
    "Hindalco":       "HINDALCO.NS",
    "Coal India":     "COALINDIA.NS",
    "Vedanta":        "VEDL.NS",
    "SAIL":           "SAIL.NS",
    # Infra / Defence
    "L&T":            "LT.NS",
    "Adani Ports":    "ADANIPORTS.NS",
    "DLF":            "DLF.NS",
    "UltraTech":      "ULTRACEMCO.NS",
    "HAL":            "HAL.NS",
    "BEL":            "BEL.NS",
    "IRCTC":          "IRCTC.NS",
    "RVNL":           "RVNL.NS",
    "IRFC":           "IRFC.NS",
    "Mazagon Dock":   "MAZDOCK.NS",
    # Telecom / Consumer
    "Bharti Airtel":  "BHARTIARTL.NS",
    "Zomato":         "ZOMATO.NS",
    "Nykaa":          "NYKAA.NS",
    "DMart":          "DMART.NS",
    "Trent":          "TRENT.NS",
    "Asian Paints":   "ASIANPAINT.NS",
    "Pidilite":       "PIDILITIND.NS",
}

SECTORS = {
    "🏆 Top 20 F&O": [
        "NIFTY 50","BANK NIFTY","Reliance","HDFC Bank",
        "ICICI Bank","TCS","Infosys","SBI","Tata Motors DVR",
        "Bajaj Finance","ITC","Sun Pharma","L&T","Maruti",
        "Coal India","NTPC","Bharti Airtel","Tata Steel",
        "Axis Bank","Wipro",
    ],
    "🏦 Banking":  ["HDFC Bank","ICICI Bank","SBI","Kotak Bank",
                    "Axis Bank","IndusInd Bank","Bajaj Finance",
                    "PNB","Bank of Baroda","Canara Bank",
                    "Federal Bank","IDFC First"],
    "💻 IT":       ["TCS","Infosys","Wipro","HCL Tech",
                    "Tech Mahindra","Persistent","Coforge",
                    "LTIMindtree"],
    "🛢️ Energy":  ["Reliance","ONGC","Indian Oil","BPCL",
                    "NTPC","Power Grid","Tata Power","Gail"],
    "🚗 Auto":     ["Tata Motors DVR","Maruti","M&M","Hero MotoCorp",
                    "Bajaj Auto","TVS Motor","Eicher Motors"],
    "💊 Pharma":   ["Sun Pharma","Dr Reddy","Cipla",
                    "Divi's Lab","Lupin","Apollo Hosp"],
    "🛒 FMCG":     ["HUL","ITC","Nestle","Britannia",
                    "Dabur","Tata Consumer"],
    "⚙️ Metals":   ["Tata Steel","JSW Steel","Hindalco",
                    "Coal India","Vedanta","SAIL"],
    "🛡️ Defence":  ["HAL","BEL","Mazagon Dock","RVNL","IRFC"],
}

LOT_SIZES = {
    "^NSEI":250,"^NSEBANK":15,
    "RELIANCE.NS":250,"TCS.NS":150,
    "HDFCBANK.NS":550,"ICICIBANK.NS":700,
    "INFY.NS":300,"SBIN.NS":1500,
    "BAJFINANCE.NS":125,"ASHOKLEY.NS":1500,
    "AXISBANK.NS":1200,"ITC.NS":3200,
    "WIPRO.NS":3000,"KOTAKBANK.NS":400,
    "HCLTECH.NS":350,"TECHM.NS":600,
}

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def now_ist():
    return datetime.now(IST)

def market_open():
    n = now_ist()
    if n.weekday() >= 5:
        return False
    return (n.replace(hour=9,  minute=15, second=0) <= n <=
            n.replace(hour=15, minute=30, second=0))

def best_trading_time():
    n   = now_ist()
    hr  = n.hour
    mn  = n.minute
    t   = hr * 60 + mn
    if t < 9*60+15:  return "pre_market"
    if t < 9*60+30:  return "avoid"       # opening 15 min
    if t < 10*60+30: return "best"        # 9:30–10:30
    if t < 12*60:    return "good"        # 10:30–12:00
    if t < 13*60+30: return "ok"          # lunch
    if t < 14*60+30: return "good"        # afternoon
    if t < 15*60:    return "caution"     # 2:30–3:00
    if t <= 15*60+30:return "avoid"       # last 30 min
    return "closed"

@st.cache_data(ttl=60)
def live_price(sym: str) -> dict:
    try:
        fi = yf.Ticker(sym).fast_info
        p  = float(fi.last_price)
        pv = float(fi.previous_close)
        ch = round(((p-pv)/pv)*100, 2)
        return {"ok":True, "p":round(p,2), "prev":round(pv,2),
                "chg":ch, "chg_abs":round(p-pv,2),
                "high":round(float(fi.day_high or pv),2),
                "low": round(float(fi.day_low  or pv),2)}
    except:
        return {"ok":False,"p":0,"prev":0,"chg":0,
                "chg_abs":0,"high":0,"low":0}

@st.cache_data(ttl=120)
def candles(sym: str, interval: str) -> pd.DataFrame:
    days = {"5m":4,"15m":20,"30m":40,"1h":59,"1d":300
            }.get(interval, 20)
    end   = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(sym,
            start=start.strftime("%Y-%m-%d"),
            end=(end+timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        if not df.empty:
            df.index = (df.index.tz_convert(IST)
                        if df.index.tzinfo
                        else df.index.tz_localize("UTC")
                                     .tz_convert(IST))
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def bulk_prices(names: list) -> pd.DataFrame:
    syms = [STOCKS[n] for n in names if n in STOCKS]
    nmap = {STOCKS[n]:n for n in names if n in STOCKS}
    if not syms: return pd.DataFrame()
    try:
        import warnings
        warnings.filterwarnings("ignore")
        raw = yf.download(syms, period="2d", interval="1d",
                          auto_adjust=True, progress=False,
                          group_by="ticker")
        rows = []
        for s in syms:
            nm = nmap.get(s, s)
            try:
                # Handle both single and multi ticker downloads
                if len(syms) == 1:
                    cc = raw["Close"].dropna()
                    hh = raw["High"].dropna()
                    ll = raw["Low"].dropna()
                elif s in raw.columns.get_level_values(1):
                    cc = raw["Close"][s].dropna()
                    hh = raw["High"][s].dropna()
                    ll = raw["Low"][s].dropna()
                elif hasattr(raw, 'columns') and s in raw:
                    cc = raw[s]["Close"].dropna()
                    hh = raw[s]["High"].dropna()
                    ll = raw[s]["Low"].dropna()
                else:
                    raise ValueError("ticker not in data")
                if len(cc) < 2: raise ValueError("not enough rows")
                pr = float(cc.iloc[-1])
                pv = float(cc.iloc[-2])
                ch = round(((pr-pv)/pv)*100, 2)
                rows.append({"Name":nm,"Sym":s,
                    "Price":round(pr,2),"Chg%":ch,
                    "Chg₹":round(pr-pv,2),
                    "High":round(float(hh.iloc[-1]),2),
                    "Low": round(float(ll.iloc[-1]),2)})
            except:
                rows.append({"Name":nm,"Sym":s,"Price":None,
                             "Chg%":None,"Chg₹":None,
                             "High":None,"Low":None})
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_news(q: str) -> list:
    try:
        url  = (f"https://news.google.com/rss/search"
                f"?q={q.replace(' ','+')}"
                f"&hl=en-IN&gl=IN&ceid=IN:en")
        resp = requests.get(
            url, headers={"User-Agent":"Mozilla/5.0"},
            timeout=6)
        root = ET.fromstring(resp.content)
        out  = []
        for item in root.findall(".//item")[:10]:
            ttl = item.findtext("title","")
            lnk = item.findtext("link","")
            dt  = item.findtext("pubDate","")[:22]
            src = ""
            if " - " in ttl:
                ttl, src = ttl.rsplit(" - ",1)
            out.append({"title":ttl.strip(),"link":lnk,
                        "date":dt,"src":src.strip()})
        return out
    except:
        return []

# ══════════════════════════════════════════════════════════
# SIGNAL ENGINE  (indicators + 11-factor checklist)
# ══════════════════════════════════════════════════════════
def compute_all(df: pd.DataFrame, lp: dict) -> dict | None:
    if df is None or len(df) < 55: return None
    try:
        c = df["Close"].squeeze().astype(float)
        h = df["High"].squeeze().astype(float)
        l = df["Low"].squeeze().astype(float)
        v = df["Volume"].squeeze().astype(float)
        o = df["Open"].squeeze().astype(float)

        # ── Indicators ────────────────────────────────────
        e9   = ta.trend.ema_indicator(c, window=9)
        e21  = ta.trend.ema_indicator(c, window=21)
        e50  = ta.trend.ema_indicator(c, window=50)
        rsi  = ta.momentum.rsi(c, window=14)
        macd = ta.trend.macd(c)
        msig = ta.trend.macd_signal(c)
        adx  = ta.trend.adx(h, l, c, window=14)
        vwap = (c * v).cumsum() / v.cumsum()
        bbu  = ta.volatility.bollinger_hband(c, window=20)
        bbl  = ta.volatility.bollinger_lband(c, window=20)
        atr  = ta.volatility.average_true_range(
                   h, l, c, window=14)
        cmf  = ta.volume.chaikin_money_flow(
                   h, l, c, v, window=20)
        obv  = ta.volume.on_balance_volume(c, v)

        cp   = lp["p"] if lp["ok"] else float(c.iloc[-1])
        e9v  = float(e9.iloc[-1])
        e21v = float(e21.iloc[-1])
        e50v = float(e50.iloc[-1])
        rv   = float(rsi.iloc[-1])
        rv1  = float(rsi.iloc[-2])
        mv   = float(macd.iloc[-1])
        msv  = float(msig.iloc[-1])
        mv1  = float(macd.iloc[-2])
        msv1 = float(msig.iloc[-2])
        adxv = float(adx.iloc[-1])
        vwv  = float(vwap.iloc[-1])
        atrv = float(atr.iloc[-1])
        cmfv = float(cmf.iloc[-1])
        bbup = float(bbu.iloc[-1])
        bblw = float(bbl.iloc[-1])

        vol_avg   = float(v.tail(20).mean())
        vol_ratio = round(float(v.iloc[-1])/(vol_avg+1e-9),2)
        vsurge    = vol_ratio >= 1.2

        # OBV trend
        obv_bull = float(obv.iloc[-1]) > float(obv.iloc[-5])

        # Support / Resistance
        window = 5
        s_lvls, r_lvls = [], []
        for i in range(window, len(df)-window):
            if float(l.iloc[i]) == float(
                    l.iloc[i-window:i+window+1].min()):
                s_lvls.append(float(l.iloc[i]))
            if float(h.iloc[i]) == float(
                    h.iloc[i-window:i+window+1].max()):
                r_lvls.append(float(h.iloc[i]))
        sup = round(max([s for s in s_lvls if s < cp],
                        default=cp*0.98), 2)
        res = round(min([r for r in r_lvls if r > cp],
                        default=cp*1.02), 2)

        risk_pts   = round(cp - sup, 2)
        reward_pts = round(res - cp, 2)
        rr_ratio   = round(reward_pts/(risk_pts+0.001), 2)

        # ATR-based SL / targets
        sl_long   = round(cp - atrv * 1.5, 2)
        sl_short  = round(cp + atrv * 1.5, 2)
        tgt1      = round(cp + atrv * 1.5, 2)
        tgt2      = round(cp + atrv * 2.5, 2)
        tgt3      = round(cp + atrv * 4.0, 2)
        tgt1s     = round(cp - atrv * 1.5, 2)
        tgt2s     = round(cp - atrv * 2.5, 2)
        tgt3s     = round(cp - atrv * 4.0, 2)

        # Liquidity sweep
        rh20 = float(h.tail(20).max())
        rl20 = float(l.tail(20).min())
        sweep_low  = (float(l.iloc[-1]) < rl20*1.001 and
                      cp > rl20*1.002)
        sweep_high = (float(h.iloc[-1]) > rh20*0.999 and
                      cp < rh20*0.998)

        # BOS
        bos_bull = (float(h.iloc[-1]) > float(h.iloc[-2]) >
                    float(h.iloc[-3]) and
                    float(v.iloc[-1]) > vol_avg*1.3)
        bos_bear = (float(l.iloc[-1]) < float(l.iloc[-2]) <
                    float(l.iloc[-3]) and
                    float(v.iloc[-1]) > vol_avg*1.3)

        # Candlestick patterns
        body0  = abs(cp - float(o.iloc[-1]))
        avg_b  = float(abs(c - o).rolling(10).mean().iloc[-1])
        uw     = float(h.iloc[-1]) - max(cp, float(o.iloc[-1]))
        lw     = min(cp, float(o.iloc[-1])) - float(l.iloc[-1])
        c1v    = float(c.iloc[-2])
        o1v    = float(o.iloc[-2])
        h1v    = float(h.iloc[-2])
        l1v    = float(l.iloc[-2])
        c2v    = float(c.iloc[-3])
        o2v    = float(o.iloc[-3])

        patterns = []
        if body0 < avg_b*0.2:
            patterns.append(("⚡ Doji","neutral",
                             "Indecision — big move coming"))
        if lw > body0*2 and uw < body0*0.5 and c1v < o1v:
            patterns.append(("🔨 Hammer","bullish",
                             "Buyers rejected lows — reversal"))
        if uw > body0*2 and lw < body0*0.5 and c1v > o1v:
            patterns.append(("⭐ Shooting Star","bearish",
                             "Sellers rejected highs — reversal"))
        if (cp>float(o.iloc[-1]) and c1v<o1v and
                float(o.iloc[-1])<c1v and cp>o1v):
            patterns.append(("🟢 Bullish Engulfing","bullish",
                             "Bulls overwhelmed bears — strong buy"))
        if (cp<float(o.iloc[-1]) and c1v>o1v and
                float(o.iloc[-1])>c1v and cp<o1v):
            patterns.append(("🔴 Bearish Engulfing","bearish",
                             "Bears overwhelmed bulls — strong sell"))
        if (c2v<o2v and abs(c1v-o1v)<avg_b*0.4 and
                cp>float(o.iloc[-1]) and cp>(o2v+c2v)/2):
            patterns.append(("🌅 Morning Star","bullish",
                             "3-candle reversal — trend change"))
        if (c2v>o2v and abs(c1v-o1v)<avg_b*0.4 and
                cp<float(o.iloc[-1]) and cp<(o2v+c2v)/2):
            patterns.append(("🌆 Evening Star","bearish",
                             "3-candle reversal — trend change"))
        if (body0>avg_b*2.5 and uw<body0*0.1 and lw<body0*0.1):
            bias = "bullish" if cp>float(o.iloc[-1]) else "bearish"
            patterns.append((
                f"💪 {'Bull' if bias=='bullish' else 'Bear'} Marubozu",
                bias, "Strong momentum — continuation likely"))

        # ── 11-FACTOR CHECKLIST ───────────────────────────
        # Each factor: (label, passed, value_str, why)

        # Uptrend score (10 conditions)
        up_conds = {
            "EMA9>EMA21":  e9v > e21v,
            "EMA21>EMA50": e21v > e50v,
            "Price>EMA9":  cp > e9v,
            "Price>VWAP":  cp > vwv,
            "RSI 55-75":   55 < rv < 75,
            "RSI Rising":  rv > rv1,
            "MACD>Signal": mv > msv,
            "MACD Cross":  (mv>msv and mv1<=msv1),
            "ADX>20":      adxv > 20,
            "Vol Surge":   vsurge,
        }
        up_score = sum(up_conds.values())

        dn_conds = {
            "EMA9<EMA21":  e9v < e21v,
            "EMA21<EMA50": e21v < e50v,
            "Price<EMA9":  cp < e9v,
            "Price<VWAP":  cp < vwv,
            "RSI 25-45":   25 < rv < 45,
            "RSI Falling": rv < rv1,
            "MACD<Signal": mv < msv,
            "MACD Cross↓": (mv<msv and mv1>=msv1),
            "ADX>20":      adxv > 20,
            "Vol Surge":   vsurge,
        }
        dn_score = sum(dn_conds.values())

        direction = (
            "UPTREND"   if up_score >= 6 else
            "DOWNTREND" if dn_score >= 6 else
            "SIDEWAYS"
        )
        score = up_score if direction != "DOWNTREND" else dn_score

        # The 11 checklist items for CE trade
        tt = best_trading_time()
        good_time = tt in ("best","good","ok")

        ce_checklist = [
            ("1. Uptrend Score ≥ 7",
             up_score >= 7,
             f"{up_score}/10",
             "Core trend condition"),
            ("2. RSI 55–68 (momentum zone)",
             55 < rv < 68,
             f"{rv:.1f}",
             "Bullish but not overbought"),
            ("3. Price above VWAP",
             cp > vwv,
             f"Price ₹{cp:,.0f} | VWAP ₹{vwv:,.0f}",
             "Buyers in control today"),
            ("4. Volume Surge ≥ 1.2×",
             vsurge,
             f"{vol_ratio:.2f}× avg",
             "Institutional participation"),
            ("5. MACD above Signal",
             mv > msv,
             f"MACD {mv:.2f} | Sig {msv:.2f}",
             "Bullish momentum confirmed"),
            ("6. ADX > 20 (trending)",
             adxv > 20,
             f"ADX {adxv:.1f}",
             "Market is trending, not sideways"),
            ("7. CMF > 0 (money flowing in)",
             cmfv > 0,
             f"CMF {cmfv:+.3f}",
             "Institutional money entering"),
            ("8. OBV Rising",
             obv_bull,
             "Yes" if obv_bull else "No",
             "Volume confirming price"),
            ("9. Risk-Reward ≥ 1.5",
             rr_ratio >= 1.5,
             f"{rr_ratio}:1",
             "Reward must justify risk"),
            ("10. No Bearish Pattern",
             not any(p[1]=="bearish" for p in patterns),
             ", ".join(p[0] for p in patterns) or "None",
             "Candle pattern must not contradict"),
            ("11. Good Trading Time",
             good_time,
             tt.replace("_"," ").upper(),
             "Avoid first 15 min and last 30 min"),
        ]

        # PE checklist (mirror)
        pe_checklist = [
            ("1. Downtrend Score ≥ 7",
             dn_score >= 7,
             f"{dn_score}/10",
             "Core trend condition"),
            ("2. RSI 32–45 (bearish zone)",
             32 < rv < 45,
             f"{rv:.1f}",
             "Bearish but not oversold"),
            ("3. Price below VWAP",
             cp < vwv,
             f"Price ₹{cp:,.0f} | VWAP ₹{vwv:,.0f}",
             "Sellers in control today"),
            ("4. Volume Surge ≥ 1.2×",
             vsurge,
             f"{vol_ratio:.2f}× avg",
             "Institutional participation"),
            ("5. MACD below Signal",
             mv < msv,
             f"MACD {mv:.2f} | Sig {msv:.2f}",
             "Bearish momentum confirmed"),
            ("6. ADX > 20 (trending)",
             adxv > 20,
             f"ADX {adxv:.1f}",
             "Market is trending, not sideways"),
            ("7. CMF < 0 (money flowing out)",
             cmfv < 0,
             f"CMF {cmfv:+.3f}",
             "Institutional money exiting"),
            ("8. OBV Falling",
             not obv_bull,
             "Yes" if not obv_bull else "No",
             "Volume confirming bearish move"),
            ("9. Risk-Reward ≥ 1.5",
             rr_ratio >= 1.5,
             f"{rr_ratio}:1",
             "Reward must justify risk"),
            ("10. No Bullish Pattern",
             not any(p[1]=="bullish" for p in patterns),
             ", ".join(p[0] for p in patterns) or "None",
             "Candle pattern must not contradict"),
            ("11. Good Trading Time",
             good_time,
             tt.replace("_"," ").upper(),
             "Avoid first 15 min and last 30 min"),
        ]

        ce_pass = sum(1 for c in ce_checklist if c[1])
        pe_pass = sum(1 for c in pe_checklist if c[1])

        return dict(
            cp=cp, direction=direction, score=score,
            up_score=up_score, dn_score=dn_score,
            up_conds=up_conds, dn_conds=dn_conds,
            rv=rv, adxv=adxv, atrv=atrv,
            e9v=e9v, e21v=e21v, e50v=e50v, vwv=vwv,
            bbup=bbup, bblw=bblw, cmfv=cmfv,
            vol_ratio=vol_ratio, vsurge=vsurge,
            obv_bull=obv_bull, rr_ratio=rr_ratio,
            sup=sup, res=res,
            sl_long=sl_long, sl_short=sl_short,
            tgt1=tgt1, tgt2=tgt2, tgt3=tgt3,
            tgt1s=tgt1s, tgt2s=tgt2s, tgt3s=tgt3s,
            sweep_low=sweep_low, sweep_high=sweep_high,
            bos_bull=bos_bull, bos_bear=bos_bear,
            patterns=patterns,
            ce_checklist=ce_checklist, pe_checklist=pe_checklist,
            ce_pass=ce_pass, pe_pass=pe_pass,
            good_time=good_time, time_state=tt,
            # series
            e9s=e9, e21s=e21, e50s=e50,
            vwaps=vwap, rsis=rsi,
            macds=macd, msigs=msig,
            bbus=bbu, bbls=bbl,
            cmfs=cmf, obvs=obv,
        )
    except Exception as e:
        return None

# ══════════════════════════════════════════════════════════
# BLACK-SCHOLES
# ══════════════════════════════════════════════════════════
def bs(S,K,T,r,sig,t="CE"):
    if T<=0: return max(S-K,0) if t=="CE" else max(K-S,0)
    d1=(math.log(S/K)+(r+.5*sig**2)*T)/(sig*math.sqrt(T))
    d2=d1-sig*math.sqrt(T)
    if t=="CE":
        return max(round(S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2),2),0)
    return max(round(K*math.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1),2),0)

def greeks(S,K,T,r,sig,t="CE"):
    if T<=0: return {"d":0,"g":0,"th":0,"v":0}
    d1=(math.log(S/K)+(r+.5*sig**2)*T)/(sig*math.sqrt(T))
    d2=d1-sig*math.sqrt(T)
    g =norm.pdf(d1)/(S*sig*math.sqrt(T))
    vg=S*norm.pdf(d1)*math.sqrt(T)/100
    if t=="CE":
        dlt=norm.cdf(d1)
        th=(-(S*norm.pdf(d1)*sig)/(2*math.sqrt(T))
            -r*K*math.exp(-r*T)*norm.cdf(d2))/365
    else:
        dlt=norm.cdf(d1)-1
        th=(-(S*norm.pdf(d1)*sig)/(2*math.sqrt(T))
            +r*K*math.exp(-r*T)*norm.cdf(-d2))/365
    return {"d":round(dlt,4),"g":round(g,6),
            "th":round(th,2),"v":round(vg,2)}

# ══════════════════════════════════════════════════════════
# HEADER  (status bar + index row)
# ══════════════════════════════════════════════════════════
n      = now_ist()
mopen  = market_open()
tstate = best_trading_time()
tclr   = {"best":"#16a34a","good":"#15803d",
           "ok":"#d97706","caution":"#ea580c",
           "avoid":"#dc2626","closed":"#94a3b8",
           "pre_market":"#64748b"}.get(tstate,"#64748b")
tmsg   = {"best":"✅ BEST TIME — 9:30–10:30 AM",
           "good":"🟡 GOOD TIME — steady market",
           "ok":"🟡 OK — lunch hours",
           "caution":"⚠️ CAUTION — choppy",
           "avoid":"❌ AVOID — too volatile",
           "closed":"⛔ MARKET CLOSED",
           "pre_market":"🕐 PRE-MARKET"
          }.get(tstate,"—")

st.markdown(f"""
<div style='background:linear-gradient(135deg,#1e3a5f,#1d4ed8);
     border-radius:12px;padding:14px 22px;
     display:flex;justify-content:space-between;
     align-items:center;margin-bottom:12px;flex-wrap:wrap;
     gap:8px;box-shadow:0 4px 12px rgba(29,78,216,0.25)'>
  <span style='font-size:18px;font-weight:700;color:#ffffff;
               letter-spacing:-0.3px'>
      🎯 Intraday &amp; Options Terminal
  </span>
  <span style='background:{"rgba(22,163,74,0.25)" if mopen else "rgba(220,38,38,0.25)"};
               color:{"#86efac" if mopen else "#fca5a5"};
               font-weight:700;padding:4px 12px;
               border-radius:20px;font-size:13px;
               border:1px solid {"#86efac44" if mopen else "#fca5a544"}'>
      {"🟢 MARKET OPEN" if mopen else "🔴 MARKET CLOSED"}
  </span>
  <span style='color:#93c5fd;font-weight:600;
               font-size:13px'>{tmsg}</span>
  <span style='color:#bfdbfe;font-size:12px'>
      🕐 {n.strftime("%d %b %Y  %H:%M IST")}
  </span>
</div>
""", unsafe_allow_html=True)

# Index cards
idx = {"NIFTY":"^NSEI","BANKNIFTY":"^NSEBANK",
       "SENSEX":"^BSESN","NIFTY IT":"^CNXIT"}
ic  = st.columns(len(idx))
for i,(nm,sy) in enumerate(idx.items()):
    ip = live_price(sy)
    if ip["ok"]:
        col = "#00ff88" if ip["chg"]>=0 else "#ff4455"
        arr = "▲" if ip["chg"]>=0 else "▼"
        ic[i].markdown(f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;
             border-radius:12px;padding:12px;text-align:center;
             box-shadow:0 1px 3px rgba(0,0,0,0.06)'>
          <div style='color:#64748b;font-size:11px;font-weight:500;
                      letter-spacing:0.5px'>{nm}</div>
          <div style='color:#1e293b;font-size:22px;font-weight:700;
                      margin:4px 0'>
              {ip['p']:,.0f}</div>
          <div style='color:{col};font-size:13px;font-weight:600'>
              {arr}{abs(ip['chg']):.2f}%
              <span style='color:#94a3b8;font-size:11px;
                           font-weight:400'>
                  {ip['chg_abs']:+.1f}
              </span>
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin:6px 0'></div>",
            unsafe_allow_html=True)

# ── Open in new window panel ───────────────────────────────
with st.expander("🔗 Open any tab in a separate browser window"):
    st.caption(
        "Click any link below to open that tab in a new browser "
        "window. You can have multiple tabs open side by side — "
        "e.g. Trade Setup in one window and Scanner in another."
    )
    # Get the current base URL
    _base = "http://localhost:8501"

    link_cols = st.columns(5)
    link_data = [
        ("📋 Watchlist",     "watchlist", "#3b82f6"),
        ("🎯 Trade Setup",   "setup",     "#16a34a"),
        ("🔍 Auto Scanner",  "scanner",   "#9333ea"),
        ("🤖 ML Prediction", "ml",        "#0891b2"),
        ("🏦 Smart Money",   "smart",     "#d97706"),
        ("🧮 P&L Calc",      "calc",      "#dc2626"),
        ("📰 News",          "news",      "#475569"),
        ("📝 Paper Trade",   "paper",     "#15803d"),
        ("📓 Journal",       "journal",   "#1d4ed8"),
    ]
    for idx, (lname, lkey, lcolor) in enumerate(link_data):
        col_idx = idx % 5
        with link_cols[col_idx]:
            st.markdown(
                f"<a href='{_base}/?tab={lkey}' "
                f"target='_blank' "
                f"style='display:block;"
                f"background:{lcolor};"
                f"color:white;"
                f"text-decoration:none;"
                f"padding:8px 12px;"
                f"border-radius:8px;"
                f"font-size:13px;"
                f"font-weight:600;"
                f"text-align:center;"
                f"margin:3px 0;"
                f"font-family:Inter,sans-serif'>"
                f"{lname}</a>",
                unsafe_allow_html=True
            )
    st.markdown(
        "<div style='margin-top:8px;font-size:12px;"
        "color:#64748b'>"
        "💡 Tip: Bookmark your most used tabs for instant access. "
        "Works on same WiFi network too — use your Network URL "
        "from the terminal instead of localhost.</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ Terminal")

# Stock search
srch = st.sidebar.text_input(
    "🔍 Search stock",
    placeholder="e.g. Reliance, TCS, HDFC..."
)
if srch:
    q    = srch.strip().lower()
    hits = {k:v for k,v in STOCKS.items()
            if q in k.lower() or q in v.lower()}
    if hits:
        pk = st.sidebar.selectbox("Results",list(hits.keys()))
        if st.sidebar.button("✅ Load",type="primary"):
            st.session_state["sn"] = pk
            st.session_state["st"] = hits[pk]

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick pick**")

# Flat list — no expanders (expanders corrupt label text in Streamlit)
SIDEBAR_PICKS = {
    "Top F&O": [
        "NIFTY 50","BANK NIFTY","Reliance","HDFC Bank",
        "ICICI Bank","TCS","Infosys","SBI",
    ],
    "Banking": [
        "HDFC Bank","ICICI Bank","SBI","Axis Bank",
        "Kotak Bank","Bajaj Finance",
    ],
    "IT": [
        "TCS","Infosys","Wipro","HCL Tech",
        "Tech Mahindra","Persistent",
    ],
    "Energy": [
        "Reliance","ONGC","NTPC","Power Grid",
        "Tata Power","Gail",
    ],
}

# Show as a selectbox for sector then buttons for stocks
sidebar_sector = st.sidebar.selectbox(
    "Sector",
    list(SIDEBAR_PICKS.keys()),
    key="sidebar_sector_pick",
    label_visibility="collapsed"
)
for ni, nm in enumerate(SIDEBAR_PICKS[sidebar_sector]):
    sk = STOCKS.get(nm, "")
    if st.sidebar.button(
        nm,
        key=f"sb_{sidebar_sector[:3]}_{ni}",
        use_container_width=True
    ):
        st.session_state["sn"] = nm
        st.session_state["st"] = sk

st.sidebar.markdown("---")
tf = st.sidebar.selectbox(
    "⏱ Timeframe",
    ["1m","5m","15m","30m","1h","1d"], index=2)

auto_rf = st.sidebar.toggle("🔄 Auto Refresh (2 min)",False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Open in new window**")
_base_url = "http://localhost:8501"
for _lname, _lkey in [
    ("📋 Watchlist",    "watchlist"),
    ("🎯 Trade Setup",  "setup"),
    ("🔍 Scanner",      "scanner"),
    ("🤖 ML",           "ml"),
    ("📝 Paper Trade",  "paper"),
    ("📓 Journal",      "journal"),
]:
    st.sidebar.markdown(
        f"<a href='{_base_url}/?tab={_lkey}' "
        f"target='_blank' "
        f"style='display:block;"
        f"background:#f8fafc;"
        f"border:1px solid #e2e8f0;"
        f"color:#374151;"
        f"text-decoration:none;"
        f"padding:7px 12px;"
        f"border-radius:7px;"
        f"font-size:12px;"
        f"font-weight:500;"
        f"margin:2px 0;"
        f"font-family:Inter,sans-serif'>"
        f"{_lname} ↗</a>",
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎯 Entry Rules Summary
✅ CE (Buy Call):
- Score ≥ 7 + RSI 55–68
- Price > VWAP
- Volume surge ✅
- MACD > Signal
- 9 or more checks GREEN

✅ PE (Buy Put):
- Score ≥ 7 + RSI 32–45
- Price < VWAP
- Volume surge ✅
- MACD < Signal
- 9 or more checks GREEN

⛔ Never trade when:
- Score < 6
- RSI > 70 or < 30
- Time shows ❌ AVOID
""")

if "sn" not in st.session_state:
    st.session_state["sn"] = "NIFTY 50"
    st.session_state["st"] = "^NSEI"

sname = st.session_state["sn"]
stick = st.session_state["st"]

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
T1,T2,T3,T4,T5,T6,T7,T8,T9 = st.tabs(TAB_NAMES)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — WATCHLIST                                  ║
# ╚══════════════════════════════════════════════════════╝
with T1:
    st.markdown("### 📋 Live Stock Prices")
    wc1,wc2,wc3 = st.columns([2,1,1])
    with wc1:
        wsec = st.selectbox("Sector",list(SECTORS.keys()),
                            key="wsec_t1")
    with wc2:
        if st.button("🔄 Refresh",type="primary",key="wrf_t1"):
            st.cache_data.clear()
    with wc3:
        showall = st.checkbox("All stocks", key="showall_wl")

    slist = list(STOCKS.keys()) if showall else SECTORS[wsec]
    with st.spinner("Fetching live prices..."):
        pdf = bulk_prices(slist)

    if pdf.empty:
        st.error("Could not fetch. Check internet.")
    else:
        valid = pdf[pdf["Price"].notna()].copy()

        # ── Grid of cards ─────────────────────────────────
        n_cols = 4
        btn_idx = 0
        for chunk_start in range(0,len(valid),n_cols):
            chunk = valid.iloc[chunk_start:chunk_start+n_cols]
            cols  = st.columns(n_cols)
            for ci,(_, row) in enumerate(chunk.iterrows()):
                chg = row["Chg%"] or 0
                col = "#00ff88" if chg>=0 else "#ff4455"
                arr = "▲" if chg>=0 else "▼"
                with cols[ci]:
                    if st.button(
                        f"**{row['Name']}**\n"
                        f"₹{row['Price']:,.2f}  "
                        f"{arr}{abs(chg):.2f}%",
                        key=f"wlb_{btn_idx}",
                        width="stretch"):
                        st.session_state["sn"] = row["Name"]
                        st.session_state["st"] = row["Sym"]
                        st.rerun()
                btn_idx += 1

        # ── Full table ────────────────────────────────────
        st.markdown("---")
        tbl = valid[["Name","Price","Chg%","Chg₹",
                     "High","Low"]].copy()
        tbl.columns = ["Stock","Price ₹","Change %",
                       "Change ₹","Day High","Day Low"]
        tbl["Price ₹"]  = tbl["Price ₹"].apply(
            lambda x: f"₹{x:,.2f}")
        tbl["Day High"] = tbl["Day High"].apply(
            lambda x: f"₹{x:,.2f}")
        tbl["Day Low"]  = tbl["Day Low"].apply(
            lambda x: f"₹{x:,.2f}")
        tbl["Change %"] = tbl["Change %"].apply(
            lambda x: f"{x:+.2f}%")
        tbl["Change ₹"] = tbl["Change ₹"].apply(
            lambda x: f"{x:+.2f}")
        st.dataframe(tbl, width="stretch", hide_index=True)

with T1:
    st.markdown("### 📋 Live Stock Prices")
    wc1,wc2,wc3 = st.columns([2,1,1])
    with wc1:
        wsec = st.selectbox("Sector",list(SECTORS.keys()),
                            key="wsec")
    with wc2:
        if st.button("🔄 Refresh",type="primary",key="wrf"):
            st.cache_data.clear()
    with wc3:
        showall = st.checkbox("All stocks")

    slist = list(STOCKS.keys()) if showall else SECTORS[wsec]
    with st.spinner("Fetching live prices..."):
        pdf = bulk_prices(slist)

    if pdf.empty:
        st.error("Could not fetch. Check internet.")
    else:
        valid = pdf[pdf["Price"].notna()].copy()

        # ── Grid of cards ─────────────────────────────────
        n_cols = 4
        btn_idx = 0
        for chunk_start in range(0,len(valid),n_cols):
            chunk = valid.iloc[chunk_start:chunk_start+n_cols]
            cols  = st.columns(n_cols)
            for ci,(_, row) in enumerate(chunk.iterrows()):
                chg = row["Chg%"] or 0
                col = "#00ff88" if chg>=0 else "#ff4455"
                arr = "▲" if chg>=0 else "▼"
                with cols[ci]:
                    if st.button(
                        f"**{row['Name']}**\n"
                        f"₹{row['Price']:,.2f}  "
                        f"{arr}{abs(chg):.2f}%",
                        key=f"wla_{btn_idx}",
                        width="stretch"):
                        st.session_state["sn"] = row["Name"]
                        st.session_state["st"] = row["Sym"]
                        st.rerun()
                btn_idx += 1

        # ── Full table ────────────────────────────────────
        st.markdown("---")
        tbl = valid[["Name","Price","Chg%","Chg₹",
                     "High","Low"]].copy()
        tbl.columns = ["Stock","Price ₹","Change %",
                       "Change ₹","Day High","Day Low"]
        tbl["Price ₹"]  = tbl["Price ₹"].apply(
            lambda x: f"₹{x:,.2f}")
        tbl["Day High"] = tbl["Day High"].apply(
            lambda x: f"₹{x:,.2f}")
        tbl["Day Low"]  = tbl["Day Low"].apply(
            lambda x: f"₹{x:,.2f}")
        tbl["Change %"] = tbl["Change %"].apply(
            lambda x: f"{x:+.2f}%")
        tbl["Change ₹"] = tbl["Change ₹"].apply(
            lambda x: f"{x:+.2f}")
        st.dataframe(tbl, width="stretch", hide_index=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — TRADE SETUP                                ║
# ╚══════════════════════════════════════════════════════╝
with T2:
    # ── Live price card ───────────────────────────────────
    lp = live_price(stick)
    if lp["ok"]:
        pc  = "#00ff88" if lp["chg"]>=0 else "#ff4455"
        arr = "▲" if lp["chg"]>=0 else "▼"
        st.markdown(f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)' style='display:flex;
             justify-content:space-between;
             align-items:center;flex-wrap:wrap;gap:12px'>
          <div>
            <div style='color:#64748b;font-size:13px'>
                {sname}
                <span style='color:#94a3b8;
                             margin-left:8px'>{stick}</span>
            </div>
            <div style='font-size:34px;font-weight:700;line-height:1.1' style='color:#1e293b'>
                ₹{lp['p']:,.2f}
            </div>
            <div style='color:{pc};font-size:17px;
                        margin-top:4px'>
                {arr} {abs(lp['chg']):.2f}%
                <span style='color:#94a3b8;font-size:13px'>
                    ({lp['chg_abs']:+.2f})
                </span>
            </div>
          </div>
          <div style='color:#64748b;font-size:13px;
                      line-height:2.2;text-align:right'>
            High <b style='color:#1e293b'>
                ₹{lp['high']:,}</b><br>
            Low  <b style='color:#1e293b'>
                ₹{lp['low']:,}</b><br>
            Prev <b style='color:#1e293b'>
                ₹{lp['prev']:,}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Could not fetch live price")

    # Fetch + compute
    with st.spinner("Analysing..."):
        df = candles(stick, tf)

    col_info, col_ref = st.columns([4,1])
    with col_info:
        if not df.empty:
            st.caption(
                f"✅ {len(df)} candles | "
                f"Last: {df.index[-1].strftime('%d %b %H:%M')} IST"
                f" | ⚠️ ~15 min delay"
            )
    with col_ref:
        if st.button("🔄 Refresh",type="primary",
                     key="t2_ref",width="stretch"):
            st.cache_data.clear(); st.rerun()

    if df.empty or len(df) < 55:
        st.error("Not enough data. Try 1d timeframe.")
        st.stop()

    sig = compute_all(df, lp)
    if not sig:
        st.error("Could not calculate signals.")
        st.stop()

    cp = sig["cp"]

    # ── Trade direction cards ──────────────────────────────
    st.markdown("---")
    d1c, d2c = st.columns(2)

    def score_bar(score, max_score=10):
        pct  = int(score/max_score*100)
        col  = ("#00ff88" if pct>=70
                else "#ffcc00" if pct>=50
                else "#ff4455")
        return (f"<div style='background:#f1f5f9;border-radius:4px;"
                f"height:8px;width:100%;margin-top:6px'>"
                f"<div style='background:{col};width:{pct}%;"
                f"height:8px;border-radius:4px'></div></div>")

    with d1c:
        uc = ("#00ff88" if sig['up_score']>=7
              else "#ffcc00" if sig['up_score']>=5
              else "#ff4455")
        st.markdown(f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
          <div style='font-size:11px;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>UPTREND SCORE</div>
          <div style='font-size:52px;font-weight:700;
                      color:{uc};line-height:1'>
              {sig['up_score']}/10
          </div>
          <div style='color:{uc};font-size:13px;
                      margin-top:4px'>
              {'🔥 BUY CE' if sig['up_score']>=7
               else '⏳ WAIT' if sig['up_score']>=5
               else '❌ NO CE TRADE'}
          </div>
          {score_bar(sig['up_score'])}
          <div style='margin-top:10px;font-size:13px;
                      color:#555'>
              CE checklist: {sig['ce_pass']}/11 passed
          </div>
        </div>
        """, unsafe_allow_html=True)

    with d2c:
        dc = ("#ff4455" if sig['dn_score']>=7
              else "#ffcc00" if sig['dn_score']>=5
              else "#555")
        st.markdown(f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
          <div style='font-size:11px;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>DOWNTREND SCORE</div>
          <div style='font-size:52px;font-weight:700;
                      color:{dc};line-height:1'>
              {sig['dn_score']}/10
          </div>
          <div style='color:{dc};font-size:13px;
                      margin-top:4px'>
              {'🔥 BUY PE' if sig['dn_score']>=7
               else '⏳ WAIT' if sig['dn_score']>=5
               else '❌ NO PE TRADE'}
          </div>
          {score_bar(sig['dn_score'])}
          <div style='margin-top:10px;font-size:13px;
                      color:#555'>
              PE checklist: {sig['pe_pass']}/11 passed
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 11-FACTOR CHECKLISTS ───────────────────────────────
    st.markdown("---")
    st.markdown("### ✅ Pre-Trade Checklist")
    st.caption(
        "All 11 factors must be green before entering. "
        "Even one red factor means wait."
    )

    cl1, cl2 = st.columns(2)

    def render_checklist(checklist, title, color):
        passed = sum(1 for c in checklist if c[1])
        total  = len(checklist)
        pct    = int(passed/total*100)
        hdr_col= ("#00ff88" if pct>=82
                  else "#ffcc00" if pct>=60
                  else "#ff4455")
        st.markdown(f"""
        <div style='font-size:15px;font-weight:600;
                    color:{hdr_col};margin-bottom:8px'>
            {title} — {passed}/{total} passed
            <span style='font-size:12px;color:#64748b;
                         margin-left:8px'>
                ({pct}% ready)
            </span>
        </div>
        """, unsafe_allow_html=True)
        for label, passed_, value, why in checklist:
            css_style = ("background:#071407;border-left:3px solid #00ff88" if passed_ 
                     else "background:#140707;border-left:3px solid #ff4455")
            ico = "✅" if passed_ else "❌"
            st.markdown(f"""
            <div style='{css_style};border-radius:6px;padding:9px 14px;margin:3px 0;font-size:14px;color:#ccc'>
              <span style='font-weight:600'>{ico} {label}</span>
              <span style='float:right;color:#6b7280;
                           font-size:12px'>{value}</span>
              <div style='font-size:11px;color:#64748b;
                          margin-top:2px'>{why}</div>
            </div>
            """, unsafe_allow_html=True)

    with cl1:
        render_checklist(
            sig["ce_checklist"],
            "📈 CE (CALL) — Bullish",
            "#00ff88"
        )
    with cl2:
        render_checklist(
            sig["pe_checklist"],
            "📉 PE (PUT) — Bearish",
            "#ff4455"
        )

    # ── ENTRY / EXIT BOXES ────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Entry & Exit Points")

    # Smart money alerts first
    if sig["sweep_low"]:
        st.success(
            "🚀 **LIQUIDITY SWEEP OF LOWS** — "
            "Institutions just triggered stop losses and bought. "
            "Highest probability CE setup right now!"
        )
    if sig["sweep_high"]:
        st.error(
            "⚠️ **LIQUIDITY SWEEP OF HIGHS** — "
            "Institutions just triggered stop losses and sold. "
            "Highest probability PE setup right now!"
        )
    if sig["bos_bull"]:
        st.success(
            "📊 **BULLISH BREAK OF STRUCTURE** — "
            "Institutions confirmed uptrend with high volume."
        )
    if sig["bos_bear"]:
        st.error(
            "📊 **BEARISH BREAK OF STRUCTURE** — "
            "Institutions confirmed downtrend with high volume."
        )

    # Candlestick patterns
    if sig["patterns"]:
        for pname, pbias, pmeaning in sig["patterns"]:
            pc_ = ("#00ff88" if pbias=="bullish"
                   else "#ff4455" if pbias=="bearish"
                   else "#ffcc00")
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)' style='border-left:3px solid {pc_}'>
              <span style='color:{pc_};font-weight:600'>
                  {pname}
              </span>
              <span style='color:#666;font-size:13px;
                           margin-left:10px'>
                  {pmeaning}
              </span>
            </div>
            """, unsafe_allow_html=True)

    # Determine recommended action
    ce_ready = sig["ce_pass"] >= 9 and sig["up_score"] >= 7
    pe_ready = sig["pe_pass"] >= 9 and sig["dn_score"] >= 7

    if ce_ready:
        # ── BUY CE entry box ──────────────────────────────
        st.success(
            f"🟢 BUY CE (CALL OPTION) — "
            f"{sig['ce_pass']}/11 checks passed"
        )

        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.markdown(
                "<div style='background:#f0fdf4;border-radius:8px;"
                "padding:16px;text-align:center;height:140px'>"
                "<div style='font-size:11px;color:#64748b;letter-spacing:1px;"
                "text-transform:uppercase;margin-bottom:8px'>Entry Zone</div>"
                f"<div style='color:#00ff88;font-size:22px;font-weight:700'>"
                f"₹{sig['e9v']:,.2f}</div>"
                "<div style='color:#64748b;font-size:12px;margin-top:8px'>"
                "Wait for pullback to EMA9<br>then enter on next green candle"
                "</div></div>",
                unsafe_allow_html=True
            )
        with ec2:
            st.markdown(
                "<div style='background:#fef2f2;border-radius:8px;"
                "padding:16px;text-align:center;height:140px'>"
                "<div style='font-size:11px;color:#64748b;letter-spacing:1px;"
                "text-transform:uppercase;margin-bottom:8px'>Stop Loss</div>"
                f"<div style='color:#ff4455;font-size:22px;font-weight:700'>"
                f"₹{sig['sl_long']:,.2f}</div>"
                "<div style='color:#64748b;font-size:12px;margin-top:8px'>"
                "ATR-based SL<br>"
                f"Exit if price closes below EMA21<br>(₹{sig['e21v']:,.2f})"
                "</div></div>",
                unsafe_allow_html=True
            )
        with ec3:
            st.markdown(
                "<div style='background:#eff6ff;border-radius:8px;"
                "padding:16px;text-align:center;height:140px'>"
                "<div style='font-size:11px;color:#64748b;letter-spacing:1px;"
                "text-transform:uppercase;margin-bottom:8px'>Targets</div>"
                f"<div style='color:#88aaff;font-size:15px;font-weight:600;"
                f"line-height:1.8'>T1 ₹{sig['tgt1']:,.2f}<br>"
                f"T2 ₹{sig['tgt2']:,.2f}<br>"
                f"T3 ₹{sig['tgt3']:,.2f}</div>"
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            f"**Risk-Reward:** {sig['rr_ratio']}:1 &nbsp;|&nbsp; "
            f"**ATR:** ₹{sig['atrv']:,.2f} &nbsp;|&nbsp; "
            f"**Support:** ₹{sig['sup']:,} &nbsp;|&nbsp; "
            f"**Resistance:** ₹{sig['res']:,}"
        )

        st.markdown("#### ⚠️ Exit Rules — Follow without exception")
        st.warning(
            f"🔴 **Stop loss hit** → Price closes below "
            f"₹{sig['sl_long']:,.2f} → Exit immediately  \n"
            f"📊 **RSI overbought** → RSI crosses above 72 → Book profit  \n"
            f"🎯 **Target reached** → At T1 (₹{sig['tgt1']:,.2f}) book 50% qty  \n"
            f"&nbsp;&nbsp; At T2 (₹{sig['tgt2']:,.2f}) book another 30%  \n"
            f"⏰ **Time exit** → Exit ALL positions by 2:45 PM  \n"
            f"📉 **EMA21 break** → Price closes below ₹{sig['e21v']:,.2f} → Exit"
        )

    elif pe_ready:
        # ── BUY PE entry box ──────────────────────────────
        st.error(
            f"🔴 BUY PE (PUT OPTION) — "
            f"{sig['pe_pass']}/11 checks passed"
        )

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown(
                "<div style='background:#fef2f2;border-radius:8px;"
                "padding:16px;text-align:center;height:140px'>"
                "<div style='font-size:11px;color:#64748b;letter-spacing:1px;"
                "text-transform:uppercase;margin-bottom:8px'>Entry Zone</div>"
                f"<div style='color:#ff4455;font-size:22px;font-weight:700'>"
                f"₹{sig['e9v']:,.2f}</div>"
                "<div style='color:#64748b;font-size:12px;margin-top:8px'>"
                "Wait for bounce to EMA9<br>then enter on next red candle"
                "</div></div>",
                unsafe_allow_html=True
            )
        with pc2:
            st.markdown(
                "<div style='background:#1a0800;border-radius:8px;"
                "padding:16px;text-align:center;height:140px'>"
                "<div style='font-size:11px;color:#64748b;letter-spacing:1px;"
                "text-transform:uppercase;margin-bottom:8px'>Stop Loss</div>"
                f"<div style='color:#ff8844;font-size:22px;font-weight:700'>"
                f"₹{sig['sl_short']:,.2f}</div>"
                "<div style='color:#64748b;font-size:12px;margin-top:8px'>"
                "ATR-based SL<br>"
                f"Exit if price closes above EMA21<br>(₹{sig['e21v']:,.2f})"
                "</div></div>",
                unsafe_allow_html=True
            )
        with pc3:
            st.markdown(
                "<div style='background:#eff6ff;border-radius:8px;"
                "padding:16px;text-align:center;height:140px'>"
                "<div style='font-size:11px;color:#64748b;letter-spacing:1px;"
                "text-transform:uppercase;margin-bottom:8px'>Targets</div>"
                f"<div style='color:#88aaff;font-size:15px;font-weight:600;"
                f"line-height:1.8'>T1 ₹{sig['tgt1s']:,.2f}<br>"
                f"T2 ₹{sig['tgt2s']:,.2f}<br>"
                f"T3 ₹{sig['tgt3s']:,.2f}</div>"
                "</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            f"**Risk-Reward:** {sig['rr_ratio']}:1 &nbsp;|&nbsp; "
            f"**ATR:** ₹{sig['atrv']:,.2f} &nbsp;|&nbsp; "
            f"**Support:** ₹{sig['sup']:,} &nbsp;|&nbsp; "
            f"**Resistance:** ₹{sig['res']:,}"
        )

        st.markdown("#### ⚠️ Exit Rules — Follow without exception")
        st.warning(
            f"🔴 **Stop loss hit** → Price closes above "
            f"₹{sig['sl_short']:,.2f} → Exit immediately  \n"
            f"📊 **RSI oversold** → RSI crosses below 28 → Book profit  \n"
            f"🎯 **Target reached** → At T1 (₹{sig['tgt1s']:,.2f}) book 50%  \n"
            f"&nbsp;&nbsp; At T2 (₹{sig['tgt2s']:,.2f}) book another 30%  \n"
            f"⏰ **Time exit** → Exit ALL positions by 2:45 PM  \n"
            f"📈 **EMA21 break** → Price closes above ₹{sig['e21v']:,.2f} → Exit"
        )

    else:
        # ── No trade ready ────────────────────────────────
        missing_ce = [c[0] for c in sig["ce_checklist"] if not c[1]]
        missing_pe = [c[0] for c in sig["pe_checklist"] if not c[1]]

        st.warning(
            f"⏳ **NO TRADE YET** — "
            f"CE {sig['ce_pass']}/11 | PE {sig['pe_pass']}/11  \n\n"
            f"**Missing for CE:** {', '.join(missing_ce[:4])}  \n"
            f"**Missing for PE:** {', '.join(missing_pe[:4])}  \n\n"
            "Wait for score to reach 7+ and 9+ checks green before entering."
        )

    # ── CHART ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🕯️ {sname} — {tf}")

    plot_df = df.tail(100).copy()
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.55,0.15,0.15,0.15],
        vertical_spacing=0.02,
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"],  high=plot_df["High"],
        low=plot_df["Low"],    close=plot_df["Close"],
        name="Price",
        increasing_line_color="lime",
        decreasing_line_color="#ff4455",
    ), row=1, col=1)

    # EMAs
    for ser,col_,nm in [
        (sig["e9s"],"yellow","EMA9"),
        (sig["e21s"],"orange","EMA21"),
        (sig["e50s"],"cyan","EMA50"),
    ]:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=ser.tail(100),
            line=dict(color=col_,width=1.2),name=nm
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=sig["vwaps"].tail(100),
        line=dict(color="white",width=1.2,dash="dash"),
        name="VWAP"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=sig["bbus"].tail(100),
        line=dict(color="rgba(60,200,100,0.4)",
                  width=1,dash="dot"),name="BB Upper"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=sig["bbls"].tail(100),
        line=dict(color="rgba(200,60,60,0.4)",
                  width=1,dash="dot"),name="BB Lower",
        fill="tonexty",
        fillcolor="rgba(255,255,255,0.02)"
    ), row=1, col=1)

    # SL lines on chart
    if ce_ready:
        fig.add_hline(
            y=sig["sl_long"],
            line_dash="dash",line_color="#ff4455",
            opacity=0.7,
            annotation_text=f"SL ₹{sig['sl_long']:,}",
            annotation_position="right",
            row=1, col=1)
        for i,(tv,tn) in enumerate([
                (sig["tgt1"],"T1"),
                (sig["tgt2"],"T2"),
                (sig["tgt3"],"T3")]):
            fig.add_hline(
                y=tv,
                line_dash="dot",line_color="#00ff88",
                opacity=0.5,
                annotation_text=f"{tn} ₹{tv:,}",
                annotation_position="right",
                row=1, col=1)

    elif pe_ready:
        fig.add_hline(
            y=sig["sl_short"],
            line_dash="dash",line_color="#ff8844",
            opacity=0.7,
            annotation_text=f"SL ₹{sig['sl_short']:,}",
            annotation_position="right",
            row=1, col=1)
        for tv,tn in [
                (sig["tgt1s"],"T1"),
                (sig["tgt2s"],"T2"),
                (sig["tgt3s"],"T3")]:
            fig.add_hline(
                y=tv,
                line_dash="dot",line_color="#ff4455",
                opacity=0.5,
                annotation_text=f"{tn} ₹{tv:,}",
                annotation_position="right",
                row=1, col=1)

    # Volume
    vc = ["lime" if float(plot_df["Close"].iloc[i]) >=
                    float(plot_df["Open"].iloc[i])
          else "#ff4455"
          for i in range(len(plot_df))]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["Volume"],
        marker_color=vc, name="Volume", opacity=0.7
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=sig["rsis"].tail(100),
        line=dict(color="violet",width=1.5), name="RSI"
    ), row=3, col=1)
    for lvl,col_ in [(70,"red"),(68,"lime"),
                     (45,"orange"),(30,"red")]:
        fig.add_hline(y=lvl, line_dash="dash",
                      line_color=col_,opacity=0.4,
                      row=3, col=1)
    fig.add_hrect(y0=55,y1=68,
                  fillcolor="rgba(0,255,100,0.05)",
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=32,y1=45,
                  fillcolor="rgba(255,60,60,0.05)",
                  line_width=0, row=3, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=sig["macds"].tail(100),
        line=dict(color="yellow",width=1.3), name="MACD"
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=sig["msigs"].tail(100),
        line=dict(color="#ff4455",width=1.3), name="Signal"
    ), row=4, col=1)
    hist = (sig["macds"]-sig["msigs"]).tail(100)
    fig.add_trace(go.Bar(
        x=plot_df.index, y=hist,
        marker_color=["lime" if v>=0 else "#ff4455"
                      for v in hist],
        name="Hist", opacity=0.6
    ), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot",
                  line_color="#333", row=4, col=1)

    fig.update_layout(
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=740,
        margin=dict(l=10,r=10,t=30,b=10),
        legend=dict(orientation="h",y=1.01,x=0,
                    font=dict(size=11)),
        title=(f"{sname} | {tf} | "
               "🟡EMA9 🟠EMA21 🔵EMA50 ⬜VWAP | "
               "Green zone=Buy RSI | Red zone=Sell RSI")
    )
    fig.update_yaxes(title_text="₹",     row=1, col=1)
    fig.update_yaxes(title_text="Vol",   row=2, col=1)
    fig.update_yaxes(title_text="RSI",
                     range=[0,100],      row=3, col=1)
    fig.update_yaxes(title_text="MACD",  row=4, col=1)
    st.plotly_chart(fig, width="stretch")

    # Key levels summary
    st.markdown("### 📌 Key Levels Summary")
    kl1,kl2,kl3,kl4,kl5,kl6 = st.columns(6)
    kl1.metric("Support",    f"₹{sig['sup']:,}")
    kl2.metric("Resistance", f"₹{sig['res']:,}")
    kl3.metric("VWAP",       f"₹{sig['vwv']:,.0f}")
    kl4.metric("EMA9",       f"₹{sig['e9v']:,.0f}")
    kl5.metric("EMA21",      f"₹{sig['e21v']:,.0f}")
    kl6.metric("ATR",        f"₹{sig['atrv']:,.1f}")

    with st.expander("📋 Last 20 candles"):
        t20 = df.tail(20)[["Open","High","Low",
                            "Close","Volume"]].copy()
        t20["Chg%"] = (t20["Close"].pct_change()*100).round(2)
        t20.index   = t20.index.strftime("%d-%b %H:%M")
        st.dataframe(t20.round(2), width="stretch")

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — SMART MONEY                                ║
# ╚══════════════════════════════════════════════════════╝

with T3:
    st.markdown("### 🔍 Auto Scanner — 30 Stocks Live")
    st.caption(
        "Scans up to 30 stocks simultaneously every few minutes. "
        "Scans 30 stocks simultaneously. Sends Telegram alerts when strong signals fire."
    )

    # ── Scanner stock universe ─────────────────────────────
    SCANNER_UNIVERSE = {
        "🏆 Top 30 F&O": [
            "NIFTY 50","BANK NIFTY","Reliance","HDFC Bank",
            "ICICI Bank","TCS","Infosys","SBI","Wipro",
            "Bajaj Finance","ITC","Sun Pharma","L&T","Maruti",
            "Coal India","NTPC","Bharti Airtel","Tata Steel",
            "Axis Bank","HCL Tech","Power Grid","Adani Ports",
            "Hindalco","ONGC","Bajaj Auto","Titan","Grasim",
            "JSW Steel","UltraTech","BEL"
        ],
        "🏦 Banking 15": [
            "HDFC Bank","ICICI Bank","SBI","Kotak Bank",
            "Axis Bank","IndusInd Bank","Bajaj Finance",
            "PNB","Bank of Baroda","Canara Bank",
            "Federal Bank","IDFC First","Bajaj Finserv",
            "Yes Bank","HDFC Bank"
        ],
        "💻 IT 12": [
            "TCS","Infosys","Wipro","HCL Tech",
            "Tech Mahindra","Persistent","Coforge",
            "LTIMindtree","Mphasis","Tata Elxsi",
            "L&T","KPIT Tech"
        ],
        "🚗 Auto + Energy 15": [
            "Tata Motors DVR","Maruti","M&M","Hero MotoCorp",
            "Bajaj Auto","TVS Motor","Eicher Motors",
            "Reliance","ONGC","Indian Oil","BPCL",
            "NTPC","Power Grid","Tata Power","Gail"
        ],
    }

    # ── Telegram setup ────────────────────────────────────
    st.markdown("#### 📱 Telegram Alert Setup")
    with st.expander(
        "Configure Telegram alerts — click to setup",
        expanded=False
    ):
        st.markdown("""
        **Your own private Telegram bot — 100% free, no third party.**

        **One-time setup (3 minutes):**

        **Step 1 — Create your bot:**
        - Open Telegram on your phone
        - Search for **@BotFather**
        - Send: `/newbot`
        - Give it a name e.g. `My Trading Bot`
        - Give it a username e.g. `ranjith_trading_bot`
        - BotFather sends you a **Token** — copy it
          (looks like: `7123456789:AAHxxxxxxxxxxxxxxx`)

        **Step 2 — Get your Chat ID:**
        - Search for **@userinfobot** in Telegram
        - Send any message to it
        - It replies with your **Chat ID** (a number like `987654321`)

        **Step 3 — Enter both below and test:**
        """)

        tg1, tg2 = st.columns(2)
        with tg1:
            tg_token = st.text_input(
                "Bot Token",
                placeholder="7123456789:AAHxxxxxxxxxxx",
                type="password",
                key="tg_token",
                help="From @BotFather on Telegram"
            )
        with tg2:
            tg_chat = st.text_input(
                "Your Chat ID",
                placeholder="987654321",
                key="tg_chat",
                help="From @userinfobot on Telegram"
            )

        if tg_token and tg_chat:
            if st.button("Send Test Message", key="tg_test",
                         type="primary"):
                try:
                    test_url = (
                        f"https://api.telegram.org/bot{tg_token}"
                        f"/sendMessage"
                    )
                    resp = requests.post(test_url, json={
                        "chat_id": tg_chat,
                        "text": (
                            "Trading Terminal connected! "
                            "You will now receive trade signals here. "
                            "Score 8+ signals will be sent automatically."
                        ),
                        "parse_mode": "HTML"
                    }, timeout=10)
                    if resp.status_code == 200:
                        st.success(
                            "Test message sent! "
                            "Check your Telegram."
                        )
                        st.session_state["tg_token_saved"] = tg_token
                        st.session_state["tg_chat_saved"]  = tg_chat
                    else:
                        err = resp.json().get("description","")
                        st.error(f"Failed: {err}. Check token and chat ID.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.session_state["tg_token_saved"] = tg_token
                st.session_state["tg_chat_saved"]  = tg_chat
                st.success(
                    "Telegram configured. "
                    "Click Test to verify, "
                    "then alerts will fire automatically at score 8+."
                )

    st.markdown("---")

    # ── Scanner controls ───────────────────────────────────
    sc1, sc2, sc3, sc4 = st.columns([2,1,1,1])
    with sc1:
        scan_group = st.selectbox(
            "Stock group to scan",
            list(SCANNER_UNIVERSE.keys()),
            key="scan_group"
        )
    with sc2:
        scan_tf = st.selectbox(
            "Timeframe",
            ["15m","30m","1h","1d"],
            index=0,
            key="scan_tf"
        )
    with sc3:
        min_score_scan = st.slider(
            "Min score",
            0, 10, 6,
            key="min_score_scan"
        )
    with sc4:
        alert_score = st.slider(
            "Alert at score",
            6, 10, 8,
            key="alert_score",
            help="Send Telegram alert when score reaches this"
        )

    run_col, auto_col = st.columns([1,1])
    with run_col:
        run_scanner = st.button(
            "🚀 Scan All Stocks Now",
            type="primary",
            key="run_scanner",
            use_container_width=True
        )
    with auto_col:
        auto_scan = st.toggle(
            "🔄 Auto scan every 5 min",
            value=False,
            key="auto_scan"
        )

    # ── Helper: send Telegram ─────────────────────────────
    def send_telegram(token, chat_id, message):
        """
        Sends a message via your private Telegram bot.
        Uses official Telegram Bot API — no third party.
        """
        try:
            url  = (f"https://api.telegram.org/bot{token}"
                    f"/sendMessage")
            resp = requests.post(url, json={
                "chat_id":    chat_id,
                "text":       message,
                "parse_mode": "HTML"
            }, timeout=8)
            return resp.status_code == 200
        except:
            return False

    # ── Run scanner ────────────────────────────────────────
    def run_scan_engine(stocks_to_scan, timeframe, min_sc, alert_sc):
        results   = []
        alerted   = []
        prog      = st.progress(0, text="Starting scan...")
        total     = len(stocks_to_scan)

        for i, sname in enumerate(stocks_to_scan):
            sym = STOCKS.get(sname)
            if not sym:
                prog.progress(int((i+1)/total*100),
                              text=f"Skipping {sname}...")
                continue

            prog.progress(
                int((i+1)/total*100),
                text=f"Scanning {sname}... ({i+1}/{total})"
            )

            try:
                sdf = candles(sym, timeframe)
                if sdf.empty or len(sdf) < 55:
                    continue

                slp  = live_price(sym)
                sig  = compute_all(sdf, slp)
                if sig is None:
                    continue

                up_s = sig["up_score"]
                dn_s = sig["dn_score"]
                best = max(up_s, dn_s)

                if best < min_sc:
                    continue

                direction = (
                    "UPTREND"   if up_s > dn_s else
                    "DOWNTREND" if dn_s > up_s else
                    "MIXED"
                )
                action = (
                    "BUY CE" if direction == "UPTREND"
                    else "BUY PE" if direction == "DOWNTREND"
                    else "WAIT"
                )
                signal_strength = (
                    "STRONG" if best >= 8 else
                    "GOOD"   if best >= 6 else
                    "WATCH"
                )

                result = {
                    "Stock":      sname,
                    "Sym":        sym,
                    "Score":      best,
                    "Direction":  direction,
                    "Action":     action,
                    "Strength":   signal_strength,
                    "Price":      sig["cp"],
                    "Change%":    sig["change"] if "change" in sig else 0,
                    "RSI":        sig["rv"],
                    "ADX":        sig["adxv"],
                    "VolSurge":   sig["vsurge"],
                    "EMA9":       sig["e9v"],
                    "SL":         sig["sl_long"] if direction=="UPTREND"
                                  else sig["sl_short"],
                    "T1":         sig["tgt1"] if direction=="UPTREND"
                                  else sig["tgt1s"],
                    "T2":         sig["tgt2"] if direction=="UPTREND"
                                  else sig["tgt2s"],
                }
                results.append(result)

                # Telegram alert for strong signals
                if (best >= alert_sc and
                        "tg_token_saved" in st.session_state and
                        "tg_chat_saved"  in st.session_state):
                    token   = st.session_state["tg_token_saved"]
                    chat_id = st.session_state["tg_chat_saved"]
                    if token and chat_id:
                        sl_val = result["SL"]
                        t1_val = result["T1"]
                        t2_val = result["T2"]
                        msg = (
                            f"<b>SIGNAL: {sname}</b>\n"
                            f"Direction: {direction}\n"
                            f"Action: <b>{action}</b>\n"
                            f"Score: {best}/10\n"
                            f"Price: Rs {sig['cp']:,.2f}\n"
                            f"Entry EMA9: Rs {sig['e9v']:,.2f}\n"
                            f"Stop Loss: Rs {sl_val:,.2f}\n"
                            f"Target 1: Rs {t1_val:,.2f}\n"
                            f"Target 2: Rs {t2_val:,.2f}\n"
                            f"RSI: {sig['rv']:.1f} | ADX: {sig['adxv']:.1f}\n"
                            f"Vol: {'Yes' if sig['vsurge'] else 'No'} | TF: {timeframe}"
                        )
                        sent = send_telegram(token, chat_id, msg)
                        if sent:
                            alerted.append(sname)

            except Exception:
                continue

        prog.empty()
        return results, alerted

    # ── Display results ────────────────────────────────────
    if run_scanner or auto_scan:
        if auto_scan and not run_scanner:
            st.info("Auto scan active — running every 5 minutes")

        stocks_to_scan = SCANNER_UNIVERSE[scan_group]
        st.markdown(f"**Scanning {len(stocks_to_scan)} stocks "
                    f"on {scan_tf} timeframe...**")

        with st.spinner(""):
            results, alerted = run_scan_engine(
                stocks_to_scan, scan_tf,
                min_score_scan, alert_score
            )

        if not results:
            st.warning(
                "No stocks met the minimum score. "
                "Try lowering the min score or use 1d timeframe."
            )
        else:
            # Sort by score
            results.sort(key=lambda x: x["Score"], reverse=True)

            # Summary metrics
            strong   = [r for r in results if r["Score"] >= 8]
            good     = [r for r in results if 6 <= r["Score"] < 8]
            watch    = [r for r in results if r["Score"] < 6]
            ce_list  = [r for r in results if r["Direction"]=="UPTREND"]
            pe_list  = [r for r in results if r["Direction"]=="DOWNTREND"]

            sm1,sm2,sm3,sm4,sm5 = st.columns(5)
            sm1.metric("Scanned",    len(stocks_to_scan))
            sm2.metric("Signals",    len(results))
            sm3.metric("Strong 8+",  len(strong))
            sm4.metric("BUY CE",     len(ce_list))
            sm5.metric("BUY PE",     len(pe_list))

            if alerted:
                st.success(
                    f"Telegram alerts sent for: "
                    f"{', '.join(alerted)}"
                )

            st.markdown(f"*Scanned at "
                        f"{now_ist().strftime('%H:%M:%S IST')} | "
                        f"Timeframe: {scan_tf}*")

            # ── Strong signals (score 8+) ──────────────────
            if strong:
                st.markdown("---")
                st.markdown("### STRONG SIGNALS — Score 8–10")
                st.caption("These are your highest priority trades")

                for r in strong:
                    dir_col = ("#00ff88" if r["Direction"]=="UPTREND"
                               else "#ff4455" if r["Direction"]=="DOWNTREND"
                               else "#ffcc00")
                    chg_col = "#00ff88" if r["Change%"] >= 0 else "#ff4455"
                    arr     = "▲" if r["Change%"] >= 0 else "▼"

                    rc1, rc2, rc3 = st.columns([3, 2, 1])
                    with rc1:
                        st.markdown(f"""
                        <div style='background:#ffffff;border:1px solid #e2e8f0;
                             border-radius:10px;padding:14px 18px'>
                          <div style='display:flex;justify-content:space-between;
                                      align-items:center'>
                            <span style='font-size:16px;font-weight:700;
                                         color:#1e293b'>{r['Stock']}</span>
                            <span style='background:#dcfce7;color:{dir_col};
                                         padding:3px 10px;border-radius:12px;
                                         font-size:12px;font-weight:600'>
                                {r['Action']}
                            </span>
                          </div>
                          <div style='margin-top:6px;font-size:13px;color:#555'>
                            Score
                            <b style='color:{dir_col};font-size:18px'>
                                {r['Score']}/10
                            </b>
                            &nbsp;|&nbsp;
                            <span style='color:#1e293b'>₹{r['Price']:,.2f}</span>
                            &nbsp;
                            <span style='color:{chg_col}'>
                                {arr}{abs(r['Change%']):.2f}%
                            </span>
                            &nbsp;|&nbsp;
                            RSI <b style='color:#1e293b'>{r['RSI']:.0f}</b>
                            &nbsp;|&nbsp;
                            Vol {'✅' if r['VolSurge'] else '❌'}
                          </div>
                          <div style='margin-top:6px;font-size:12px;color:#444'>
                            Entry ₹{r['EMA9']:,.2f} &nbsp;|&nbsp;
                            SL ₹{r['SL']:,.2f} &nbsp;|&nbsp;
                            T1 ₹{r['T1']:,.2f} &nbsp;|&nbsp;
                            T2 ₹{r['T2']:,.2f}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with rc2:
                        st.markdown(f"""
                        <div style='background:#ffffff;border:1px solid #e2e8f0;
                             border-radius:10px;padding:14px 18px;height:80px;
                             display:flex;flex-direction:column;
                             justify-content:center'>
                          <div style='font-size:11px;color:#555'>
                              DIRECTION
                          </div>
                          <div style='font-size:20px;font-weight:600;
                                      color:{dir_col}'>
                              {r['Direction']}
                          </div>
                          <div style='font-size:11px;color:#555'>
                              ADX {r['ADX']:.0f}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with rc3:
                        if st.button(
                            "Analyse",
                            key=f"scan_an_{r['Stock']}",
                            use_container_width=True
                        ):
                            st.session_state["sn"] = r["Stock"]
                            st.session_state["st"] = r["Sym"]
                            st.rerun()

                    st.markdown(
                        "<div style='margin:4px 0'></div>",
                        unsafe_allow_html=True
                    )

            # ── Good signals (6-7) ─────────────────────────
            if good:
                st.markdown("---")
                st.markdown("### GOOD SIGNALS — Score 6–7")
                st.caption("Worth watching — wait for score to reach 8")

                cols_per_row = 3
                for i in range(0, len(good), cols_per_row):
                    chunk = good[i:i+cols_per_row]
                    gcols = st.columns(cols_per_row)
                    for ci, r in enumerate(chunk):
                        dc = ("#00ff88" if r["Direction"]=="UPTREND"
                              else "#ff4455" if r["Direction"]=="DOWNTREND"
                              else "#ffcc00")
                        with gcols[ci]:
                            st.markdown(f"""
                            <div style='background:#ffffff;
                                 border:1px solid #e2e8f0;
                                 border-radius:8px;padding:12px'>
                              <div style='color:#1e293b;font-weight:500'>
                                  {r['Stock']}
                              </div>
                              <div style='color:{dc};font-size:18px;
                                          font-weight:600'>
                                  {r['Score']}/10
                              </div>
                              <div style='font-size:12px;color:#555'>
                                  {r['Action']} |
                                  ₹{r['Price']:,.0f} |
                                  RSI {r['RSI']:.0f}
                              </div>
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(
                                "View",
                                key=f"scan_vw_{i}_{ci}",
                                use_container_width=True
                            ):
                                st.session_state["sn"] = r["Stock"]
                                st.session_state["st"] = r["Sym"]
                                st.rerun()

            # ── Score bar chart ────────────────────────────
            if results:
                st.markdown("---")
                st.markdown("### Score Overview — All Signals")
                import plotly.graph_objects as go_scan
                bar_colors = [
                    "#00ff88" if r["Direction"]=="UPTREND"
                    else "#ff4455" if r["Direction"]=="DOWNTREND"
                    else "#ffcc00"
                    for r in results
                ]
                fig_scan = go_scan.Figure(go_scan.Bar(
                    x=[r["Stock"] for r in results],
                    y=[r["Score"] for r in results],
                    marker_color=bar_colors,
                    text=[f"{r['Score']}/10" for r in results],
                    textposition="outside",
                    customdata=[[r["Action"],r["Price"]]
                                for r in results],
                ))
                fig_scan.add_hline(
                    y=8, line_dash="dash",
                    line_color="#00ff88",
                    annotation_text="Strong zone (8+)"
                )
                fig_scan.add_hline(
                    y=6, line_dash="dot",
                    line_color="#ffcc00",
                    annotation_text="Good zone (6+)"
                )
                fig_scan.update_layout(
                    template="plotly_white",
                    height=350,
                    yaxis_range=[0,11],
                    margin=dict(l=10,r=10,t=20,b=80),
                    xaxis_tickangle=-35,
                    showlegend=False
                )
                st.plotly_chart(fig_scan, use_container_width=True)

                # Full results table
                with st.expander("Full results table"):
                    df_res = pd.DataFrame([{
                        "Stock":     r["Stock"],
                        "Score":     r["Score"],
                        "Direction": r["Direction"],
                        "Action":    r["Action"],
                        "Price":     f"Rs{r['Price']:,.2f}",
                        "RSI":       round(r["RSI"],1),
                        "ADX":       round(r["ADX"],1),
                        "Vol":       "Yes" if r["VolSurge"] else "No",
                        "Entry":     f"Rs{r['EMA9']:,.2f}",
                        "SL":        f"Rs{r['SL']:,.2f}",
                        "T1":        f"Rs{r['T1']:,.2f}",
                    } for r in results])
                    st.dataframe(
                        df_res,
                        use_container_width=True,
                        hide_index=True
                    )

        # Auto scan loop
        if auto_scan:
            st.info("Next scan in 5 minutes...")
            time.sleep(300)
            st.rerun()

    else:
        # Default screen
        st.markdown("""
        ### How the auto scanner works

        1. Select a stock group (up to 30 stocks)
        2. Choose timeframe — **15m** for intraday signals
        3. Click **Scan All Stocks Now**
        4. Results appear sorted by score in under 60 seconds
        5. Strong signals (8+) show full entry, SL and targets
        6. Click **Analyse** on any signal to deep-dive into that stock

        ### Telegram alerts

        Setup Telegram alerts above so your phone buzzes
        automatically when any stock hits a score of 8 or above —
        even when you are not watching the screen.

        ### Best times to scan

        | Time | What to do |
        |------|------------|
        | 9:30 AM | First scan of day — find morning setups |
        | 11:00 AM | Mid-morning scan — confirms trends |
        | 1:30 PM | Afternoon scan — fresh setups |
        | Enable auto scan | Scanner runs every 5 min automatically |
        """)


with T4:
    st.markdown("### 🤖 ML Prediction Engine")
    st.caption(
        "Trained on historical candle data using Random Forest + "
        "Gradient Boosting. Predicts next 3-candle direction."
    )

    if df.empty or len(df) < 100:
        st.error(
            "Need at least 100 candles for ML training. "
            "Switch to **1d** timeframe."
        )
    else:
        # ── Real-time approximation ───────────────────────
        st.markdown("### ⚡ Real-Time Approximation")
        st.caption(
            "Bridges the 15-minute data delay by injecting "
            "the live price into indicator calculations."
        )

        rt = approximate_realtime(df, lp["p"] if lp["ok"] else 0)

        if rt.get("ok"):
            bias_col = rt["bias_color"]

            rt1, rt2, rt3 = st.columns(3)

            with rt1:
                st.markdown(f"""
                <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)' style='border:2px solid {bias_col};
                     text-align:center'>
                  <div style='font-size:11px;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>LIVE BIAS</div>
                  <div style='font-size:36px;font-weight:700;
                              color:{bias_col};line-height:1.1'>
                      {rt["live_bias"]}
                  </div>
                  <div style='font-size:13px;color:#64748b;
                              margin-top:6px'>
                      {rt["bull_count"]} bull / {rt["bear_count"]} bear signals
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with rt2:
                since_col = "#00ff88" if rt["since_close"] >= 0 else "#ff4455"
                st.markdown(f"""
                <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
                  <div style='font-size:11px;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>LIVE vs LAST CANDLE</div>
                  <div style='font-size:28px;font-weight:700;
                              color:{since_col}'>
                      {rt["since_close"]:+.3f}%
                  </div>
                  <div style='font-size:13px;color:#64748b;margin-top:6px'>
                    Candle position: {rt["candle_pos"]:.0f}%
                    ({rt["candle_zone"]})<br>
                    Micro trend: <b style='color:#1e293b'>{rt["micro_trend"]}</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with rt3:
                st.markdown(f"""
                <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
                  <div style='font-size:11px;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>LIVE INDICATORS</div>
                  <table style='width:100%;font-size:13px'>
                    <tr><td style='color:#555'>Live RSI</td>
                        <td style='text-align:right;color:#1e293b'>
                            {rt["rsi_live"]}</td></tr>
                    <tr><td style='color:#555'>Live EMA9</td>
                        <td style='text-align:right;color:yellow'>
                            ₹{rt["ema9_live"]:,}</td></tr>
                    <tr><td style='color:#555'>Live EMA21</td>
                        <td style='text-align:right;color:orange'>
                            ₹{rt["ema21_live"]:,}</td></tr>
                    <tr><td style='color:#555'>Live VWAP</td>
                        <td style='text-align:right;color:#1e293b'>
                            ₹{rt["vwap_live"]:,}</td></tr>
                    <tr><td style='color:#555'>VWAP Dev</td>
                        <td style='text-align:right;
                            color:{"#00ff88" if rt["vwap_dev"]>0 else "#ff4455"}'>
                            {rt["vwap_dev"]:+.2f}%</td></tr>
                  </table>
                </div>
                """, unsafe_allow_html=True)

            # Live signals list
            st.markdown("#### 📡 Live Signal Breakdown")
            for s in rt["live_signals"]:
                col_ = ("#071407" if "✅" in s
                        else "#140707" if "❌" in s
                        else "#141007")
                st.markdown(
                    f"<div style='background:{col_};"
                    f"border-radius:6px;padding:8px 14px;"
                    f"margin:3px 0;font-size:13px;color:#ccc'>"
                    f"{s}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning(
                "Live price not available. "
                "Real-time approximation disabled."
            )

        st.markdown("---")

        # ── ML Training + Prediction ──────────────────────
        st.markdown("### 🧠 ML Model Training & Prediction")

        ml_col1, ml_col2 = st.columns([1, 2])

        with ml_col1:
            run_ml = st.button(
                "🚀 Train & Predict",
                type="primary",
                key="run_ml",
                use_container_width=True
            )
            st.caption(
                f"Will train on {len(df)} candles of "
                f"{sname} history"
            )

        with ml_col2:
            st.info(
                "Click **Train & Predict** to run the ML model. "
                "Training takes 5–15 seconds. "
                "Uses Random Forest + Gradient Boosting ensemble "
                "trained on 30+ technical features."
            )

        if run_ml or "ml_result" in st.session_state:
            if run_ml:
                with st.spinner(
                    "Training ML model on historical data..."
                ):
                    model_data = train_model(df)
                    if model_data["ok"]:
                        pred = predict_next_move(df, model_data)
                        st.session_state["ml_result"]     = pred
                        st.session_state["ml_model_data"] = model_data
                    else:
                        st.error(
                            f"Training failed: {model_data.get('reason')}"
                        )
                        st.stop()

            pred       = st.session_state.get("ml_result", {})
            model_data = st.session_state.get("ml_model_data", {})

            if pred and pred.get("ok"):
                pc_   = pred["sig_color"]
                conf  = pred["confidence"]

                # ── Main prediction card ──────────────────
                st.markdown(f"""
                <div style='background:#f8fafc;
                     border:2px solid {pc_};
                     border-radius:14px;padding:24px;
                     text-align:center;margin:12px 0'>
                  <div style='font-size:12px;color:#64748b;
                              letter-spacing:2px'>
                      ML PREDICTION — NEXT 3 CANDLES
                  </div>
                  <div style='font-size:56px;font-weight:700;
                              color:{pc_};line-height:1.1;
                              margin:8px 0'>
                      {pred["prediction"]}
                  </div>
                  <div style='font-size:26px;font-weight:600;
                              color:{pc_}'>
                      {pred["signal"]}
                  </div>
                  <div style='font-size:15px;color:#6b7280;
                              margin-top:8px'>
                      Confidence: {pred["reliability"]} ({conf}%)
                  </div>
                  <div style='margin-top:12px;
                              display:flex;justify-content:center;
                              gap:12px;flex-wrap:wrap'>
                    <span style='background:#dcfce7;
                         color:#00ff88;padding:4px 14px;
                         border-radius:12px;font-size:13px'>
                        📈 Uptrend: {pred["probabilities"].get("UPTREND",0):.1f}%
                    </span>
                    <span style='background:#fee2e2;
                         color:#ff4455;padding:4px 14px;
                         border-radius:12px;font-size:13px'>
                        📉 Downtrend: {pred["probabilities"].get("DOWNTREND",0):.1f}%
                    </span>
                    <span style='background:#fef9c3;
                         color:#ffcc00;padding:4px 14px;
                         border-radius:12px;font-size:13px'>
                        ➡️ Sideways: {pred["probabilities"].get("SIDEWAYS",0):.1f}%
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Model stats
                ms1, ms2, ms3 = st.columns(3)
                ms1.metric(
                    "Model Accuracy",
                    f"{pred['model_accuracy']}%" if pred["model_accuracy"] else "N/A",
                    help="Cross-validated accuracy on historical data"
                )
                ms2.metric(
                    "Trained on",
                    f"{pred['n_trained']} candles"
                )
                ms3.metric(
                    "Timeframe",
                    tf
                )

                # Probability bar chart
                st.markdown("#### 📊 Probability Distribution")
                probs = pred["probabilities"]
                fig_prob = go.Figure(go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color=[
                        "#00ff88" if k=="UPTREND"
                        else "#ff4455" if k=="DOWNTREND"
                        else "#ffcc00"
                        for k in probs.keys()
                    ],
                    text=[f"{v:.1f}%" for v in probs.values()],
                    textposition="outside"
                ))
                fig_prob.update_layout(
                    template="plotly_white",
                    height=280,
                    yaxis_range=[0,100],
                    yaxis_title="Probability %",
                    margin=dict(l=10,r=10,t=10,b=10)
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # Top features
                st.markdown("#### 🔬 Top Contributing Features")
                st.caption(
                    "The 5 most important indicators the ML model "
                    "used to make this prediction"
                )
                for feat in pred["top_contrib"]:
                    imp  = feat["importance"]
                    bar  = int(imp * 3)
                    col_ = ("#00ff88" if imp >= 10
                            else "#ffcc00" if imp >= 5
                            else "#888")
                    st.markdown(f"""
                    <div style='background:#ffffff;
                         border-radius:6px;padding:10px 14px;
                         margin:4px 0'>
                      <div style='display:flex;
                                  justify-content:space-between'>
                        <span style='color:#374151;font-size:13px'>
                            {feat["feature"]}
                        </span>
                        <span style='color:{col_};font-size:13px;
                                      font-weight:600'>
                            {imp:.1f}% importance
                        </span>
                      </div>
                      <div style='background:#f1f5f9;height:4px;
                                  border-radius:2px;margin-top:6px'>
                        <div style='background:{col_};
                             width:{min(imp*5,100):.0f}%;
                             height:4px;border-radius:2px'></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Combined signal (ML + technical)
                st.markdown("---")
                st.markdown("### 🎯 Combined Signal (ML + Technical)")

                ml_dir = pred["prediction"]
                tech_dir = sig["direction"] if sig else "SIDEWAYS"
                rt_bias  = rt.get("live_bias","NEUTRAL") if rt.get("ok") else "NEUTRAL"

                agree_bull = (ml_dir=="UPTREND" and
                              tech_dir=="UPTREND" and
                              rt_bias=="BULLISH")
                agree_bear = (ml_dir=="DOWNTREND" and
                              tech_dir=="DOWNTREND" and
                              rt_bias=="BEARISH")

                if agree_bull:
                    st.success(f"""
                    🔥 **ALL THREE AGREE — BULLISH**

                    ML Model: UPTREND ({conf}% confidence) ✅
                    Technical Score: {sig["up_score"]}/10 ✅
                    Live Bias: BULLISH ✅

                    **This is the highest quality CE setup.**
                    All three signal sources confirm uptrend.
                    Enter on pullback to EMA9 (₹{sig["e9v"]:,})
                    with SL below ₹{sig["sl_long"]:,}
                    """)
                elif agree_bear:
                    st.error(f"""
                    🔥 **ALL THREE AGREE — BEARISH**

                    ML Model: DOWNTREND ({conf}% confidence) ✅
                    Technical Score: {sig["dn_score"]}/10 ✅
                    Live Bias: BEARISH ✅

                    **This is the highest quality PE setup.**
                    All three signal sources confirm downtrend.
                    Enter on bounce to EMA9 (₹{sig["e9v"]:,})
                    with SL above ₹{sig["sl_short"]:,}
                    """)
                elif ml_dir != "SIDEWAYS" and tech_dir != "SIDEWAYS":
                    if ml_dir == tech_dir:
                        col_ = "#00ff88" if ml_dir=="UPTREND" else "#ff4455"
                        st.markdown(f"""
                        <div style='background:#fffbeb;border:1.5px solid #ffcc00;border-radius:10px;padding:18px;margin:6px 0'>
                          <b style='color:{col_}'>
                              ⚡ ML + Technical agree: {ml_dir}
                          </b><br>
                          <span style='color:#6b7280;font-size:13px'>
                              Live bias is {rt_bias} — wait for it to confirm
                              before entering.
                          </span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(
                            f"⚠️ **Mixed signals** — "
                            f"ML says {ml_dir} but "
                            f"technical says {tech_dir}. "
                            "Do not trade when signals conflict."
                        )
                else:
                    st.warning(
                        "⏳ **No clear combined signal** — "
                        "At least one model shows SIDEWAYS. "
                        "Wait for alignment."
                    )

                with st.expander("📖 How to use ML predictions"):
                    st.markdown("""
                    ### How the ML model works

                    The model is trained on YOUR stock's own
                    historical data — not generic data.
                    It learns the specific patterns of that
                    stock by analyzing 30+ technical features
                    including RSI, MACD, EMA ratios, volume
                    patterns, Bollinger Band position,
                    Stochastic, Williams %R and more.

                    ### Three levels of confirmation

                    | Level | Condition | Action |
                    |-------|-----------|--------|
                    | **Highest** | ML + Technical + Live all agree | Enter trade |
                    | **Good** | ML + Technical agree | Wait for live bias to confirm |
                    | **Avoid** | ML and Technical disagree | Do not trade |

                    ### Model accuracy guide

                    | Accuracy | Meaning |
                    |----------|---------|
                    | > 65% | Strong — trust the signal |
                    | 55–65% | Moderate — use with technical confirmation |
                    | < 55% | Weak — rely more on technical signals |

                    ### Important limitations

                    - ML predicts probability, not certainty
                    - Always use stop loss regardless of ML signal
                    - Retrain the model daily for fresh predictions
                    - Works best on 1d timeframe with 300+ candles
                    """)

with T5:
    st.markdown("### 🏦 Smart Money & Institutional Analysis")

    if "sig" not in dir() or sig is None:
        st.info("Select a stock in Tab 2 first.")
    else:
        sm1, sm2 = st.columns(2)

        with sm1:
            st.markdown("#### 📊 Money Flow Indicators")
            cmf_col = ("#00ff88" if sig["cmfv"] > 0.05
                       else "#ff4455" if sig["cmfv"] < -0.05
                       else "#ffcc00")
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
              <div style='font-size:11px;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>CHAIKIN MONEY FLOW</div>
              <div style='font-size:40px;font-weight:700;
                          color:{cmf_col}'>
                  {sig['cmfv']:+.3f}
              </div>
              <div style='color:#64748b;font-size:13px;
                          margin-top:4px'>
                  {'🟢 Institutions BUYING' if sig['cmfv']>0.05
                   else '🔴 Institutions SELLING' if sig['cmfv']<-0.05
                   else '🟡 Neutral'}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)' style='margin-top:8px'>
              <table style='width:100%;font-size:13px'>
                <tr>
                  <td style='color:#555'>OBV Direction</td>
                  <td style='text-align:right;
                      color:{"#00ff88" if sig["obv_bull"] else "#ff4455"}'>
                      {"▲ Rising" if sig["obv_bull"] else "▼ Falling"}
                  </td>
                </tr>
                <tr>
                  <td style='color:#555'>Volume Ratio</td>
                  <td style='text-align:right;
                      color:{"#00ff88" if sig["vol_ratio"]>=1.5 else "#fff"}'>
                      {sig['vol_ratio']}× avg
                  </td>
                </tr>
                <tr>
                  <td style='color:#555'>VWAP Deviation</td>
                  <td style='text-align:right;color:#1e293b'>
                      {((sig['cp']-sig['vwv'])/sig['vwv']*100):+.2f}%
                  </td>
                </tr>
                <tr>
                  <td style='color:#555'>Liquidity Sweep↓</td>
                  <td style='text-align:right'>
                      {"🚀 YES — BUY" if sig['sweep_low'] else "—"}
                  </td>
                </tr>
                <tr>
                  <td style='color:#555'>Liquidity Sweep↑</td>
                  <td style='text-align:right'>
                      {"⚠️ YES — SELL" if sig['sweep_high'] else "—"}
                  </td>
                </tr>
                <tr>
                  <td style='color:#555'>BOS Bullish</td>
                  <td style='text-align:right;color:#00ff88'>
                      {"✅ YES" if sig['bos_bull'] else "—"}
                  </td>
                </tr>
                <tr>
                  <td style='color:#555'>BOS Bearish</td>
                  <td style='text-align:right;color:#ff4455'>
                      {"✅ YES" if sig['bos_bear'] else "—"}
                  </td>
                </tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

        with sm2:
            st.markdown("#### 🕯️ Candlestick Patterns")
            if sig["patterns"]:
                for pname, pbias, pmeaning in sig["patterns"]:
                    pc_ = ("#00ff88" if pbias=="bullish"
                           else "#ff4455" if pbias=="bearish"
                           else "#ffcc00")
                    st.markdown(f"""
                    <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)'
                         style='border-left:3px solid {pc_};
                                margin-bottom:8px'>
                      <div style='color:{pc_};font-weight:600'>
                          {pname}
                      </div>
                      <div style='color:#6b7280;font-size:12px;
                                  margin-top:4px'>
                          {pmeaning}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No strong patterns on current candle")

            st.markdown("#### 📈 Support & Resistance")
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
              <div style='display:flex;
                          justify-content:space-between;
                          margin-bottom:12px'>
                <div style='text-align:center'>
                  <div style='font-size:11px;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>SUPPORT</div>
                  <div style='color:#00ff88;font-size:22px;
                              font-weight:700'>
                      ₹{sig['sup']:,}
                  </div>
                </div>
                <div style='text-align:center'>
                  <div style='font-size:11px;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>CURRENT</div>
                  <div style='color:#1e293b;font-size:22px;
                              font-weight:700'>
                      ₹{sig['cp']:,}
                  </div>
                </div>
                <div style='text-align:center'>
                  <div style='font-size:11px;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px'>RESISTANCE</div>
                  <div style='color:#ff4455;font-size:22px;
                              font-weight:700'>
                      ₹{sig['res']:,}
                  </div>
                </div>
              </div>
              <div style='font-size:13px;color:#555'>
                  Risk: ₹{sig['res']-sig['cp']:.1f} to resistance &nbsp;|&nbsp;
                  Reward: ₹{sig['cp']-sig['sup']:.1f} to support
              </div>
            </div>
            """, unsafe_allow_html=True)

        # CMF chart
        st.markdown("---")
        st.markdown("#### 📊 Money Flow Chart (CMF)")
        cmf_s = sig["cmfs"].tail(80)
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(
            x=df.tail(80).index, y=cmf_s,
            marker_color=["#00ff88" if v>=0 else "#ff4455"
                          for v in cmf_s],
            name="CMF", opacity=0.85
        ))
        fig_c.add_hline(y= 0.1, line_dash="dash",
                        line_color="#00ff88", opacity=0.5,
                        annotation_text="Inst. buying (0.1)")
        fig_c.add_hline(y=-0.1, line_dash="dash",
                        line_color="#ff4455", opacity=0.5,
                        annotation_text="Inst. selling (-0.1)")
        fig_c.add_hline(y=0, line_color="#333",
                        line_dash="dot")
        fig_c.update_layout(
            template="plotly_white", height=250,
            margin=dict(l=10,r=10,t=20,b=10),
            title="Chaikin Money Flow — "
                  "Green = institutions buying | "
                  "Red = institutions selling",
            yaxis_range=[-0.5,0.5]
        )
        st.plotly_chart(fig_c, width="stretch")

        with st.expander(
            "📖 How to read Smart Money signals"
        ):
            st.markdown("""
            | Signal | Meaning | Action |
            |--------|---------|--------|
            | **CMF > 0.1** | Big money flowing IN | 🟢 CE signal |
            | **CMF < -0.1** | Big money flowing OUT | 🔴 PE signal |
            | **OBV Rising** | Volume confirms price rise | 🟢 Bullish |
            | **OBV Falling** | Volume confirms price fall | 🔴 Bearish |
            | **Liquidity Sweep ↓** | Stop hunt below lows — institutions bought | 🚀 Strong CE |
            | **Liquidity Sweep ↑** | Stop hunt above highs — institutions sold | ⚠️ Strong PE |
            | **BOS Bullish** | Break of structure upward with volume | 🟢 CE confirmed |
            | **BOS Bearish** | Break of structure downward with volume | 🔴 PE confirmed |
            | **Vol Ratio > 2.0** | Institutional block trade | 📊 Watch direction |
            """)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 4 — P&L CALCULATOR                             ║
# ╚══════════════════════════════════════════════════════╝
with T6:
    st.markdown("### 🧮 Options P&L Calculator")
    st.caption(
        "Enter trade details to see exact profit, loss "
        "and breakeven before placing any trade"
    )

    strat = st.selectbox("Strategy", [
        "📈 Buy CE (Bullish)",
        "📉 Buy PE (Bearish)",
        "🔀 Bull Call Spread",
        "🔀 Bear Put Spread",
        "🦋 Long Straddle",
    ])

    st.markdown("---")
    pi1,pi2,pi3 = st.columns(3)
    with pi1:
        sp  = st.number_input(
            "Spot Price ₹",
            value=float(lp["p"]) if lp["ok"] else 22500.0,
            step=50.0)
        iv  = st.slider("IV %", 5, 80, 15) / 100
    with pi2:
        sk  = st.number_input(
            "Strike ₹",
            value=round((float(lp["p"])
                         if lp["ok"] else 22500)/50)*50.0,
            step=50.0)
        dte = st.number_input("Days to Expiry",0,90,7)
    with pi3:
        lots  = st.number_input("Lots",1,50,1)
        lsz   = st.number_input(
            "Lot Size",
            value=int(LOT_SIZES.get(stick,25)),
            min_value=1)

    T_yr  = max(dte/365, 0.001)
    r_r   = 0.065
    units = lots * lsz

    def pl_chart(prices, pnls, color, be,
                 spot_v, title="P&L"):
        fig_ = go.Figure()
        fig_.add_trace(go.Scatter(
            x=list(prices), y=pnls,
            line=dict(color=color,width=2),
            fill="tozeroy",
            fillcolor=f"rgba({'0,255,100' if color=='lime' else '255,60,80'},0.06)",
            name="P&L"))
        fig_.add_hline(y=0,line_dash="dash",
                       line_color="white",opacity=0.3)
        fig_.add_vline(x=be,line_dash="dot",
                       line_color="yellow",
                       annotation_text=f"BE ₹{be:,.0f}",
                       annotation_position="top right")
        fig_.add_vline(x=spot_v,line_dash="dot",
                       line_color="cyan",
                       annotation_text=f"Spot ₹{spot_v:,.0f}",
                       annotation_position="top left")
        fig_.update_layout(
            template="plotly_white",height=320,
            xaxis_title="Spot at Expiry ₹",
            yaxis_title="P&L ₹",
            margin=dict(l=10,r=10,t=20,b=10))
        return fig_

    if strat == "📈 Buy CE (Bullish)":
        prem = st.number_input(
            "CE Premium ₹",
            value=float(bs(sp,sk,T_yr,r_r,iv,"CE")),
            step=0.5)
        tc   = prem*units
        be_  = sk+prem
        g    = greeks(sp,sk,T_yr,r_r,iv,"CE")
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Cost",  f"₹{tc:,.0f}")
        m2.metric("Breakeven",   f"₹{be_:,.1f}")
        m3.metric("Max Loss",    f"₹{tc:,.0f}")
        m4.metric("Max Profit",  "Unlimited ∞")
        m5.metric("Delta",       g["d"])
        g1,g2,g3,g4 = st.columns(4)
        g1.metric("Delta",      g["d"])
        g2.metric("Gamma",      g["g"])
        g3.metric("Theta/day",  f"₹{abs(g['th']*units):,.1f}")
        g4.metric("Vega/1%IV",  f"₹{g['v']*units:,.1f}")
        prices = np.linspace(sp*0.88,sp*1.12,200)
        pnls_  = [(max(p-sk,0)-prem)*units for p in prices]
        st.plotly_chart(
            pl_chart(prices,pnls_,"lime",be_,sp),
            width="stretch")
        st.info(f"""
        💡 Max risk ₹{tc:,.0f} | Need > ₹{be_:,.1f} to profit |
        Theta burns ₹{abs(g['th']*units):,.1f}/day |
        Best exit at 60–80% profit
        """)

    elif strat == "📉 Buy PE (Bearish)":
        prem = st.number_input(
            "PE Premium ₹",
            value=float(bs(sp,sk,T_yr,r_r,iv,"PE")),
            step=0.5)
        tc  = prem*units
        be_ = sk-prem
        g   = greeks(sp,sk,T_yr,r_r,iv,"PE")
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Cost",  f"₹{tc:,.0f}")
        m2.metric("Breakeven",   f"₹{be_:,.1f}")
        m3.metric("Max Loss",    f"₹{tc:,.0f}")
        m4.metric("Max Profit",  f"₹{be_*units:,.0f}")
        m5.metric("Delta",       g["d"])
        g1,g2,g3,g4 = st.columns(4)
        g1.metric("Delta",      g["d"])
        g2.metric("Gamma",      g["g"])
        g3.metric("Theta/day",  f"₹{abs(g['th']*units):,.1f}")
        g4.metric("Vega/1%IV",  f"₹{g['v']*units:,.1f}")
        prices = np.linspace(sp*0.88,sp*1.12,200)
        pnls_  = [(max(sk-p,0)-prem)*units for p in prices]
        st.plotly_chart(
            pl_chart(prices,pnls_,"#ff4455",be_,sp),
            width="stretch")
        st.info(f"""
        💡 Max risk ₹{tc:,.0f} | Need < ₹{be_:,.1f} to profit |
        Theta burns ₹{abs(g['th']*units):,.1f}/day
        """)

    elif strat == "🔀 Bull Call Spread":
        ca,cb = st.columns(2)
        with ca:
            bk = st.number_input("Buy CE Strike",
                                 value=sk,step=50.0)
            bp = st.number_input(
                "Buy CE Premium",
                value=float(bs(sp,bk,T_yr,r_r,iv,"CE")),
                step=0.5)
        with cb:
            sk2= st.number_input("Sell CE Strike",
                                  value=sk+100,step=50.0)
            sp2= st.number_input(
                "Sell CE Premium",
                value=float(bs(sp,sk2,T_yr,r_r,iv,"CE")),
                step=0.5)
        net  = (bp-sp2)*units
        maxp = (sk2-bk-bp+sp2)*units
        be_  = bk+bp-sp2
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Net Cost",   f"₹{net:,.0f}")
        m2.metric("Breakeven",  f"₹{be_:,.1f}")
        m3.metric("Max Loss",   f"₹{net:,.0f}")
        m4.metric("Max Profit", f"₹{maxp:,.0f}")
        prices = np.linspace(sp*0.9,sp*1.1,200)
        pnls_  = [(max(p-bk,0)-max(p-sk2,0)
                   -bp+sp2)*units for p in prices]
        st.plotly_chart(
            pl_chart(prices,pnls_,"lime",be_,sp),
            width="stretch")
        st.success(f"Pay ₹{net:,.0f} | Max profit ₹{maxp:,.0f} above ₹{sk2:,.0f}")

    elif strat == "🔀 Bear Put Spread":
        ca,cb = st.columns(2)
        with ca:
            bk = st.number_input("Buy PE Strike",
                                 value=sk,step=50.0)
            bp = st.number_input(
                "Buy PE Premium",
                value=float(bs(sp,bk,T_yr,r_r,iv,"PE")),
                step=0.5)
        with cb:
            sk2= st.number_input("Sell PE Strike",
                                  value=sk-100,step=50.0)
            sp2= st.number_input(
                "Sell PE Premium",
                value=float(bs(sp,sk2,T_yr,r_r,iv,"PE")),
                step=0.5)
        net  = (bp-sp2)*units
        maxp = (bk-sk2-bp+sp2)*units
        be_  = bk-bp+sp2
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Net Cost",   f"₹{net:,.0f}")
        m2.metric("Breakeven",  f"₹{be_:,.1f}")
        m3.metric("Max Loss",   f"₹{net:,.0f}")
        m4.metric("Max Profit", f"₹{maxp:,.0f}")
        prices = np.linspace(sp*0.9,sp*1.1,200)
        pnls_  = [(max(bk-p,0)-max(sk2-p,0)
                   -bp+sp2)*units for p in prices]
        st.plotly_chart(
            pl_chart(prices,pnls_,"#ff4455",be_,sp),
            width="stretch")
        st.error(f"Pay ₹{net:,.0f} | Max profit ₹{maxp:,.0f} below ₹{sk2:,.0f}")

    elif strat == "🦋 Long Straddle":
        cep = st.number_input(
            "CE Premium",
            value=float(bs(sp,sk,T_yr,r_r,iv,"CE")),
            step=0.5)
        pep = st.number_input(
            "PE Premium",
            value=float(bs(sp,sk,T_yr,r_r,iv,"PE")),
            step=0.5)
        tc   = (cep+pep)*units
        ube  = sk+cep+pep
        lbe  = sk-cep-pep
        mv_n = round((ube-sp)/sp*100,1)
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Cost",   f"₹{tc:,.0f}")
        m2.metric("Upper BE",     f"₹{ube:,.1f}")
        m3.metric("Lower BE",     f"₹{lbe:,.1f}")
        m4.metric("Max Loss",     f"₹{tc:,.0f}")
        m5.metric("Move Needed",  f"{mv_n}%")
        prices = np.linspace(sp*0.85,sp*1.15,200)
        pnls_  = [(max(p-sk,0)+max(sk-p,0)
                   -cep-pep)*units for p in prices]
        fig_st = go.Figure()
        fig_st.add_trace(go.Scatter(
            x=list(prices),y=pnls_,
            line=dict(color="violet",width=2),
            fill="tozeroy",
            fillcolor="rgba(150,0,200,0.06)",
            name="P&L"))
        fig_st.add_hline(y=0,line_dash="dash",
                         line_color="white",opacity=0.3)
        for bv,lb in [(ube,"Upper BE"),(lbe,"Lower BE")]:
            fig_st.add_vline(x=bv,line_dash="dot",
                             line_color="lime",
                             annotation_text=f"{lb} ₹{bv:,.0f}")
        fig_st.update_layout(
            template="plotly_white",height=300,
            xaxis_title="Spot ₹",yaxis_title="P&L ₹",
            margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_st, width="stretch")
        st.warning(f"""
        Need {mv_n}% move either direction to profit.
        Best used BEFORE: RBI Policy | Results | Budget | US Fed.
        Exit immediately AFTER the event — IV collapses!
        """)

    # Lot size reference
    with st.expander("📋 NSE Lot Size Reference"):
        lst = pd.DataFrame([
            {"Index/Stock":"NIFTY 50","Lot":25,"Approx Margin":"₹1.2L"},
            {"Index/Stock":"BANK NIFTY","Lot":15,"Approx Margin":"₹45K"},
            {"Index/Stock":"Reliance","Lot":250,"Approx Margin":"₹60K"},
            {"Index/Stock":"TCS","Lot":150,"Approx Margin":"₹55K"},
            {"Index/Stock":"HDFC Bank","Lot":550,"Approx Margin":"₹90K"},
            {"Index/Stock":"Infosys","Lot":300,"Approx Margin":"₹50K"},
            {"Index/Stock":"SBI","Lot":1500,"Approx Margin":"₹95K"},
            {"Index/Stock":"Bajaj Finance","Lot":125,"Approx Margin":"₹85K"},
            {"Index/Stock":"ITC","Lot":3200,"Approx Margin":"₹75K"},
            {"Index/Stock":"Tata Motors DVR","Lot":1425,"Approx Margin":"₹90K"},
        ])
        st.dataframe(lst, width="stretch", hide_index=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 5 — NEWS & EVENTS                              ║
# ╚══════════════════════════════════════════════════════╝
with T7:
    st.markdown("### 📰 News & Market Events")

    nt1,nt2,nt3,nt4 = st.tabs([
        "🇮🇳 Market News",
        f"📌 {sname[:12]}",
        "🌍 Global",
        "📅 Daily Checklist",
    ])

    def render_news(arts):
        if not arts:
            st.info("No news found."); return
        for a in arts:
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)'>
              <a href='{a['link']}' target='_blank'
                 style='color:#374151;font-size:14px;
                        font-weight:500;text-decoration:none'>
                  {a['title']}
              </a>
              <div style='color:#94a3b8;font-size:11px;
                          margin-top:5px'>
                  📰 {a['src']} &nbsp;|&nbsp; 🕐 {a['date']}
              </div>
            </div>""", unsafe_allow_html=True)

    with nt1:
        with st.spinner("Loading..."):
            render_news(get_news(
                "NSE BSE Nifty Sensex India stock market"))
    with nt2:
        with st.spinner("Loading..."):
            render_news(get_news(f"{sname} NSE India stock"))
    with nt3:
        with st.spinner("Loading..."):
            render_news(get_news(
                "US Fed Dow Nasdaq global markets"))
    with nt4:
        tstate_now = best_trading_time()
        tclr_now   = {"best":"#00ff88","good":"#88ff44",
                      "ok":"#ffcc00","caution":"#ff8844",
                      "avoid":"#ff4455","closed":"#555",
                      "pre_market":"#aaa"}.get(tstate_now,"#aaa")
        tmsg_now   = {"best":"✅ BEST TIME — enter now",
                      "good":"🟡 GOOD TIME",
                      "ok":"🟡 OK — lunch hours",
                      "caution":"⚠️ CAUTION — choppy",
                      "avoid":"❌ AVOID — too volatile",
                      "closed":"⛔ MARKET CLOSED",
                      "pre_market":"🕐 PRE-MARKET — prepare"
                     }.get(tstate_now,"—")

        st.markdown(f"""
        <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:10px;box-shadow:0 1px 4px rgba(0,0,0,0.06)' style='border:1.5px solid {tclr_now};
             text-align:center;padding:20px'>
          <div style='font-size:24px;font-weight:700;
                      color:{tclr_now}'>{tmsg_now}</div>
          <div style='color:#64748b;font-size:13px;margin-top:6px'>
              {now_ist().strftime('%H:%M IST')}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ☀️ Pre-Market Checklist (8:45–9:15 AM)")
        checklist_items = [
            "Check Gift Nifty direction on Google",
            "Check US market close (Dow, Nasdaq, S&P)",
            "Check Crude Oil price (affects energy stocks)",
            "Check Gold price (affects MCX traders)",
            "Look at Tab 5 for any big events today",
            "Check PCR ratio in Options Chain tab",
            "Note first 15-min candle high and low",
        ]
        for item in checklist_items:
            st.checkbox(item, key=f"pre_{item[:20]}")

        st.markdown("---")
        st.markdown("""
        ### ⏰ Trading Time Guide

        | Time | Recommendation | Why |
        |------|----------------|-----|
        | 9:15–9:30 AM | ❌ Do not trade | Opening chaos |
        | 9:30–10:30 AM | ✅ Best window | Strong trends form |
        | 10:30–12:00 PM | ✅ Good | Clear price action |
        | 12:00–1:30 PM | 🟡 OK | Slow lunch hours |
        | 1:30–2:30 PM | ✅ Good | Afternoon momentum |
        | 2:30–3:00 PM | ⚠️ Caution | Choppy |
        | 3:00–3:30 PM | ❌ Do not trade | Erratic closing |

        ### 📅 Events — Avoid trading options on these days

        | Event | Impact |
        |-------|--------|
        | RBI Monetary Policy | 🔴 Very High — avoid |
        | US Fed Meeting | 🔴 Very High — avoid |
        | Union Budget (Feb 1) | 🔴 Very High — avoid |
        | Weekly F&O Expiry (Thu) | 🟠 High — be careful |
        | Monthly Expiry (last Thu) | 🔴 Very High — avoid |
        | Quarterly Results | 🟠 High — be careful |
        | US CPI data | 🟡 Medium |
        | India CPI / WPI | 🟡 Medium |

        ### 📋 Pre-Trade Rules (check all before every trade)

        1. Uptrend/Downtrend score ≥ 7 ✅
        2. RSI in correct zone (55–68 for CE, 32–45 for PE) ✅
        3. Price above/below VWAP ✅
        4. Volume surge ≥ 1.2× ✅
        5. MACD confirming direction ✅
        6. ADX > 20 ✅
        7. CMF confirming direction ✅
        8. OBV confirming direction ✅
        9. Risk-Reward ≥ 1.5 ✅
        10. No contradicting candlestick pattern ✅
        11. Trading time is green ✅
        12. No major event today ✅
        """)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 8 — PAPER TRADING                              ║
# ╚══════════════════════════════════════════════════════╝
with T8:
    st.markdown("### 📝 Paper Trading — Practice Without Real Money")
    st.caption(
        "Record virtual trades based on your terminal signals. "
        "Track your win rate, P&L and learn before using real money."
    )

    # ── Initialise session state ───────────────────────
    if "pt_trades" not in st.session_state:
        st.session_state["pt_trades"] = []
    if "pt_capital" not in st.session_state:
        st.session_state["pt_capital"] = 100000.0

    trades = st.session_state["pt_trades"]

    # ── Capital setup ──────────────────────────────────
    with st.expander("Capital & Settings", expanded=len(trades)==0):
        cap_col1, cap_col2 = st.columns(2)
        with cap_col1:
            capital = st.number_input(
                "Starting capital (Rs)",
                value=float(st.session_state["pt_capital"]),
                step=10000.0,
                min_value=10000.0,
                key="pt_capital_input"
            )
            st.session_state["pt_capital"] = capital
        with cap_col2:
            st.info(
                "Start with Rs 1,00,000 virtual money. "
                "Trade as if it were real. "
                "Never risk more than 2% per trade."
            )
        max_risk = round(capital * 0.02, 0)
        st.markdown(
            f"Max risk per trade (2%): "
            f"**Rs {max_risk:,.0f}**"
        )

    st.markdown("---")

    # ── Add new trade ──────────────────────────────────
    st.markdown("#### Add a Paper Trade")
    st.caption(
        "When your terminal gives a signal, record it here "
        "before entering. Fill in exactly what the terminal shows."
    )

    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        pt_stock = st.text_input(
            "Stock / Index",
            placeholder="e.g. NIFTY 50, HDFC Bank",
            key="pt_stock"
        )
        pt_type = st.selectbox(
            "Trade type",
            ["CE (Call)", "PE (Put)", "Intraday Buy", "Intraday Sell"],
            key="pt_type"
        )
        pt_signal = st.slider(
            "Signal score at entry",
            0, 10, 7,
            key="pt_signal"
        )
    with nc2:
        pt_entry = st.number_input(
            "Entry price (Rs)",
            value=0.0, step=0.5,
            min_value=0.0,
            key="pt_entry"
        )
        pt_sl = st.number_input(
            "Stop loss (Rs)",
            value=0.0, step=0.5,
            min_value=0.0,
            key="pt_sl"
        )
        pt_target = st.number_input(
            "Target 1 (Rs)",
            value=0.0, step=0.5,
            min_value=0.0,
            key="pt_target"
        )
    with nc3:
        pt_qty = st.number_input(
            "Quantity / Lots",
            value=1, step=1,
            min_value=1,
            key="pt_qty"
        )
        pt_date = st.date_input(
            "Trade date",
            value=now_ist().date(),
            key="pt_date"
        )
        pt_notes = st.text_area(
            "Notes (why did you enter?)",
            placeholder="Signal score 8, RSI 61, Vol surge, EMA9 pullback...",
            height=80,
            key="pt_notes"
        )

    if st.button(
        "Record Trade",
        type="primary",
        key="pt_add",
        use_container_width=True
    ):
        if pt_stock and pt_entry > 0 and pt_sl > 0:
            risk_per_unit = abs(pt_entry - pt_sl)
            total_risk    = risk_per_unit * pt_qty
            rr = round(
                abs(pt_target - pt_entry) /
                (risk_per_unit + 0.001), 2
            ) if pt_target > 0 else 0

            trade = {
                "id":       len(trades) + 1,
                "date":     str(pt_date),
                "stock":    pt_stock,
                "type":     pt_type,
                "score":    pt_signal,
                "entry":    pt_entry,
                "sl":       pt_sl,
                "target":   pt_target,
                "qty":      pt_qty,
                "risk":     round(total_risk, 2),
                "rr":       rr,
                "notes":    pt_notes,
                "status":   "OPEN",
                "exit":     0.0,
                "pnl":      0.0,
                "result":   "",
            }
            trades.append(trade)
            st.session_state["pt_trades"] = trades
            st.success(
                f"Trade recorded: {pt_stock} {pt_type} "
                f"@ Rs{pt_entry} | "
                f"Risk: Rs{total_risk:,.0f} | "
                f"R:R {rr}:1"
            )
            if total_risk > capital * 0.02:
                st.warning(
                    f"Risk Rs{total_risk:,.0f} exceeds 2% rule "
                    f"(Rs{capital*0.02:,.0f}). "
                    "Consider reducing quantity."
                )
        else:
            st.error(
                "Please fill stock name, entry price and stop loss."
            )

    st.markdown("---")

    # ── Open trades ────────────────────────────────────
    open_trades = [t for t in trades if t["status"] == "OPEN"]
    if open_trades:
        st.markdown(f"#### Open Trades ({len(open_trades)})")
        for t in open_trades:
            oc1, oc2, oc3, oc4 = st.columns([3,2,2,1])
            with oc1:
                st.markdown(
                    f"**{t['stock']}** {t['type']} | "
                    f"Entry Rs{t['entry']:,.2f} | "
                    f"SL Rs{t['sl']:,.2f} | "
                    f"Score {t['score']}/10"
                )
            with oc2:
                lp_pt = live_price(STOCKS.get(t["stock"], ""))
                if lp_pt["ok"]:
                    cur = lp_pt["p"]
                    unreal = round(
                        (cur - t["entry"]) * t["qty"]
                        if "Buy" in t["type"] or "CE" in t["type"]
                        else (t["entry"] - cur) * t["qty"], 2
                    )
                    col_ = "#00ff88" if unreal >= 0 else "#ff4455"
                    st.markdown(
                        f"Live Rs{cur:,} | "
                        f"<span style='color:{col_}'>"
                        f"Unrealised Rs{unreal:+,.0f}</span>",
                        unsafe_allow_html=True
                    )
            with oc3:
                exit_price = st.number_input(
                    "Exit at Rs",
                    value=0.0, step=0.5,
                    key=f"exit_{t['id']}",
                    label_visibility="collapsed"
                )
            with oc4:
                if st.button(
                    "Close",
                    key=f"close_{t['id']}",
                    type="primary"
                ):
                    if exit_price > 0:
                        if "Buy" in t["type"] or "CE" in t["type"]:
                            pnl = round(
                                (exit_price - t["entry"]) * t["qty"], 2
                            )
                        else:
                            pnl = round(
                                (t["entry"] - exit_price) * t["qty"], 2
                            )
                        result = "WIN" if pnl > 0 else "LOSS"
                        for tr in st.session_state["pt_trades"]:
                            if tr["id"] == t["id"]:
                                tr["status"] = "CLOSED"
                                tr["exit"]   = exit_price
                                tr["pnl"]    = pnl
                                tr["result"] = result
                        st.rerun()
                    else:
                        st.error("Enter exit price first")
        st.markdown("---")

    # ── Performance dashboard ──────────────────────────
    closed = [t for t in trades if t["status"] == "CLOSED"]

    if closed:
        st.markdown("#### Performance Dashboard")

        total_pnl  = sum(t["pnl"] for t in closed)
        wins       = [t for t in closed if t["result"] == "WIN"]
        losses     = [t for t in closed if t["result"] == "LOSS"]
        win_rate   = round(len(wins)/len(closed)*100, 1)
        avg_win    = round(
            sum(t["pnl"] for t in wins)/len(wins), 2
        ) if wins else 0
        avg_loss   = round(
            sum(t["pnl"] for t in losses)/len(losses), 2
        ) if losses else 0
        expectancy = round(
            (win_rate/100 * avg_win) +
            ((1-win_rate/100) * avg_loss), 2
        )
        profit_factor = round(
            sum(t["pnl"] for t in wins) /
            abs(sum(t["pnl"] for t in losses) + 0.001), 2
        ) if losses else 999

        # Metrics row
        pm1,pm2,pm3,pm4,pm5,pm6 = st.columns(6)
        pnl_col = "normal" if total_pnl >= 0 else "inverse"
        pm1.metric("Total P&L",
                   f"Rs{total_pnl:+,.0f}",
                   delta_color=pnl_col)
        pm2.metric("Win Rate",   f"{win_rate}%")
        pm3.metric("Trades",     len(closed))
        pm4.metric("Avg Win",    f"Rs{avg_win:+,.0f}")
        pm5.metric("Avg Loss",   f"Rs{avg_loss:+,.0f}")
        pm6.metric("Profit Factor", f"{profit_factor}")

        # Capital curve
        import plotly.graph_objects as go_pt
        running = capital
        curve_x = ["Start"]
        curve_y = [capital]
        for t in sorted(closed, key=lambda x: x["date"]):
            running += t["pnl"]
            curve_x.append(f"#{t['id']} {t['stock'][:8]}")
            curve_y.append(round(running, 2))

        fig_curve = go_pt.Figure()
        fig_curve.add_trace(go_pt.Scatter(
            x=curve_x, y=curve_y,
            mode="lines+markers",
            line=dict(
                color="#00ff88" if curve_y[-1] >= capital
                else "#ff4455",
                width=2
            ),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(0,255,136,0.05)"
            if curve_y[-1] >= capital
            else "rgba(255,68,85,0.05)",
            name="Capital"
        ))
        fig_curve.add_hline(
            y=capital,
            line_dash="dash",
            line_color="white",
            opacity=0.3,
            annotation_text="Starting capital"
        )
        fig_curve.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=10,r=10,t=20,b=60),
            xaxis_tickangle=-30,
            yaxis_title="Capital Rs",
            title="Capital curve — how your account grew trade by trade"
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Signal score analysis
        st.markdown("#### Signal Score vs Result")
        score_win  = [t["score"] for t in wins]
        score_loss = [t["score"] for t in losses]

        if score_win or score_loss:
            fig_sc = go_pt.Figure()
            if score_win:
                fig_sc.add_trace(go_pt.Box(
                    y=score_win,
                    name="Winning trades",
                    marker_color="#00ff88",
                    boxmean=True
                ))
            if score_loss:
                fig_sc.add_trace(go_pt.Box(
                    y=score_loss,
                    name="Losing trades",
                    marker_color="#ff4455",
                    boxmean=True
                ))
            fig_sc.update_layout(
                template="plotly_white",
                height=260,
                margin=dict(l=10,r=10,t=20,b=10),
                yaxis_title="Signal score",
                yaxis_range=[0,11],
                title=(
                    "Were higher-score trades more profitable? "
                    "This tells you if the signal works."
                )
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # Expectancy insight
        st.markdown("#### What your numbers tell you")
        if win_rate >= 50 and profit_factor >= 1.5:
            st.success(
                f"Win rate {win_rate}% and profit factor "
                f"{profit_factor} — your system is working. "
                "You are ready to try small real money."
            )
        elif win_rate >= 40 and profit_factor >= 1.0:
            st.warning(
                f"Win rate {win_rate}% with profit factor "
                f"{profit_factor}. "
                "Getting better. Keep paper trading. "
                "Target 50%+ win rate before going live."
            )
        else:
            st.error(
                f"Win rate {win_rate}%. "
                "The system needs more work before real money. "
                "Review your losing trades — were you following "
                "all 11 checklist rules?"
            )

        if expectancy > 0:
            st.info(
                f"Expectancy: Rs{expectancy:+.2f} per trade. "
                "This means on average each trade makes you "
                f"Rs{abs(expectancy):.0f}. "
                "Positive expectancy is the goal."
            )

        # Closed trades table
        with st.expander("All closed trades"):
            df_pt = pd.DataFrame([{
                "Date":   t["date"],
                "Stock":  t["stock"],
                "Type":   t["type"],
                "Score":  t["score"],
                "Entry":  f"Rs{t['entry']:,.2f}",
                "Exit":   f"Rs{t['exit']:,.2f}",
                "P&L":    f"Rs{t['pnl']:+,.0f}",
                "Result": t["result"],
                "R:R":    t["rr"],
                "Notes":  t["notes"][:30] if t["notes"] else "",
            } for t in closed])
            st.dataframe(
                df_pt,
                use_container_width=True,
                hide_index=True
            )

    elif trades:
        st.info("Close your open trades to see performance analytics.")
    else:
        st.markdown("""
        ### How to use paper trading

        Paper trading means making pretend trades with virtual
        money to practise before risking real capital.

        **Your daily routine:**

        1. Open the Trade Setup tab as normal
        2. When you see a signal (score 7+, 9+ checks green)
           record it here instead of your broker app
        3. Enter the exact entry, stop loss and target
           shown by your terminal
        4. When the trade closes (SL hit or target reached),
           come back and record the exit price
        5. Watch your win rate and capital curve build up

        **When are you ready for real money?**

        | Metric | Target before going live |
        |--------|--------------------------|
        | Win rate | 45% or above |
        | Profit factor | 1.5 or above |
        | Trades completed | At least 20 |
        | Followed all 11 rules | Every single trade |
        | Consistent for weeks | 3+ weeks |

        Most successful traders paper trade for 1 to 3 months
        before using real money. The discipline you build
        here is worth more than any indicator.
        """)

    # ── Excel Download ────────────────────────────────
    st.markdown("---")
    if trades:
        st.markdown("#### Download Paper Trading Data")
        dl1, dl2 = st.columns(2)

        with dl1:
            if st.button(
                "Download as Excel",
                key="pt_download_excel",
                type="primary",
                use_container_width=True
            ):
                import io, openpyxl
                from openpyxl.styles import (
                    PatternFill, Font, Alignment, Border, Side
                )
                from openpyxl.utils import get_column_letter

                wb = openpyxl.Workbook()

                # ── Sheet 1: All Trades ────────────────
                ws1 = wb.active
                ws1.title = "All Trades"

                # Header style
                hdr_fill = PatternFill("solid", fgColor="1E40AF")
                hdr_font = Font(color="FFFFFF", bold=True, size=11)
                hdr_align= Alignment(horizontal="center",
                                     vertical="center")
                thin = Side(style="thin", color="CBD5E1")
                border= Border(left=thin, right=thin,
                               top=thin, bottom=thin)

                headers = [
                    "Date","Stock","Type","Score",
                    "Entry Rs","SL Rs","Target Rs",
                    "Exit Rs","P&L Rs","Result",
                    "Risk Rs","R:R","Notes"
                ]
                ws1.append(headers)
                for ci, h in enumerate(headers, 1):
                    cell = ws1.cell(1, ci)
                    cell.fill    = hdr_fill
                    cell.font    = hdr_font
                    cell.alignment = hdr_align
                    cell.border  = border

                # Data rows
                win_fill  = PatternFill("solid", fgColor="DCFCE7")
                loss_fill = PatternFill("solid", fgColor="FEE2E2")
                open_fill = PatternFill("solid", fgColor="EFF6FF")

                for t in trades:
                    row = [
                        t["date"], t["stock"], t["type"],
                        t["score"],
                        t["entry"], t["sl"], t["target"],
                        t["exit"] if t["exit"] else "",
                        t["pnl"]  if t["pnl"]  else "",
                        t["result"] if t["result"] else "OPEN",
                        t["risk"], t["rr"], t["notes"]
                    ]
                    ws1.append(row)
                    row_num = ws1.max_row
                    fill = (win_fill  if t["result"]=="WIN"
                            else loss_fill if t["result"]=="LOSS"
                            else open_fill)
                    for ci in range(1, len(headers)+1):
                        cell = ws1.cell(row_num, ci)
                        cell.fill   = fill
                        cell.border = border
                        cell.alignment = Alignment(
                            horizontal="center"
                        )

                # Column widths
                col_widths = [12,18,14,8,12,12,12,12,
                              12,10,10,8,30]
                for ci, w in enumerate(col_widths, 1):
                    ws1.column_dimensions[
                        get_column_letter(ci)
                    ].width = w

                ws1.freeze_panes = "A2"
                ws1.auto_filter.ref = ws1.dimensions

                # ── Sheet 2: Summary ───────────────────
                ws2 = wb.create_sheet("Summary")
                closed_t = [t for t in trades
                            if t["status"]=="CLOSED"]
                wins_t   = [t for t in closed_t
                            if t["result"]=="WIN"]
                losses_t = [t for t in closed_t
                            if t["result"]=="LOSS"]

                total_pnl_t = sum(t["pnl"] for t in closed_t)
                win_rate_t  = (round(len(wins_t)/
                               len(closed_t)*100,1)
                               if closed_t else 0)
                avg_win_t   = (round(sum(t["pnl"]
                               for t in wins_t)/
                               len(wins_t),2)
                               if wins_t else 0)
                avg_loss_t  = (round(sum(t["pnl"]
                               for t in losses_t)/
                               len(losses_t),2)
                               if losses_t else 0)
                pf_t = (round(
                    sum(t["pnl"] for t in wins_t) /
                    abs(sum(t["pnl"] for t in losses_t)+0.001)
                    ,2) if losses_t else 999)

                title_font = Font(bold=True, size=14,
                                  color="1E293B")
                label_font = Font(bold=True, size=11,
                                  color="475569")
                val_font   = Font(size=11, color="1E293B")

                ws2["A1"] = "Paper Trading Summary"
                ws2["A1"].font = title_font
                ws2.merge_cells("A1:C1")

                summary_rows = [
                    ("Total Trades",    len(trades)),
                    ("Closed Trades",   len(closed_t)),
                    ("Open Trades",     len(trades)-len(closed_t)),
                    ("Winning Trades",  len(wins_t)),
                    ("Losing Trades",   len(losses_t)),
                    ("Win Rate %",      win_rate_t),
                    ("Total P&L (Rs)",  total_pnl_t),
                    ("Avg Win (Rs)",    avg_win_t),
                    ("Avg Loss (Rs)",   avg_loss_t),
                    ("Profit Factor",   pf_t),
                    ("Starting Capital",
                     st.session_state.get("pt_capital",100000)),
                    ("Final Capital",
                     st.session_state.get("pt_capital",100000)
                     + total_pnl_t),
                ]
                for ri, (label, val) in enumerate(
                        summary_rows, 3):
                    ws2[f"A{ri}"] = label
                    ws2[f"A{ri}"].font = label_font
                    ws2[f"B{ri}"] = val
                    ws2[f"B{ri}"].font = val_font
                    if "P&L" in label or "Win" in label                             or "Loss" in label                             or "Capital" in label:
                        ws2[f"B{ri}"].number_format = (
                            '#,##0.00'
                        )

                ws2.column_dimensions["A"].width = 22
                ws2.column_dimensions["B"].width = 16

                # Save to buffer
                buf = io.BytesIO()
                wb.save(buf)
                buf.seek(0)

                st.download_button(
                    label="Click to Download Excel file",
                    data=buf.getvalue(),
                    file_name=(
                        f"paper_trading_"
                        f"{now_ist().strftime('%Y%m%d')}"
                        f".xlsx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-"
                        "officedocument.spreadsheetml.sheet"
                    ),
                    key="pt_xl_dl"
                )

        with dl2:
            if st.button(
                "Clear all trades (start fresh)",
                key="pt_clear",
                type="secondary",
                use_container_width=True
            ):
                st.session_state["pt_trades"] = []
                st.rerun()



# ╔══════════════════════════════════════════════════════╗
# ║  TAB 9 — TRADE JOURNAL                              ║
# ╚══════════════════════════════════════════════════════╝
with T9:
    st.markdown("### 📓 Trade Journal — Real Money Trades")
    st.caption(
        "Record every real trade you place. "
        "Track performance, review mistakes and improve over time."
    )

    # ── Session state init ─────────────────────────────
    if "tj_trades" not in st.session_state:
        st.session_state["tj_trades"] = []
    if "tj_capital" not in st.session_state:
        st.session_state["tj_capital"] = 100000.0

    tj = st.session_state["tj_trades"]

    # ── Capital setup ──────────────────────────────────
    with st.expander("Account Settings", expanded=len(tj)==0):
        tj_cap = st.number_input(
            "Starting capital (Rs)",
            value=float(st.session_state["tj_capital"]),
            step=10000.0, min_value=1000.0,
            key="tj_capital_input"
        )
        st.session_state["tj_capital"] = tj_cap
        st.info(
            f"Max risk per trade (2%): "
            f"**Rs {tj_cap*0.02:,.0f}**  |  "
            f"Never risk more than this on a single trade."
        )

    st.markdown("---")

    # ── Add new journal entry ──────────────────────────
    st.markdown("#### Record a Real Trade")

    ja, jb, jc = st.columns(3)
    with ja:
        tj_stock  = st.text_input(
            "Stock / Index",
            placeholder="e.g. NIFTY 50, HDFC Bank",
            key="tj_stock"
        )
        tj_type   = st.selectbox(
            "Trade type",
            ["CE (Call)","PE (Put)",
             "Intraday Buy","Intraday Sell",
             "Positional Buy","Positional Sell"],
            key="tj_type"
        )
        tj_score  = st.slider(
            "Signal score at entry",
            0, 10, 7, key="tj_score"
        )
        tj_date   = st.date_input(
            "Trade date",
            value=now_ist().date(),
            key="tj_date"
        )
        tj_time   = st.time_input(
            "Entry time",
            key="tj_time"
        )

    with jb:
        tj_entry  = st.number_input(
            "Entry price (Rs)",
            value=0.0, step=0.5,
            min_value=0.0, key="tj_entry"
        )
        tj_sl     = st.number_input(
            "Stop loss (Rs)",
            value=0.0, step=0.5,
            min_value=0.0, key="tj_sl"
        )
        tj_target = st.number_input(
            "Target (Rs)",
            value=0.0, step=0.5,
            min_value=0.0, key="tj_target"
        )
        tj_exit   = st.number_input(
            "Exit price (Rs) — 0 if still open",
            value=0.0, step=0.5,
            min_value=0.0, key="tj_exit"
        )
        tj_qty    = st.number_input(
            "Quantity / Lots",
            value=1, step=1,
            min_value=1, key="tj_qty"
        )

    with jc:
        tj_emotion = st.selectbox(
            "Emotional state at entry",
            ["Calm and confident",
             "Slightly anxious",
             "FOMO (fear of missing out)",
             "Revenge trading",
             "Overconfident",
             "Uncertain but entered anyway"],
            key="tj_emotion"
        )
        tj_followed = st.selectbox(
            "Followed all 11 checklist rules?",
            ["Yes — all rules followed",
             "No — skipped 1-2 rules",
             "No — skipped 3+ rules",
             "Entered on gut feeling"],
            key="tj_followed"
        )
        tj_setup = st.text_area(
            "Why did you enter? (signal details)",
            placeholder=(
                "Score 8/10, RSI 62, Vol surge, "
                "EMA9 pullback, MACD cross..."
            ),
            height=70, key="tj_setup"
        )
        tj_lesson = st.text_area(
            "What did you learn from this trade?",
            placeholder=(
                "Entered too early before confirmation, "
                "should have waited for candle close..."
            ),
            height=70, key="tj_lesson"
        )

    if st.button(
        "Save Trade to Journal",
        type="primary",
        key="tj_add",
        use_container_width=True
    ):
        if tj_stock and tj_entry > 0 and tj_sl > 0:
            risk_unit = abs(tj_entry - tj_sl)
            tot_risk  = round(risk_unit * tj_qty, 2)
            rr_val    = round(
                abs(tj_target - tj_entry) /
                (risk_unit + 0.001), 2
            ) if tj_target > 0 else 0

            # Calculate P&L if exit given
            if tj_exit > 0:
                if "Buy" in tj_type or "CE" in tj_type:
                    pnl_val = round(
                        (tj_exit - tj_entry) * tj_qty, 2
                    )
                else:
                    pnl_val = round(
                        (tj_entry - tj_exit) * tj_qty, 2
                    )
                res_val = "WIN" if pnl_val > 0 else "LOSS"
                status  = "CLOSED"
            else:
                pnl_val = 0.0
                res_val = ""
                status  = "OPEN"

            entry = {
                "id":       len(tj) + 1,
                "date":     str(tj_date),
                "time":     str(tj_time),
                "stock":    tj_stock,
                "type":     tj_type,
                "score":    tj_score,
                "entry":    tj_entry,
                "sl":       tj_sl,
                "target":   tj_target,
                "exit":     tj_exit,
                "qty":      tj_qty,
                "risk":     tot_risk,
                "rr":       rr_val,
                "pnl":      pnl_val,
                "result":   res_val,
                "status":   status,
                "emotion":  tj_emotion,
                "followed": tj_followed,
                "setup":    tj_setup,
                "lesson":   tj_lesson,
            }
            tj.append(entry)
            st.session_state["tj_trades"] = tj
            st.success(
                f"Saved: {tj_stock} {tj_type} "
                f"@ Rs{tj_entry} | "
                f"Risk Rs{tot_risk:,.0f} | "
                f"R:R {rr_val}:1"
                + (f" | P&L Rs{pnl_val:+,.0f} ({res_val})"
                   if tj_exit > 0 else "")
            )
            if tot_risk > tj_cap * 0.02:
                st.warning(
                    f"Risk Rs{tot_risk:,.0f} exceeds 2% rule "
                    f"(Rs{tj_cap*0.02:,.0f}). "
                    "Review your position sizing."
                )
        else:
            st.error("Fill stock name, entry price and SL.")

    st.markdown("---")

    # ── Open trades ────────────────────────────────────
    open_tj = [t for t in tj if t["status"]=="OPEN"]
    if open_tj:
        st.markdown(f"#### Open Positions ({len(open_tj)})")
        for t in open_tj:
            ot1,ot2,ot3,ot4 = st.columns([3,2,2,1])
            with ot1:
                st.markdown(
                    f"**{t['stock']}** {t['type']}  |  "
                    f"Entry Rs{t['entry']:,.2f}  |  "
                    f"SL Rs{t['sl']:,.2f}  |  "
                    f"Score {t['score']}/10"
                )
            with ot2:
                lp_tj = live_price(STOCKS.get(t["stock"],""))
                if lp_tj["ok"]:
                    cur = lp_tj["p"]
                    unr = round(
                        (cur - t["entry"]) * t["qty"]
                        if "Buy" in t["type"]
                            or "CE" in t["type"]
                        else
                        (t["entry"] - cur) * t["qty"], 2
                    )
                    uc = ("#16a34a" if unr >= 0
                          else "#dc2626")
                    st.markdown(
                        f"Live Rs{cur:,}  "
                        f"<span style='color:{uc}'>"
                        f"Rs{unr:+,.0f}</span>",
                        unsafe_allow_html=True
                    )
            with ot3:
                close_px = st.number_input(
                    "Exit at Rs",
                    value=0.0, step=0.5,
                    key=f"tj_exit_{t['id']}",
                    label_visibility="collapsed"
                )
            with ot4:
                if st.button(
                    "Close",
                    key=f"tj_close_{t['id']}",
                    type="primary"
                ):
                    if close_px > 0:
                        if ("Buy" in t["type"]
                                or "CE" in t["type"]):
                            pnl_c = round(
                                (close_px-t["entry"])
                                * t["qty"], 2
                            )
                        else:
                            pnl_c = round(
                                (t["entry"]-close_px)
                                * t["qty"], 2
                            )
                        for tr in st.session_state[
                                "tj_trades"]:
                            if tr["id"] == t["id"]:
                                tr["status"] = "CLOSED"
                                tr["exit"]   = close_px
                                tr["pnl"]    = pnl_c
                                tr["result"] = (
                                    "WIN" if pnl_c > 0
                                    else "LOSS"
                                )
                        st.rerun()
                    else:
                        st.error("Enter exit price first")

    # ── Performance dashboard ──────────────────────────
    closed_tj = [t for t in tj if t["status"]=="CLOSED"]

    if closed_tj:
        st.markdown("---")
        st.markdown("### Performance Report")

        tot_pnl   = sum(t["pnl"] for t in closed_tj)
        wins_tj   = [t for t in closed_tj
                     if t["result"]=="WIN"]
        losses_tj = [t for t in closed_tj
                     if t["result"]=="LOSS"]
        wr_tj     = round(
            len(wins_tj)/len(closed_tj)*100, 1
        )
        avg_w_tj  = round(
            sum(t["pnl"] for t in wins_tj)/
            (len(wins_tj)+0.001), 2
        )
        avg_l_tj  = round(
            sum(t["pnl"] for t in losses_tj)/
            (len(losses_tj)+0.001), 2
        )
        pf_tj = round(
            sum(t["pnl"] for t in wins_tj) /
            abs(sum(t["pnl"] for t in losses_tj)+0.001),
            2
        ) if losses_tj else 999
        exp_tj = round(
            (wr_tj/100 * avg_w_tj) +
            ((1-wr_tj/100) * avg_l_tj), 2
        )
        cur_cap = tj_cap + tot_pnl
        roi_pct = round(tot_pnl/tj_cap*100, 2)

        # Metrics
        mc = st.columns(6)
        mc[0].metric(
            "Total P&L",
            f"Rs{tot_pnl:+,.0f}",
            delta=f"{roi_pct:+.1f}%"
        )
        mc[1].metric("Win Rate",      f"{wr_tj}%")
        mc[2].metric("Profit Factor", f"{pf_tj}")
        mc[3].metric("Avg Win",       f"Rs{avg_w_tj:+,.0f}")
        mc[4].metric("Avg Loss",      f"Rs{avg_l_tj:+,.0f}")
        mc[5].metric("Expectancy",    f"Rs{exp_tj:+.0f}")

        # ── Charts ────────────────────────────────────
        import plotly.graph_objects as go_tj
        ch1, ch2 = st.columns(2)

        with ch1:
            # Capital curve
            running = tj_cap
            cx, cy  = ["Start"], [tj_cap]
            for t in sorted(closed_tj,
                            key=lambda x: x["date"]):
                running += t["pnl"]
                cx.append(f"#{t['id']} {t['stock'][:6]}")
                cy.append(round(running, 2))

            fig_cap = go_tj.Figure()
            col_line= ("#16a34a" if cy[-1]>=tj_cap
                       else "#dc2626")
            fig_cap.add_trace(go_tj.Scatter(
                x=cx, y=cy,
                mode="lines+markers",
                line=dict(color=col_line, width=2.5),
                marker=dict(size=7, color=col_line),
                fill="tozeroy",
                fillcolor=(
                    "rgba(22,163,74,0.08)"
                    if cy[-1]>=tj_cap
                    else "rgba(220,38,38,0.08)"
                ),
                name="Capital"
            ))
            fig_cap.add_hline(
                y=tj_cap,
                line_dash="dash",
                line_color="#94a3b8",
                opacity=0.5,
                annotation_text="Starting capital"
            )
            fig_cap.update_layout(
                template="plotly_white",
                height=280,
                title="Capital curve",
                xaxis_tickangle=-30,
                yaxis_title="Rs",
                margin=dict(l=10,r=10,t=40,b=60),
                showlegend=False
            )
            st.plotly_chart(
                fig_cap, use_container_width=True
            )

        with ch2:
            # Win vs Loss bar
            months = {}
            for t in closed_tj:
                m = t["date"][:7]
                if m not in months:
                    months[m] = {"win":0,"loss":0,"pnl":0}
                if t["result"]=="WIN":
                    months[m]["win"]  += 1
                else:
                    months[m]["loss"] += 1
                months[m]["pnl"] += t["pnl"]

            if months:
                fig_m = go_tj.Figure()
                fig_m.add_trace(go_tj.Bar(
                    x=list(months.keys()),
                    y=[v["win"] for v in months.values()],
                    name="Wins",
                    marker_color="#16a34a",
                    opacity=0.85
                ))
                fig_m.add_trace(go_tj.Bar(
                    x=list(months.keys()),
                    y=[v["loss"] for v in months.values()],
                    name="Losses",
                    marker_color="#dc2626",
                    opacity=0.85
                ))
                fig_m.update_layout(
                    barmode="group",
                    template="plotly_white",
                    height=280,
                    title="Monthly wins vs losses",
                    margin=dict(l=10,r=10,t=40,b=30),
                    yaxis_title="Trades",
                    legend=dict(
                        orientation="h",
                        y=1.1
                    )
                )
                st.plotly_chart(
                    fig_m, use_container_width=True
                )

        # ── Emotion & Discipline analysis ─────────────
        st.markdown("#### Behaviour Analysis")
        ba1, ba2 = st.columns(2)

        with ba1:
            st.markdown("##### Emotion vs Result")
            emotion_data = {}
            for t in closed_tj:
                em = t.get("emotion","Unknown")
                if em not in emotion_data:
                    emotion_data[em] = {"w":0,"l":0}
                if t["result"]=="WIN":
                    emotion_data[em]["w"] += 1
                else:
                    emotion_data[em]["l"] += 1

            for em, data in emotion_data.items():
                total_em = data["w"] + data["l"]
                wr_em    = round(data["w"]/total_em*100)
                col_em   = ("#16a34a" if wr_em>=50
                            else "#dc2626")
                st.markdown(
                    f"<div style='background:#f8fafc;"
                    f"border:1px solid #e2e8f0;"
                    f"border-radius:8px;"
                    f"padding:10px 14px;margin:4px 0;"
                    f"display:flex;"
                    f"justify-content:space-between'>"
                    f"<span style='color:#374151;"
                    f"font-size:13px'>{em}</span>"
                    f"<span style='color:{col_em};"
                    f"font-weight:700;font-size:13px'>"
                    f"{wr_em}% win rate "
                    f"({total_em} trades)</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        with ba2:
            st.markdown("##### Discipline vs Result")
            disc_data = {}
            for t in closed_tj:
                df_ = t.get("followed","Unknown")
                if df_ not in disc_data:
                    disc_data[df_] = {"w":0,"l":0}
                if t["result"]=="WIN":
                    disc_data[df_]["w"] += 1
                else:
                    disc_data[df_]["l"] += 1

            for df_, data in disc_data.items():
                total_d = data["w"] + data["l"]
                wr_d    = round(data["w"]/total_d*100)
                col_d   = ("#16a34a" if wr_d>=50
                           else "#dc2626")
                label   = df_[:35]+"..." if len(df_)>35                           else df_
                st.markdown(
                    f"<div style='background:#f8fafc;"
                    f"border:1px solid #e2e8f0;"
                    f"border-radius:8px;"
                    f"padding:10px 14px;margin:4px 0;"
                    f"display:flex;"
                    f"justify-content:space-between'>"
                    f"<span style='color:#374151;"
                    f"font-size:13px'>{label}</span>"
                    f"<span style='color:{col_d};"
                    f"font-weight:700;font-size:13px'>"
                    f"{wr_d}% win "
                    f"({total_d})</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # ── Lessons learned section ────────────────────
        lessons = [t for t in closed_tj
                   if t.get("lesson","").strip()]
        if lessons:
            st.markdown("---")
            st.markdown("#### Lessons Learned")
            for t in lessons[-5:]:
                col_r = ("#16a34a" if t["result"]=="WIN"
                         else "#dc2626")
                st.markdown(
                    f"<div style='background:#f8fafc;"
                    f"border-left:3px solid {col_r};"
                    f"border-radius:0 8px 8px 0;"
                    f"padding:10px 14px;margin:6px 0'>"
                    f"<div style='font-size:12px;"
                    f"color:#64748b;margin-bottom:4px'>"
                    f"{t['date']} | {t['stock']} | "
                    f"{t['result']}</div>"
                    f"<div style='font-size:13px;"
                    f"color:#374151'>{t['lesson']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # ── All trades table ───────────────────────────
        with st.expander("All journal entries"):
            df_tj = pd.DataFrame([{
                "Date":      t["date"],
                "Stock":     t["stock"],
                "Type":      t["type"],
                "Score":     t["score"],
                "Emotion":   t.get("emotion","")[:20],
                "Followed":  "Yes" if "Yes" in
                             t.get("followed","") else "No",
                "Entry":     f"Rs{t['entry']:,.2f}",
                "Exit":      f"Rs{t['exit']:,.2f}"
                             if t["exit"] else "Open",
                "P&L":       f"Rs{t['pnl']:+,.0f}"
                             if t["pnl"] else "Open",
                "Result":    t["result"] or "Open",
            } for t in tj])
            st.dataframe(
                df_tj,
                use_container_width=True,
                hide_index=True
            )

    # ── Readiness verdict ──────────────────────────────
    if closed_tj and len(closed_tj) >= 5:
        st.markdown("---")
        wr_check = wr_tj >= 45
        pf_check = pf_tj >= 1.5
        disc_pct = sum(
            1 for t in closed_tj
            if "Yes" in t.get("followed","")
        ) / len(closed_tj) * 100

        if wr_check and pf_check and disc_pct >= 80:
            st.success(
                f"Win rate {wr_tj}% | Profit factor {pf_tj} | "
                f"Discipline {disc_pct:.0f}% — "
                "Your trading is consistent. "
                "Ready to increase position size gradually."
            )
        elif wr_check or pf_check:
            st.warning(
                f"Win rate {wr_tj}% | Profit factor {pf_tj} — "
                "Getting better. Follow all 11 rules every trade."
            )
        else:
            st.error(
                f"Win rate {wr_tj}% | Profit factor {pf_tj} — "
                "Review losing trades. Are you following all 11 rules?"
            )

    # ── Download Excel ─────────────────────────────────
    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        if tj and st.button(
            "Download Journal as Excel",
            type="primary",
            key="tj_download",
            use_container_width=True
        ):
            import io, openpyxl
            from openpyxl.styles import (
                PatternFill, Font, Alignment, Border, Side
            )
            from openpyxl.utils import get_column_letter

            wb2 = openpyxl.Workbook()

            # Sheet 1: All entries
            ws_all = wb2.active
            ws_all.title = "All Trades"
            hf = PatternFill("solid", fgColor="1E40AF")
            hft= Font(color="FFFFFF", bold=True, size=11)
            bd = Border(
                left=Side(style="thin",color="CBD5E1"),
                right=Side(style="thin",color="CBD5E1"),
                top=Side(style="thin",color="CBD5E1"),
                bottom=Side(style="thin",color="CBD5E1")
            )
            hdrs2 = [
                "Date","Time","Stock","Type","Score",
                "Entry Rs","SL Rs","Target Rs","Exit Rs",
                "Qty","Risk Rs","R:R","P&L Rs","Result",
                "Emotion","Followed Rules","Setup","Lesson"
            ]
            ws_all.append(hdrs2)
            for ci2, h2 in enumerate(hdrs2, 1):
                c2 = ws_all.cell(1, ci2)
                c2.fill = hf
                c2.font = hft
                c2.alignment = Alignment(horizontal="center")
                c2.border = bd

            wf = PatternFill("solid", fgColor="DCFCE7")
            lf = PatternFill("solid", fgColor="FEE2E2")
            of = PatternFill("solid", fgColor="EFF6FF")

            for t in tj:
                row2 = [
                    t["date"], t.get("time",""),
                    t["stock"], t["type"], t["score"],
                    t["entry"], t["sl"], t["target"],
                    t["exit"] or "", t["qty"],
                    t["risk"], t["rr"],
                    t["pnl"] or "", t["result"] or "OPEN",
                    t.get("emotion",""),
                    t.get("followed",""),
                    t.get("setup",""),
                    t.get("lesson","")
                ]
                ws_all.append(row2)
                rn2 = ws_all.max_row
                fill2 = (wf if t["result"]=="WIN"
                         else lf if t["result"]=="LOSS"
                         else of)
                for ci2 in range(1, len(hdrs2)+1):
                    cell2 = ws_all.cell(rn2, ci2)
                    cell2.fill   = fill2
                    cell2.border = bd
                    cell2.alignment = Alignment(
                        horizontal="center",
                        wrap_text=True
                    )

            widths2 = [12,10,18,14,8,12,12,12,12,
                       8,10,8,12,10,22,20,30,30]
            for ci2, w2 in enumerate(widths2, 1):
                ws_all.column_dimensions[
                    get_column_letter(ci2)
                ].width = w2
            ws_all.freeze_panes = "A2"
            ws_all.auto_filter.ref = ws_all.dimensions

            # Sheet 2: Monthly summary
            ws_m = wb2.create_sheet("Monthly Summary")
            ws_m.append(["Month","Trades","Wins",
                         "Losses","Win%","Total P&L"])
            months2 = {}
            for t in closed_tj:
                m2 = t["date"][:7]
                if m2 not in months2:
                    months2[m2] = {
                        "t":0,"w":0,"l":0,"pnl":0
                    }
                months2[m2]["t"] += 1
                if t["result"]=="WIN":
                    months2[m2]["w"] += 1
                else:
                    months2[m2]["l"] += 1
                months2[m2]["pnl"] += t["pnl"]

            for m2, d2 in sorted(months2.items()):
                wr2 = round(d2["w"]/d2["t"]*100, 1)
                ws_m.append([
                    m2, d2["t"], d2["w"], d2["l"],
                    wr2, round(d2["pnl"],2)
                ])

            ws_m.column_dimensions["A"].width = 12
            ws_m.column_dimensions["F"].width = 14

            buf2 = io.BytesIO()
            wb2.save(buf2)
            buf2.seek(0)

            st.download_button(
                label="Click to Download Journal Excel",
                data=buf2.getvalue(),
                file_name=(
                    f"trade_journal_"
                    f"{now_ist().strftime('%Y%m%d')}.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                key="tj_xl_dl"
            )

    with dl_col2:
        if tj and st.button(
            "Clear journal (start fresh)",
            type="secondary",
            key="tj_clear",
            use_container_width=True
        ):
            st.session_state["tj_trades"] = []
            st.rerun()

    if not tj:
        st.markdown("""
        ### How to use the Trade Journal

        Record **every real money trade** here — not just the
        good ones. The whole point of a journal is honesty.

        **What makes this journal powerful:**

        It tracks not just profit and loss but also your
        **emotional state** and whether you **followed your
        rules**. After 20 trades you will see clearly:

        - Do you win more when you are calm vs anxious?
        - Do trades where you skipped rules lose money?
        - Which stocks work best for your style?
        - What time of day do you trade best?

        **The single most important column is Lessons Learned.**
        Write one sentence after every trade — good or bad.
        In 3 months this becomes your personal trading manual
        worth more than any course or book.

        **Download to Excel anytime** to keep a permanent
        record even if you clear the app.
        """)


# ── Auto refresh ──────────────────────────────────────────
if auto_rf:
    st.sidebar.success("🔄 Refreshing every 2 min...")
    time.sleep(120)
    st.rerun()
