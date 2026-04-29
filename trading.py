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

# ── Persistent credential storage using JSON file ─────────
import json, os

CREDS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".trading_creds.json"
)

def load_creds():
    """Load saved credentials from local JSON file."""
    try:
        if os.path.exists(CREDS_FILE):
            with open(CREDS_FILE, "r") as f:
                data = json.load(f)
            # Load into session state if not already set
            if ("tg_token_saved" not in st.session_state
                    and data.get("tg_token")):
                st.session_state["tg_token_saved"] = (
                    data["tg_token"]
                )
            if ("tg_chat_saved" not in st.session_state
                    and data.get("tg_chat")):
                st.session_state["tg_chat_saved"] = (
                    data["tg_chat"]
                )
    except Exception:
        pass

def save_creds(token: str, chat: str):
    """Save credentials to local JSON file permanently."""
    try:
        with open(CREDS_FILE, "w") as f:
            json.dump({
                "tg_token": token,
                "tg_chat":  chat
            }, f)
        return True
    except Exception:
        return False

# Load credentials on every page load
load_creds()

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
}
TAB_NAMES = [
    "📋 Watchlist",
    "🎯 Trade Setup",
    "🔍 Auto Scanner",
    "🤖 ML Prediction",
    "🏦 Smart Money",
    "🧮 P&L Calculator",
    "📰 News & Events",
]
TAB_ICONS = ["📋","🎯","🔍","🤖","🏦","🧮","📰"]
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

/* ── Mobile-first layout ──────────────────────────── */
.block-container {
    padding-top: 0.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 1400px !important;
}

/* Streamlit header - make transparent on all devices */
header[data-testid="stHeader"] {
    background: rgba(244,246,249,0.95) !important;
    border-bottom: 1px solid #e2e8f0 !important;
    backdrop-filter: blur(8px) !important;
    z-index: 999 !important;
}

/* Push content below fixed header */
.main .block-container {
    padding-top: 1rem !important;
}

/* ── Mobile responsive — comprehensive fix ──────── */
@media (max-width: 768px) {
    /* Layout */
    .block-container {
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
        padding-top: 0.3rem !important;
    }

    /* Top navigation bar — make visible on mobile */
    header[data-testid="stHeader"] {
        background: #1e3a5f !important;
        border-bottom: 2px solid #1d4ed8 !important;
    }
    header[data-testid="stHeader"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    /* Hamburger menu icon */
    button[kind="header"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    [data-testid="stToolbar"] {
        background: transparent !important;
    }
    [data-testid="stToolbar"] svg {
        fill: #ffffff !important;
    }
    /* Make all header icons white */
    header svg, header button {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    /* Tabs — horizontal scroll, no wrap */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        gap: 2px !important;
        padding: 3px !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        white-space: nowrap !important;
        font-size: 11px !important;
        padding: 5px 8px !important;
        flex-shrink: 0 !important;
        min-width: fit-content !important;
    }
    .stTabs [aria-selected="true"] {
        font-size: 11px !important;
    }

    /* Headings */
    h1 { font-size: 18px !important; }
    h2 { font-size: 16px !important; }
    h3 { font-size: 15px !important; }
    h4 { font-size: 14px !important; }
    p, .stMarkdown p {
        font-size: 13px !important;
        line-height: 1.5 !important;
    }
    caption, .stCaption {
        font-size: 11px !important;
    }

    /* Buttons */
    .stButton button {
        font-size: 12px !important;
        padding: 7px 10px !important;
        border-radius: 8px !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        padding: 8px 10px !important;
        border-radius: 8px !important;
    }
    div[data-testid="metric-container"] label {
        font-size: 10px !important;
    }
    div[data-testid="metric-container"]
        [data-testid="stMetricValue"] {
        font-size: 16px !important;
        font-weight: 700 !important;
    }

    /* Inputs — prevent iOS zoom */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"] div {
        font-size: 16px !important;
        border-radius: 8px !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 260px !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        font-size: 12px !important;
        padding: 6px 10px !important;
    }

    /* Plotly charts full width */
    .js-plotly-plot {
        width: 100% !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 13px !important;
        padding: 8px 10px !important;
    }

    /* Alert boxes */
    div[data-testid="stSuccess"],
    div[data-testid="stWarning"],
    div[data-testid="stError"],
    div[data-testid="stInfo"] {
        padding: 10px 14px !important;
        font-size: 13px !important;
        border-radius: 8px !important;
    }

    /* Dataframe */
    .stDataFrame {
        font-size: 11px !important;
    }

    /* Columns gap */
    [data-testid="column"] {
        padding: 0 2px !important;
    }
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
    # ── Indices ───────────────────────────────────────
    "NIFTY 50":         "^NSEI",
    "BANK NIFTY":       "^NSEBANK",
    "SENSEX":           "^BSESN",
    "NIFTY IT":         "^CNXIT",
    "NIFTY AUTO":       "^CNXAUTO",
    "NIFTY PHARMA":     "^CNXPHARMA",
    "NIFTY METAL":      "^CNXMETAL",
    "NIFTY FMCG":       "^CNXFMCG",
    "NIFTY MIDCAP":     "^NSEMDCP50",
    "INDIA VIX":        "^INDIAVIX",
    # ── Banking & Finance ─────────────────────────────
    "HDFC Bank":        "HDFCBANK.NS",
    "ICICI Bank":       "ICICIBANK.NS",
    "SBI":              "SBIN.NS",
    "Kotak Bank":       "KOTAKBANK.NS",
    "Axis Bank":        "AXISBANK.NS",
    "IndusInd Bank":    "INDUSINDBK.NS",
    "Yes Bank":         "YESBANK.NS",
    "PNB":              "PNB.NS",
    "Bank of Baroda":   "BANKBARODA.NS",
    "Canara Bank":      "CANBK.NS",
    "Federal Bank":     "FEDERALBNK.NS",
    "IDFC First":       "IDFCFIRSTB.NS",
    "Bajaj Finance":    "BAJFINANCE.NS",
    "Bajaj Finserv":    "BAJAJFINSV.NS",
    "Shriram Finance":  "SHRIRAMFIN.NS",
    "Muthoot Finance":  "MUTHOOTFIN.NS",
    "LIC Housing":      "LICHSGFIN.NS",
    "HDFC Life":        "HDFCLIFE.NS",
    "SBI Life":         "SBILIFE.NS",
    "ICICI Lombard":    "ICICIGI.NS",
    # ── IT & Technology ──────────────────────────────
    "TCS":              "TCS.NS",
    "Infosys":          "INFY.NS",
    "Wipro":            "WIPRO.NS",
    "HCL Tech":         "HCLTECH.NS",
    "Tech Mahindra":    "TECHM.NS",
    "Persistent":       "PERSISTENT.NS",
    "Coforge":          "COFORGE.NS",
    "LTIMindtree":      "LTIM.NS",
    "Mphasis":          "MPHASIS.NS",
    "Tata Elxsi":       "TATAELXSI.NS",
    "KPIT Tech":        "KPITTECH.NS",
    "Cyient":           "CYIENT.NS",
    # ── Energy & Power ───────────────────────────────
    "Reliance":         "RELIANCE.NS",
    "ONGC":             "ONGC.NS",
    "Indian Oil":       "IOC.NS",
    "BPCL":             "BPCL.NS",
    "NTPC":             "NTPC.NS",
    "Power Grid":       "POWERGRID.NS",
    "Adani Green":      "ADANIGREEN.NS",
    "Adani Power":      "ADANIPOWER.NS",
    "Adani Energy":     "ADANIENSOL.NS",
    "Tata Power":       "TATAPOWER.NS",
    "Gail":             "GAIL.NS",
    "Coal India":       "COALINDIA.NS",
    "NHPC":             "NHPC.NS",
    "SJVN":             "SJVN.NS",
    "Torrent Power":    "TORNTPOWER.NS",
    "JSW Energy":       "JSWENERGY.NS",
    # ── Auto & Auto Ancillary ─────────────────────────
    "Maruti":           "MARUTI.NS",
    "M&M":              "M&M.NS",
    "Hero MotoCorp":    "HEROMOTOCO.NS",
    "Bajaj Auto":       "BAJAJ-AUTO.NS",
    "TVS Motor":        "TVSMOTOR.NS",
    "Eicher Motors":    "EICHERMOT.NS",
    "Ashok Leyland":    "ASHOKLEY.NS",
    "Tata Motors":      "TATAMOTORS.NS",
    "Samvardhana":      "MOTHERSON.NS",
    "Bosch":            "BOSCHLTD.NS",
    "Balkrishna Ind":   "BALKRISIND.NS",
    "MRF":              "MRF.NS",
    "Apollo Tyres":     "APOLLOTYRE.NS",
    # ── Pharma & Healthcare ───────────────────────────
    "Sun Pharma":       "SUNPHARMA.NS",
    "Dr Reddy":         "DRREDDY.NS",
    "Cipla":            "CIPLA.NS",
    "Divi's Lab":       "DIVISLAB.NS",
    "Lupin":            "LUPIN.NS",
    "Apollo Hosp":      "APOLLOHOSP.NS",
    "Mankind Pharma":   "MANKIND.NS",
    "Zydus Life":       "ZYDUSLIFE.NS",
    "Alkem Lab":        "ALKEM.NS",
    "Torrent Pharma":   "TORNTPHARM.NS",
    "Aurobindo":        "AUROPHARMA.NS",
    "Max Healthcare":   "MAXHEALTH.NS",
    # ── FMCG & Consumer ──────────────────────────────
    "HUL":              "HINDUNILVR.NS",
    "ITC":              "ITC.NS",
    "Nestle":           "NESTLEIND.NS",
    "Britannia":        "BRITANNIA.NS",
    "Dabur":            "DABUR.NS",
    "Tata Consumer":    "TATACONSUM.NS",
    "Marico":           "MARICO.NS",
    "Colgate":          "COLPAL.NS",
    "Emami":            "EMAMILTD.NS",
    "Godrej Consumer":  "GODREJCP.NS",
    "United Spirits":   "MCDOWELL-N.NS",
    # ── Metals & Mining ──────────────────────────────
    "Tata Steel":       "TATASTEEL.NS",
    "JSW Steel":        "JSWSTEEL.NS",
    "Hindalco":         "HINDALCO.NS",
    "Vedanta":          "VEDL.NS",
    "SAIL":             "SAIL.NS",
    "NMDC":             "NMDC.NS",
    "Jindal Steel":     "JSPL.NS",
    "APL Apollo":       "APLAPOLLO.NS",
    "Hindustan Zinc":   "HINDZINC.NS",
    "National Aluminium":"NATIONALUM.NS",
    # ── Infrastructure & Realty ───────────────────────
    "L&T":              "LT.NS",
    "UltraTech":        "ULTRACEMCO.NS",
    "DLF":              "DLF.NS",
    "Godrej Properties":"GODREJPROP.NS",
    "Prestige Estate":  "PRESTIGE.NS",
    "Oberoi Realty":    "OBEROIRLTY.NS",
    "Macrotech":        "LODHA.NS",
    "ACC":              "ACC.NS",
    "Ambuja Cement":    "AMBUJACEM.NS",
    "Shree Cement":     "SHREECEM.NS",
    "Dalmia Bharat":    "DALBHARAT.NS",
    # ── Adani Group ──────────────────────────────────
    "Adani Ports":      "ADANIPORTS.NS",
    "Adani Enterprises":"ADANIENT.NS",
    "Adani Total Gas":  "ATGL.NS",
    "Adani Wilmar":     "AWL.NS",
    "NDTV":             "NDTV.NS",
    # ── Tata Group ───────────────────────────────────
    "Tata Chemicals":   "TATACHEM.NS",
    "Titan":            "TITAN.NS",
    "Trent":            "TRENT.NS",
    "Tata Comm":        "TATACOMM.NS",
    "Indian Hotels":    "INDHOTEL.NS",
    "Voltas":           "VOLTAS.NS",
    # ── Defence & Railways ───────────────────────────
    "HAL":              "HAL.NS",
    "BEL":              "BEL.NS",
    "Mazagon Dock":     "MAZDOCK.NS",
    "RVNL":             "RVNL.NS",
    "IRFC":             "IRFC.NS",
    "IRCTC":            "IRCTC.NS",
    "IRCON":            "IRCON.NS",
    "BEML":             "BEML.NS",
    "Cochin Shipyard":  "COCHINSHIP.NS",
    "Garden Reach":     "GRSE.NS",
    "BDL":              "BDL.NS",
    "Data Patterns":    "DATAPATTNS.NS",
    # ── Telecom ──────────────────────────────────────
    "Bharti Airtel":    "BHARTIARTL.NS",
    "Vodafone Idea":    "IDEA.NS",
    "Indus Towers":     "INDUSTOWER.NS",
    # ── New Age & Consumer Tech ───────────────────────
    "Zomato":           "ZOMATO.NS",
    "Nykaa":            "NYKAA.NS",
    "Paytm":            "PAYTM.NS",
    "DMart":            "DMART.NS",
    "PB Fintech":       "POLICYBZR.NS",
    "CarTrade":         "CARTRADE.NS",
    # ── Chemicals & Specialty ─────────────────────────
    "Asian Paints":     "ASIANPAINT.NS",
    "Pidilite":         "PIDILITIND.NS",
    "SRF":              "SRF.NS",
    "Navin Fluorine":   "NAVINFLUOR.NS",
    "Deepak Nitrite":   "DEEPAKNTR.NS",
    "Atul Ltd":         "ATUL.NS",
    # ── Capital Goods & Engineering ───────────────────
    "Siemens":          "SIEMENS.NS",
    "ABB India":        "ABB.NS",
    "Havells":          "HAVELLS.NS",
    "Crompton":         "CROMPTON.NS",
    "Thermax":          "THERMAX.NS",
    "Cummins":          "CUMMINSIND.NS",
    "Bharat Forge":     "BHARATFORG.NS",
    "Schaeffler":       "SCHAEFFLER.NS",
    # ── Media & Entertainment ─────────────────────────
    "Sun TV":           "SUNTV.NS",
    "Zee Entertainment":"ZEEL.NS",
    "PVR Inox":         "PVRINOX.NS",
    # ── Additional Banking & NBFC ─────────────────────
    "RBL Bank":         "RBLBANK.NS",
    "Karnataka Bank":   "KTKBANK.NS",
    "DCB Bank":         "DCBBANK.NS",
    "Ujjivan Small":    "UJJIVANSFB.NS",
    "Equitas Small":    "EQUITASBNK.NS",
    "AU Small Finance": "AUBANK.NS",
    "Cholamandalam":    "CHOLAFIN.NS",
    "Mahindra Finance": "M&MFIN.NS",
    "Piramal Enterprises":"PIRAMALENT.NS",
    "Aditya Birla Cap": "ABCAPITAL.NS",
    "Manappuram":       "MANAPPURAM.NS",
    "IIFL Finance":     "IIFL.NS",
    # ── Additional IT ─────────────────────────────────
    "Oracle India":     "OFSS.NS",
    "Wipro":            "WIPRO.NS",
    "Hexaware":         "HEXAWARE.NS",
    "Sonata Software":  "SONATSOFTW.NS",
    "Mastek":           "MASTEK.NS",
    "Tanla Platforms":  "TANLA.NS",
    "Route Mobile":     "ROUTE.NS",
    "Intellect Design": "INTELLECT.NS",
    # ── Additional Pharma ─────────────────────────────
    "Biocon":           "BIOCON.NS",
    "Ipca Labs":        "IPCALAB.NS",
    "Ajanta Pharma":    "AJANTPHARM.NS",
    "Alembic Pharma":   "APLLTD.NS",
    "Gland Pharma":     "GLAND.NS",
    "Syngene":          "SYNGENE.NS",
    "Aarti Drugs":      "AARTIDRUGS.NS",
    "Laurus Labs":      "LAURUSLABS.NS",
    "Granules India":   "GRANULES.NS",
    "Suven Pharma":     "SUVENPHAR.NS",
    # ── Additional FMCG & Consumer ────────────────────
    "Varun Beverages":  "VBL.NS",
    "United Breweries": "UBL.NS",
    "Jubilant Food":    "JUBLFOOD.NS",
    "Westlife Food":    "WESTLIFE.NS",
    "Restaurant Brands":"RBASKETSS.NS",
    "Devyani Intl":     "DEVYANI.NS",
    "Bikaji Foods":     "BIKAJI.NS",
    "Mrs Bectors":      "BECTORFOOD.NS",
    "Radico Khaitan":   "RADICO.NS",
    "Globus Spirits":   "GLOBUSSPR.NS",
    # ── Additional Auto ───────────────────────────────
    "Mahindra CIE":     "MAHINDCIE.NS",
    "Endurance Tech":   "ENDURANCE.NS",
    "Minda Corp":       "MINDACORP.NS",
    "Suprajit Engg":    "SUPRAJIT.NS",
    "Craftsman Auto":   "CRAFTSMAN.NS",
    "Sona BLW":         "SONACOMS.NS",
    "Uno Minda":        "UNOMINDA.NS",
    # ── Additional Metals ─────────────────────────────
    "Welspun Corp":     "WELCORP.NS",
    "Ratnamani Metals": "RATNAMANI.NS",
    "Mishra Dhatu":     "MIDHANI.NS",
    "Gravita India":    "GRAVITA.NS",
    "Hindustan Copper": "HINDCOPPER.NS",
    # ── Textiles ──────────────────────────────────────
    "Page Industries":  "PAGEIND.NS",
    "Trident":          "TRIDENT.NS",
    "Vardhman Textile": "VTL.NS",
    "GHCL":             "GHCL.NS",
    "KPR Mill":         "KPRMILL.NS",
    "Welspun India":    "WELSPUNIND.NS",
    # ── Real Estate & Construction ────────────────────
    "NCC Ltd":          "NCC.NS",
    "KNR Constructions":"KNRCON.NS",
    "PNC Infratech":    "PNCINFRA.NS",
    "IRB Infra":        "IRB.NS",
    "Ashoka Buildcon":  "ASHOKA.NS",
    "PSP Projects":     "PSPPROJECT.NS",
    "Brigade Enterprises":"BRIGADE.NS",
    "Sobha":            "SOBHA.NS",
    "Signature Global": "SIGNATURE.NS",
    # ── Power & Utilities ─────────────────────────────
    "Tata Consultancy": "CESC.NS",
    "CESC":             "CESC.NS",
    "Torrent Power":    "TORNTPOWER.NS",
    "GIPCL":            "GIPCL.NS",
    "GE Vernova":       "GEVERNOVA.NS",
    "Hitachi Energy":   "POWERINDIA.NS",
    "KPI Green":        "KPIGREEN.NS",
    "Waaree Energies":  "WAAREEENER.NS",
    "Premier Energies": "PREMIERENS.NS",
    # ── Agrochemicals & Fertilisers ───────────────────
    "UPL":              "UPL.NS",
    "PI Industries":    "PIIND.NS",
    "Coromandel Intl":  "COROMANDEL.NS",
    "Bayer Cropscience":"BAYERCROP.NS",
    "Chambal Fert":     "CHAMBLFERT.NS",
    "Rallis India":     "RALLIS.NS",
    "Astec Lifesciences":"ASTEC.NS",
    "Dhanuka Agritech": "DHANUKA.NS",
    # ── Logistics & Shipping ──────────────────────────
    "Container Corp":   "CONCOR.NS",
    "VRL Logistics":    "VRLLOG.NS",
    "TCI Express":      "TCIEXP.NS",
    "Delhivery":        "DELHIVERY.NS",
    "Blue Dart":        "BLUEDART.NS",
    "Mahindra Logistics":"MAHLOG.NS",
    "Gateway Distriparks":"GDL.NS",
    "Shipping Corp":    "SCI.NS",
    # ── Healthcare Devices & Hospitals ────────────────
    "Narayana Hrudayalaya":"NH.NS",
    "Fortis Healthcare":"FORTIS.NS",
    "Krishna Institute":"KIMS.NS",
    "Global Health":    "MEDANTA.NS",
    "Aster DM":         "ASTERDM.NS",
    "Poly Medicure":    "POLYMED.NS",
    "Vijaya Diagnostic":"VIJAYA.NS",
    # ── Specialty Finance ─────────────────────────────
    "BSE Ltd":          "BSE.NS",
    "CDSL":             "CDSL.NS",
    "CAMS":             "CAMS.NS",
    "KFintech":         "KFINTECH.NS",
    "Angel One":        "ANGELONE.NS",
    "5Paisa Capital":   "5PAISA.NS",
    "MCX":              "MCX.NS",
    "Multi Comm Exch":  "MCX.NS",
    # ── Hotels & Tourism ──────────────────────────────
    "EIH Ltd":          "EIHOTEL.NS",
    "Lemon Tree":       "LEMONTREE.NS",
    "Chalet Hotels":    "CHALET.NS",
    "Mahindra Holidays":"MHRIL.NS",
    "Thomas Cook":      "THOMASCOOK.NS",
    # ── Retail ────────────────────────────────────────
    "Titan":            "TITAN.NS",
    "Kalyan Jewellers": "KALYANKJIL.NS",
    "Senco Gold":       "SENCO.NS",
    "PC Jeweller":      "PCJEWELLER.NS",
    "Shopper Stop":     "SHOPERSTOP.NS",
    "V-Mart Retail":    "VMART.NS",
    "Aditya Birla Fash":"ABFRL.NS",
    "Vedant Fashions":  "MANYAVAR.NS",
    # ── Miscellaneous ─────────────────────────────────
    "3M India":         "3MINDIA.NS",
    "Honeywell Auto":   "HONAUT.NS",
    "ABB India":        "ABB.NS",
    "Thermax":          "THERMAX.NS",
    "Triveni Turbine":  "TRITURBINE.NS",
    "Lakshmi Machine":  "LMWLTD.NS",
    "TD Power Systems": "TDPOWERSYS.NS",
    "Elecon Engg":      "ELECON.NS",
    "Rexnord Elect":    "REXNORD.NS",
    "Kaynes Tech":      "KAYNES.NS",
    "Syrma SGS":        "SYRMA.NS",
    "Avalon Tech":      "AVALON.NS",
}

SECTORS = {
    "🏆 Top 30 F&O": [
        "NIFTY 50","BANK NIFTY","Reliance","HDFC Bank",
        "ICICI Bank","TCS","Infosys","SBI","Wipro",
        "Bajaj Finance","ITC","Sun Pharma","L&T","Maruti",
        "Coal India","NTPC","Bharti Airtel","Tata Steel",
        "Axis Bank","HCL Tech","Power Grid","Adani Ports",
        "Hindalco","ONGC","Bajaj Auto","Titan",
        "JSW Steel","UltraTech","BEL","Adani Enterprises",
    ],
    "🏦 Banking": [
        "HDFC Bank","ICICI Bank","SBI","Kotak Bank",
        "Axis Bank","IndusInd Bank","Bajaj Finance",
        "PNB","Bank of Baroda","Canara Bank",
        "Federal Bank","IDFC First","Bajaj Finserv",
        "Yes Bank","Shriram Finance","Muthoot Finance",
    ],
    "💻 IT": [
        "TCS","Infosys","Wipro","HCL Tech",
        "Tech Mahindra","Persistent","Coforge",
        "LTIMindtree","Mphasis","Tata Elxsi","KPIT Tech",
    ],
    "🛢️ Energy & Power": [
        "Reliance","ONGC","Indian Oil","BPCL",
        "NTPC","Power Grid","Adani Green","Adani Power",
        "Tata Power","Gail","NHPC","JSW Energy",
        "Torrent Power","Coal India",
    ],
    "🚗 Auto": [
        "Maruti","M&M","Hero MotoCorp","Bajaj Auto",
        "TVS Motor","Eicher Motors","Ashok Leyland",
        "Tata Motors","Bosch","MRF","Apollo Tyres",
    ],
    "💊 Pharma": [
        "Sun Pharma","Dr Reddy","Cipla","Divi's Lab",
        "Lupin","Apollo Hosp","Mankind Pharma",
        "Zydus Life","Torrent Pharma","Aurobindo",
    ],
    "🛒 FMCG": [
        "HUL","ITC","Nestle","Britannia","Dabur",
        "Tata Consumer","Marico","Colgate","Emami",
    ],
    "⚙️ Metals": [
        "Tata Steel","JSW Steel","Hindalco","Vedanta",
        "SAIL","NMDC","Jindal Steel","Hindustan Zinc",
    ],
    "🛡️ Defence": [
        "HAL","BEL","Mazagon Dock","RVNL","IRFC",
        "IRCTC","BEML","Cochin Shipyard","Garden Reach","BDL",
    ],
    "🏗️ Infra & Cement": [
        "L&T","UltraTech","DLF","Adani Ports",
        "Godrej Properties","ACC","Ambuja Cement",
        "Shree Cement","Prestige Estate",
    ],
    "⚡ Adani Group": [
        "Adani Ports","Adani Enterprises","Adani Green",
        "Adani Power","Adani Total Gas","Adani Energy",
    ],
    "🔬 Chemicals": [
        "Asian Paints","Pidilite","SRF","Navin Fluorine",
        "Deepak Nitrite","Atul Ltd",
    ],
    "🏭 Capital Goods": [
        "Siemens","ABB India","Havells","Thermax",
        "Cummins","Bharat Forge","Crompton",
        "Kaynes Tech","Syrma SGS","Triveni Turbine",
    ],
    "🌾 Agro & Fertilisers": [
        "UPL","PI Industries","Coromandel Intl",
        "Chambal Fert","Rallis India","Dhanuka Agritech",
    ],
    "🚚 Logistics": [
        "Container Corp","Delhivery","Blue Dart",
        "VRL Logistics","TCI Express","Gateway Distriparks",
    ],
    "🏨 Hotels & Retail": [
        "Indian Hotels","EIH Ltd","Lemon Tree",
        "Titan","Kalyan Jewellers","Senco Gold",
        "Vedant Fashions","DMart","Trent",
    ],
    "🏥 Healthcare": [
        "Apollo Hosp","Narayana Hrudayalaya","Fortis Healthcare",
        "Global Health","Aster DM","Max Healthcare",
        "Krishna Institute","Vijaya Diagnostic",
    ],
    "💹 Broking & Exchanges": [
        "BSE Ltd","CDSL","CAMS","Angel One","MCX",
        "KFintech","5Paisa Capital",
    ],
    "🧵 Textiles": [
        "Page Industries","Trident","Vardhman Textile",
        "KPR Mill","Welspun India","GHCL",
    ],
    "☀️ Renewable Energy": [
        "Adani Green","Adani Power","Tata Power",
        "NHPC","JSW Energy","KPI Green",
        "Waaree Energies","Premier Energies",
    ],
    "🏗️ Construction": [
        "NCC Ltd","KNR Constructions","IRB Infra",
        "Ashoka Buildcon","PSP Projects","RVNL",
        "PNC Infratech","L&T",
    ],
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
# TELEGRAM HELPER  (global — used by scanner + test button)
# ══════════════════════════════════════════════════════════
def send_telegram(token: str, chat_id: str,
                  message: str) -> bool:
    """Sends message via official Telegram Bot API."""
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id":    str(chat_id).strip(),
                "text":       message,
                "parse_mode": "HTML"
            },
            timeout=10
        )
        return resp.status_code == 200
    except Exception:
        return False

def tg_configured() -> bool:
    """True if token + chat_id are saved in session state."""
    return (
        bool(st.session_state.get("tg_token_saved", "")) and
        bool(st.session_state.get("tg_chat_saved",  ""))
    )

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

        # ── CPR (Central Pivot Range) ─────────────────────
        # Uses previous day candle for CPR calculation
        # If intraday tf: use previous completed session
        prev_h = float(h.iloc[-2])
        prev_l = float(l.iloc[-2])
        prev_c = float(c.iloc[-2])

        cpr_pivot = round((prev_h + prev_l + prev_c) / 3, 2)
        cpr_bc    = round((prev_h + prev_l) / 2, 2)
        cpr_tc    = round(cpr_pivot - cpr_bc + cpr_pivot, 2)

        # Make sure TC > BC
        if cpr_tc < cpr_bc:
            cpr_tc, cpr_bc = cpr_bc, cpr_tc

        cpr_width    = round(cpr_tc - cpr_bc, 2)
        cpr_width_pct= round(cpr_width / cpr_pivot * 100, 2)

        # CPR width classification
        # Narrow = trending day, Wide = sideways day
        if cpr_width_pct < 0.25:
            cpr_type = "Narrow (Trending day expected)"
        elif cpr_width_pct < 0.5:
            cpr_type = "Moderate"
        else:
            cpr_type = "Wide (Sideways day expected)"

        # Price position relative to CPR
        if cp > cpr_tc:
            cpr_position = "ABOVE"   # bullish
            cpr_bias     = "Bullish"
        elif cp < cpr_bc:
            cpr_position = "BELOW"   # bearish
            cpr_bias     = "Bearish"
        else:
            cpr_position = "INSIDE"  # sideways
            cpr_bias     = "Sideways"

        # Virgin CPR — price never touched yesterday's CPR
        # (strong magnet for price today)
        virgin_cpr = not (
            float(l.iloc[-1]) <= cpr_tc and
            float(h.iloc[-1]) >= cpr_bc
        )

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
            ("9b. CPR — Price above CPR",
             cpr_position == "ABOVE",
             f"{cpr_bias} | {cpr_type[:8]}",
             "Central Pivot Range bias must be bullish"),
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
            ("9b. CPR — Price above CPR",
             cpr_position == "ABOVE",
             f"{cpr_bias} | {cpr_type[:8]}",
             "Central Pivot Range bias must be bullish"),
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
            # CPR
            cpr_pivot=cpr_pivot, cpr_tc=cpr_tc, cpr_bc=cpr_bc,
            cpr_width=cpr_width, cpr_width_pct=cpr_width_pct,
            cpr_type=cpr_type, cpr_bias=cpr_bias,
            cpr_position=cpr_position, virgin_cpr=virgin_cpr,
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
     border-radius:12px;padding:12px 16px;
     margin-bottom:10px;
     box-shadow:0 4px 12px rgba(29,78,216,0.25)'>
  <div style='display:flex;justify-content:space-between;
       align-items:center;flex-wrap:wrap;gap:6px'>
    <span style='font-size:16px;font-weight:700;
                 color:#ffffff;letter-spacing:-0.3px'>
        🎯 Intraday &amp; Options Terminal
    </span>
    <span style='background:{"rgba(22,163,74,0.3)" if mopen else "rgba(220,38,38,0.3)"};
                 color:{"#86efac" if mopen else "#fca5a5"};
                 font-weight:700;padding:3px 10px;
                 border-radius:20px;font-size:12px'>
        {"🟢 OPEN" if mopen else "🔴 CLOSED"}
    </span>
    <span style='color:#93c5fd;font-weight:600;
                 font-size:12px'>{tmsg}</span>
    <span style='color:#bfdbfe;font-size:11px'>
        🕐 {n.strftime("%d %b  %H:%M IST")}
    </span>
  </div>
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
    ("🏦 Smart Money",  "smart"),
    ("🧮 P&L Calc",     "calc"),
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
T1,T2,T3,T4,T5,T6,T7 = st.tabs(TAB_NAMES)

# ── Reusable inline stock search widget ──────────────────
def inline_stock_search(tab_key: str):
    """
    Shows a compact search box + sector picker inline.
    Returns True if a new stock was selected.
    """
    changed = False
    sc1, sc2, sc3 = st.columns([3, 2, 1])
    with sc1:
        q = st.text_input(
            "Search stock",
            placeholder="Adani Ports, TCS, NIFTY...",
            key=f"inline_srch_{tab_key}",
            label_visibility="collapsed"
        )
        if q:
            hits = {k:v for k,v in STOCKS.items()
                    if q.strip().lower() in k.lower()
                    or q.strip().lower() in v.lower()}
            if hits:
                pick = st.selectbox(
                    "Results",
                    list(hits.keys()),
                    key=f"inline_pick_{tab_key}",
                    label_visibility="collapsed"
                )
                if st.button(
                    "Load",
                    key=f"inline_load_{tab_key}",
                    type="primary"
                ):
                    st.session_state["sn"] = pick
                    st.session_state["st"] = hits[pick]
                    changed = True
                    st.rerun()
            elif q:
                st.caption("No results found")
    with sc2:
        sector_opts = [""] + list(SECTORS.keys())
        chosen_sec  = st.selectbox(
            "Sector",
            sector_opts,
            key=f"inline_sec_{tab_key}",
            label_visibility="collapsed",
            format_func=lambda x: "Browse by sector..." if x=="" else x
        )
        if chosen_sec:
            sec_stocks = SECTORS[chosen_sec]
            chosen_st  = st.selectbox(
                "Stock",
                sec_stocks,
                key=f"inline_sec_st_{tab_key}",
                label_visibility="collapsed"
            )
            if st.button(
                "Load",
                key=f"inline_sec_load_{tab_key}",
                type="primary"
            ):
                st.session_state["sn"] = chosen_st
                st.session_state["st"] = STOCKS.get(
                    chosen_st, ""
                )
                changed = True
                st.rerun()
    with sc3:
        st.markdown(
            f"<div style='padding:6px 0;font-size:13px;"
            f"color:#64748b'>"
            f"📍 <b style='color:#1e293b'>"
            f"{st.session_state.get('sn','NIFTY 50')}"
            f"</b></div>",
            unsafe_allow_html=True
        )
    return changed

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
    # ── Inline stock search ───────────────────────────────
    st.markdown(
        "<div style='background:#f0f9ff;border:1px solid "
        "#bae6fd;border-radius:10px;padding:10px 14px;"
        "margin-bottom:12px'>"
        "<span style='font-size:13px;color:#0369a1;"
        "font-weight:600'>🔍 Search or browse stocks</span>"
        "</div>",
        unsafe_allow_html=True
    )
    inline_stock_search("t2")
    st.markdown("---")

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

    # CPR lines on chart
    if sig:
        # CPR TC (top)
        fig.add_hline(
            y=sig["cpr_tc"],
            line_dash="dot", line_color="#f59e0b",
            line_width=1.5, opacity=0.8,
            annotation_text=f"CPR TC {sig['cpr_tc']:,}",
            annotation_position="left",
            row=1, col=1
        )
        # CPR Pivot
        fig.add_hline(
            y=sig["cpr_pivot"],
            line_dash="solid", line_color="#f59e0b",
            line_width=2, opacity=0.9,
            annotation_text=f"Pivot {sig['cpr_pivot']:,}",
            annotation_position="left",
            row=1, col=1
        )
        # CPR BC (bottom)
        fig.add_hline(
            y=sig["cpr_bc"],
            line_dash="dot", line_color="#f59e0b",
            line_width=1.5, opacity=0.8,
            annotation_text=f"CPR BC {sig['cpr_bc']:,}",
            annotation_position="left",
            row=1, col=1
        )

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

    # CPR section
    st.markdown("#### 🔄 CPR — Central Pivot Range")
    cpr_col = (
        "#16a34a" if sig["cpr_position"] == "ABOVE"
        else "#dc2626" if sig["cpr_position"] == "BELOW"
        else "#f59e0b"
    )
    cpr_bg = (
        "#f0fdf4" if sig["cpr_position"] == "ABOVE"
        else "#fef2f2" if sig["cpr_position"] == "BELOW"
        else "#fffbeb"
    )
    cp1,cp2,cp3,cp4,cp5 = st.columns(5)
    cp1.metric("CPR Pivot",  f"₹{sig['cpr_pivot']:,}")
    cp2.metric("CPR Top (TC)",f"₹{sig['cpr_tc']:,}")
    cp3.metric("CPR Bot (BC)",f"₹{sig['cpr_bc']:,}")
    cp4.metric("CPR Width",  f"{sig['cpr_width_pct']:.2f}%")
    cp5.metric("Bias",       sig["cpr_bias"])

    st.markdown(
        f"<div style='background:{cpr_bg};"
        f"border:1.5px solid {cpr_col};"
        f"border-radius:10px;padding:12px 18px;"
        f"margin:8px 0;display:flex;gap:20px;"
        f"flex-wrap:wrap;align-items:center'>"
        f"<span style='font-size:16px;font-weight:700;"
        f"color:{cpr_col}'>Price is {sig['cpr_position']} CPR</span>"
        f"<span style='font-size:13px;color:#475569'>"
        f"{sig['cpr_type']}</span>"
        + ("<span style='background:#7c3aed;color:white;"
           "padding:3px 10px;border-radius:12px;"
           "font-size:12px'>✨ Virgin CPR</span>"
           if sig['virgin_cpr'] else "")
        + "</div>",
        unsafe_allow_html=True
    )

    with st.expander("📖 How to use CPR"):
        st.markdown("""
        **CPR (Central Pivot Range)** is calculated from
        yesterday's High, Low and Close.

        | Position | Meaning | Trade |
        |----------|---------|-------|
        | Price **above** TC | Strong bullish | Buy CE |
        | Price **inside** CPR | Sideways — avoid | Wait |
        | Price **below** BC | Strong bearish | Buy PE |

        **CPR Width tells you what kind of day to expect:**
        - **Narrow CPR (< 0.25%)** — Strong trending day.
          Big moves expected. Best for directional trades.
        - **Wide CPR (> 0.5%)** — Choppy sideways day.
          Avoid options. Wait for breakout.

        **Virgin CPR** — Price never touched yesterday's CPR.
        This acts as a strong magnet — price is likely to
        come back and test it today. High probability level.
        """)

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
        "Scans stocks simultaneously across sectors. "
        "Click Scan, review results, then manually send signals to Telegram."
    )

    # ── Scanner stock universe — uses same SECTORS as watchlist ──
    SCANNER_UNIVERSE = {
        k: v for k, v in SECTORS.items()
    }
    # Add a "All Sectors" option
    SCANNER_UNIVERSE = {
        "🌐 All Sectors (top 5 each)": [
            stock
            for sector_stocks in list(SECTORS.values())
            for stock in sector_stocks[:5]
        ],
        **SECTORS
    }

    # ── Telegram setup ────────────────────────────────────
    # Show configured badge if already set
    if tg_configured():
        st.success(
            "📱 Telegram configured ✅ — "
            "Use the Send buttons on each signal card to "
            "send alerts manually. Expand below to change settings."
        )
    st.markdown("#### 📱 Telegram Alert Setup")
    with st.expander(
        "Configure Telegram alerts — click to setup",
        expanded=not tg_configured()
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

        # ── Input fields (pre-filled if saved) ────────
        saved_token = st.session_state.get(
            "tg_token_saved", ""
        )
        saved_chat  = st.session_state.get(
            "tg_chat_saved", ""
        )

        if saved_token and saved_chat:
            st.success(
                "✅ Telegram credentials loaded from saved file. "
                "You don't need to re-enter them."
            )

        tg_token = st.text_input(
            "Step 1 — Bot Token",
            value=saved_token,
            placeholder="7123456789:AAHxxxxxxxxxxx",
            type="password",
            key="tg_token",
            help="Get this from @BotFather on Telegram"
        )
        tg_chat = st.text_input(
            "Step 2 — Chat ID",
            value=saved_chat,
            placeholder="987654321",
            key="tg_chat",
            help="Get this from @userinfobot on Telegram"
        )

        # ── Save credentials whenever typed ───────────
        if tg_token:
            st.session_state["tg_token_saved"] = tg_token
        if tg_chat:
            st.session_state["tg_chat_saved"] = tg_chat
        # Auto-save both if both are filled
        if tg_token and tg_chat:
            save_creds(tg_token, tg_chat)

        st.markdown("---")

        # ── Test button — always visible ───────────────
        st.markdown(
            "<div style='font-size:13px;color:#374151;"
            "margin-bottom:8px'>"
            "Step 3 — Click the button below to verify "
            "your connection:</div>",
            unsafe_allow_html=True
        )

        if st.button(
            "📲 Send Test Message to Telegram",
            key="tg_test",
            type="primary",
            use_container_width=True
        ):
            tok = (tg_token or
                   st.session_state.get("tg_token_saved",""))
            cid = (tg_chat or
                   st.session_state.get("tg_chat_saved",""))

            if not tok or not cid:
                st.error(
                    "❌ Please enter both Bot Token "
                    "and Chat ID above first."
                )
            else:
                with st.spinner("Sending test message..."):
                    ok = send_telegram(
                        tok, cid,
                        "🎯 <b>Trading Terminal Connected!</b>\n"
                        "You will now receive trade signals here.\n"
                        "Use Send buttons in the scanner to send signals manually."
                    )
                if ok:
                    st.session_state["tg_token_saved"] = tok
                    st.session_state["tg_chat_saved"]  = cid
                    # Save permanently to file
                    saved = save_creds(tok, cid)
                    st.success(
                        "✅ Test message sent! "
                        "Check your Telegram. "
                        "Credentials saved permanently — "
                        "you won't need to enter them again."
                        + (" 💾" if saved else "")
                    )
                    st.balloons()
                else:
                    # Try to get specific error
                    try:
                        resp = requests.post(
                            f"https://api.telegram.org/bot{tok}"
                            f"/sendMessage",
                            json={"chat_id": cid,
                                  "text": "test"},
                            timeout=10
                        )
                        err = resp.json().get(
                            "description", resp.text
                        )
                    except Exception as e:
                        err = str(e)
                    st.error(
                        f"❌ Failed: {err}\n\n"
                        "**Common fixes:**\n"
                        "1. Open Telegram → find your bot "
                        "→ press START\n"
                        "2. Chat ID must be a plain number "
                        "from @userinfobot\n"
                        "3. Token format: "
                        "`1234567890:AAHxxxxxxxx`"
                    )

        # Show current status
        if tg_configured():
            st.success(
                "✅ Telegram is configured and active. "
                "You will receive alerts when scanner "
                "finds score 8+ signals."
            )

    st.markdown("---")

    # ── Scanner controls ───────────────────────────────────
    sc1, sc2 = st.columns([2, 1])
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

    sl1, sl2 = st.columns(2)
    with sl1:
        min_score_scan = st.slider(
            "Minimum score to show",
            min_value=0,
            max_value=10,
            value=6,
            key="min_score_scan",
            help="Only show stocks with score above this"
        )
    with sl2:
        min_combined_scan = st.slider(
            "Minimum combined score",
            min_value=0,
            max_value=10,
            value=5,
            key="min_combined_scan",
            help=(
                "Combined = 60% technical + 40% historical. "
                "Higher = more consistent signal."
            )
        )

    alert_score = min_score_scan  # used for display only

    # Mobile-friendly controls - stack vertically on small screens
    run_scanner = st.button(
        "🚀 Scan All Stocks Now",
        type="primary",
        key="run_scanner",
        use_container_width=True
    )
    st.markdown(
        "<div style='margin:6px 0;padding:10px 14px;"
        "background:#f0f9ff;border:1px solid #bae6fd;"
        "border-radius:8px;font-size:13px;color:#0369a1'>"
        "🔄 <b>Auto scan every 5 min</b> — "
        "Toggle below to enable automatic scanning</div>",
        unsafe_allow_html=True
    )
    auto_scan = st.checkbox(
        "Enable auto scan every 5 minutes",
        value=False,
        key="auto_scan"
    )

    # send_telegram is now a global function

    # ── Run scanner ────────────────────────────────────────
    def get_option_levels(price: float, direction: str) -> dict:
        """Calculate ATM / ITM / OTM strike levels."""
        if price > 30000: step = 500
        elif price > 10000: step = 100
        elif price > 3000:  step = 50
        elif price > 500:   step = 20
        else:               step = 10

        atm = round(price / step) * step

        if direction == "UPTREND":
            return {
                "opt_type": "CE (Call)",
                "ITM": atm - step,
                "ATM": atm,
                "OTM": atm + step,
                "recommend": atm,
                "step": step,
            }
        else:
            return {
                "opt_type": "PE (Put)",
                "ITM": atm + step,
                "ATM": atm,
                "OTM": atm - step,
                "recommend": atm,
                "step": step,
            }

    def historical_consistency(sdf: pd.DataFrame,
                               direction: str) -> dict:
        """
        Validates signal across last 3, 5, 10 candles.
        Ensures results dont flip every 15 minutes.
        Returns a consistency score 0-10.
        """
        try:
            c   = sdf["Close"].squeeze().astype(float)
            v   = sdf["Volume"].squeeze().astype(float)
            e9  = ta.trend.ema_indicator(c, 9)
            e21 = ta.trend.ema_indicator(c, 21)
            rsi = ta.momentum.rsi(c, 14)
            vol_avg = v.rolling(20).mean()

            # 1. Last 3 candles price vs EMA9 and EMA21
            c3 = 0
            for i in range(1, 4):
                if direction == "UPTREND":
                    if (float(c.iloc[-i]) > float(e9.iloc[-i])
                            and float(e9.iloc[-i]) >
                            float(e21.iloc[-i])):
                        c3 += 1
                else:
                    if (float(c.iloc[-i]) < float(e9.iloc[-i])
                            and float(e9.iloc[-i]) <
                            float(e21.iloc[-i])):
                        c3 += 1

            # 2. Last 5 candles price vs EMA21
            c5 = 0
            for i in range(1, 6):
                if direction == "UPTREND":
                    if float(c.iloc[-i]) > float(e21.iloc[-i]):
                        c5 += 1
                else:
                    if float(c.iloc[-i]) < float(e21.iloc[-i]):
                        c5 += 1

            # 3. RSI in correct zone last 5 candles
            rsi_ok = 0
            for i in range(1, 6):
                rv = float(rsi.iloc[-i])
                if direction == "UPTREND" and 45 < rv < 78:
                    rsi_ok += 1
                elif direction == "DOWNTREND" and 22 < rv < 55:
                    rsi_ok += 1

            # 4. Volume above average last 3 candles
            vol_ok = sum(
                1 for i in range(1, 4)
                if float(v.iloc[-i]) > float(vol_avg.iloc[-i])
            )

            # Weighted consistency score
            score = round(
                (c3/3)*4 + (c5/5)*3 +
                (rsi_ok/5)*2 + (vol_ok/3)*1, 1
            )

            reliability = (
                "🔥 Very High" if score >= 8 else
                "✅ High"      if score >= 6 else
                "📈 Moderate"  if score >= 4 else
                "⚠️ Low — wait"
            )
            return {
                "score":       score,
                "reliability": reliability,
                "candles_3":   c3,
                "candles_5":   c5,
                "rsi_ok":      rsi_ok,
                "vol_ok":      vol_ok,
            }
        except:
            return {"score":0, "reliability":"—",
                    "candles_3":0, "candles_5":0,
                    "rsi_ok":0, "vol_ok":0}

    def run_scan_engine(stocks_to_scan, timeframe,
                        min_sc, alert_sc):
        results = []
        alerted = []
        prog    = st.progress(0, text="Starting scan...")
        total   = len(stocks_to_scan)

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

                slp = live_price(sym)
                sig = compute_all(sdf, slp)
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
                if direction == "MIXED":
                    continue

                action = ("BUY CE" if direction == "UPTREND"
                          else "BUY PE")

                # Historical consistency (key fix for
                # results changing every 15 min)
                hist = historical_consistency(sdf, direction)

                # Skip low consistency signals
                if hist["score"] < 3:
                    continue

                # Option levels
                opt = get_option_levels(sig["cp"], direction)

                # SL and targets
                sl_v = (sig["sl_long"]
                        if direction == "UPTREND"
                        else sig["sl_short"])
                t1_v = (sig["tgt1"]
                        if direction == "UPTREND"
                        else sig["tgt1s"])
                t2_v = (sig["tgt2"]
                        if direction == "UPTREND"
                        else sig["tgt2s"])
                t3_v = (sig["tgt3"]
                        if direction == "UPTREND"
                        else sig["tgt3s"])

                # Risk reward
                risk   = abs(sig["cp"] - sl_v)
                reward = abs(t1_v - sig["cp"])
                rr     = round(reward/(risk+0.001), 2)

                # Combined score
                combined = round(
                    best*0.6 + hist["score"]*0.4, 1
                )

                # Price change
                try:
                    chg = round(
                        ((sig["cp"] -
                          float(sdf["Close"].iloc[-2])) /
                         float(sdf["Close"].iloc[-2]))*100, 2
                    )
                except:
                    chg = 0

                result = {
                    "Stock":       sname,
                    "Sym":         sym,
                    "Score":       best,
                    "HistScore":   hist["score"],
                    "Combined":    combined,
                    "Reliability": hist["reliability"],
                    "Direction":   direction,
                    "Action":      action,
                    "Price":       sig["cp"],
                    "Change%":     chg,
                    "RSI":         sig["rv"],
                    "ADX":         sig["adxv"],
                    "VolSurge":    sig["vsurge"],
                    "Entry":       sig["e9v"],
                    "SL":          sl_v,
                    "T1":          t1_v,
                    "T2":          t2_v,
                    "T3":          t3_v,
                    "RR":          rr,
                    "OptType":     opt["opt_type"],
                    "ATM":         opt["ATM"],
                    "ITM":         opt["ITM"],
                    "OTM":         opt["OTM"],
                    "Consist3":    hist["candles_3"],
                    "Consist5":    hist["candles_5"],
                    # CPR
                    "CPR_Bias":    sig.get("cpr_bias","—"),
                    "CPR_Pos":     sig.get("cpr_position","—"),
                    "CPR_Type":    sig.get("cpr_type","—"),
                    "CPR_Pivot":   sig.get("cpr_pivot",0),
                    "Virgin_CPR":  sig.get("virgin_cpr",False),
                }
                results.append(result)

                # No auto alerts — user sends manually

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
        # Filter by combined score
        if "min_combined_scan" in st.session_state:
            results = [
                r for r in results
                if r.get("Combined", 0) >=
                st.session_state["min_combined_scan"]
            ]

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



            st.markdown(f"*Scanned at "
                        f"{now_ist().strftime('%H:%M:%S IST')} | "
                        f"Timeframe: {scan_tf}*")

            # ── Sort by combined score ─────────────────────
            results.sort(
                key=lambda x: x["Combined"], reverse=True
            )

            # ── Strong signals (score 8+) ──────────────────
            strong_r = [r for r in results
                        if r["Score"] >= 8]
            good_r   = [r for r in results
                        if 6 <= r["Score"] < 8]

            if strong_r:
                st.markdown("---")
                sh1, sh2 = st.columns([3, 1])
                with sh1:
                    st.markdown(
                        f"### 🔥 STRONG SIGNALS — "
                        f"{len(strong_r)} found"
                    )
                    st.caption(
                        "Confirmed by both technical score AND "
                        "historical consistency. Highest priority."
                    )
                with sh2:
                    if tg_configured():
                        if st.button(
                            "📱 Send All to Telegram",
                            key="scan_tg_all",
                            type="primary",
                            use_container_width=True
                        ):
                            tok = st.session_state.get(
                                "tg_token_saved", ""
                            )
                            cid = st.session_state.get(
                                "tg_chat_saved", ""
                            )
                            sent_count = 0
                            for r_all in strong_r:
                                msg_all = (
                                    f"<b>{r_all['Stock']}</b>"
                                    f" — {r_all['Action']}\n"
                                    f"Score: {r_all['Score']}/10"
                                    f" | Combined: "
                                    f"{r_all['Combined']}/10\n"
                                    f"Price: Rs "
                                    f"{r_all['Price']:,.2f}\n"
                                    f"Entry: Rs "
                                    f"{r_all['Entry']:,.2f} | "
                                    f"SL: Rs {r_all['SL']:,.2f}\n"
                                    f"T1: Rs {r_all['T1']:,.2f}"
                                    f" | T2: Rs "
                                    f"{r_all['T2']:,.2f}\n"
                                    f"ATM: {r_all['ATM']}"
                                )
                                if send_telegram(tok, cid,
                                                 msg_all):
                                    sent_count += 1
                            st.success(
                                f"✅ Sent {sent_count}/"
                                f"{len(strong_r)} signals!"
                            )
                    else:
                        st.caption(
                            "Setup Telegram above to enable"
                        )

                for r in strong_r:
                    dir_col  = ("#16a34a"
                                if r["Direction"]=="UPTREND"
                                else "#dc2626")
                    bg_light = ("#f0fdf4"
                                if r["Direction"]=="UPTREND"
                                else "#fef2f2")
                    chg_col  = ("#16a34a"
                                if r["Change%"] >= 0
                                else "#dc2626")
                    arr      = "▲" if r["Change%"]>=0 else "▼"

                    # Main card
                    st.markdown(
                        f"<div style='background:#ffffff;"
                        f"border:1.5px solid "
                        f"{'#86efac' if r['Direction']=='UPTREND' else '#fca5a5'};"
                        f"border-radius:12px;padding:16px 20px;"
                        f"margin-bottom:8px;"
                        f"box-shadow:0 2px 8px rgba(0,0,0,0.06)'>"

                        # Row 1: Name + Action + Reliability
                        f"<div style='display:flex;"
                        f"justify-content:space-between;"
                        f"align-items:center;flex-wrap:wrap;gap:6px'>"
                        f"<span style='font-size:18px;font-weight:700;"
                        f"color:#1e293b'>{r['Stock']}</span>"
                        f"<span style='background:{bg_light};"
                        f"color:{dir_col};padding:4px 14px;"
                        f"border-radius:20px;font-size:13px;"
                        f"font-weight:700'>{r['Action']}</span>"
                        f"<span style='font-size:12px;"
                        f"color:#64748b'>{r['Reliability']}</span>"
                        f"</div>"

                        # Row 2: Scores
                        f"<div style='margin:10px 0 8px;"
                        f"display:flex;gap:20px;flex-wrap:wrap'>"
                        f"<div><div style='font-size:10px;color:#94a3b8;"
                        f"text-transform:uppercase;letter-spacing:0.5px'>"
                        f"Signal Score</div>"
                        f"<div style='font-size:22px;font-weight:700;"
                        f"color:{dir_col}'>{r['Score']}/10</div></div>"
                        f"<div><div style='font-size:10px;color:#94a3b8;"
                        f"text-transform:uppercase;letter-spacing:0.5px'>"
                        f"Combined Score</div>"
                        f"<div style='font-size:22px;font-weight:700;"
                        f"color:#1e293b'>{r['Combined']}/10</div></div>"
                        f"<div><div style='font-size:10px;color:#94a3b8;"
                        f"text-transform:uppercase;letter-spacing:0.5px'>"
                        f"Consistency</div>"
                        f"<div style='font-size:14px;font-weight:600;"
                        f"color:#374151'>{r['Consist3']}/3 candles</div></div>"
                        f"<div><div style='font-size:10px;color:#94a3b8;"
                        f"text-transform:uppercase;letter-spacing:0.5px'>"
                        f"R:R Ratio</div>"
                        f"<div style='font-size:14px;font-weight:600;"
                        f"color:#374151'>{r['RR']}:1</div></div>"
                        f"</div>"

                        # Row 3: Price info
                        f"<div style='background:#f8fafc;"
                        f"border-radius:8px;padding:10px 14px;"
                        f"margin-bottom:10px;font-size:13px;"
                        f"color:#475569'>"
                        f"Price <b style='color:#1e293b'>"
                        f"₹{r['Price']:,.2f}</b>"
                        f"  <span style='color:{chg_col}'>"
                        f"{arr}{abs(r['Change%']):.2f}%</span>"
                        f"  &nbsp;|&nbsp;  RSI "
                        f"<b style='color:#1e293b'>{r['RSI']:.0f}</b>"
                        f"  &nbsp;|&nbsp;  ADX "
                        f"<b style='color:#1e293b'>{r['ADX']:.0f}</b>"
                        f"  &nbsp;|&nbsp;  Vol "
                        f"{'✅' if r['VolSurge'] else '❌'}"
                        f"</div>"

                        # Row 4: Entry SL Targets
                        f"<div style='display:grid;"
                        f"grid-template-columns:repeat(5,1fr);"
                        f"gap:8px;margin-bottom:10px'>"

                        f"<div style='background:#f0fdf4;"
                        f"border-radius:8px;padding:8px;text-align:center'>"
                        f"<div style='font-size:10px;color:#64748b;"
                        f"text-transform:uppercase'>Entry</div>"
                        f"<div style='font-size:13px;font-weight:700;"
                        f"color:#16a34a'>₹{r['Entry']:,.2f}</div>"
                        f"<div style='font-size:10px;color:#94a3b8'>"
                        f"EMA9 pullback</div></div>"

                        f"<div style='background:#fef2f2;"
                        f"border-radius:8px;padding:8px;text-align:center'>"
                        f"<div style='font-size:10px;color:#64748b;"
                        f"text-transform:uppercase'>Stop Loss</div>"
                        f"<div style='font-size:13px;font-weight:700;"
                        f"color:#dc2626'>₹{r['SL']:,.2f}</div>"
                        f"<div style='font-size:10px;color:#94a3b8'>"
                        f"ATR-based</div></div>"

                        f"<div style='background:#eff6ff;"
                        f"border-radius:8px;padding:8px;text-align:center'>"
                        f"<div style='font-size:10px;color:#64748b;"
                        f"text-transform:uppercase'>Target 1</div>"
                        f"<div style='font-size:13px;font-weight:700;"
                        f"color:#1d4ed8'>₹{r['T1']:,.2f}</div>"
                        f"<div style='font-size:10px;color:#94a3b8'>"
                        f"1.5× ATR</div></div>"

                        f"<div style='background:#eff6ff;"
                        f"border-radius:8px;padding:8px;text-align:center'>"
                        f"<div style='font-size:10px;color:#64748b;"
                        f"text-transform:uppercase'>Target 2</div>"
                        f"<div style='font-size:13px;font-weight:700;"
                        f"color:#1d4ed8'>₹{r['T2']:,.2f}</div>"
                        f"<div style='font-size:10px;color:#94a3b8'>"
                        f"2.5× ATR</div></div>"

                        f"<div style='background:#eff6ff;"
                        f"border-radius:8px;padding:8px;text-align:center'>"
                        f"<div style='font-size:10px;color:#64748b;"
                        f"text-transform:uppercase'>Target 3</div>"
                        f"<div style='font-size:13px;font-weight:700;"
                        f"color:#1d4ed8'>₹{r['T3']:,.2f}</div>"
                        f"<div style='font-size:10px;color:#94a3b8'>"
                        f"4× ATR</div></div>"

                        f"</div>"

                        # Row 4b: CPR info
                        f"<div style='background:"
                        f"{'#f0fdf4' if r.get('CPR_Pos')=='ABOVE' else '#fef2f2' if r.get('CPR_Pos')=='BELOW' else '#fffbeb'};"
                        f"border-radius:8px;padding:8px 14px;"
                        f"margin-bottom:8px;font-size:13px'>"
                        f"<b style='color:#7c3aed'>CPR</b> — "
                        f"Price is <b>{r.get('CPR_Pos','—')}</b> "
                        f"Central Pivot Range | "
                        f"Bias: <b>{r.get('CPR_Bias','—')}</b> | "
                        f"{r.get('CPR_Type','—')[:20]}"
                        f"{'  ✨ <b>Virgin CPR</b>' if r.get('Virgin_CPR') else ''}"
                        f"</div>"

                        # Row 5: Options ATM/ITM/OTM
                        f"<div style='background:#faf5ff;"
                        f"border-radius:8px;padding:10px 14px;"
                        f"margin-bottom:10px'>"
                        f"<div style='font-size:11px;color:#7c3aed;"
                        f"font-weight:700;margin-bottom:6px'>"
                        f"OPTIONS — {r['OptType']}</div>"
                        f"<div style='display:flex;gap:16px;"
                        f"flex-wrap:wrap;font-size:13px'>"
                        f"<span><b style='color:#16a34a'>✅ ATM {r['ATM']}</b>"
                        f" (Recommended)</span>"
                        f"<span style='color:#475569'>ITM {r['ITM']}"
                        f" (safer, costly)</span>"
                        f"<span style='color:#94a3b8'>OTM {r['OTM']}"
                        f" (risky, cheap)</span>"
                        f"</div></div>"

                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # Action buttons below card
                    btn_c1, btn_c2 = st.columns(2)

                    with btn_c1:
                        if st.button(
                            f"📊 Analyse {r['Stock']}",
                            key=f"scan_an_{r['Stock']}",
                            type="primary",
                            use_container_width=True
                        ):
                            st.session_state["sn"] = r["Stock"]
                            st.session_state["st"] = r["Sym"]
                            st.rerun()

                    with btn_c2:
                        if st.button(
                            f"📱 Send to Telegram",
                            key=f"scan_tg_{r['Stock']}",
                            use_container_width=True,
                            disabled=not tg_configured()
                        ):
                            tok = st.session_state.get(
                                "tg_token_saved", ""
                            )
                            cid = st.session_state.get(
                                "tg_chat_saved", ""
                            )
                            msg = (
                                f"<b>{r['Stock']}</b> — "
                                f"{r['Action']}\n"
                                f"Score: {r['Score']}/10 | "
                                f"Combined: {r['Combined']}/10\n"
                                f"Reliability: {r['Reliability']}\n"
                                f"Price: Rs {r['Price']:,.2f}\n"
                                f"Entry: Rs {r['Entry']:,.2f}\n"
                                f"SL: Rs {r['SL']:,.2f}\n"
                                f"T1: Rs {r['T1']:,.2f} | "
                                f"T2: Rs {r['T2']:,.2f}\n"
                                f"ATM: {r['ATM']} | "
                                f"ITM: {r['ITM']} | "
                                f"OTM: {r['OTM']}\n"
                                f"RSI: {r['RSI']:.0f} | "
                                f"R:R {r['RR']}:1"
                            )
                            if send_telegram(tok, cid, msg):
                                st.success(
                                    f"✅ Sent {r['Stock']} "
                                    f"signal to Telegram!"
                                )
                            else:
                                st.error(
                                    "❌ Failed. Check Telegram "
                                    "setup in scanner tab."
                                )

                    if not tg_configured():
                        st.caption(
                            "⚠️ Setup Telegram in the "
                            "section above to enable alerts"
                        )

                    st.markdown(
                        "<div style='margin:8px 0'></div>",
                        unsafe_allow_html=True
                    )

            # ── Good signals (6-7) ─────────────────────────
            if good_r:
                st.markdown("---")
                st.markdown(
                    f"### 📈 GOOD SIGNALS — {len(good_r)} found"
                )
                st.caption(
                    "Forming but not yet confirmed. "
                    "Watch these — enter when score reaches 8."
                )
                cols_g = 2
                for gi in range(0, len(good_r), cols_g):
                    chunk_g = good_r[gi:gi+cols_g]
                    gcols   = st.columns(cols_g)
                    for ci, r in enumerate(chunk_g):
                        dc = ("#16a34a"
                              if r["Direction"]=="UPTREND"
                              else "#dc2626")
                        with gcols[ci]:
                            st.markdown(
                                f"<div style='background:#ffffff;"
                                f"border:1px solid #e2e8f0;"
                                f"border-radius:10px;"
                                f"padding:14px;margin-bottom:6px'>"
                                f"<div style='display:flex;"
                                f"justify-content:space-between;"
                                f"align-items:center'>"
                                f"<b style='color:#1e293b;font-size:15px'>"
                                f"{r['Stock']}</b>"
                                f"<span style='color:{dc};"
                                f"font-weight:700;font-size:16px'>"
                                f"{r['Score']}/10</span></div>"
                                f"<div style='font-size:12px;"
                                f"color:#64748b;margin-top:6px'>"
                                f"{r['Action']} | ₹{r['Price']:,.2f} | "
                                f"RSI {r['RSI']:.0f} | "
                                f"R:R {r['RR']}:1</div>"
                                f"<div style='font-size:11px;"
                                f"color:#94a3b8;margin-top:4px'>"
                                f"Entry ₹{r['Entry']:,.2f} | "
                                f"SL ₹{r['SL']:,.2f}</div>"
                                f"<div style='font-size:11px;"
                                f"color:#7c3aed;margin-top:4px'>"
                                f"{r['OptType']} ATM {r['ATM']}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            if st.button(
                                "View",
                                key=f"scan_vw_{gi}_{ci}",
                                use_container_width=True
                            ):
                                st.session_state["sn"] = r["Stock"]
                                st.session_state["st"] = r["Sym"]
                                st.rerun()

            # ── Score chart ────────────────────────────────
            if results:
                st.markdown("---")
                st.markdown("### Score Overview")
                import plotly.graph_objects as go_scan
                results_sorted = sorted(
                    results,
                    key=lambda x: x["Combined"],
                    reverse=True
                )
                fig_scan = go_scan.Figure(go_scan.Bar(
                    x=[r["Stock"] for r in results_sorted],
                    y=[r["Combined"] for r in results_sorted],
                    marker_color=[
                        "#16a34a" if r["Direction"]=="UPTREND"
                        else "#dc2626"
                        for r in results_sorted
                    ],
                    text=[
                        f"{r['Combined']}/10"
                        for r in results_sorted
                    ],
                    textposition="outside",
                ))
                fig_scan.add_hline(
                    y=8, line_dash="dash",
                    line_color="#16a34a",
                    annotation_text="Strong zone (8+)"
                )
                fig_scan.add_hline(
                    y=6, line_dash="dot",
                    line_color="#f59e0b",
                    annotation_text="Good zone (6+)"
                )
                fig_scan.update_layout(
                    template="plotly_white",
                    height=320,
                    yaxis_range=[0, 11],
                    margin=dict(l=10,r=10,t=20,b=80),
                    xaxis_tickangle=-35,
                    showlegend=False,
                    title="Combined Score (Technical 60% + Historical 40%)"
                )
                st.plotly_chart(
                    fig_scan, use_container_width=True
                )

                # Full table
                with st.expander("Full results table"):
                    df_scan = pd.DataFrame([{
                        "Stock":       r["Stock"],
                        "Score":       r["Score"],
                        "Combined":    r["Combined"],
                        "Reliability": r["Reliability"],
                        "Action":      r["Action"],
                        "Price":       f"₹{r['Price']:,.2f}",
                        "Entry":       f"₹{r['Entry']:,.2f}",
                        "SL":          f"₹{r['SL']:,.2f}",
                        "T1":          f"₹{r['T1']:,.2f}",
                        "T2":          f"₹{r['T2']:,.2f}",
                        "R:R":         r["RR"],
                        "ATM Strike":  r["ATM"],
                        "ITM Strike":  r["ITM"],
                        "OTM Strike":  r["OTM"],
                        "RSI":         round(r["RSI"],1),
                        "ADX":         round(r["ADX"],1),
                        "Vol":  "✅" if r["VolSurge"] else "❌",
                    } for r in results_sorted])
                    st.dataframe(
                        df_scan,
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
        "Trained on historical candle data using "
        "Random Forest + Gradient Boosting. "
        "Predicts next 3-candle direction."
    )

    # ── Inline stock search ───────────────────────────────
    st.markdown(
        "<div style='background:#f0f9ff;border:1px solid "
        "#bae6fd;border-radius:10px;padding:10px 14px;"
        "margin-bottom:12px'>"
        "<span style='font-size:13px;color:#0369a1;"
        "font-weight:600'>🔍 Search or browse stocks</span>"
        "</div>",
        unsafe_allow_html=True
    )
    inline_stock_search("t4")
    st.markdown("---")

    # ── Stock info bar ─────────────────────────────────
    ml_lp = live_price(stick)
    pc_ml = "#16a34a" if ml_lp["chg"] >= 0 else "#dc2626"
    arr_ml= "▲" if ml_lp["chg"] >= 0 else "▼"

    if ml_lp["ok"]:
        st.markdown(
            f"<div style='background:#1e3a5f;"
            f"border-radius:10px;padding:12px 18px;"
            f"display:flex;justify-content:space-between;"
            f"align-items:center;margin-bottom:12px'>"
            f"<span style='color:#fff;font-size:16px;"
            f"font-weight:700'>{sname}</span>"
            f"<span style='color:#fff;font-size:18px;"
            f"font-weight:700'>₹{ml_lp['p']:,.2f}</span>"
            f"<span style='color:{pc_ml};font-weight:600'>"
            f"{arr_ml}{abs(ml_lp['chg']):.2f}%</span>"
            f"<span style='color:#93c5fd;font-size:12px'>"
            f"{stick}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── Controls ───────────────────────────────────────
    ctl1, ctl2, ctl3 = st.columns([2, 2, 2])

    with ctl1:
        ml_tf_opt = st.selectbox(
            "Timeframe",
            ["15m", "30m", "1h", "1d"],
            index=3,
            key="ml_tf_select",
            help="1d gives most candles = better accuracy"
        )
    with ctl2:
        run_ml = st.button(
            "🚀 Train & Predict",
            type="primary",
            key="run_ml",
            use_container_width=True
        )
    with ctl3:
        # Clear old results when stock changes
        if st.button(
            "🔄 Reset",
            key="ml_reset",
            use_container_width=True,
            help="Clear old results and start fresh"
        ):
            for k in ["ml_result","ml_model_data",
                      "ml_stock","ml_tf"]:
                st.session_state.pop(k, None)
            st.rerun()

    # Clear cached results if stock or timeframe changed
    prev_stock = st.session_state.get("ml_stock", "")
    prev_tf    = st.session_state.get("ml_tf", "")
    if prev_stock != sname or prev_tf != ml_tf_opt:
        for k in ["ml_result", "ml_model_data"]:
            st.session_state.pop(k, None)
        st.session_state["ml_stock"] = sname
        st.session_state["ml_tf"]    = ml_tf_opt

    # ── Load data and run ──────────────────────────────
    if run_ml:
        with st.spinner(
            f"Loading {sname} candle data "
            f"({ml_tf_opt} timeframe)..."
        ):
            ml_df = candles(stick, ml_tf_opt)

        if ml_df.empty or len(ml_df) < 100:
            st.error(
                f"Only {len(ml_df)} candles available for "
                f"{sname} on {ml_tf_opt}. "
                f"Need at least 100. "
                f"Switch to **1d** timeframe."
            )
        else:
            st.caption(
                f"✅ {len(ml_df)} candles loaded | "
                f"Last: "
                f"{ml_df.index[-1].strftime('%d %b %H:%M')} "
                f"IST | Timeframe: {ml_tf_opt}"
            )

            # Real-time approximation
            st.markdown("---")
            st.markdown("#### ⚡ Real-Time Approximation")
            st.caption(
                "Live price injected into indicators "
                "to bridge the 15-minute delay."
            )

            rt = approximate_realtime(
                ml_df, ml_lp["p"] if ml_lp["ok"] else 0
            )

            if rt.get("ok"):
                ra, rb, rc_ = st.columns(3)
                bc = rt["bias_color"]

                with ra:
                    bias_bg = (
                        "#f0fdf4" if rt["live_bias"]=="BULLISH"
                        else "#fef2f2"
                        if rt["live_bias"]=="BEARISH"
                        else "#fffbeb"
                    )
                    st.markdown(
                        f"<div style='background:{bias_bg};"
                        f"border-radius:10px;padding:16px;"
                        f"text-align:center'>"
                        f"<div style='font-size:11px;"
                        f"color:#64748b;text-transform:uppercase;"
                        f"letter-spacing:1px'>Live Bias</div>"
                        f"<div style='font-size:28px;"
                        f"font-weight:700;color:{bc}'>"
                        f"{rt['live_bias']}</div>"
                        f"<div style='font-size:12px;"
                        f"color:#64748b;margin-top:4px'>"
                        f"{rt['bull_count']} bull / "
                        f"{rt['bear_count']} bear</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with rb:
                    sc_col = (
                        "#16a34a"
                        if rt["since_close"] >= 0
                        else "#dc2626"
                    )
                    st.markdown(
                        f"<div style='background:#f8fafc;"
                        f"border-radius:10px;padding:16px;"
                        f"text-align:center'>"
                        f"<div style='font-size:11px;"
                        f"color:#64748b;text-transform:uppercase;"
                        f"letter-spacing:1px'>"
                        f"Live vs Last Candle</div>"
                        f"<div style='font-size:28px;"
                        f"font-weight:700;color:{sc_col}'>"
                        f"{rt['since_close']:+.3f}%</div>"
                        f"<div style='font-size:12px;"
                        f"color:#64748b;margin-top:4px'>"
                        f"Candle pos: {rt['candle_pos']:.0f}% "
                        f"({rt['candle_zone']})<br>"
                        f"Micro trend: "
                        f"<b>{rt['micro_trend']}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with rc_:
                    st.markdown(
                        f"<div style='background:#f8fafc;"
                        f"border-radius:10px;padding:16px'>"
                        f"<div style='font-size:11px;"
                        f"color:#64748b;text-transform:uppercase;"
                        f"letter-spacing:1px;margin-bottom:8px'>"
                        f"Live Indicators</div>"
                        f"<table style='width:100%;"
                        f"font-size:13px'>"
                        f"<tr><td style='color:#64748b'>"
                        f"RSI</td><td style='text-align:right;"
                        f"font-weight:600;color:#1e293b'>"
                        f"{rt['rsi_live']}</td></tr>"
                        f"<tr><td style='color:#64748b'>"
                        f"EMA9</td><td style='text-align:right;"
                        f"font-weight:600;color:#ca8a04'>"
                        f"₹{rt['ema9_live']:,}</td></tr>"
                        f"<tr><td style='color:#64748b'>"
                        f"EMA21</td><td style='text-align:right;"
                        f"font-weight:600;color:#ea580c'>"
                        f"₹{rt['ema21_live']:,}</td></tr>"
                        f"<tr><td style='color:#64748b'>"
                        f"VWAP</td><td style='text-align:right;"
                        f"font-weight:600;color:#1e293b'>"
                        f"₹{rt['vwap_live']:,}</td></tr>"
                        f"<tr><td style='color:#64748b'>"
                        f"VWAP Dev</td>"
                        f"<td style='text-align:right;"
                        f"font-weight:600;"
                        f"color:{'#16a34a' if rt['vwap_dev']>0 else '#dc2626'}'>"
                        f"{rt['vwap_dev']:+.2f}%</td></tr>"
                        f"</table></div>",
                        unsafe_allow_html=True
                    )

                # Live signals
                st.markdown("#### 📡 Live Signal Breakdown")
                for sig_txt in rt["live_signals"]:
                    c_ = ("#f0fdf4" if "✅" in sig_txt
                          else "#fef2f2" if "❌" in sig_txt
                          else "#fffbeb")
                    st.markdown(
                        f"<div style='background:{c_};"
                        f"border-radius:6px;padding:8px 14px;"
                        f"margin:3px 0;font-size:13px;"
                        f"color:#374151'>{sig_txt}</div>",
                        unsafe_allow_html=True
                    )

            # ML training
            st.markdown("---")
            st.markdown("#### 🧠 ML Prediction")

            with st.spinner(
                "Training Random Forest + "
                "Gradient Boosting model..."
            ):
                model_data = train_model(ml_df)

            if not model_data.get("ok"):
                st.error(
                    f"Training failed: "
                    f"{model_data.get('reason','Unknown error')}"
                )
            else:
                pred = predict_next_move(ml_df, model_data)
                st.session_state["ml_result"]     = pred
                st.session_state["ml_model_data"] = model_data

                if pred and pred.get("ok"):
                    pc_   = pred["sig_color"]
                    conf  = pred["confidence"]

                    # Prediction card
                    pred_bg = (
                        "#f0fdf4" if pred["prediction"]=="UPTREND"
                        else "#fef2f2"
                        if pred["prediction"]=="DOWNTREND"
                        else "#fffbeb"
                    )
                    st.markdown(
                        f"<div style='background:{pred_bg};"
                        f"border:2px solid {pc_};"
                        f"border-radius:14px;padding:24px;"
                        f"text-align:center;margin:12px 0'>"
                        f"<div style='font-size:12px;"
                        f"color:#64748b;letter-spacing:2px;"
                        f"text-transform:uppercase'>"
                        f"ML PREDICTION — NEXT 3 CANDLES</div>"
                        f"<div style='font-size:52px;"
                        f"font-weight:700;color:{pc_};"
                        f"line-height:1.1;margin:8px 0'>"
                        f"{pred['prediction']}</div>"
                        f"<div style='font-size:26px;"
                        f"font-weight:600;color:{pc_}'>"
                        f"{pred['signal']}</div>"
                        f"<div style='font-size:15px;"
                        f"color:#64748b;margin-top:8px'>"
                        f"Confidence: {pred['reliability']} "
                        f"({conf}%)</div>"
                        f"<div style='margin-top:12px;"
                        f"display:flex;justify-content:center;"
                        f"gap:12px;flex-wrap:wrap'>"
                        f"<span style='background:#dcfce7;"
                        f"color:#16a34a;padding:4px 14px;"
                        f"border-radius:12px;font-size:13px'>"
                        f"📈 Uptrend: "
                        f"{pred['probabilities'].get('UPTREND',0):.1f}%"
                        f"</span>"
                        f"<span style='background:#fee2e2;"
                        f"color:#dc2626;padding:4px 14px;"
                        f"border-radius:12px;font-size:13px'>"
                        f"📉 Downtrend: "
                        f"{pred['probabilities'].get('DOWNTREND',0):.1f}%"
                        f"</span>"
                        f"<span style='background:#fef9c3;"
                        f"color:#854d0e;padding:4px 14px;"
                        f"border-radius:12px;font-size:13px'>"
                        f"➡️ Sideways: "
                        f"{pred['probabilities'].get('SIDEWAYS',0):.1f}%"
                        f"</span>"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )

                    # Stats
                    ms1, ms2, ms3 = st.columns(3)
                    ms1.metric(
                        "Model Accuracy",
                        f"{pred['model_accuracy']}%"
                        if pred["model_accuracy"] else "N/A"
                    )
                    ms2.metric(
                        "Trained on",
                        f"{pred['n_trained']} candles"
                    )
                    ms3.metric("Timeframe", ml_tf_opt)

                    # Probability chart
                    st.markdown("#### 📊 Probability Distribution")
                    probs = pred["probabilities"]
                    import plotly.graph_objects as go_ml
                    fig_prob = go_ml.Figure(go_ml.Bar(
                        x=list(probs.keys()),
                        y=list(probs.values()),
                        marker_color=[
                            "#16a34a" if k=="UPTREND"
                            else "#dc2626" if k=="DOWNTREND"
                            else "#f59e0b"
                            for k in probs.keys()
                        ],
                        text=[f"{v:.1f}%" for v in probs.values()],
                        textposition="outside"
                    ))
                    fig_prob.update_layout(
                        template="plotly_white",
                        height=260,
                        yaxis_range=[0, 100],
                        yaxis_title="Probability %",
                        margin=dict(l=10,r=10,t=10,b=10)
                    )
                    st.plotly_chart(
                        fig_prob, use_container_width=True
                    )

                    # Top features
                    st.markdown("#### 🔬 Top Features Used")
                    for feat in pred["top_contrib"]:
                        imp = feat["importance"]
                        fc  = ("#16a34a" if imp >= 10
                               else "#f59e0b" if imp >= 5
                               else "#94a3b8")
                        st.markdown(
                            f"<div style='background:#f8fafc;"
                            f"border-radius:6px;padding:10px 14px;"
                            f"margin:4px 0'>"
                            f"<div style='display:flex;"
                            f"justify-content:space-between'>"
                            f"<span style='color:#475569;"
                            f"font-size:13px'>{feat['feature']}"
                            f"</span>"
                            f"<span style='color:{fc};"
                            f"font-size:13px;font-weight:600'>"
                            f"{imp:.1f}% importance</span></div>"
                            f"<div style='background:#e2e8f0;"
                            f"height:4px;border-radius:2px;"
                            f"margin-top:6px'>"
                            f"<div style='background:{fc};"
                            f"width:{min(imp*5,100):.0f}%;"
                            f"height:4px;border-radius:2px'>"
                            f"</div></div></div>",
                            unsafe_allow_html=True
                        )

                    # Combined signal
                    st.markdown("---")
                    st.markdown("### 🎯 Combined Signal")

                    ml_dir   = pred["prediction"]
                    tech_sig = compute_all(ml_df, ml_lp)
                    tech_dir = (tech_sig["direction"]
                                if tech_sig else "SIDEWAYS")
                    rt_bias  = (rt.get("live_bias","NEUTRAL")
                                if rt and rt.get("ok")
                                else "NEUTRAL")

                    all_bull = (
                        ml_dir   == "UPTREND" and
                        tech_dir == "UPTREND" and
                        rt_bias  == "BULLISH"
                    )
                    all_bear = (
                        ml_dir   == "DOWNTREND" and
                        tech_dir == "DOWNTREND" and
                        rt_bias  == "BEARISH"
                    )

                    if all_bull and tech_sig:
                        st.success(
                            f"🔥 ALL THREE AGREE — BULLISH\n\n"
                            f"ML: UPTREND ({conf}%) ✅ | "
                            f"Technical: {tech_sig['up_score']}/10 ✅ | "
                            f"Live: BULLISH ✅\n\n"
                            f"Highest quality CE setup. "
                            f"Enter on pullback to "
                            f"EMA9 (₹{tech_sig['e9v']:,}) "
                            f"with SL ₹{tech_sig['sl_long']:,}"
                        )
                    elif all_bear and tech_sig:
                        st.error(
                            f"🔥 ALL THREE AGREE — BEARISH\n\n"
                            f"ML: DOWNTREND ({conf}%) ✅ | "
                            f"Technical: {tech_sig['dn_score']}/10 ✅ | "
                            f"Live: BEARISH ✅\n\n"
                            f"Highest quality PE setup. "
                            f"Enter on bounce to "
                            f"EMA9 (₹{tech_sig['e9v']:,}) "
                            f"with SL ₹{tech_sig['sl_short']:,}"
                        )
                    elif ml_dir == tech_dir and ml_dir != "SIDEWAYS":
                        col_ = ("#16a34a"
                                if ml_dir == "UPTREND"
                                else "#dc2626")
                        st.markdown(
                            f"<div style='background:#f8fafc;"
                            f"border:1px solid #e2e8f0;"
                            f"border-radius:10px;"
                            f"padding:14px 18px'>"
                            f"<b style='color:{col_}'>"
                            f"⚡ ML + Technical agree: {ml_dir}"
                            f"</b><br>"
                            f"<span style='color:#64748b;"
                            f"font-size:13px'>"
                            f"Live bias is {rt_bias}. "
                            f"Wait for it to confirm "
                            f"before entering.</span></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning(
                            f"⚠️ Mixed signals — "
                            f"ML says {ml_dir} but "
                            f"Technical says {tech_dir}. "
                            "Do not trade when signals conflict."
                        )

    elif "ml_result" in st.session_state:
        # Show cached result with note
        st.info(
            f"Showing previous result for "
            f"{st.session_state.get('ml_stock', sname)}. "
            f"Click **Train & Predict** to get fresh prediction "
            f"for {sname}."
        )
        pred = st.session_state["ml_result"]
        if pred and pred.get("ok"):
            pc_ = pred["sig_color"]
            st.markdown(
                f"<div style='background:#f8fafc;"
                f"border:2px solid {pc_};"
                f"border-radius:12px;padding:20px;"
                f"text-align:center'>"
                f"<div style='font-size:42px;font-weight:700;"
                f"color:{pc_}'>{pred['prediction']}</div>"
                f"<div style='font-size:20px;color:{pc_}'>"
                f"{pred['signal']}</div>"
                f"<div style='font-size:14px;color:#64748b;"
                f"margin-top:6px'>"
                f"Confidence: {pred['confidence']}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info(
            f"Stock selected: **{sname}**  \n\n"
            f"Click **🚀 Train & Predict** above to run "
            f"the ML model and get a prediction."
        )

        with st.expander("📖 How ML prediction works"):
            st.markdown("""
            The model trains on YOUR stock's own historical
            data — not generic market data. It learns the
            specific patterns of that stock by analyzing
            30+ technical features.

            | Level | Condition | Action |
            |-------|-----------|--------|
            | **Best** | ML + Technical + Live agree | Enter trade |
            | **Good** | ML + Technical agree | Wait for live |
            | **Avoid** | ML and Technical disagree | No trade |

            **Use 1d timeframe** for best accuracy —
            more candles = better trained model.
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
# ── Auto refresh ──────────────────────────────────────────
if auto_rf:
    st.sidebar.success("🔄 Refreshing every 2 min...")
    time.sleep(120)
    st.rerun()