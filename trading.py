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

# ── Zerodha Kite Connect ──────────────────────────────────
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

# ── Load API credentials ──────────────────────────────────
# Works both locally (.env file) and on Streamlit Cloud (secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import os

def _get_secret(key: str, default: str = "") -> str:
    """Load from Streamlit secrets first, then .env, then env vars."""
    try:
        # Streamlit Cloud secrets (production)
        return st.secrets[key]
    except:
        pass
    # Local .env or environment variable
    return os.getenv(key, default)

KITE_API_KEY    = _get_secret("KITE_API_KEY")
KITE_API_SECRET = _get_secret("KITE_API_SECRET")


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
    """
    Load saved credentials.
    Priority: Streamlit secrets > local JSON file > nothing
    """
    # Load Telegram from Streamlit secrets (cloud deployment)
    try:
        if ("tg_token_saved" not in st.session_state
                and st.secrets.get("tg_token")):
            st.session_state["tg_token_saved"] = (
                st.secrets["tg_token"]
            )
        if ("tg_chat_saved" not in st.session_state
                and st.secrets.get("tg_chat")):
            st.session_state["tg_chat_saved"] = (
                st.secrets["tg_chat"]
            )
    except:
        pass

    # Load from local JSON file (local use)
    try:
        if os.path.exists(CREDS_FILE):
            with open(CREDS_FILE, "r") as f:
                data = json.load(f)
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

# ── Kite session management ───────────────────────────────
def get_kite() -> "KiteConnect | None":
    """Returns authenticated KiteConnect instance or None."""
    if not KITE_AVAILABLE or not KITE_API_KEY:
        return None
    token = st.session_state.get("kite_access_token", "")
    if not token:
        # Try loading from saved file
        try:
            kt_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                ".kite_token.json"
            )
            if os.path.exists(kt_file):
                import json as _json
                kt_data = _json.load(open(kt_file))
                # Check if token is from today
                from datetime import date as _date
                if kt_data.get("date") == str(_date.today()):
                    token = kt_data.get("token","")
                    if token:
                        st.session_state["kite_access_token"] = token
        except:
            pass
    if not token:
        return None
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(token)
        return kite
    except:
        return None

def kite_is_connected() -> bool:
    """True if Kite session is active today."""
    return bool(st.session_state.get("kite_access_token",""))

def save_kite_token(token: str):
    """Save access token to file for today."""
    try:
        import json as _json
        from datetime import date as _date
        kt_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".kite_token.json"
        )
        _json.dump({"token": token,
                    "date": str(_date.today())},
                   open(kt_file, "w"))
    except:
        pass

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
    "pulse":     7,
    "options":   8,
    "backtest":  9,
}
TAB_NAMES = [
    "📋 Watchlist",
    "🎯 Trade Setup",
    "🔍 Auto Scanner",
    "🤖 ML Prediction",
    "🏦 Smart Money",
    "🧮 P&L Calculator",
    "📰 News & Events",
    "📊 Market Pulse",
    "🔗 Options Chain",
    "🧪 Backtest",
]
TAB_ICONS = ["📋","🎯","🔍","🤖","🏦","🧮","📰","📊","🔗","🧪"]
TAB_KEYS  = list(TAB_ROUTES.keys())

# Read current tab from URL
_qp = st.query_params
_tab_key = _qp.get("tab", "watchlist")
_default_tab = TAB_ROUTES.get(_tab_key, 0)

# ══════════════════════════════════════════════════════════
# MARKET DATA HELPERS
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_india_vix() -> dict:
    """Get India VIX from Yahoo Finance."""
    try:
        vix_tk = yf.Ticker("^INDIAVIX")
        fi = vix_tk.fast_info
        p  = float(fi.last_price)
        pc = float(fi.previous_close) if fi.previous_close else p
        ch = round(((p-pc)/pc)*100, 2) if pc else 0
        level = (
            "🔴 EXTREME FEAR" if p > 25 else
            "🟠 HIGH FEAR"    if p > 20 else
            "🟡 ELEVATED"     if p > 15 else
            "🟢 CALM"         if p > 10 else
            "🟢 VERY CALM"
        )
        advice = (
            "Avoid buying options — premiums very expensive"
            if p > 20 else
            "Good time to buy CE/PE — premiums reasonable"
            if p < 15 else
            "Moderate conditions — trade carefully"
        )
        return {
            "ok": True, "vix": round(p, 2),
            "prev": round(pc, 2), "chg": ch,
            "level": level, "advice": advice
        }
    except Exception as e:
        return {"ok": False, "vix": 0, "level": "Unknown",
                "advice": "VIX data unavailable", "chg": 0}

@st.cache_data(ttl=600)
def get_fii_dii() -> dict:
    """Get FII/DII data from NSE website."""
    try:
        session_ = requests.Session()
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/"
        }
        session_.get("https://www.nseindia.com", headers=hdrs, timeout=8)
        r = session_.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=hdrs, timeout=8
        )
        if r.status_code == 200:
            data = r.json()
            if data:
                latest = data[0]
                fii_buy  = float(latest.get("fiiBuy", 0) or 0)
                fii_sell = float(latest.get("fiiSell", 0) or 0)
                dii_buy  = float(latest.get("diiBuy", 0) or 0)
                dii_sell = float(latest.get("diiSell", 0) or 0)
                fii_net  = round(fii_buy - fii_sell, 2)
                dii_net  = round(dii_buy - dii_sell, 2)
                return {
                    "ok": True,
                    "date":     latest.get("date", ""),
                    "fii_buy":  fii_buy,
                    "fii_sell": fii_sell,
                    "fii_net":  fii_net,
                    "dii_buy":  dii_buy,
                    "dii_sell": dii_sell,
                    "dii_net":  dii_net,
                    "data":     data[:10]
                }
        return {"ok": False}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@st.cache_data(ttl=180)
def get_options_chain(symbol: str = "NIFTY") -> dict:
    """Get options chain from NSE."""
    try:
        session_ = requests.Session()
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/"
        }
        session_.get("https://www.nseindia.com", headers=hdrs, timeout=8)
        url = (f"https://www.nseindia.com/api/option-chain-indices"
               f"?symbol={symbol}")
        r = session_.get(url, headers=hdrs, timeout=8)
        if r.status_code == 200:
            data = r.json()
            records = data.get("records", {})
            exp_dates = records.get("expiryDates", [])
            spot = float(records.get("underlyingValue", 0))
            oc_data = records.get("data", [])
            return {
                "ok": True,
                "spot": spot,
                "expiries": exp_dates,
                "data": oc_data,
                "timestamp": records.get("timestamp", "")
            }
        return {"ok": False}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@st.cache_data(ttl=3600)
def get_nifty_history_for_backtest(months: int = 6) -> pd.DataFrame:
    """Get historical data for backtesting."""
    try:
        end   = datetime.now()
        start = end - timedelta(days=months*30)
        df = yf.download(
            "^NSEI", start=start, end=end,
            interval="1d", progress=False,
            auto_adjust=True
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()

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

/* ── Layout ───────────────────────────────────────── */
.block-container {
    padding-top: 0.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 1400px !important;
}

/* Streamlit header — transparent, no height, no click blocking */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
    pointer-events: none !important;
}

/* Toolbar — allow clicks through */
[data-testid="stToolbar"] {
    pointer-events: all !important;
    z-index: 999 !important;
}

/* Main content area */
.main .block-container {
    padding-top: 0.5rem !important;
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
/* Hide sidebar collapse/expand arrow that corrupts text */
[data-testid="stSidebarCollapsedControl"] { display:none !important; }
button[data-testid="collapsedControl"] { display:none !important; }
section[data-testid="stSidebar"] > div > div:first-child {
    padding-top: 2px !important;
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
/* Fix label text corruption in sidebar */
section[data-testid="stSidebar"] label {
    font-size: 12px !important;
    color: #374151 !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stCheckbox label {
    display: none !important;
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
/* ── Sidebar ALL arrow/corruption fixes ────────────── */
/* Hide collapse arrow buttons */
[data-testid="stSidebarCollapsedControl"] { display:none !important; }
[data-testid="stSidebarCollapseButton"]   { display:none !important; }
button[data-testid="collapsedControl"]    { display:none !important; }

/* Hide expander arrow icon in sidebar — this was causing _arrow_right */
section[data-testid="stSidebar"] details summary {
    list-style: none !important;
}
section[data-testid="stSidebar"] details summary::-webkit-details-marker {
    display: none !important;
}
section[data-testid="stSidebar"] .streamlit-expanderHeader svg {
    display: none !important;
}

/* Hide text input labels in sidebar */
section[data-testid="stSidebar"] .stTextInput label { 
    display: none !important; 
}

/* Hide selectbox labels in sidebar */
section[data-testid="stSidebar"] .stSelectbox label { 
    display: none !important; 
}

/* Remove top padding */
section[data-testid="stSidebar"] > div:first-child { 
    padding-top: 0 !important; 
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

    # ══════════════════════════════════════════════════════
    # COMMODITIES — MCX (Multi Commodity Exchange)
    # Yahoo Finance tickers for Indian commodities
    # ══════════════════════════════════════════════════════

    # ── Precious Metals ───────────────────────────────────
    "Gold (MCX)":           "GC=F",     # Gold Futures (USD) - global
    "Silver (MCX)":         "SI=F",     # Silver Futures (USD) - global
    "Gold Spot":            "GLD",      # SPDR Gold ETF proxy
    "Silver Spot":          "SLV",      # iShares Silver ETF proxy

    # ── India Gold/Silver ETFs on NSE ─────────────────────
    "Nippon Gold ETF":      "GOLDBEES.NS",
    "SBI Gold ETF":         "SBIGETS.NS",
    "HDFC Gold ETF":        "HDFCMFGETF.NS",
    "Nippon Silver ETF":    "SILVERBEES.NS",

    # ── Energy ────────────────────────────────────────────
    "Crude Oil (MCX)":      "CL=F",     # WTI Crude Futures
    "Brent Crude":          "BZ=F",     # Brent Crude Futures
    "Natural Gas (MCX)":    "NG=F",     # Natural Gas Futures

    # ── Base Metals ───────────────────────────────────────
    "Copper (MCX)":         "HG=F",     # Copper Futures
    "Aluminium":            "ALI=F",    # Aluminium Futures
    "Nickel":               "NI=F",     # Nickel Futures
    "Zinc":                 "ZNC=F",    # Zinc Futures
    "Lead":                 "LE=F",     # Lead Futures

    # ── Agricultural ──────────────────────────────────────
    "Cotton (MCX)":         "CT=F",     # Cotton Futures
    "Mentha Oil":           "MO=F",     # Mentha Oil
    "Cardamom":             "CD=F",     # Cardamom
    "Castor Seed":          "CS=F",     # Castor Seed
    "Crude Palm Oil":       "KO=F",     # Palm Oil

    # ── Global Commodity ETFs (tradeable via NSE) ─────────
    "Mirae Commodity ETF":  "MCXCMMDTY.NS",

    # ── Currency & Forex ──────────────────────────────────
    "USD/INR":              "USDINR=X",
    "EUR/INR":              "EURINR=X",
    "GBP/INR":              "GBPINR=X",
    "JPY/INR":              "JPYINR=X",

    # ── Global Indices (for context) ──────────────────────
    "Dow Jones":            "^DJI",
    "S&P 500":              "^GSPC",
    "NASDAQ":               "^IXIC",
    "Nikkei 225":           "^N225",
    "Hang Seng":            "^HSI",
    "FTSE 100":             "^FTSE",
    "Gift Nifty":           "^NSEMDCP50", # proxy
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

    # ── Commodity Sectors ─────────────────────────────────
    "🥇 Precious Metals": [
        "Gold (MCX)","Silver (MCX)",
        "Nippon Gold ETF","SBI Gold ETF",
        "HDFC Gold ETF","Nippon Silver ETF",
    ],
    "🛢️ Energy Commodities": [
        "Crude Oil (MCX)","Brent Crude","Natural Gas (MCX)",
    ],
    "⚙️ Base Metals MCX": [
        "Copper (MCX)","Aluminium","Nickel","Zinc","Lead",
    ],
    "🌾 Agricultural MCX": [
        "Cotton (MCX)","Mentha Oil","Castor Seed",
        "Crude Palm Oil","Cardamom",
    ],
    "💱 Currency": [
        "USD/INR","EUR/INR","GBP/INR","JPY/INR",
    ],
    "🌍 Global Indices": [
        "Dow Jones","S&P 500","NASDAQ",
        "Nikkei 225","Hang Seng","FTSE 100",
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

def live_price(sym: str) -> dict:
    """
    Gets live price — uses Kite if connected, else Yahoo Finance.
    Kite = real-time tick data (zero delay).
    Yahoo = ~15 second delay via fast_info.
    No cache when Kite is connected — always fresh.
    """
    # ── Try Kite first (real-time) ────────────────────────
    kite = get_kite()
    if kite and not sym.startswith("^"):
        try:
            # Convert Yahoo symbol to NSE symbol
            # e.g. HDFCBANK.NS -> NSE:HDFCBANK
            nse_sym = sym.replace(".NS","").replace(".BO","")
            exchange = "BSE" if ".BO" in sym else "NSE"
            quote_key = f"{exchange}:{nse_sym}"
            quote = kite.quote([quote_key])
            q     = quote[quote_key]
            p     = float(q["last_price"])
            pv    = float(q["ohlc"]["close"])
            ch    = round(((p-pv)/pv)*100, 2) if pv else 0
            return {
                "ok":True,
                "p":round(p,2),
                "prev":round(pv,2),
                "chg":ch,
                "chg_abs":round(p-pv,2),
                "high":round(float(q["ohlc"]["high"]),2),
                "low": round(float(q["ohlc"]["low"]),2),
                "source":"kite"
            }
        except Exception as _kite_err:
            # Store error for debugging
            st.session_state["kite_candle_error"] = str(_kite_err)
            pass  # Fall through to Yahoo

    # ── Yahoo Finance fallback ────────────────────────────
    @st.cache_data(ttl=15)
    def _yahoo_price(sym_: str) -> dict:
        try:
            tk = yf.Ticker(sym_)
            fi = tk.fast_info
            p  = float(fi.last_price)
            pv = float(fi.previous_close)
            if not p or p <= 0:
                hist = tk.history(period="2d", interval="1d")
                if not hist.empty:
                    p  = float(hist["Close"].iloc[-1])
                    pv = float(hist["Close"].iloc[-2]) if len(hist)>1 else p
            ch = round(((p-pv)/pv)*100, 2) if pv else 0
            return {
                "ok":True, "p":round(p,2), "prev":round(pv,2),
                "chg":ch, "chg_abs":round(p-pv,2),
                "high":round(float(fi.day_high or p),2),
                "low": round(float(fi.day_low  or p),2),
                "source":"yahoo"
            }
        except:
            return {"ok":False,"p":0,"prev":0,"chg":0,
                    "chg_abs":0,"high":0,"low":0,
                    "source":"yahoo"}
    return _yahoo_price(sym)

@st.cache_data(ttl=3600)
def get_kite_instruments(_token: str = "") -> dict:
    """
    Cache NSE instruments list for 1 hour.
    _token parameter busts cache when session changes.
    Returns dict: {tradingsymbol: instrument_token}
    """
    if not _token or not KITE_AVAILABLE or not KITE_API_KEY:
        return {}
    try:
        kite_inst = KiteConnect(api_key=KITE_API_KEY)
        kite_inst.set_access_token(_token)
        instruments = kite_inst.instruments("NSE")
        result = {
            inst["tradingsymbol"]: inst["instrument_token"]
            for inst in instruments
        }
        st.session_state["kite_instruments_count"] = len(result)
        return result
    except Exception as e:
        st.session_state["kite_inst_error"] = str(e)
        return {}

def candles(sym: str, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV candles — Kite if connected (real-time),
    else Yahoo Finance (~15 min delay for intraday).
    """
    # ── Try Kite first ────────────────────────────────────
    kite = get_kite()
    if kite and not sym.startswith("^"):
        try:
            nse_sym = sym.replace(".NS","").replace(".BO","")

            # Kite interval mapping
            kite_interval = {
                "1m":  "minute",
                "5m":  "5minute",
                "15m": "15minute",
                "30m": "30minute",
                "1h":  "60minute",
                "1d":  "day",
            }.get(interval, "15minute")

            # Date range
            days_ = {
                "minute":2,"5minute":20,
                "15minute":60,"30minute":60,
                "60minute":200,"day":2000
            }.get(kite_interval, 60)

            from_dt = datetime.now() - timedelta(days=days_)
            to_dt   = datetime.now()

            # Use cached instruments lookup
            # Pass token so cache busts when new login happens
            _cur_token = st.session_state.get(
                "kite_access_token", ""
            )
            instruments_map = get_kite_instruments(_cur_token)
            inst_token = instruments_map.get(nse_sym)

            # Debug: store which path was taken
            if inst_token:
                st.session_state["kite_data_source"] = (
                    f"Kite: {nse_sym} token {inst_token}"
                )
            else:
                st.session_state["kite_data_source"] = (
                    f"Kite: {nse_sym} NOT FOUND in instruments"
                )

            if inst_token:
                hist = kite.historical_data(
                    inst_token,
                    from_date=from_dt,
                    to_date=to_dt,
                    interval=kite_interval,
                    continuous=False
                )
                if hist:
                    df_k = pd.DataFrame(hist)
                    df_k.rename(columns={
                        "date":"Date",
                        "open":"Open","high":"High",
                        "low":"Low","close":"Close",
                        "volume":"Volume"
                    }, inplace=True)
                    df_k.set_index("Date", inplace=True)
                    if df_k.index.tzinfo is None:
                        df_k.index = df_k.index.tz_localize(IST)
                    else:
                        df_k.index = df_k.index.tz_convert(IST)
                    df_k.dropna(inplace=True)
                    return df_k
        except Exception as _kite_err:
            # Store error for debugging
            st.session_state["kite_candle_error"] = str(_kite_err)
            pass  # Fall through to Yahoo

    # ── Yahoo Finance fallback (cached) ───────────────────
    @st.cache_data(ttl=60)
    def _yahoo_candles(sym_: str, interval_: str) -> pd.DataFrame:
        days = {"1m":1,"5m":4,"15m":20,"30m":40,
                "1h":59,"1d":300}.get(interval_, 20)
        end   = datetime.now()
        start = end - timedelta(days=days)
        try:
            df = yf.download(sym_,
                start=start.strftime("%Y-%m-%d"),
                end=(end+timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval_,
                auto_adjust=True, progress=False,
                threads=False)
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
    return _yahoo_candles(sym, interval)

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

        # ── Supertrend Indicator ─────────────────────────
        # Supertrend = ATR-based trend following indicator
        # Period=7, Multiplier=3 (standard settings)
        st_period = 7
        st_mult   = 3.0
        try:
            st_atr = ta.volatility.AverageTrueRange(
                h, l, c, window=st_period
            ).average_true_range()

            hl2 = (h + l) / 2
            upper_band = hl2 + (st_mult * st_atr)
            lower_band = hl2 - (st_mult * st_atr)

            # Calculate Supertrend
            st_vals  = [0.0] * len(c)
            st_trend = [1]   * len(c)  # 1=uptrend, -1=downtrend

            for idx in range(1, len(c)):
                # Upper band
                if (upper_band.iloc[idx] < upper_band.iloc[idx-1] or
                        float(c.iloc[idx-1]) > upper_band.iloc[idx-1]):
                    ub = float(upper_band.iloc[idx])
                else:
                    ub = float(upper_band.iloc[idx-1])

                # Lower band
                if (lower_band.iloc[idx] > lower_band.iloc[idx-1] or
                        float(c.iloc[idx-1]) < lower_band.iloc[idx-1]):
                    lb = float(lower_band.iloc[idx])
                else:
                    lb = float(lower_band.iloc[idx-1])

                # Trend direction
                if st_trend[idx-1] == -1:
                    if float(c.iloc[idx]) > ub:
                        st_trend[idx] = 1
                        st_vals[idx]  = lb
                    else:
                        st_trend[idx] = -1
                        st_vals[idx]  = ub
                else:
                    if float(c.iloc[idx]) < lb:
                        st_trend[idx] = -1
                        st_vals[idx]  = ub
                    else:
                        st_trend[idx] = 1
                        st_vals[idx]  = lb

            st_value    = round(st_vals[-1], 2)
            st_dir      = st_trend[-1]   # 1=buy, -1=sell
            st_bull     = st_dir == 1
            st_crossed  = st_trend[-1] != st_trend[-2]  # fresh crossover
            st_signal   = ("BUY" if st_bull else "SELL")
            st_label    = (f"Supertrend BUY ₹{st_value:,.2f}"
                           if st_bull
                           else f"Supertrend SELL ₹{st_value:,.2f}")
        except Exception:
            st_value   = 0.0
            st_bull    = cp > float(c.mean())
            st_crossed = False
            st_signal  = "BUY" if st_bull else "SELL"
            st_label   = st_signal
            st_dir     = 1 if st_bull else -1

        # ── Fibonacci Retracement Levels ──────────────────
        # Calculate from recent swing high and swing low (last 50 candles)
        fib_period = min(50, len(c))
        fib_high   = float(h.tail(fib_period).max())
        fib_low    = float(l.tail(fib_period).min())
        fib_range  = fib_high - fib_low

        # Key Fibonacci levels
        fib_236 = round(fib_high - 0.236 * fib_range, 2)
        fib_382 = round(fib_high - 0.382 * fib_range, 2)
        fib_500 = round(fib_high - 0.500 * fib_range, 2)
        fib_618 = round(fib_high - 0.618 * fib_range, 2)
        fib_786 = round(fib_high - 0.786 * fib_range, 2)

        # Find nearest Fibonacci level to current price
        fib_levels = {
            "23.6%": fib_236,
            "38.2%": fib_382,
            "50.0%": fib_500,
            "61.8%": fib_618,
            "78.6%": fib_786,
        }
        nearest_fib = min(
            fib_levels.items(),
            key=lambda x: abs(x[1] - cp)
        )
        fib_nearest_name = nearest_fib[0]
        fib_nearest_val  = nearest_fib[1]
        fib_distance_pct = round(
            abs(cp - fib_nearest_val) / cp * 100, 2
        )

        # Is price near a key Fibonacci level (within 0.5%)?
        near_fib = fib_distance_pct < 0.5
        # Is price bouncing from Fibonacci support?
        fib_support = (near_fib and
                       cp > fib_nearest_val and
                       float(l.iloc[-1]) <= fib_nearest_val * 1.005)

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
            ("9c. Supertrend BUY signal",
             st_bull,
             f"{st_signal} ₹{st_value:,.0f}"
             + (" FRESH!" if st_crossed else ""),
             "Supertrend must be in BUY zone"),
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
            ("9c. Supertrend BUY signal",
             st_bull,
             f"{st_signal} ₹{st_value:,.0f}"
             + (" FRESH!" if st_crossed else ""),
             "Supertrend must be in BUY zone"),
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
            # Supertrend
            st_value=st_value, st_bull=st_bull,
            st_crossed=st_crossed, st_signal=st_signal,
            st_label=st_label, st_dir=st_dir,
            # Fibonacci
            fib_high=fib_high, fib_low=fib_low,
            fib_236=fib_236, fib_382=fib_382,
            fib_500=fib_500, fib_618=fib_618, fib_786=fib_786,
            fib_nearest_name=fib_nearest_name,
            fib_nearest_val=fib_nearest_val,
            fib_distance_pct=fib_distance_pct,
            near_fib=near_fib, fib_support=fib_support,
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
# Auto-detect base URL (before expander)
import socket as _socket
try:
    _hn = _socket.gethostname()
    _is_local = (_hn.startswith("DESKTOP") or
                 _hn.startswith("LAPTOP") or
                 "ranjith" in _hn.lower() or
                 _hn == "localhost")
except:
    _is_local = False

_base = ("http://localhost:8501" if _is_local
         else "https://ranjithour-sketch-trading-terminal-trading.streamlit.app")

with st.expander("Open any tab in a separate browser window"):
    st.caption(
        "Click any link below to open that tab in a new browser "
        "window. You can have multiple tabs open side by side."
    )
    link_cols = st.columns(5)
    link_data = [
        ("📋 Watchlist",    "watchlist", "#3b82f6"),
        ("🎯 Trade Setup",  "setup",     "#16a34a"),
        ("🔍 Scanner",      "scanner",   "#9333ea"),
        ("🤖 ML",           "ml",        "#0891b2"),
        ("🏦 Smart Money",  "smart",     "#d97706"),
        ("🧮 P&L Calc",     "calc",      "#dc2626"),
        ("📰 News",         "news",      "#475569"),
        ("📊 Market Pulse", "pulse",     "#0f766e"),
        ("🔗 Options",      "options",   "#7c3aed"),
        ("🧪 Backtest",     "backtest",  "#1d4ed8"),
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
# ── Zerodha Kite Connection ───────────────────────────────
if KITE_AVAILABLE and KITE_API_KEY:
    if kite_is_connected():
        st.sidebar.success("Kite LIVE — Real-time data")
        if st.sidebar.button("Disconnect", key="kite_disc"):
            st.session_state.pop("kite_access_token", None)
            try:
                _kt = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    ".kite_token.json"
                )
                if os.path.exists(_kt):
                    os.remove(_kt)
            except:
                pass
            st.rerun()
    else:
        st.sidebar.warning("Yahoo data (15-min delay)")
        # Check if redirected back from Kite with request_token
        _qp = st.query_params
        if "request_token" in _qp:
            _req = _qp["request_token"]
            try:
                _k2 = KiteConnect(api_key=KITE_API_KEY)
                _sess = _k2.generate_session(
                    _req,
                    api_secret=KITE_API_SECRET
                )
                _tok = _sess["access_token"]
                st.session_state["kite_access_token"] = _tok
                save_kite_token(_tok)
                st.query_params.clear()
                st.sidebar.success("Kite connected!")
                st.rerun()
            except Exception as _e:
                st.sidebar.error(f"Login failed: {_e}")


        # Login URL - v=3 is required by Kite Connect
        _login_url = (
            f"https://kite.trade/connect/login"
            f"?api_key={KITE_API_KEY}&v=3"
        )

        st.sidebar.link_button(
            "Login with Zerodha Kite",
            url=_login_url,
            use_container_width=True,
            type="primary"
        )
        st.sidebar.caption(
            "After login, Zerodha redirects back here automatically."
        )
        st.sidebar.caption(
            "Ensure kite.trade Redirect URL is set to your app URL."
        )
else:
    st.sidebar.info("Yahoo Finance (delayed)")

st.sidebar.markdown("---")

# Stock search
# Use empty string "" as label — avoids _arrow_right corruption
# Do NOT use label_visibility="collapsed" — that causes the bug
with st.sidebar:
    st.markdown(
        "<p style='font-size:12px;color:#64748b;"
        "margin:0 0 4px 0'>Search stock</p>",
        unsafe_allow_html=True
    )
    srch = st.text_input(
        "Search stock",
        placeholder="Reliance, TCS, Gold...",
        key="sidebar_search_main",
        label_visibility="hidden"
    )
    if srch:
        q    = srch.strip().lower()
        hits = {k:v for k,v in STOCKS.items()
                if q in k.lower() or q in v.lower()}
        if hits:
            st.markdown(
                "<p style='font-size:11px;color:#94a3b8;"
                "margin:4px 0 2px 0'>Results</p>",
                unsafe_allow_html=True
            )
            pk = st.selectbox(
                "Select stock",
                list(hits.keys()),
                key="sidebar_search_result",
                label_visibility="hidden"
            )
            if st.button(
                "Load", type="primary", key="sb_load",
                use_container_width=True
            ):
                st.session_state["sn"] = pk
                st.session_state["st"] = hits[pk]
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<b>Quick pick</b>", unsafe_allow_html=True)

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
    "Commodities": [
        "Gold (MCX)","Silver (MCX)",
        "Crude Oil (MCX)","Copper (MCX)",
        "USD/INR","Natural Gas (MCX)",
    ],
    "Global": [
        "Dow Jones","S&P 500","NASDAQ",
        "Nikkei 225","Hang Seng",
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
    "Timeframe",
    ["1m","5m","15m","30m","1h","1d"], index=2)

auto_rf = st.sidebar.checkbox("Auto Refresh (2 min)", False)

st.sidebar.markdown("---")
st.sidebar.markdown("<b>Open in new window</b>", unsafe_allow_html=True)
_base_url = _base
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
### Entry Rules Summary
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
T1,T2,T3,T4,T5,T6,T7,T8,T9,T10 = st.tabs(TAB_NAMES)

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

    # Show data source indicator
    _kite_on = kite_is_connected()
    _kite_src = st.session_state.get("kite_data_source","")
    _kite_err = st.session_state.get("kite_candle_error","")
    _src_col  = "#f0fdf4" if _kite_on else "#fffbeb"
    _src_bdr  = "#86efac" if _kite_on else "#fcd34d"
    _src_txt  = "#166534" if _kite_on else "#92400e"

    if _kite_on and _kite_src.startswith("Kite:") and "token" in _kite_src:
        _src_msg = (
            "⚡ <b>Kite LIVE</b> — Real-time candles active. "
            "All signals are live."
        )
    elif _kite_on:
        _src_msg = (
            "⚡ Kite connected but loading candle data... "
            "First load takes ~30 seconds to fetch instruments."
            + (f" Error: {_kite_err}" if _kite_err else "")
        )
        _src_col = "#fffbeb"
        _src_bdr = "#fcd34d"
        _src_txt = "#92400e"
    else:
        _src_msg = (
            "📊 <b>Yahoo Finance</b> — 15 min delay. "
            "Login with Kite in sidebar for live signals."
        )
    st.markdown(
        f"<div style='background:{_src_col};border:1px solid {_src_bdr};"
        f"border-radius:8px;padding:8px 14px;margin-bottom:8px;"
        f"font-size:12px;color:{_src_txt}'>{_src_msg}</div>",
        unsafe_allow_html=True
    )
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

    # ── Multi-Timeframe Confirmation ──────────────────────
    st.markdown("---")
    st.markdown("### 🕐 Multi-Timeframe Confirmation")
    st.caption(
        "Checks 15m, 1h and 1d simultaneously. "
        "All 3 agreeing = highest confidence trade."
    )

    mtf_tfs   = [("15m","Intraday"),("1h","Short-term"),("1d","Medium-term")]
    mtf_cols  = st.columns(3)
    mtf_results = []

    for (mtf_tf, mtf_label), mtf_col in zip(mtf_tfs, mtf_cols):
        with st.spinner(f"Loading {mtf_tf}..."):
            mtf_df  = candles(stick, mtf_tf)
            mtf_sig = None
            if mtf_df is not None and len(mtf_df) >= 55:
                try:
                    mtf_sig = compute_all(mtf_df, live_price(stick))
                except:
                    pass

        mtf_dir = mtf_sig["direction"] if mtf_sig else "UNKNOWN"
        mtf_results.append(mtf_dir)
        mtf_col_hex = (
            "#16a34a" if mtf_dir=="UPTREND"
            else "#dc2626" if mtf_dir=="DOWNTREND"
            else "#f59e0b"
        )
        mtf_bg = (
            "#f0fdf4" if mtf_dir=="UPTREND"
            else "#fef2f2" if mtf_dir=="DOWNTREND"
            else "#fffbeb"
        )
        with mtf_col:
            if mtf_sig:
                st.markdown(
                    f"<div style='background:{mtf_bg};"
                    f"border:1.5px solid {mtf_col_hex};"
                    f"border-radius:10px;padding:14px;"
                    f"text-align:center'>"
                    f"<div style='font-size:11px;color:#64748b'>"
                    f"{mtf_tf} — {mtf_label}</div>"
                    f"<div style='font-size:18px;font-weight:700;"
                    f"color:{mtf_col_hex};margin:4px 0'>{mtf_dir}</div>"
                    f"<div style='font-size:12px;color:#475569'>"
                    f"Score {max(mtf_sig['up_score'],mtf_sig['dn_score'])}/10</div>"
                    f"<div style='font-size:11px;color:#64748b'>"
                    f"ST: {'BUY' if mtf_sig['st_bull'] else 'SELL'} | "
                    f"RSI: {mtf_sig['rv']:.0f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background:#f8fafc;"
                    f"border:1px solid #e2e8f0;border-radius:10px;"
                    f"padding:14px;text-align:center'>"
                    f"<div style='font-size:11px;color:#64748b'>"
                    f"{mtf_tf}</div>"
                    f"<div style='color:#94a3b8'>No data</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # MTF verdict
    bull_c = mtf_results.count("UPTREND")
    bear_c = mtf_results.count("DOWNTREND")

    if bull_c == 3:
        st.success(
            "🔥 ALL 3 TIMEFRAMES BULLISH — "
            "Highest confidence BUY CE setup!"
        )
    elif bear_c == 3:
        st.error(
            "🔥 ALL 3 TIMEFRAMES BEARISH — "
            "Highest confidence BUY PE setup!"
        )
    elif bull_c == 2:
        st.warning(
            "⚡ 2/3 timeframes BULLISH — "
            "Good CE setup. Wait for 15m to confirm."
        )
    elif bear_c == 2:
        st.warning(
            "⚡ 2/3 timeframes BEARISH — "
            "Good PE setup. Wait for 15m to confirm."
        )
    else:
        st.info(
            "⚠️ Timeframes conflicting — "
            "mixed signals. Wait for alignment."
        )

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

    # Supertrend line on chart
    if sig and sig["st_value"] > 0:
        st_color = "#16a34a" if sig["st_bull"] else "#dc2626"
        st_series = [sig["st_value"]] * len(plot_df)
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=st_series,
            line=dict(color=st_color, width=2, dash="dot"),
            name=f"Supertrend {sig['st_signal']}",
            opacity=0.8
        ), row=1, col=1)

        # Fibonacci levels on chart
        fib_data = [
            (sig["fib_236"], "Fib 23.6%", "#94a3b8"),
            (sig["fib_382"], "Fib 38.2%", "#f59e0b"),
            (sig["fib_500"], "Fib 50.0%", "#f59e0b"),
            (sig["fib_618"], "Fib 61.8%", "#ef4444"),
            (sig["fib_786"], "Fib 78.6%", "#ef4444"),
        ]
        for fib_val, fib_name, fib_col in fib_data:
            if sig["fib_low"] < fib_val < sig["fib_high"]:
                fig.add_hline(
                    y=fib_val,
                    line_dash="dot",
                    line_color=fib_col,
                    line_width=1,
                    opacity=0.5,
                    annotation_text=fib_name,
                    annotation_position="right",
                    row=1, col=1
                )

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
    # ── Supertrend Display ────────────────────────────────
    if sig:
        st_col = "#16a34a" if sig["st_bull"] else "#dc2626"
        st_bg  = "#f0fdf4" if sig["st_bull"] else "#fef2f2"
        st_crossed_badge = (
            "<span style='background:#f59e0b;color:#fff;"
            "padding:2px 8px;border-radius:10px;"
            "font-size:11px;margin-left:8px'>FRESH CROSSOVER</span>"
            if sig["st_crossed"] else ""
        )
        st.markdown(
            f"<div style='background:{st_bg};border:1.5px solid "
            f"{st_col};border-radius:10px;padding:12px 18px;"
            f"margin-bottom:10px;display:flex;"
            f"justify-content:space-between;align-items:center'>"
            f"<div>"
            f"<span style='font-size:13px;font-weight:700;"
            f"color:{st_col}'>⚡ SUPERTREND — {sig['st_signal']}"
            f"</span>{st_crossed_badge}</div>"
            f"<div style='font-size:14px;font-weight:700;"
            f"color:{st_col}'>₹{sig['st_value']:,.2f}</div>"
            f"<div style='font-size:12px;color:#64748b'>"
            f"{'Price above Supertrend — Bullish' if sig['st_bull'] else 'Price below Supertrend — Bearish'}"
            f"</div></div>",
            unsafe_allow_html=True
        )

    # ── Fibonacci Levels ───────────────────────────────────
    if sig:
        st.markdown("#### 📐 Fibonacci Retracement Levels")
        st.caption(
            f"Swing High ₹{sig['fib_high']:,.2f} → "
            f"Swing Low ₹{sig['fib_low']:,.2f} "
            f"| Nearest: {sig['fib_nearest_name']} "
            f"₹{sig['fib_nearest_val']:,.2f} "
            f"({sig['fib_distance_pct']:.2f}% away)"
            + (" ← Price near Fibonacci!" if sig["near_fib"] else "")
        )

        fc1,fc2,fc3,fc4,fc5 = st.columns(5)
        fib_display = [
            (fc1, "23.6%", sig["fib_236"], "#94a3b8", "Weak support"),
            (fc2, "38.2%", sig["fib_382"], "#f59e0b", "Key support"),
            (fc3, "50.0%", sig["fib_500"], "#f59e0b", "Mid point"),
            (fc4, "61.8%", sig["fib_618"], "#ef4444", "Golden ratio"),
            (fc5, "78.6%", sig["fib_786"], "#ef4444", "Deep support"),
        ]
        cur = sig["cp"]
        for col, label, val, col_hex, desc in fib_display:
            diff = round(((cur - val) / val) * 100, 2)
            is_near = abs(diff) < 0.5
            bg = "#fffbeb" if is_near else "#f8fafc"
            border = f"2px solid {col_hex}" if is_near else "1px solid #e2e8f0"
            col.markdown(
                f"<div style='background:{bg};border:{border};"
                f"border-radius:8px;padding:10px;text-align:center'>"
                f"<div style='font-size:11px;color:#64748b'>{label}</div>"
                f"<div style='font-size:14px;font-weight:700;"
                f"color:{col_hex}'>₹{val:,.0f}</div>"
                f"<div style='font-size:10px;color:#94a3b8'>{desc}</div>"
                f"<div style='font-size:10px;color:"
                f"{'#16a34a' if diff>=0 else '#dc2626'}'>"
                f"{diff:+.1f}%</div>"
                + ('<div style="font-size:9px;color:#f59e0b">◀ NEAR</div>' if is_near else '')
                + "</div>",
                unsafe_allow_html=True
            )

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

    # ══════════════════════════════════════════════════════
    # DURATION-BASED SIGNAL ENGINE
    # ══════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### ⏱️ Duration Signal — Hold or Exit?")
    st.caption(
        "Enter your trade details below. "
        "The terminal will tell you every day whether "
        "to hold your position or exit early."
    )

    # ── Trade details input ────────────────────────────────
    with st.expander(
        "Enter your trade details",
        expanded="dur_expiry" not in st.session_state
    ):
        di1, di2, di3 = st.columns(3)
        with di1:
            dur_type = st.selectbox(
                "Option type",
                ["CE (Call — Bullish)",
                 "PE (Put — Bearish)"],
                key="dur_type"
            )
            dur_duration = st.selectbox(
                "Holding duration",
                ["Intraday (exit by 2:45 PM)",
                 "Weekly (next Thursday expiry)",
                 "Monthly (last Thursday expiry)",
                 "Custom expiry date"],
                key="dur_duration"
            )
        with di2:
            dur_entry = st.number_input(
                "Your entry price (Rs)",
                value=0.0, step=0.5,
                min_value=0.0,
                key="dur_entry"
            )
            dur_sl = st.number_input(
                "Your stop loss (Rs)",
                value=sig["sl_long"]
                if "CE" in dur_type else sig["sl_short"]
                if sig else 0.0,
                step=0.5,
                key="dur_sl"
            )
        with di3:
            dur_target = st.number_input(
                "Your target price (Rs)",
                value=sig["tgt1"]
                if "CE" in dur_type else sig["tgt1s"]
                if sig else 0.0,
                step=0.5,
                key="dur_target"
            )
            if "Custom" in dur_duration:
                dur_expiry = st.date_input(
                    "Custom expiry date",
                    key="dur_custom_expiry"
                )
            else:
                # Auto-calculate next Thursday
                today   = now_ist().date()
                days_to_thu = (3 - today.weekday()) % 7
                if days_to_thu == 0:
                    days_to_thu = 7
                if "Monthly" in dur_duration:
                    # Last Thursday of current month
                    import calendar
                    last_day = calendar.monthrange(
                        today.year, today.month
                    )[1]
                    from datetime import date as date_type
                    last_thu = max(
                        date_type(today.year, today.month, d)
                        for d in range(1, last_day+1)
                        if date_type(
                            today.year, today.month, d
                        ).weekday() == 3
                    )
                    dur_expiry = last_thu
                elif "Intraday" in dur_duration:
                    dur_expiry = today
                else:
                    from datetime import date as date_type
                    dur_expiry = (
                        today +
                        timedelta(days=days_to_thu)
                    )
                st.date_input(
                    "Expiry date",
                    value=dur_expiry,
                    disabled=True,
                    key="dur_expiry_display"
                )

        if st.button(
            "Generate Hold/Exit Analysis",
            type="primary",
            key="dur_analyse",
            use_container_width=True
        ):
            st.session_state["dur_expiry"]  = str(dur_expiry)
            st.session_state["dur_entry"]   = dur_entry
            st.session_state["dur_sl"]      = dur_sl
            st.session_state["dur_target"]  = dur_target
            st.session_state["dur_type"]    = dur_type
            st.session_state["dur_active"]  = True

    # ── Show analysis if active ────────────────────────────
    if st.session_state.get("dur_active") and sig:
        from datetime import date as date_type, datetime as dt_type
        import math

        # Load saved values
        try:
            expiry_date = date_type.fromisoformat(
                st.session_state.get("dur_expiry", "")
            )
        except:
            expiry_date = now_ist().date()

        s_entry  = st.session_state.get("dur_entry",  0)
        s_sl     = st.session_state.get("dur_sl",     0)
        s_target = st.session_state.get("dur_target", 0)
        s_type   = st.session_state.get("dur_type",   "CE")
        today    = now_ist().date()

        days_left  = (expiry_date - today).days
        days_total = max(
            (expiry_date - (
                expiry_date - timedelta(days=7)
            )).days, 1
        )
        is_ce = "CE" in s_type

        # ── Current price & P&L ───────────────────────────
        cur_price = lp["p"] if lp["ok"] else sig["cp"]
        pnl_pts   = (
            cur_price - s_entry if is_ce
            else s_entry - cur_price
        )
        pnl_pct   = round(
            pnl_pts / (s_entry + 0.001) * 100, 2
        ) if s_entry > 0 else 0

        # ── Hold/Exit decision engine ──────────────────────
        hold_signals   = []
        exit_signals   = []
        warning_signals= []

        # 1. Price vs EMA21
        if is_ce:
            if cur_price > sig["e21v"]:
                hold_signals.append(
                    f"✅ Price ₹{cur_price:,} is above EMA21 "
                    f"₹{sig['e21v']:,} — trend intact"
                )
            else:
                exit_signals.append(
                    f"❌ Price ₹{cur_price:,} broke below "
                    f"EMA21 ₹{sig['e21v']:,} — exit now"
                )
        else:
            if cur_price < sig["e21v"]:
                hold_signals.append(
                    f"✅ Price ₹{cur_price:,} is below EMA21 "
                    f"₹{sig['e21v']:,} — trend intact"
                )
            else:
                exit_signals.append(
                    f"❌ Price ₹{cur_price:,} broke above "
                    f"EMA21 ₹{sig['e21v']:,} — exit now"
                )

        # 2. Stop loss check
        if s_sl > 0:
            if is_ce and cur_price <= s_sl:
                exit_signals.append(
                    f"🚨 Stop loss hit — price ₹{cur_price:,} "
                    f"at or below SL ₹{s_sl:,} — EXIT IMMEDIATELY"
                )
            elif not is_ce and cur_price >= s_sl:
                exit_signals.append(
                    f"🚨 Stop loss hit — price ₹{cur_price:,} "
                    f"at or above SL ₹{s_sl:,} — EXIT IMMEDIATELY"
                )
            else:
                dist = abs(cur_price - s_sl)
                dist_pct = round(dist / cur_price * 100, 2)
                hold_signals.append(
                    f"✅ Stop loss safe — "
                    f"₹{dist:,.0f} ({dist_pct}%) away"
                )

        # 3. Target check
        if s_target > 0:
            if is_ce and cur_price >= s_target:
                warning_signals.append(
                    f"🎯 Target ₹{s_target:,} reached — "
                    f"book at least 50% quantity now"
                )
            elif not is_ce and cur_price <= s_target:
                warning_signals.append(
                    f"🎯 Target ₹{s_target:,} reached — "
                    f"book at least 50% quantity now"
                )

        # 4. CPR bias
        if sig["cpr_bias"] == "Bullish" and is_ce:
            hold_signals.append(
                "✅ CPR bias Bullish — supports CE position"
            )
        elif sig["cpr_bias"] == "Bearish" and not is_ce:
            hold_signals.append(
                "✅ CPR bias Bearish — supports PE position"
            )
        elif sig["cpr_position"] == "INSIDE":
            warning_signals.append(
                "⚠️ Price inside CPR — choppy zone, "
                "reduce position size"
            )
        else:
            exit_signals.append(
                f"❌ CPR bias {sig['cpr_bias']} "
                f"is against your position"
            )

        # 5. RSI check
        rsi_v = sig["rv"]
        if is_ce:
            if rsi_v > 75:
                warning_signals.append(
                    f"⚠️ RSI {rsi_v:.0f} is overbought — "
                    "consider booking partial profit"
                )
            elif 50 <= rsi_v <= 75:
                hold_signals.append(
                    f"✅ RSI {rsi_v:.0f} in bullish zone — "
                    "momentum intact"
                )
            else:
                warning_signals.append(
                    f"⚠️ RSI {rsi_v:.0f} weakening — "
                    "watch closely"
                )
        else:
            if rsi_v < 25:
                warning_signals.append(
                    f"⚠️ RSI {rsi_v:.0f} is oversold — "
                    "consider booking partial profit"
                )
            elif 25 <= rsi_v <= 50:
                hold_signals.append(
                    f"✅ RSI {rsi_v:.0f} in bearish zone — "
                    "momentum intact"
                )
            else:
                warning_signals.append(
                    f"⚠️ RSI {rsi_v:.0f} weakening — "
                    "watch closely"
                )

        # 6. Time decay warning
        if days_left <= 1:
            exit_signals.append(
                "🚨 Expiry tomorrow — exit today by 2:45 PM "
                "unless deeply profitable"
            )
        elif days_left <= 2:
            warning_signals.append(
                f"⚠️ Only {days_left} days to expiry — "
                "theta decay is very high now, "
                "consider exiting if not in profit"
            )
        elif days_left <= 4:
            warning_signals.append(
                f"⚠️ {days_left} days to expiry — "
                "time decay accelerating"
            )
        else:
            hold_signals.append(
                f"✅ {days_left} days to expiry — "
                "enough time for the move to develop"
            )

        # 7. Volume check
        if sig["vsurge"]:
            hold_signals.append(
                "✅ Volume surge confirms price move"
            )

        # 8. Signal score
        tech_score = (
            sig["up_score"] if is_ce else sig["dn_score"]
        )
        if tech_score >= 7:
            hold_signals.append(
                f"✅ Technical score {tech_score}/10 — "
                "strong signal, hold"
            )
        elif tech_score >= 5:
            warning_signals.append(
                f"⚠️ Technical score {tech_score}/10 — "
                "weakening, watch"
            )
        else:
            exit_signals.append(
                f"❌ Technical score {tech_score}/10 — "
                "signal too weak, consider exiting"
            )

        # ── Final verdict ──────────────────────────────────
        if exit_signals:
            verdict      = "EXIT NOW"
            verdict_col  = "#dc2626"
            verdict_bg   = "#fef2f2"
            verdict_icon = "🔴"
        elif len(warning_signals) >= 2:
            verdict      = "REDUCE POSITION"
            verdict_col  = "#d97706"
            verdict_bg   = "#fffbeb"
            verdict_icon = "🟡"
        elif len(hold_signals) >= 3:
            verdict      = "HOLD"
            verdict_col  = "#16a34a"
            verdict_bg   = "#f0fdf4"
            verdict_icon = "🟢"
        else:
            verdict      = "MONITOR CLOSELY"
            verdict_col  = "#d97706"
            verdict_bg   = "#fffbeb"
            verdict_icon = "🟡"

        # ── Display verdict ────────────────────────────────
        st.markdown(
            f"<div style='background:{verdict_bg};"
            f"border:2px solid {verdict_col};"
            f"border-radius:14px;padding:20px 24px;"
            f"margin:12px 0;text-align:center'>"
            f"<div style='font-size:13px;color:#64748b;"
            f"letter-spacing:2px;text-transform:uppercase'>"
            f"Today's Recommendation — {today.strftime('%d %b %Y')}"
            f"</div>"
            f"<div style='font-size:48px;font-weight:700;"
            f"color:{verdict_col};margin:8px 0;line-height:1'>"
            f"{verdict_icon} {verdict}</div>"
            f"<div style='font-size:14px;color:#475569'>"
            f"{sname} {s_type} | "
            f"Entry ₹{s_entry:,} | "
            f"Expiry {expiry_date.strftime('%d %b %Y')} "
            f"({days_left} days left)"
            f"</div></div>",
            unsafe_allow_html=True
        )

        # ── P&L display ────────────────────────────────────
        pl1, pl2, pl3, pl4 = st.columns(4)
        pnl_col_m = "normal" if pnl_pts >= 0 else "inverse"
        pl1.metric(
            "Current Price",
            f"₹{cur_price:,.2f}",
            delta=f"{pnl_pts:+.2f} pts",
            delta_color=pnl_col_m
        )
        pl2.metric("Entry", f"₹{s_entry:,}")
        pl3.metric("Stop Loss", f"₹{s_sl:,}")
        pl4.metric("Target", f"₹{s_target:,}")

        # P&L %
        pnl_c = "#16a34a" if pnl_pct >= 0 else "#dc2626"
        st.markdown(
            f"<div style='background:#f8fafc;"
            f"border-radius:8px;padding:10px 16px;"
            f"font-size:14px;margin:8px 0'>"
            f"Unrealised P&L: "
            f"<b style='color:{pnl_c}'>{pnl_pct:+.2f}%"
            f" ({pnl_pts:+.2f} pts)</b>"
            f"</div>",
            unsafe_allow_html=True
        )

        # ── Signal breakdown ───────────────────────────────
        st.markdown("#### Signal breakdown")

        col_hold, col_warn, col_exit = st.columns(3)
        with col_hold:
            st.markdown(
                f"<div style='font-size:13px;font-weight:700;"
                f"color:#16a34a;margin-bottom:8px'>"
                f"✅ Hold signals ({len(hold_signals)})</div>",
                unsafe_allow_html=True
            )
            for hs in hold_signals:
                st.markdown(
                    f"<div style='background:#f0fdf4;"
                    f"border-left:3px solid #86efac;"
                    f"border-radius:0 6px 6px 0;"
                    f"padding:8px 12px;margin:4px 0;"
                    f"font-size:12px;color:#166534'>"
                    f"{hs}</div>",
                    unsafe_allow_html=True
                )

        with col_warn:
            st.markdown(
                f"<div style='font-size:13px;font-weight:700;"
                f"color:#d97706;margin-bottom:8px'>"
                f"⚠️ Warnings ({len(warning_signals)})</div>",
                unsafe_allow_html=True
            )
            for ws in warning_signals:
                st.markdown(
                    f"<div style='background:#fffbeb;"
                    f"border-left:3px solid #fcd34d;"
                    f"border-radius:0 6px 6px 0;"
                    f"padding:8px 12px;margin:4px 0;"
                    f"font-size:12px;color:#92400e'>"
                    f"{ws}</div>",
                    unsafe_allow_html=True
                )

        with col_exit:
            st.markdown(
                f"<div style='font-size:13px;font-weight:700;"
                f"color:#dc2626;margin-bottom:8px'>"
                f"❌ Exit signals ({len(exit_signals)})</div>",
                unsafe_allow_html=True
            )
            for es in exit_signals:
                st.markdown(
                    f"<div style='background:#fef2f2;"
                    f"border-left:3px solid #fca5a5;"
                    f"border-radius:0 6px 6px 0;"
                    f"padding:8px 12px;margin:4px 0;"
                    f"font-size:12px;color:#991b1b'>"
                    f"{es}</div>",
                    unsafe_allow_html=True
                )

        # ── Daily check reminder ───────────────────────────
        st.markdown("---")
        st.markdown("#### 📅 Daily holding checklist")
        _dir_word  = "above" if is_ce else "below"
        _bias_word = "Bullish" if is_ce else "Bearish"
        st.info(
            f"Check these every morning at 9:30 AM "
            f"while holding {sname} {s_type}:\n\n"
            f"**1.** Is price still {_dir_word} "
            f"EMA21 (₹{sig['e21v']:,})?\n"
            f"**2.** Is CPR bias still {_bias_word}?\n"
            f"**3.** Is RSI still in the right zone?\n"
            f"**4.** Did any bad news come overnight?\n"
            f"**5.** Days to expiry: **{days_left}** — "
            f"exit by 2:45 PM on expiry day no matter what."
        )

        if st.button(
            "Clear — start new trade",
            key="dur_clear",
            type="secondary"
        ):
            for k in ["dur_active","dur_expiry","dur_entry",
                      "dur_sl","dur_target","dur_type"]:
                st.session_state.pop(k, None)
            st.rerun()

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — SMART MONEY                                ║
# ╚══════════════════════════════════════════════════════╝

with T3:
    st.markdown("### 🔍 Auto Scanner — 30 Stocks Live")

    # ── Quick stock search inside scanner ────────────────
    with st.expander("🔍 Search a specific stock", expanded=False):
        sc_search = st.text_input(
            "Type stock name",
            placeholder="e.g. Adani Ports, TCS, HDFC...",
            key="scanner_search_box"
        )
        if sc_search:
            hits = {k:v for k,v in STOCKS.items()
                    if sc_search.strip().lower() in k.lower()}
            if hits:
                picked = st.selectbox(
                    "Select stock",
                    list(hits.keys()),
                    key="scanner_search_pick",
                    label_visibility="collapsed"
                )
                sc1, sc2 = st.columns(2)
                with sc1:
                    if st.button(
                        f"📊 Analyse {picked}",
                        key="scanner_search_analyse",
                        type="primary",
                        use_container_width=True
                    ):
                        st.session_state["sn"] = picked
                        st.session_state["st"] = hits[picked]
                        st.rerun()
                with sc2:
                    if tg_configured() and st.button(
                        "📱 Quick Signal",
                        key="scanner_search_tg",
                        use_container_width=True
                    ):
                        sym_q = hits[picked]
                        lp_q  = live_price(sym_q)
                        if lp_q["ok"]:
                            msg_q = (
                                f"<b>Quick Signal: {picked}</b>\n"
                                f"Price: Rs {lp_q['p']:,.2f}\n"
                                f"Change: {lp_q['chg']:+.2f}%\n"
                                f"Check Trade Setup tab for full analysis."
                            )
                            if send_telegram(
                                st.session_state["tg_token_saved"],
                                st.session_state["tg_chat_saved"],
                                msg_q
                            ):
                                st.success("Sent ✅")
                            else:
                                st.error("Failed ❌")
            elif sc_search:
                st.caption("No results found")

    _kite_scan = kite_is_connected()
    if _kite_scan:
        st.success(
            "⚡ Kite LIVE connected — Scanner using real-time candles"
        )
    else:
        st.warning(
            "📊 Yahoo Finance data (15-min delay) — "
            "Login with Zerodha Kite in sidebar for live scanner signals"
        )
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

    st.markdown("---")

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
                    except Exception as ex:
                        err = str(ex)
                    st.error(
                        f"❌ Failed: {err}  \n"
                        "Fix: Open Telegram → find your bot "
                        "→ press START first."
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
    if st.button(
        "🔄 Clear cache & refresh data",
        key="scan_clear_cache",
        help="Forces Yahoo Finance to fetch fresh data"
    ):
        st.cache_data.clear()
        st.rerun()
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
    # ── Show previous scan results if available ───────
    # This keeps results visible when buttons are clicked
    if "scan_results" in st.session_state and not run_scanner:
        results   = st.session_state["scan_results"]
        scan_time = st.session_state.get("scan_time","")
        grp_used  = st.session_state.get("scan_group_used","")
        tf_used   = st.session_state.get("scan_tf_used","")
        st.info(
            f"📊 Showing last scan results — "
            f"{len(results)} signals | "
            f"{grp_used} | {tf_used} | {scan_time}  "
            f"*(Click Scan to refresh)*"
        )
        # Display cached results
        total_scanned = len(SCANNER_UNIVERSE.get(grp_used,[]))

        # ── Summary metrics ───────────────────────────────
        results_sorted = sorted(
            results, key=lambda x: x["Score"], reverse=True
        )
        strong_r = [r for r in results_sorted if r["Score"] >= 8]
        good_r   = [r for r in results_sorted if 6 <= r["Score"] < 8]
        ce_list  = [r for r in results_sorted if r["Direction"]=="UPTREND"]
        pe_list  = [r for r in results_sorted if r["Direction"]=="DOWNTREND"]

        sm1,sm2,sm3,sm4,sm5 = st.columns(5)
        sm1.metric("Scanned",   total_scanned if total_scanned else len(results))
        sm2.metric("Signals",   len(results))
        sm3.metric("Strong 8+", len(strong_r))
        sm4.metric("BUY CE",    len(ce_list))
        sm5.metric("BUY PE",    len(pe_list))

        # ── Strong signals ─────────────────────────────────
        if strong_r:
            st.markdown("---")
            sh1, sh2 = st.columns([3,1])
            with sh1:
                st.markdown(f"### 🔥 STRONG SIGNALS — {len(strong_r)} found")
                st.caption("Confirmed by technical score AND historical consistency.")
            with sh2:
                if tg_configured():
                    if st.button("📱 Send All", key="scan_tg_all",
                                 type="primary", use_container_width=True):
                        tok = st.session_state.get("tg_token_saved","")
                        cid = st.session_state.get("tg_chat_saved","")
                        sent = 0
                        for r_s in strong_r:
                            msg_s = (
                                f"<b>{r_s['Stock']}</b> — {r_s['Action']}\n"
                                f"Score: {r_s['Score']}/10 | Combined: {r_s['Combined']}/10\n"
                                f"Price: Rs {r_s['Price']:,.2f}\n"
                                f"Entry: Rs {r_s['Entry']:,.2f} | SL: Rs {r_s['SL']:,.2f}\n"
                                f"T1: Rs {r_s['T1']:,.2f} | T2: Rs {r_s['T2']:,.2f}\n"
                                f"ATM: {r_s['ATM']} | RSI: {r_s['RSI']:.0f} | R:R {r_s['RR']}:1"
                            )
                            if send_telegram(tok, cid, msg_s):
                                sent += 1
                        if sent > 0:
                            st.success(f"✅ Sent {sent}/{len(strong_r)} signals!")
                        else:
                            st.error("❌ Failed. Check Telegram setup.")

            for idx_r, r in enumerate(strong_r):
                dir_col  = "#16a34a" if r["Direction"]=="UPTREND" else "#dc2626"
                bg_light = "#f0fdf4" if r["Direction"]=="UPTREND" else "#fef2f2"
                chg_col  = "#16a34a" if r["Change%"] >= 0 else "#dc2626"
                arr      = "▲" if r["Change%"] >= 0 else "▼"
                border_col = "#86efac" if r["Direction"]=="UPTREND" else "#fca5a5"

                st.markdown(
                    f"<div style='background:#ffffff;border:1.5px solid {border_col};"
                    f"border-radius:12px;padding:16px 18px;margin-bottom:10px'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:8px'>"
                    f"<span style='font-size:17px;font-weight:700;color:#1e293b'>{r['Stock']}</span>"
                    f"<span style='background:{bg_light};color:{dir_col};padding:4px 14px;"
                    f"border-radius:20px;font-size:13px;font-weight:700'>{r['Action']}</span>"
                    f"<span style='font-size:12px;color:#64748b'>{r['Reliability']}</span>"
                    f"</div>"
                    f"<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px'>"
                    f"<div><div style='font-size:10px;color:#94a3b8'>Signal</div>"
                    f"<div style='font-size:20px;font-weight:700;color:{dir_col}'>{r['Score']}/10</div></div>"
                    f"<div><div style='font-size:10px;color:#94a3b8'>Combined</div>"
                    f"<div style='font-size:20px;font-weight:700;color:#1e293b'>{r['Combined']}/10</div></div>"
                    f"<div><div style='font-size:10px;color:#94a3b8'>R:R</div>"
                    f"<div style='font-size:14px;font-weight:600;color:#374151'>{r['RR']}:1</div></div>"
                    f"<div><div style='font-size:10px;color:#94a3b8'>Price</div>"
                    f"<div style='font-size:14px;font-weight:600;color:#1e293b'>"
                    f"₹{r['Price']:,.2f} <span style='color:{chg_col}'>{arr}{abs(r['Change%']):.1f}%</span></div></div>"
                    f"</div>"
                    f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px'>"
                    f"<div style='background:#f0fdf4;border-radius:6px;padding:8px;text-align:center'>"
                    f"<div style='font-size:10px;color:#64748b'>Entry</div>"
                    f"<div style='font-size:13px;font-weight:700;color:#16a34a'>₹{r['Entry']:,.0f}</div></div>"
                    f"<div style='background:#fef2f2;border-radius:6px;padding:8px;text-align:center'>"
                    f"<div style='font-size:10px;color:#64748b'>Stop Loss</div>"
                    f"<div style='font-size:13px;font-weight:700;color:#dc2626'>₹{r['SL']:,.0f}</div></div>"
                    f"<div style='background:#eff6ff;border-radius:6px;padding:8px;text-align:center'>"
                    f"<div style='font-size:10px;color:#64748b'>Target 1</div>"
                    f"<div style='font-size:13px;font-weight:700;color:#1d4ed8'>₹{r['T1']:,.0f}</div></div>"
                    f"<div style='background:#eff6ff;border-radius:6px;padding:8px;text-align:center'>"
                    f"<div style='font-size:10px;color:#64748b'>Target 2</div>"
                    f"<div style='font-size:13px;font-weight:700;color:#1d4ed8'>₹{r['T2']:,.0f}</div></div>"
                    f"</div>"
                    f"<div style='background:#faf5ff;border-radius:6px;padding:8px 12px;margin-bottom:8px;"
                    f"font-size:12px;color:#7c3aed'>"
                    f"<b>{r['OptType']}</b> | ATM ✅ {r['ATM']} | ITM {r['ITM']} | OTM {r['OTM']}"
                    f"{'  ✨ Virgin CPR' if r.get('Virgin_CPR') else ''}"
                    f" | CPR: Price {r.get('CPR_Pos','—')}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    if st.button(
                        "🔬 Full Analysis",
                        key=f"scan_full_{idx_r}",
                        type="primary",
                        use_container_width=True
                    ):
                        # Toggle expanded analysis for this stock
                        key = f"expand_{idx_r}"
                        st.session_state[key] = not st.session_state.get(key, False)
                with bc2:
                    if st.button(
                        "📊 Trade Setup →",
                        key=f"scan_an_{idx_r}",
                        use_container_width=True
                    ):
                        st.session_state["sn"] = r["Stock"]
                        st.session_state["st"] = r["Sym"]
                        st.rerun()
                with bc3:
                    if tg_configured():
                        if st.button(
                            "📱 Send Signal",
                            key=f"scan_tg_{idx_r}",
                            use_container_width=True
                        ):
                            tok = st.session_state.get("tg_token_saved","")
                            cid = st.session_state.get("tg_chat_saved","")
                            msg = (
                                f"<b>{r['Stock']}</b> — {r['Action']}\n"
                                f"Score: {r['Score']}/10 | Combined: {r['Combined']}/10\n"
                                f"Reliability: {r['Reliability']}\n"
                                f"Price: Rs {r['Price']:,.2f}\n"
                                f"Entry: Rs {r['Entry']:,.2f} | SL: Rs {r['SL']:,.2f}\n"
                                f"T1: Rs {r['T1']:,.2f} | T2: Rs {r['T2']:,.2f}\n"
                                f"ATM: {r['ATM']} | ITM: {r['ITM']} | OTM: {r['OTM']}\n"
                                f"RSI: {r['RSI']:.0f} | R:R {r['RR']}:1"
                            )
                            if send_telegram(tok, cid, msg):
                                st.success(f"✅ {r['Stock']} sent!")
                            else:
                                st.error("❌ Failed — check Telegram setup")

                # ── COMBINED ANALYSIS PANEL ────────────────────
                if st.session_state.get(f"expand_{idx_r}", False):
                    with st.container():
                        st.markdown(
                            f"<div style='background:#f0f9ff;"
                            f"border:2px solid #3b82f6;"
                            f"border-radius:14px;padding:20px;"
                            f"margin:8px 0'>"
                            f"<div style='font-size:16px;font-weight:700;"
                            f"color:#1e40af;margin-bottom:16px'>"
                            f"🔬 Full Analysis — {r['Stock']}"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )

                        # Load data for this stock
                        _sym  = r["Sym"]
                        _name = r["Stock"]
                        _lp   = live_price(_sym)

                        with st.spinner(
                            f"Loading full analysis for {_name}..."
                        ):
                            _df  = candles(_sym, scan_tf)
                            _sig = compute_all(_df, _lp) if (
                                _df is not None and
                                len(_df) >= 55
                            ) else None
                            # ML
                            _df_ml = candles(_sym, "1d")
                            _model = None
                            _pred  = None
                            if _df_ml is not None and len(_df_ml) >= 100:
                                _model = train_model(_df_ml)
                                if _model.get("ok"):
                                    _pred = predict_next_move(
                                        _df_ml, _model
                                    )
                            # Real-time approximation
                            _rt = approximate_realtime(
                                _df,
                                _lp["p"] if _lp["ok"] else 0
                            ) if _df is not None else None

                        # ── TOP ROW: Scanner + ML side by side ─────
                        col_sc, col_ml = st.columns(2)

                        with col_sc:
                            st.markdown(
                                "<div style='background:#ffffff;"
                                "border:1px solid #e2e8f0;"
                                "border-radius:12px;padding:16px'>"
                                "<div style='font-size:13px;font-weight:700;"
                                "color:#64748b;margin-bottom:12px;"
                                "text-transform:uppercase;letter-spacing:1px'>"
                                "📊 Scanner Signal</div>",
                                unsafe_allow_html=True
                            )
                            dir_c = (
                                "#16a34a"
                                if r["Direction"] == "UPTREND"
                                else "#dc2626"
                            )
                            st.markdown(
                                f"<div style='font-size:28px;"
                                f"font-weight:700;color:{dir_c}'>"
                                f"{r['Action']}</div>"
                                f"<div style='font-size:13px;"
                                f"color:#475569;margin-top:4px'>"
                                f"Score: <b>{r['Score']}/10</b> | "
                                f"Combined: <b>{r['Combined']}/10</b>"
                                f"</div>"
                                f"<div style='font-size:12px;"
                                f"color:#64748b;margin-top:8px'>"
                                f"Reliability: {r['Reliability']}</div>"
                                f"<hr style='border-color:#f1f5f9'>"
                                f"<table style='width:100%;font-size:13px'>"
                                f"<tr><td style='color:#64748b'>Entry</td>"
                                f"<td style='text-align:right;color:#16a34a;"
                                f"font-weight:700'>₹{r['Entry']:,.2f}</td></tr>"
                                f"<tr><td style='color:#64748b'>Stop Loss</td>"
                                f"<td style='text-align:right;color:#dc2626;"
                                f"font-weight:700'>₹{r['SL']:,.2f}</td></tr>"
                                f"<tr><td style='color:#64748b'>Target 1</td>"
                                f"<td style='text-align:right;color:#1d4ed8;"
                                f"font-weight:700'>₹{r['T1']:,.2f}</td></tr>"
                                f"<tr><td style='color:#64748b'>Target 2</td>"
                                f"<td style='text-align:right;color:#1d4ed8;"
                                f"font-weight:700'>₹{r['T2']:,.2f}</td></tr>"
                                f"<tr><td style='color:#64748b'>R:R Ratio</td>"
                                f"<td style='text-align:right;font-weight:700;"
                                f"color:#1e293b'>{r['RR']}:1</td></tr>"
                                f"<tr><td style='color:#64748b'>RSI</td>"
                                f"<td style='text-align:right;"
                                f"color:#1e293b'>{r['RSI']:.0f}</td></tr>"
                                f"<tr><td style='color:#64748b'>ADX</td>"
                                f"<td style='text-align:right;"
                                f"color:#1e293b'>{r['ADX']:.0f}</td></tr>"
                                f"</table>"
                                f"<hr style='border-color:#f1f5f9'>"
                                f"<div style='font-size:12px;color:#7c3aed;"
                                f"font-weight:600'>OPTIONS</div>"
                                f"<div style='font-size:12px;color:#475569;"
                                f"margin-top:4px'>"
                                f"{r['OptType']}<br>"
                                f"✅ ATM {r['ATM']} (recommended)<br>"
                                f"ITM {r['ITM']} | OTM {r['OTM']}"
                                f"</div>"
                                f"<div style='margin-top:8px;font-size:12px;"
                                f"color:#475569'>"
                                f"CPR: Price <b>{r.get('CPR_Pos','—')}</b> | "
                                f"{r.get('CPR_Bias','—')}"
                                f"{'  ✨ Virgin CPR' if r.get('Virgin_CPR') else ''}"
                                f"</div></div>",
                                unsafe_allow_html=True
                            )

                        with col_ml:
                            st.markdown(
                                "<div style='background:#ffffff;"
                                "border:1px solid #e2e8f0;"
                                "border-radius:12px;padding:16px'>"
                                "<div style='font-size:13px;font-weight:700;"
                                "color:#64748b;margin-bottom:12px;"
                                "text-transform:uppercase;letter-spacing:1px'>"
                                "🤖 ML Prediction (1d)</div>",
                                unsafe_allow_html=True
                            )

                            if _pred and _pred.get("ok"):
                                pc_ = _pred["sig_color"]
                                pred_bg = (
                                    "#f0fdf4"
                                    if _pred["prediction"]=="UPTREND"
                                    else "#fef2f2"
                                    if _pred["prediction"]=="DOWNTREND"
                                    else "#fffbeb"
                                )
                                st.markdown(
                                    f"<div style='background:{pred_bg};"
                                    f"border-radius:8px;padding:14px;"
                                    f"text-align:center;margin-bottom:12px'>"
                                    f"<div style='font-size:32px;font-weight:700;"
                                    f"color:{pc_}'>{_pred['prediction']}</div>"
                                    f"<div style='font-size:16px;color:{pc_};"
                                    f"font-weight:600'>{_pred['signal']}</div>"
                                    f"<div style='font-size:13px;color:#64748b;"
                                    f"margin-top:6px'>"
                                    f"{_pred['reliability']} "
                                    f"({_pred['confidence']}%)"
                                    f"</div></div>",
                                    unsafe_allow_html=True
                                )

                                # Probability bars
                                probs = _pred["probabilities"]
                                for pname, pval in probs.items():
                                    bc_ = (
                                        "#16a34a" if pname=="UPTREND"
                                        else "#dc2626" if pname=="DOWNTREND"
                                        else "#f59e0b"
                                    )
                                    st.markdown(
                                        f"<div style='margin:4px 0'>"
                                        f"<div style='display:flex;"
                                        f"justify-content:space-between;"
                                        f"font-size:12px;color:#475569'>"
                                        f"<span>{pname}</span>"
                                        f"<span style='font-weight:600;"
                                        f"color:{bc_}'>{pval:.1f}%</span></div>"
                                        f"<div style='background:#f1f5f9;"
                                        f"height:6px;border-radius:3px;margin-top:2px'>"
                                        f"<div style='background:{bc_};"
                                        f"width:{pval:.0f}%;height:6px;"
                                        f"border-radius:3px'></div></div></div>",
                                        unsafe_allow_html=True
                                    )

                                # Model accuracy
                                st.markdown(
                                    f"<div style='margin-top:10px;font-size:12px;"
                                    f"color:#64748b'>"
                                    f"Model accuracy: "
                                    f"<b>{_pred['model_accuracy']}%</b> | "
                                    f"Trained on "
                                    f"<b>{_pred['n_trained']}</b> candles"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

                                # Agreement check
                                scan_dir = r["Direction"]
                                ml_dir   = _pred["prediction"]
                                if scan_dir == "UPTREND" and ml_dir == "UPTREND":
                                    st.success(
                                        "🔥 Scanner + ML both agree BULLISH — "
                                        "highest confidence CE setup!"
                                    )
                                elif scan_dir == "DOWNTREND" and ml_dir == "DOWNTREND":
                                    st.error(
                                        "🔥 Scanner + ML both agree BEARISH — "
                                        "highest confidence PE setup!"
                                    )
                                else:
                                    st.warning(
                                        f"⚠️ Scanner says {scan_dir} but "
                                        f"ML says {ml_dir}. "
                                        "Wait for both to agree."
                                    )
                            else:
                                st.info(
                                    "ML needs 100+ daily candles. "
                                    "Not enough data for this stock."
                                )

                            # Live bias from real-time approximation
                            if _rt and _rt.get("ok"):
                                st.markdown("---")
                                bc_rt = _rt["bias_color"]
                                st.markdown(
                                    f"<div style='background:#f8fafc;"
                                    f"border-radius:8px;padding:10px 14px'>"
                                    f"<div style='font-size:11px;color:#64748b;"
                                    f"text-transform:uppercase;letter-spacing:1px'>"
                                    f"Live Approximation</div>"
                                    f"<div style='font-size:20px;font-weight:700;"
                                    f"color:{bc_rt};margin:4px 0'>"
                                    f"{_rt['live_bias']}</div>"
                                    f"<div style='font-size:12px;color:#64748b'>"
                                    f"RSI: {_rt['rsi_live']} | "
                                    f"Since last candle: "
                                    f"{_rt['since_close']:+.2f}% | "
                                    f"Micro: {_rt['micro_trend']}"
                                    f"</div></div>",
                                    unsafe_allow_html=True
                                )
                            st.markdown("</div>", unsafe_allow_html=True)

                        # ── MINI CHART ─────────────────────────────
                        if _df is not None and len(_df) >= 30:
                            st.markdown(
                                f"#### 🕯️ {_name} — {scan_tf} chart"
                            )
                            plot_mini = _df.tail(60).copy()
                            fig_mini  = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                row_heights=[0.75, 0.25],
                                vertical_spacing=0.03
                            )
                            # Candles
                            fig_mini.add_trace(go.Candlestick(
                                x=plot_mini.index,
                                open=plot_mini["Open"],
                                high=plot_mini["High"],
                                low=plot_mini["Low"],
                                close=plot_mini["Close"],
                                name="Price",
                                increasing_line_color="#16a34a",
                                decreasing_line_color="#dc2626",
                            ), row=1, col=1)

                            # EMAs
                            if _sig:
                                for ser, col_, nm in [
                                    (_sig["e9s"],   "#f59e0b", "EMA9"),
                                    (_sig["e21s"],  "#ea580c", "EMA21"),
                                    (_sig["vwaps"], "#64748b", "VWAP"),
                                ]:
                                    fig_mini.add_trace(go.Scatter(
                                        x=plot_mini.index,
                                        y=ser.tail(60),
                                        line=dict(color=col_, width=1.5),
                                        name=nm
                                    ), row=1, col=1)

                                # CPR lines
                                for cpr_val, cpr_nm, cpr_cl in [
                                    (_sig["cpr_tc"],    "TC",    "#f59e0b"),
                                    (_sig["cpr_pivot"], "Pivot", "#f59e0b"),
                                    (_sig["cpr_bc"],    "BC",    "#f59e0b"),
                                ]:
                                    fig_mini.add_hline(
                                        y=cpr_val,
                                        line_dash="dot",
                                        line_color=cpr_cl,
                                        line_width=1,
                                        opacity=0.7,
                                        annotation_text=cpr_nm,
                                        row=1, col=1
                                    )

                                # SL line
                                sl_plot = (
                                    _sig["sl_long"]
                                    if r["Direction"] == "UPTREND"
                                    else _sig["sl_short"]
                                )
                                fig_mini.add_hline(
                                    y=sl_plot,
                                    line_dash="dash",
                                    line_color="#dc2626",
                                    line_width=1.5,
                                    opacity=0.8,
                                    annotation_text=f"SL ₹{sl_plot:,}",
                                    row=1, col=1
                                )

                            # Volume
                            vc = [
                                "#16a34a"
                                if float(plot_mini["Close"].iloc[i]) >=
                                   float(plot_mini["Open"].iloc[i])
                                else "#dc2626"
                                for i in range(len(plot_mini))
                            ]
                            fig_mini.add_trace(go.Bar(
                                x=plot_mini.index,
                                y=plot_mini["Volume"],
                                marker_color=vc,
                                name="Vol", opacity=0.7
                            ), row=2, col=1)

                            fig_mini.update_layout(
                                template="plotly_white",
                                height=420,
                                xaxis_rangeslider_visible=False,
                                margin=dict(l=10,r=10,t=20,b=10),
                                legend=dict(
                                    orientation="h", y=1.02,
                                    font=dict(size=10)
                                ),
                            )
                            st.plotly_chart(
                                fig_mini, use_container_width=True
                            )

                st.markdown(
                    "<div style='margin:4px 0'></div>",
                    unsafe_allow_html=True
                )

        # ── Good signals ───────────────────────────────────
        if good_r:
            st.markdown("---")
            st.markdown(f"### 📈 GOOD SIGNALS — {len(good_r)} found")
            st.caption("Watch these — enter when score reaches 8+")
            for gi in range(0, len(good_r), 2):
                chunk = good_r[gi:gi+2]
                gcols = st.columns(2)
                for ci, r in enumerate(chunk):
                    dc = "#16a34a" if r["Direction"]=="UPTREND" else "#dc2626"
                    with gcols[ci]:
                        st.markdown(
                            f"<div style='background:#ffffff;border:1px solid #e2e8f0;"
                            f"border-radius:10px;padding:14px;margin-bottom:6px'>"
                            f"<div style='display:flex;justify-content:space-between'>"
                            f"<b style='color:#1e293b;font-size:15px'>{r['Stock']}</b>"
                            f"<span style='color:{dc};font-weight:700'>{r['Score']}/10</span></div>"
                            f"<div style='font-size:12px;color:#64748b;margin-top:6px'>"
                            f"{r['Action']} | ₹{r['Price']:,.0f} | RSI {r['RSI']:.0f}</div>"
                            f"<div style='font-size:11px;color:#94a3b8;margin-top:4px'>"
                            f"Entry ₹{r['Entry']:,.0f} | SL ₹{r['SL']:,.0f}</div>"
                            f"<div style='font-size:11px;color:#7c3aed;margin-top:4px'>"
                            f"{r['OptType']} ATM {r['ATM']}</div></div>",
                            unsafe_allow_html=True
                        )
                        if st.button("View", key=f"scan_vw_{gi}_{ci}",
                                     use_container_width=True):
                            st.session_state["sn"] = r["Stock"]
                            st.session_state["st"] = r["Sym"]
                            st.rerun()

        # ── Score chart ────────────────────────────────────
        if results_sorted:
            st.markdown("---")
            import plotly.graph_objects as go_scan
            fig_scan = go_scan.Figure(go_scan.Bar(
                x=[r["Stock"] for r in results_sorted],
                y=[r["Combined"] for r in results_sorted],
                marker_color=["#16a34a" if r["Direction"]=="UPTREND"
                              else "#dc2626" for r in results_sorted],
                text=[f"{r['Combined']}/10" for r in results_sorted],
                textposition="outside",
            ))
            fig_scan.add_hline(y=8, line_dash="dash", line_color="#16a34a",
                               annotation_text="Strong (8+)")
            fig_scan.add_hline(y=6, line_dash="dot", line_color="#f59e0b",
                               annotation_text="Good (6+)")
            fig_scan.update_layout(
                template="plotly_white", height=300,
                yaxis_range=[0,11],
                margin=dict(l=10,r=10,t=20,b=60),
                xaxis_tickangle=-35, showlegend=False,
                title="Combined Score (60% Technical + 40% Historical)"
            )
            st.plotly_chart(fig_scan, use_container_width=True)

    if run_scanner or auto_scan:
        if auto_scan and not run_scanner:
            st.info("Auto scan active — running every 5 minutes")

        stocks_to_scan = SCANNER_UNIVERSE[scan_group]
        st.markdown(f"**Scanning {len(stocks_to_scan)} stocks "
                f"on {scan_tf} timeframe...**")

        with st.spinner("Scanning... please wait"):
            results, alerted = run_scan_engine(
            stocks_to_scan, scan_tf,
            min_score_scan, alert_score
        )
        # Filter by combined score
        min_comb = st.session_state.get("min_combined_scan", 5)
        results = [
        r for r in results
        if r.get("Combined", 0) >= min_comb
        ]

        # ── Save results to session state ──────────────
        # This ensures results persist when buttons are clicked
        if results:
            st.session_state["scan_results"]     = results
            st.session_state["scan_group_used"]  = scan_group
            st.session_state["scan_tf_used"]     = scan_tf
            st.session_state["scan_time"]        = (
                now_ist().strftime("%H:%M:%S IST")
            )

        if not results:
            st.warning(
            "No stocks met the minimum score. "
            "Try lowering the min score or use 1d timeframe."
        )
        else:
            total_scanned = len(stocks_to_scan)

            # ── Summary metrics ───────────────────────────────
            results_sorted = sorted(
                results, key=lambda x: x["Score"], reverse=True
            )
            strong_r = [r for r in results_sorted if r["Score"] >= 8]
            good_r   = [r for r in results_sorted if 6 <= r["Score"] < 8]
            ce_list  = [r for r in results_sorted if r["Direction"]=="UPTREND"]
            pe_list  = [r for r in results_sorted if r["Direction"]=="DOWNTREND"]

            sm1,sm2,sm3,sm4,sm5 = st.columns(5)
            sm1.metric("Scanned",   total_scanned if total_scanned else len(results))
            sm2.metric("Signals",   len(results))
            sm3.metric("Strong 8+", len(strong_r))
            sm4.metric("BUY CE",    len(ce_list))
            sm5.metric("BUY PE",    len(pe_list))

            # ── Strong signals ─────────────────────────────────
            if strong_r:
                st.markdown("---")
                sh1, sh2 = st.columns([3,1])
                with sh1:
                    st.markdown(f"### 🔥 STRONG SIGNALS — {len(strong_r)} found")
                    st.caption("Confirmed by technical score AND historical consistency.")
                with sh2:
                    if tg_configured():
                        if st.button("📱 Send All", key="scan_tg_all",
                                     type="primary", use_container_width=True):
                            tok = st.session_state.get("tg_token_saved","")
                            cid = st.session_state.get("tg_chat_saved","")
                            sent = 0
                            for r_s in strong_r:
                                msg_s = (
                                    f"<b>{r_s['Stock']}</b> — {r_s['Action']}\n"
                                    f"Score: {r_s['Score']}/10 | Combined: {r_s['Combined']}/10\n"
                                    f"Price: Rs {r_s['Price']:,.2f}\n"
                                    f"Entry: Rs {r_s['Entry']:,.2f} | SL: Rs {r_s['SL']:,.2f}\n"
                                    f"T1: Rs {r_s['T1']:,.2f} | T2: Rs {r_s['T2']:,.2f}\n"
                                    f"ATM: {r_s['ATM']} | RSI: {r_s['RSI']:.0f} | R:R {r_s['RR']}:1"
                                )
                                if send_telegram(tok, cid, msg_s):
                                    sent += 1
                            if sent > 0:
                                st.success(f"✅ Sent {sent}/{len(strong_r)} signals!")
                            else:
                                st.error("❌ Failed. Check Telegram setup.")

                for idx_r, r in enumerate(strong_r):
                    dir_col  = "#16a34a" if r["Direction"]=="UPTREND" else "#dc2626"
                    bg_light = "#f0fdf4" if r["Direction"]=="UPTREND" else "#fef2f2"
                    chg_col  = "#16a34a" if r["Change%"] >= 0 else "#dc2626"
                    arr      = "▲" if r["Change%"] >= 0 else "▼"
                    border_col = "#86efac" if r["Direction"]=="UPTREND" else "#fca5a5"

                    st.markdown(
                        f"<div style='background:#ffffff;border:1.5px solid {border_col};"
                        f"border-radius:12px;padding:16px 18px;margin-bottom:10px'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:8px'>"
                        f"<span style='font-size:17px;font-weight:700;color:#1e293b'>{r['Stock']}</span>"
                        f"<span style='background:{bg_light};color:{dir_col};padding:4px 14px;"
                        f"border-radius:20px;font-size:13px;font-weight:700'>{r['Action']}</span>"
                        f"<span style='font-size:12px;color:#64748b'>{r['Reliability']}</span>"
                        f"</div>"
                        f"<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px'>"
                        f"<div><div style='font-size:10px;color:#94a3b8'>Signal</div>"
                        f"<div style='font-size:20px;font-weight:700;color:{dir_col}'>{r['Score']}/10</div></div>"
                        f"<div><div style='font-size:10px;color:#94a3b8'>Combined</div>"
                        f"<div style='font-size:20px;font-weight:700;color:#1e293b'>{r['Combined']}/10</div></div>"
                        f"<div><div style='font-size:10px;color:#94a3b8'>R:R</div>"
                        f"<div style='font-size:14px;font-weight:600;color:#374151'>{r['RR']}:1</div></div>"
                        f"<div><div style='font-size:10px;color:#94a3b8'>Price</div>"
                        f"<div style='font-size:14px;font-weight:600;color:#1e293b'>"
                        f"₹{r['Price']:,.2f} <span style='color:{chg_col}'>{arr}{abs(r['Change%']):.1f}%</span></div></div>"
                        f"</div>"
                        f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px'>"
                        f"<div style='background:#f0fdf4;border-radius:6px;padding:8px;text-align:center'>"
                        f"<div style='font-size:10px;color:#64748b'>Entry</div>"
                        f"<div style='font-size:13px;font-weight:700;color:#16a34a'>₹{r['Entry']:,.0f}</div></div>"
                        f"<div style='background:#fef2f2;border-radius:6px;padding:8px;text-align:center'>"
                        f"<div style='font-size:10px;color:#64748b'>Stop Loss</div>"
                        f"<div style='font-size:13px;font-weight:700;color:#dc2626'>₹{r['SL']:,.0f}</div></div>"
                        f"<div style='background:#eff6ff;border-radius:6px;padding:8px;text-align:center'>"
                        f"<div style='font-size:10px;color:#64748b'>Target 1</div>"
                        f"<div style='font-size:13px;font-weight:700;color:#1d4ed8'>₹{r['T1']:,.0f}</div></div>"
                        f"<div style='background:#eff6ff;border-radius:6px;padding:8px;text-align:center'>"
                        f"<div style='font-size:10px;color:#64748b'>Target 2</div>"
                        f"<div style='font-size:13px;font-weight:700;color:#1d4ed8'>₹{r['T2']:,.0f}</div></div>"
                        f"</div>"
                        f"<div style='background:#faf5ff;border-radius:6px;padding:8px 12px;margin-bottom:8px;"
                        f"font-size:12px;color:#7c3aed'>"
                        f"<b>{r['OptType']}</b> | ATM ✅ {r['ATM']} | ITM {r['ITM']} | OTM {r['OTM']}"
                        f"{'  ✨ Virgin CPR' if r.get('Virgin_CPR') else ''}"
                        f" | CPR: Price {r.get('CPR_Pos','—')}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button(f"📊 Analyse", key=f"scan_an_{idx_r}",
                                     type="primary", use_container_width=True):
                            st.session_state["sn"] = r["Stock"]
                            st.session_state["st"] = r["Sym"]
                            st.rerun()
                    with bc2:
                        if tg_configured():
                            if st.button(f"📱 Send Signal", key=f"scan_tg_{idx_r}",
                                         use_container_width=True):
                                tok = st.session_state.get("tg_token_saved","")
                                cid = st.session_state.get("tg_chat_saved","")
                                msg = (
                                    f"<b>{r['Stock']}</b> — {r['Action']}\n"
                                    f"Score: {r['Score']}/10 | Combined: {r['Combined']}/10\n"
                                    f"Reliability: {r['Reliability']}\n"
                                    f"Price: Rs {r['Price']:,.2f}\n"
                                    f"Entry: Rs {r['Entry']:,.2f} | SL: Rs {r['SL']:,.2f}\n"
                                    f"T1: Rs {r['T1']:,.2f} | T2: Rs {r['T2']:,.2f}\n"
                                    f"ATM: {r['ATM']} | ITM: {r['ITM']} | OTM: {r['OTM']}\n"
                                    f"RSI: {r['RSI']:.0f} | R:R {r['RR']}:1"
                                )
                                if send_telegram(tok, cid, msg):
                                    st.success(f"✅ {r['Stock']} sent!")
                                else:
                                    st.error("❌ Failed — check Telegram setup")
                    st.markdown("<div style='margin:4px 0'></div>", unsafe_allow_html=True)

            # ── Good signals ───────────────────────────────────
            if good_r:
                st.markdown("---")
                st.markdown(f"### 📈 GOOD SIGNALS — {len(good_r)} found")
                st.caption("Watch these — enter when score reaches 8+")
                for gi in range(0, len(good_r), 2):
                    chunk = good_r[gi:gi+2]
                    gcols = st.columns(2)
                    for ci, r in enumerate(chunk):
                        dc = "#16a34a" if r["Direction"]=="UPTREND" else "#dc2626"
                        with gcols[ci]:
                            st.markdown(
                                f"<div style='background:#ffffff;border:1px solid #e2e8f0;"
                                f"border-radius:10px;padding:14px;margin-bottom:6px'>"
                                f"<div style='display:flex;justify-content:space-between'>"
                                f"<b style='color:#1e293b;font-size:15px'>{r['Stock']}</b>"
                                f"<span style='color:{dc};font-weight:700'>{r['Score']}/10</span></div>"
                                f"<div style='font-size:12px;color:#64748b;margin-top:6px'>"
                                f"{r['Action']} | ₹{r['Price']:,.0f} | RSI {r['RSI']:.0f}</div>"
                                f"<div style='font-size:11px;color:#94a3b8;margin-top:4px'>"
                                f"Entry ₹{r['Entry']:,.0f} | SL ₹{r['SL']:,.0f}</div>"
                                f"<div style='font-size:11px;color:#7c3aed;margin-top:4px'>"
                                f"{r['OptType']} ATM {r['ATM']}</div></div>",
                                unsafe_allow_html=True
                            )
                            if st.button("View", key=f"scan_vw_{gi}_{ci}",
                                         use_container_width=True):
                                st.session_state["sn"] = r["Stock"]
                                st.session_state["st"] = r["Sym"]
                                st.rerun()

            # ── Score chart ────────────────────────────────────
            if results_sorted:
                st.markdown("---")
                import plotly.graph_objects as go_scan
                fig_scan = go_scan.Figure(go_scan.Bar(
                    x=[r["Stock"] for r in results_sorted],
                    y=[r["Combined"] for r in results_sorted],
                    marker_color=["#16a34a" if r["Direction"]=="UPTREND"
                                  else "#dc2626" for r in results_sorted],
                    text=[f"{r['Combined']}/10" for r in results_sorted],
                    textposition="outside",
                ))
                fig_scan.add_hline(y=8, line_dash="dash", line_color="#16a34a",
                                   annotation_text="Strong (8+)")
                fig_scan.add_hline(y=6, line_dash="dot", line_color="#f59e0b",
                                   annotation_text="Good (6+)")
                fig_scan.update_layout(
                    template="plotly_white", height=300,
                    yaxis_range=[0,11],
                    margin=dict(l=10,r=10,t=20,b=60),
                    xaxis_tickangle=-35, showlegend=False,
                    title="Combined Score (60% Technical + 40% Historical)"
                )
                st.plotly_chart(fig_scan, use_container_width=True)

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

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 8 — MARKET PULSE                               ║
# ╚══════════════════════════════════════════════════════╝
with T8:
    st.markdown("### 📊 Market Pulse — VIX + FII/DII")
    st.caption(
        "India VIX shows market fear. "
        "FII/DII shows institutional money flow. "
        "These two together tell you the market mood."
    )

    # ── India VIX ─────────────────────────────────────────
    st.markdown("#### ⚡ India VIX — Market Fear Index")

    if st.button("Refresh VIX + FII/DII", key="pulse_refresh",
                 type="primary"):
        get_india_vix.clear()
        get_fii_dii.clear()
        st.rerun()

    vix_data = get_india_vix()

    if vix_data["ok"]:
        vix_v = vix_data["vix"]
        vix_col = (
            "#dc2626" if vix_v > 25 else
            "#ea580c" if vix_v > 20 else
            "#d97706" if vix_v > 15 else
            "#16a34a"
        )
        vix_bg = (
            "#fef2f2" if vix_v > 25 else
            "#fff7ed" if vix_v > 20 else
            "#fffbeb" if vix_v > 15 else
            "#f0fdf4"
        )

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.markdown(
                f"<div style='background:{vix_bg};"
                f"border:2px solid {vix_col};"
                f"border-radius:14px;padding:20px;"
                f"text-align:center'>"
                f"<div style='font-size:12px;color:#64748b;"
                f"text-transform:uppercase;letter-spacing:1px'>"
                f"India VIX</div>"
                f"<div style='font-size:52px;font-weight:700;"
                f"color:{vix_col};line-height:1'>{vix_v}</div>"
                f"<div style='font-size:14px;font-weight:700;"
                f"color:{vix_col};margin-top:4px'>"
                f"{vix_data['level']}</div>"
                f"<div style='font-size:12px;color:#64748b;"
                f"margin-top:4px'>"
                f"Change: {vix_data['chg']:+.2f}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with vc2:
            st.markdown(
                f"<div style='background:#f8fafc;"
                f"border-radius:12px;padding:16px'>"
                f"<div style='font-size:13px;font-weight:700;"
                f"color:#374151;margin-bottom:10px'>"
                f"VIX Levels Guide</div>"
                f"<div style='font-size:12px;color:#475569;"
                f"line-height:2'>"
                f"🟢 Below 12 — Very calm, best for buying<br>"
                f"🟢 12–15 — Calm, good for CE/PE<br>"
                f"🟡 15–20 — Elevated, trade carefully<br>"
                f"🟠 20–25 — High fear, reduce size<br>"
                f"🔴 Above 25 — Extreme fear, avoid options"
                f"</div></div>",
                unsafe_allow_html=True
            )
        with vc3:
            st.markdown(
                f"<div style='background:#eff6ff;"
                f"border:1px solid #93c5fd;"
                f"border-radius:12px;padding:16px'>"
                f"<div style='font-size:13px;font-weight:700;"
                f"color:#1d4ed8;margin-bottom:8px'>"
                f"Today's Advice</div>"
                f"<div style='font-size:13px;color:#374151'>"
                f"{vix_data['advice']}</div>"
                f"<div style='margin-top:12px;font-size:12px;"
                f"color:#64748b'>"
                f"High VIX = expensive options = "
                f"avoid buying<br>"
                f"Low VIX = cheap options = "
                f"good time to buy</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # VIX history chart
        try:
            vix_hist = yf.download(
                "^INDIAVIX", period="3mo",
                interval="1d", progress=False
            )
            if not vix_hist.empty:
                import plotly.graph_objects as go_vix
                fig_vix = go_vix.Figure()
                vix_close = vix_hist["Close"].squeeze()
                fig_vix.add_trace(go_vix.Scatter(
                    x=vix_hist.index,
                    y=vix_close,
                    fill="tozeroy",
                    line=dict(color="#3b82f6", width=2),
                    fillcolor="rgba(59,130,246,0.1)",
                    name="India VIX"
                ))
                for lvl, col_, nm in [
                    (25, "#dc2626", "Extreme Fear"),
                    (20, "#ea580c", "High Fear"),
                    (15, "#d97706", "Elevated"),
                ]:
                    fig_vix.add_hline(
                        y=lvl,
                        line_dash="dash",
                        line_color=col_,
                        opacity=0.5,
                        annotation_text=nm
                    )
                fig_vix.update_layout(
                    template="plotly_white",
                    height=250,
                    margin=dict(l=10,r=10,t=20,b=20),
                    title="India VIX — Last 3 Months",
                    yaxis_title="VIX"
                )
                st.plotly_chart(fig_vix, use_container_width=True)
        except Exception:
            pass
    else:
        st.warning(
            "VIX data unavailable. "
            "Check internet connection or try refreshing."
        )

    # ── FII / DII Data ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏦 FII / DII — Institutional Money Flow")
    st.caption(
        "FII = Foreign Institutional Investors (foreign funds). "
        "DII = Domestic Institutional Investors (Indian mutual funds, LIC). "
        "When FII buys heavily — market usually goes up next session."
    )

    fii_data = get_fii_dii()

    if fii_data["ok"]:
        fd1, fd2, fd3, fd4 = st.columns(4)
        fii_net = fii_data["fii_net"]
        dii_net = fii_data["dii_net"]

        fd1.metric(
            "FII Buy",
            f"₹{fii_data['fii_buy']:,.0f}Cr",
        )
        fd2.metric(
            "FII Sell",
            f"₹{fii_data['fii_sell']:,.0f}Cr",
        )
        fd3.metric(
            "FII Net",
            f"₹{fii_net:,.0f}Cr",
            delta=f"{'Buying' if fii_net>0 else 'Selling'}",
            delta_color="normal" if fii_net > 0 else "inverse"
        )
        fd4.metric(
            "DII Net",
            f"₹{dii_net:,.0f}Cr",
            delta=f"{'Buying' if dii_net>0 else 'Selling'}",
            delta_color="normal" if dii_net > 0 else "inverse"
        )

        # Signal interpretation
        if fii_net > 1000 and dii_net > 0:
            st.success(
                f"🔥 Both FII and DII buying — "
                f"strong bullish signal for tomorrow. "
                f"FII net: ₹{fii_net:,.0f}Cr | "
                f"DII net: ₹{dii_net:,.0f}Cr"
            )
        elif fii_net > 500:
            st.success(
                f"✅ FII buying ₹{fii_net:,.0f}Cr — "
                f"bullish bias for tomorrow"
            )
        elif fii_net < -1000:
            st.error(
                f"🔴 FII selling ₹{abs(fii_net):,.0f}Cr — "
                f"bearish pressure. "
                f"{'DII countering with buying' if dii_net>0 else 'DII also selling — double bearish'}"
            )
        else:
            st.info(
                f"FII net: ₹{fii_net:,.0f}Cr | "
                f"DII net: ₹{dii_net:,.0f}Cr — "
                f"Neutral flow"
            )

        # Last 10 days table
        if fii_data.get("data"):
            with st.expander("Last 10 days FII/DII data"):
                rows = []
                for d in fii_data["data"][:10]:
                    fb = float(d.get("fiiBuy",0) or 0)
                    fs = float(d.get("fiiSell",0) or 0)
                    db = float(d.get("diiBuy",0) or 0)
                    ds = float(d.get("diiSell",0) or 0)
                    rows.append({
                        "Date":     d.get("date",""),
                        "FII Buy":  f"₹{fb:,.0f}Cr",
                        "FII Sell": f"₹{fs:,.0f}Cr",
                        "FII Net":  f"₹{fb-fs:+,.0f}Cr",
                        "DII Buy":  f"₹{db:,.0f}Cr",
                        "DII Sell": f"₹{ds:,.0f}Cr",
                        "DII Net":  f"₹{db-ds:+,.0f}Cr",
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.warning(
            "FII/DII data unavailable from NSE. "
            "NSE website may be blocking automated requests. "
            "Try during market hours (9 AM - 6 PM IST)."
        )
        st.info(
            "Manual check: Go to **nseindia.com** → "
            "Market Data → FII/DII Activity"
        )


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 9 — OPTIONS CHAIN                              ║
# ╚══════════════════════════════════════════════════════╝
with T9:
    st.markdown("### 🔗 Options Chain — NIFTY / BANKNIFTY")
    st.caption(
        "Shows all CE and PE strikes with OI, change in OI and IV. "
        "High Call OI = resistance. High Put OI = support."
    )

    oc1, oc2 = st.columns([2, 1])
    with oc1:
        oc_symbol = st.selectbox(
            "Index",
            ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
            key="oc_symbol"
        )
    with oc2:
        if st.button(
            "Load Options Chain",
            type="primary",
            key="oc_load",
            use_container_width=True
        ):
            get_options_chain.clear()

    oc_data = get_options_chain(oc_symbol)

    if oc_data["ok"]:
        spot = oc_data["spot"]
        expiries = oc_data["expiries"]

        st.markdown(
            f"<div style='background:#1e3a5f;color:white;"
            f"border-radius:10px;padding:10px 18px;"
            f"font-size:16px;font-weight:700;margin-bottom:12px'>"
            f"{oc_symbol} Spot: ₹{spot:,.2f}</div>",
            unsafe_allow_html=True
        )

        if expiries:
            sel_exp = st.selectbox(
                "Expiry",
                expiries[:5],
                key="oc_expiry"
            )

            # Filter data for selected expiry
            exp_data = [
                d for d in oc_data["data"]
                if d.get("expiryDate") == sel_exp
            ]

            # Calculate ATM strike
            step = 50 if oc_symbol == "NIFTY" else 100
            atm = round(spot / step) * step

            # Build chain table
            rows = []
            total_call_oi = 0
            total_put_oi  = 0

            for item in exp_data:
                strike = item.get("strikePrice", 0)
                ce = item.get("CE", {})
                pe = item.get("PE", {})
                if not ce and not pe:
                    continue

                ce_oi  = int(ce.get("openInterest", 0) or 0)
                ce_doi = int(ce.get("changeinOpenInterest", 0) or 0)
                ce_iv  = round(float(ce.get("impliedVolatility", 0) or 0), 1)
                ce_ltp = round(float(ce.get("lastPrice", 0) or 0), 2)

                pe_oi  = int(pe.get("openInterest", 0) or 0)
                pe_doi = int(pe.get("changeinOpenInterest", 0) or 0)
                pe_iv  = round(float(pe.get("impliedVolatility", 0) or 0), 1)
                pe_ltp = round(float(pe.get("lastPrice", 0) or 0), 2)

                total_call_oi += ce_oi
                total_put_oi  += pe_oi

                rows.append({
                    "CE OI":    ce_oi,
                    "CE Chg":   ce_doi,
                    "CE IV":    ce_iv,
                    "CE LTP":   ce_ltp,
                    "Strike":   strike,
                    "ATM":      "◀ ATM" if strike == atm else "",
                    "PE LTP":   pe_ltp,
                    "PE IV":    pe_iv,
                    "PE Chg":   pe_doi,
                    "PE OI":    pe_oi,
                })

            if rows:
                # PCR
                pcr = round(total_put_oi / (total_call_oi + 1), 2)
                pcr_signal = (
                    "🟢 Bullish (PCR > 1.3 = oversold)"
                    if pcr > 1.3 else
                    "🔴 Bearish (PCR < 0.7 = overbought)"
                    if pcr < 0.7 else
                    "🟡 Neutral"
                )

                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("Total Call OI", f"{total_call_oi:,}")
                pm2.metric("Total Put OI",  f"{total_put_oi:,}")
                pm3.metric(
                    "PCR",
                    f"{pcr}",
                    delta=pcr_signal,
                    delta_color="normal" if pcr > 1 else "inverse"
                )

                # Show chain near ATM (±10 strikes)
                df_chain = pd.DataFrame(rows)
                df_chain_near = df_chain[
                    (df_chain["Strike"] >= atm - step*10) &
                    (df_chain["Strike"] <= atm + step*10)
                ].copy()

                # Highlight max OI
                max_call_oi = df_chain["CE OI"].max()
                max_put_oi  = df_chain["PE OI"].max()

                st.markdown(
                    "**Max Call OI** (Resistance): "
                    f"**{df_chain.loc[df_chain['CE OI'].idxmax(), 'Strike']:,}** "
                    f"({max_call_oi:,} contracts)  |  "
                    "**Max Put OI** (Support): "
                    f"**{df_chain.loc[df_chain['PE OI'].idxmax(), 'Strike']:,}** "
                    f"({max_put_oi:,} contracts)"
                )

                st.dataframe(
                    df_chain_near.set_index("Strike"),
                    use_container_width=True,
                    height=500
                )
    else:
        st.warning(
            "Options chain data unavailable from NSE. "
            "NSE may be blocking automated requests."
        )
        st.info(
            "While connected to Zerodha Kite, "
            "options chain loads from live data. "
            "Login with Kite in the sidebar for live options data."
        )
        # Try Yahoo Finance options as fallback
        st.markdown("#### Yahoo Finance Options (Fallback)")
        yf_sym = "^NSEI" if oc_symbol=="NIFTY" else "^NSEBANK"
        try:
            tk_oc = yf.Ticker(yf_sym)
            exp_yf = tk_oc.options
            if exp_yf:
                sel_yf = st.selectbox(
                    "Expiry (Yahoo)",
                    exp_yf[:3],
                    key="oc_yf_exp"
                )
                chain_yf = tk_oc.option_chain(sel_yf)
                ycol1, ycol2 = st.columns(2)
                with ycol1:
                    st.markdown("**Calls**")
                    st.dataframe(
                        chain_yf.calls[[
                            "strike","lastPrice",
                            "openInterest","impliedVolatility"
                        ]].head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                with ycol2:
                    st.markdown("**Puts**")
                    st.dataframe(
                        chain_yf.puts[[
                            "strike","lastPrice",
                            "openInterest","impliedVolatility"
                        ]].head(20),
                        use_container_width=True,
                        hide_index=True
                    )
        except Exception as e:
            st.caption(f"Yahoo options also unavailable: {e}")


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 10 — BACKTEST                                  ║
# ╚══════════════════════════════════════════════════════╝
with T10:
    st.markdown("### 🧪 Strategy Backtest")
    st.caption(
        "Test your signal strategy on historical data. "
        "See what the win rate and profit would have been "
        "if you had traded every signal for the past 6 months."
    )

    # ── Settings ──────────────────────────────────────────
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        bt_stock = st.text_input(
            "Stock to backtest",
            value="NIFTY 50",
            key="bt_stock"
        )
        bt_sym = STOCKS.get(bt_stock, "^NSEI")
    with bc2:
        bt_months = st.slider(
            "Months of history",
            1, 12, 6,
            key="bt_months"
        )
        bt_tf = st.selectbox(
            "Timeframe",
            ["15m","1h","1d"],
            index=2,
            key="bt_tf"
        )
    with bc3:
        bt_min_score = st.slider(
            "Min signal score",
            5, 10, 7,
            key="bt_min_score"
        )
        bt_capital = st.number_input(
            "Capital per trade (Rs)",
            value=10000,
            step=1000,
            key="bt_capital"
        )

    if st.button(
        "Run Backtest",
        type="primary",
        key="bt_run",
        use_container_width=True
    ):
        with st.spinner(
            f"Running backtest on {bt_stock} "
            f"({bt_months} months, {bt_tf})..."
        ):
            # Fetch historical data
            days_map = {"15m":30,"1h":60,"1d":bt_months*30}
            bt_days  = days_map.get(bt_tf, bt_months*30)
            end_bt   = datetime.now()
            start_bt = end_bt - timedelta(days=bt_days)

            bt_df = yf.download(
                bt_sym,
                start=start_bt,
                end=end_bt,
                interval=bt_tf,
                auto_adjust=True,
                progress=False
            )
            if isinstance(bt_df.columns, pd.MultiIndex):
                bt_df.columns = bt_df.columns.get_level_values(0)
            bt_df.dropna(inplace=True)

        if bt_df.empty or len(bt_df) < 100:
            st.error(
                f"Not enough data for {bt_stock}. "
                f"Try 1d timeframe or longer period."
            )
        else:
            # Run signals on rolling windows
            trades = []
            window_size = 55
            step_size   = 5

            prog_bt = st.progress(
                0, text="Scanning historical signals..."
            )
            total_windows = (len(bt_df)-window_size)//step_size

            for wi, start_i in enumerate(
                range(0, len(bt_df)-window_size, step_size)
            ):
                prog_bt.progress(
                    min(int(wi/total_windows*100), 99),
                    text=f"Scanning... {wi}/{total_windows}"
                )
                window_df = bt_df.iloc[start_i:start_i+window_size]
                try:
                    fake_lp = {
                        "ok": True,
                        "p": float(window_df["Close"].iloc[-1]),
                        "chg": 0, "chg_abs": 0,
                        "high": float(window_df["High"].iloc[-1]),
                        "low":  float(window_df["Low"].iloc[-1]),
                        "prev": float(window_df["Close"].iloc[-2])
                    }
                    sig_bt = compute_all(window_df, fake_lp)
                    if sig_bt is None:
                        continue

                    score = max(
                        sig_bt["up_score"],
                        sig_bt["dn_score"]
                    )
                    if score < bt_min_score:
                        continue

                    direction = sig_bt["direction"]
                    if direction not in ["UPTREND","DOWNTREND"]:
                        continue

                    entry_price = sig_bt["cp"]
                    sl = (sig_bt["sl_long"]
                          if direction=="UPTREND"
                          else sig_bt["sl_short"])
                    target = (sig_bt["tgt1"]
                              if direction=="UPTREND"
                              else sig_bt["tgt1s"])
                    signal_date = window_df.index[-1]

                    # Check outcome on next candles
                    future = bt_df.iloc[
                        start_i+window_size:
                        start_i+window_size+10
                    ]
                    if future.empty:
                        continue

                    outcome  = "OPEN"
                    exit_px  = float(future["Close"].iloc[-1])
                    exit_dt  = future.index[-1]

                    for _, row_f in future.iterrows():
                        if direction == "UPTREND":
                            if float(row_f["Low"]) <= sl:
                                outcome = "LOSS"
                                exit_px = sl
                                exit_dt = row_f.name
                                break
                            if float(row_f["High"]) >= target:
                                outcome = "WIN"
                                exit_px = target
                                exit_dt = row_f.name
                                break
                        else:
                            if float(row_f["High"]) >= sl:
                                outcome = "LOSS"
                                exit_px = sl
                                exit_dt = row_f.name
                                break
                            if float(row_f["Low"]) <= target:
                                outcome = "WIN"
                                exit_px = target
                                exit_dt = row_f.name
                                break

                    if outcome == "OPEN":
                        pnl_bt = (
                            (exit_px - entry_price)
                            if direction=="UPTREND"
                            else (entry_price - exit_px)
                        )
                        outcome = "WIN" if pnl_bt > 0 else "LOSS"
                    else:
                        pnl_bt = (
                            (exit_px - entry_price)
                            if direction=="UPTREND"
                            else (entry_price - exit_px)
                        )

                    pnl_pct = round(pnl_bt/entry_price*100, 2)

                    trades.append({
                        "Date":      signal_date.strftime("%d %b"),
                        "Direction": direction,
                        "Score":     score,
                        "Entry":     round(entry_price, 2),
                        "Exit":      round(exit_px, 2),
                        "P&L pts":   round(pnl_bt, 2),
                        "P&L %":     pnl_pct,
                        "Result":    outcome,
                    })
                except Exception:
                    continue

            prog_bt.empty()

            if not trades:
                st.warning(
                    f"No signals found with score {bt_min_score}+. "
                    "Try lowering the minimum score."
                )
            else:
                df_bt = pd.DataFrame(trades)
                wins  = df_bt[df_bt["Result"]=="WIN"]
                losses= df_bt[df_bt["Result"]=="LOSS"]
                wr_bt = round(len(wins)/len(df_bt)*100,1)
                avg_w = round(wins["P&L %"].mean(),2) if len(wins)>0 else 0
                avg_l = round(losses["P&L %"].mean(),2) if len(losses)>0 else 0
                pf_bt = round(
                    abs(wins["P&L %"].sum()) /
                    (abs(losses["P&L %"].sum())+0.001), 2
                )
                total_ret = round(df_bt["P&L %"].sum(), 2)

                # Summary metrics
                bm1,bm2,bm3,bm4,bm5 = st.columns(5)
                bm1.metric("Total Trades", len(df_bt))
                bm2.metric("Win Rate",     f"{wr_bt}%")
                bm3.metric("Profit Factor",f"{pf_bt}")
                bm4.metric("Avg Win",      f"{avg_w}%")
                bm5.metric("Avg Loss",     f"{avg_l}%")

                # Verdict
                if wr_bt >= 55 and pf_bt >= 1.5:
                    st.success(
                        f"🔥 Strong strategy! Win rate {wr_bt}% | "
                        f"Profit factor {pf_bt} | "
                        f"Total return {total_ret}% over "
                        f"{bt_months} months. "
                        f"This signal system is working well."
                    )
                elif wr_bt >= 45 and pf_bt >= 1.0:
                    st.warning(
                        f"📈 Decent strategy. Win rate {wr_bt}% | "
                        f"Profit factor {pf_bt}. "
                        f"Profitable but room to improve."
                    )
                else:
                    st.error(
                        f"⚠️ Weak results. Win rate {wr_bt}% | "
                        f"Profit factor {pf_bt}. "
                        f"Try increasing min score or "
                        f"changing timeframe."
                    )

                # Equity curve
                import plotly.graph_objects as go_bt
                cumulative = [100]
                for _, row_b in df_bt.iterrows():
                    cumulative.append(
                        cumulative[-1] * (1 + row_b["P&L %"]/100)
                    )

                fig_bt = go_bt.Figure()
                fig_bt.add_trace(go_bt.Scatter(
                    y=cumulative,
                    mode="lines",
                    line=dict(
                        color="#16a34a"
                        if cumulative[-1] >= 100
                        else "#dc2626",
                        width=2
                    ),
                    fill="tozeroy",
                    fillcolor=(
                        "rgba(22,163,74,0.1)"
                        if cumulative[-1] >= 100
                        else "rgba(220,38,38,0.1)"
                    ),
                    name="Equity Curve"
                ))
                fig_bt.add_hline(
                    y=100,
                    line_dash="dash",
                    line_color="#94a3b8",
                    annotation_text="Starting capital"
                )
                fig_bt.update_layout(
                    template="plotly_white",
                    height=280,
                    title=(
                        f"Equity Curve — "
                        f"{bt_stock} {bt_tf} "
                        f"(Score {bt_min_score}+)"
                    ),
                    yaxis_title="Portfolio value (start=100)",
                    margin=dict(l=10,r=10,t=40,b=10)
                )
                st.plotly_chart(fig_bt, use_container_width=True)

                # Win/Loss chart
                fig_wl = go_bt.Figure(go_bt.Bar(
                    x=df_bt["Date"],
                    y=df_bt["P&L %"],
                    marker_color=[
                        "#16a34a" if r=="WIN" else "#dc2626"
                        for r in df_bt["Result"]
                    ],
                    name="P&L %"
                ))
                fig_wl.update_layout(
                    template="plotly_white",
                    height=220,
                    title="Individual trade P&L %",
                    margin=dict(l=10,r=10,t=40,b=40),
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_wl, use_container_width=True)

                # Full trades table
                with st.expander("All trades"):
                    st.dataframe(
                        df_bt,
                        width='stretch',
                        hide_index=True
                    )

    else:
        st.info(
            "Configure settings above and click "
            "**Run Backtest** to test your strategy "
            "on historical data."
        )
        st.markdown("""
        ### How backtesting works

        The backtest takes every 15m/1h/1d candle from the past
        and runs your signal engine on it. When score crosses
        your minimum threshold it records a virtual trade.

        Then it checks the next 10 candles to see if the trade
        hit the target (WIN) or stop loss (LOSS).

        **What good results look like:**

        | Metric | Target |
        |--------|--------|
        | Win Rate | 45% or above |
        | Profit Factor | 1.5 or above |
        | Avg Win > Avg Loss | Always |

        **Important:** Backtesting on daily (1d) timeframe
        is most reliable. 15m backtests have many false
        signals due to noise.
        """)


# ── Auto refresh ──────────────────────────────────────────
if auto_rf:
    st.sidebar.success("🔄 Refreshing every 2 min...")
    time.sleep(120)
    st.rerun()