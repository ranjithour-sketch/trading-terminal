"""
ml_engine.py — Machine Learning Prediction Engine
===================================================
Uses RandomForest + feature engineering to predict
uptrend / downtrend with confidence scores.
No internet needed — trains on historical candle data.
"""
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 30+ technical features from OHLCV data.
    These are the inputs to the ML model.
    """
    feat = pd.DataFrame(index=df.index)
    c = df["Close"].squeeze().astype(float)
    h = df["High"].squeeze().astype(float)
    l = df["Low"].squeeze().astype(float)
    v = df["Volume"].squeeze().astype(float)
    o = df["Open"].squeeze().astype(float)

    # ── Trend indicators ──────────────────────────────────
    feat["ema9"]      = ta.trend.ema_indicator(c, 9)
    feat["ema21"]     = ta.trend.ema_indicator(c, 21)
    feat["ema50"]     = ta.trend.ema_indicator(c, 50)
    feat["ema9_21"]   = feat["ema9"] / (feat["ema21"] + 1e-9)
    feat["ema21_50"]  = feat["ema21"] / (feat["ema50"] + 1e-9)
    feat["price_ema9"]= c / (feat["ema9"] + 1e-9)

    # ── Momentum ──────────────────────────────────────────
    feat["rsi"]       = ta.momentum.rsi(c, 14)
    feat["rsi_5"]     = ta.momentum.rsi(c, 5)
    feat["stoch_k"]   = ta.momentum.stoch(h, l, c)
    feat["stoch_d"]   = ta.momentum.stoch_signal(h, l, c)
    feat["williams"]  = ta.momentum.williams_r(h, l, c)
    feat["roc"]       = ta.momentum.roc(c, 10)

    # ── MACD ──────────────────────────────────────────────
    feat["macd"]      = ta.trend.macd(c)
    feat["macd_sig"]  = ta.trend.macd_signal(c)
    feat["macd_hist"] = ta.trend.macd_diff(c)
    feat["macd_cross"]= (
        (feat["macd"] > feat["macd_sig"]).astype(int) -
        (feat["macd"] < feat["macd_sig"]).astype(int)
    )

    # ── Volatility ────────────────────────────────────────
    feat["atr"]       = ta.volatility.average_true_range(h, l, c, 14)
    feat["atr_pct"]   = feat["atr"] / (c + 1e-9) * 100
    feat["bb_upper"]  = ta.volatility.bollinger_hband(c, 20)
    feat["bb_lower"]  = ta.volatility.bollinger_lband(c, 20)
    feat["bb_pct"]    = ta.volatility.bollinger_pband(c, 20)
    feat["bb_width"]  = ta.volatility.bollinger_wband(c, 20)

    # ── Volume ────────────────────────────────────────────
    vol_ma            = v.rolling(20).mean()
    feat["vol_ratio"] = v / (vol_ma + 1e-9)
    feat["obv"]       = ta.volume.on_balance_volume(c, v)
    feat["obv_sma"]   = feat["obv"].rolling(10).mean()
    feat["obv_ratio"] = feat["obv"] / (feat["obv_sma"] + 1e-9)
    feat["cmf"]       = ta.volume.chaikin_money_flow(h, l, c, v, 20)
    feat["mfi"]       = ta.volume.money_flow_index(h, l, c, v, 14)

    # ── Trend strength ────────────────────────────────────
    feat["adx"]       = ta.trend.adx(h, l, c, 14)
    feat["adx_pos"]   = ta.trend.adx_pos(h, l, c, 14)
    feat["adx_neg"]   = ta.trend.adx_neg(h, l, c, 14)
    feat["adx_diff"]  = feat["adx_pos"] - feat["adx_neg"]

    # ── VWAP deviation ────────────────────────────────────
    vwap              = (c * v).cumsum() / v.cumsum()
    feat["vwap_dev"]  = (c - vwap) / (vwap + 1e-9) * 100

    # ── Price action features ─────────────────────────────
    feat["candle_body"]  = (c - o) / (c + 1e-9) * 100
    feat["upper_wick"]   = (h - c.combine(o, max)) / (c + 1e-9) * 100
    feat["lower_wick"]   = (c.combine(o, min) - l) / (c + 1e-9) * 100
    feat["hl_range"]     = (h - l) / (c + 1e-9) * 100

    # ── Returns ───────────────────────────────────────────
    feat["ret_1"]     = c.pct_change(1) * 100
    feat["ret_3"]     = c.pct_change(3) * 100
    feat["ret_5"]     = c.pct_change(5) * 100
    feat["ret_10"]    = c.pct_change(10) * 100

    # ── Consecutive candles ───────────────────────────────
    feat["consec_up"] = (c > o).astype(int).rolling(3).sum()
    feat["consec_dn"] = (c < o).astype(int).rolling(3).sum()

    return feat.dropna()


# ══════════════════════════════════════════════════════════
# LABEL GENERATION
# ══════════════════════════════════════════════════════════
def make_labels(df: pd.DataFrame,
                forward: int = 3,
                threshold: float = 0.5) -> pd.Series:
    """
    Creates labels for supervised learning:
    1 = UPTREND  (price rises > threshold% in next N candles)
    0 = DOWNTREND/SIDEWAYS
    """
    close  = df["Close"].squeeze().astype(float)
    future = close.shift(-forward)
    ret    = (future - close) / close * 100

    labels = pd.Series(0, index=close.index)
    labels[ret >  threshold] = 1   # Uptrend
    labels[ret < -threshold] = 2   # Downtrend
    return labels


# ══════════════════════════════════════════════════════════
# ML MODEL TRAINER
# ══════════════════════════════════════════════════════════
def train_model(df: pd.DataFrame) -> dict:
    """
    Trains a RandomForest model on historical data.
    Returns model, scaler, feature names, and accuracy metrics.
    Minimum 100 candles required.
    """
    if df is None or len(df) < 100:
        return {"ok": False, "reason": "Need 100+ candles"}

    try:
        features = build_features(df)
        labels   = make_labels(df, forward=3, threshold=0.3)

        # Align indices
        common   = features.index.intersection(labels.index)
        X        = features.loc[common].values
        y        = labels.loc[common].values

        # Remove last 3 rows (no future label)
        X = X[:-3]
        y = y[:-3]

        if len(X) < 60:
            return {"ok": False, "reason": "Not enough data after processing"}

        # Remove rows with NaN
        mask = ~np.isnan(X).any(axis=1)
        X    = X[mask]
        y    = y[mask]

        # ── Model 1: Random Forest ────────────────────────
        rf_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators  = 100,
                max_depth     = 8,
                min_samples_leaf = 5,
                random_state  = 42,
                n_jobs        = -1
            ))
        ])
        rf_model.fit(X, y)

        # ── Model 2: Gradient Boosting ────────────────────
        gb_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators  = 100,
                max_depth     = 4,
                learning_rate = 0.1,
                random_state  = 42
            ))
        ])
        gb_model.fit(X, y)

        # Cross-validation accuracy
        try:
            cv_scores = cross_val_score(rf_model, X, y,
                                        cv=3, scoring="accuracy")
            accuracy  = round(float(cv_scores.mean()) * 100, 1)
        except:
            accuracy  = None

        # Feature importances
        rf_clf   = rf_model.named_steps["clf"]
        feat_imp = dict(zip(
            features.columns,
            rf_clf.feature_importances_
        ))
        top_features = sorted(feat_imp.items(),
                              key=lambda x: x[1],
                              reverse=True)[:10]

        return {
            "ok":           True,
            "rf_model":     rf_model,
            "gb_model":     gb_model,
            "feature_cols": list(features.columns),
            "accuracy":     accuracy,
            "top_features": top_features,
            "n_samples":    len(X),
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ══════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════
def predict_next_move(df: pd.DataFrame,
                      model_data: dict) -> dict:
    """
    Uses trained models to predict the next 3-candle direction.
    Returns prediction, confidence, and explanation.
    """
    if not model_data.get("ok"):
        return {"ok": False}

    try:
        features   = build_features(df)
        feat_cols  = model_data["feature_cols"]

        # Get latest row of features
        latest_row = features.reindex(
            columns=feat_cols, fill_value=0
        ).iloc[-1:].values

        if np.isnan(latest_row).any():
            latest_row = np.nan_to_num(latest_row, nan=0.0)

        # RF prediction + probability
        rf   = model_data["rf_model"]
        gb   = model_data["gb_model"]

        rf_proba  = rf.predict_proba(latest_row)[0]
        gb_proba  = gb.predict_proba(latest_row)[0]

        # Ensemble: average both models
        classes   = rf.classes_
        proba_avg = (rf_proba + gb_proba) / 2

        # Map class index to label
        class_map = {0: "SIDEWAYS", 1: "UPTREND", 2: "DOWNTREND"}
        pred_idx  = int(np.argmax(proba_avg))
        pred_cls  = int(classes[pred_idx])
        pred_label= class_map.get(pred_cls, "SIDEWAYS")

        # Confidence = max probability
        confidence = round(float(np.max(proba_avg)) * 100, 1)

        # Individual probabilities
        prob_dict = {}
        for i, cls in enumerate(classes):
            prob_dict[class_map.get(int(cls),"?")] = round(
                float(proba_avg[i]) * 100, 1
            )

        # Reliability label
        reliability = (
            "🔥 Very High" if confidence >= 75 else
            "✅ High"      if confidence >= 60 else
            "📈 Moderate"  if confidence >= 50 else
            "😐 Low"
        )

        # Signal strength
        if pred_label == "UPTREND" and confidence >= 60:
            signal     = "BUY CE"
            sig_color  = "#00ff88"
        elif pred_label == "DOWNTREND" and confidence >= 60:
            signal     = "BUY PE"
            sig_color  = "#ff4455"
        else:
            signal     = "WAIT"
            sig_color  = "#ffcc00"

        # Top contributing features
        feat_vals   = dict(zip(feat_cols, latest_row[0]))
        top_contrib = []
        for fname, fimp in model_data["top_features"][:5]:
            fval = feat_vals.get(fname, 0)
            top_contrib.append({
                "feature":    fname,
                "importance": round(float(fimp) * 100, 1),
                "value":      round(float(fval), 3),
            })

        return {
            "ok":          True,
            "prediction":  pred_label,
            "signal":      signal,
            "sig_color":   sig_color,
            "confidence":  confidence,
            "reliability": reliability,
            "probabilities": prob_dict,
            "top_contrib": top_contrib,
            "model_accuracy": model_data["accuracy"],
            "n_trained":   model_data["n_samples"],
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ══════════════════════════════════════════════════════════
# REAL-TIME APPROXIMATION ENGINE
# ══════════════════════════════════════════════════════════
def approximate_realtime(df: pd.DataFrame,
                         live_price: float) -> dict:
    """
    Approximates real-time conditions by:
    1. Injecting the live price as the latest candle close
    2. Recalculating all indicators with the live price
    3. Detecting micro-trend changes in the last 5 candles
    4. Estimating where price is within current candle

    This bridges the 15-minute data delay gap.
    """
    if df is None or len(df) < 20 or live_price <= 0:
        return {"ok": False}

    try:
        c = df["Close"].squeeze().astype(float)
        h = df["High"].squeeze().astype(float)
        l = df["Low"].squeeze().astype(float)
        v = df["Volume"].squeeze().astype(float)

        last_close  = float(c.iloc[-1])
        last_high   = float(h.iloc[-1])
        last_low    = float(l.iloc[-1])
        prev_close  = float(c.iloc[-2])

        # ── 1. Price momentum since last candle ───────────
        since_close = round((live_price - last_close) /
                            last_close * 100, 3)
        since_prev  = round((live_price - prev_close) /
                            prev_close * 100, 3)

        # ── 2. Current candle position ────────────────────
        candle_range = last_high - last_low
        if candle_range > 0:
            candle_pos = round(
                (live_price - last_low) / candle_range * 100, 1
            )
        else:
            candle_pos = 50.0

        candle_zone = (
            "upper"  if candle_pos >= 70 else
            "middle" if candle_pos >= 30 else
            "lower"
        )

        # ── 3. Micro trend (last 5 candles + live) ────────
        recent_closes = list(c.tail(5)) + [live_price]
        micro_up   = sum(1 for i in range(1, len(recent_closes))
                        if recent_closes[i] > recent_closes[i-1])
        micro_down = sum(1 for i in range(1, len(recent_closes))
                        if recent_closes[i] < recent_closes[i-1])

        micro_trend = (
            "RISING"  if micro_up >= 4 else
            "FALLING" if micro_down >= 4 else
            "MIXED"
        )

        # ── 4. Recalculate RSI with live price ────────────
        c_live    = pd.concat([c, pd.Series([live_price])])
        rsi_live  = float(
            ta.momentum.rsi(c_live, 14).iloc[-1]
        )

        # ── 5. EMA with live price ────────────────────────
        ema9_live  = float(
            ta.trend.ema_indicator(c_live, 9).iloc[-1]
        )
        ema21_live = float(
            ta.trend.ema_indicator(c_live, 21).iloc[-1]
        )

        # ── 6. VWAP with live price ───────────────────────
        v_live    = pd.concat([v, pd.Series([float(v.iloc[-1])])])
        c_vwap    = pd.concat([c, pd.Series([live_price])])
        vwap_live = float(
            (c_vwap * v_live).cumsum().iloc[-1] /
            v_live.cumsum().iloc[-1]
        )
        vwap_dev  = round(
            (live_price - vwap_live) / vwap_live * 100, 2
        )

        # ── 7. Live signal assessment ─────────────────────
        live_bullish = 0
        live_bearish = 0
        live_signals = []

        if since_close > 0.2:
            live_bullish += 1
            live_signals.append(
                f"📈 Price up {since_close:+.2f}% since last candle"
            )
        elif since_close < -0.2:
            live_bearish += 1
            live_signals.append(
                f"📉 Price down {since_close:+.2f}% since last candle"
            )

        if micro_trend == "RISING":
            live_bullish += 2
            live_signals.append(
                "🔼 Micro trend RISING — 4/5 recent candles up"
            )
        elif micro_trend == "FALLING":
            live_bearish += 2
            live_signals.append(
                "🔽 Micro trend FALLING — 4/5 recent candles down"
            )

        if live_price > ema9_live:
            live_bullish += 1
            live_signals.append(
                f"✅ Live price above EMA9 (₹{ema9_live:,.2f})"
            )
        else:
            live_bearish += 1
            live_signals.append(
                f"❌ Live price below EMA9 (₹{ema9_live:,.2f})"
            )

        if live_price > vwap_live:
            live_bullish += 1
            live_signals.append(
                f"✅ Live price above VWAP ({vwap_dev:+.2f}%)"
            )
        else:
            live_bearish += 1
            live_signals.append(
                f"❌ Live price below VWAP ({vwap_dev:+.2f}%)"
            )

        if 55 < rsi_live < 70:
            live_bullish += 1
            live_signals.append(
                f"✅ Live RSI bullish zone ({rsi_live:.1f})"
            )
        elif 30 < rsi_live < 45:
            live_bearish += 1
            live_signals.append(
                f"✅ Live RSI bearish zone ({rsi_live:.1f})"
            )
        elif rsi_live >= 70:
            live_signals.append(
                f"⚠️ Live RSI overbought ({rsi_live:.1f}) — "
                "caution on new CE"
            )
        elif rsi_live <= 30:
            live_signals.append(
                f"⚠️ Live RSI oversold ({rsi_live:.1f}) — "
                "caution on new PE"
            )

        if candle_zone == "upper":
            live_bullish += 1
            live_signals.append(
                f"📊 Price in upper {candle_pos:.0f}% of candle — bullish"
            )
        elif candle_zone == "lower":
            live_bearish += 1
            live_signals.append(
                f"📊 Price in lower {candle_pos:.0f}% of candle — bearish"
            )

        live_bias = (
            "BULLISH"  if live_bullish > live_bearish + 1 else
            "BEARISH"  if live_bearish > live_bullish + 1 else
            "NEUTRAL"
        )

        bias_color = (
            "#00ff88" if live_bias == "BULLISH" else
            "#ff4455" if live_bias == "BEARISH" else
            "#ffcc00"
        )

        return {
            "ok":           True,
            "live_price":   live_price,
            "last_close":   last_close,
            "since_close":  since_close,
            "since_prev":   since_prev,
            "candle_pos":   candle_pos,
            "candle_zone":  candle_zone,
            "micro_trend":  micro_trend,
            "rsi_live":     round(rsi_live, 1),
            "ema9_live":    round(ema9_live, 2),
            "ema21_live":   round(ema21_live, 2),
            "vwap_live":    round(vwap_live, 2),
            "vwap_dev":     vwap_dev,
            "live_bias":    live_bias,
            "bias_color":   bias_color,
            "live_signals": live_signals,
            "bull_count":   live_bullish,
            "bear_count":   live_bearish,
        }

    except Exception as e:
        return {"ok": False, "reason": str(e)}
