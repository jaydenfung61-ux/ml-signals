"""
ML Signals Generator
Runs via GitHub Actions at 9:30 AM, 2:30 PM, and 4:05 PM EDT (weekdays).
Fetches live prices, computes bias + strategy, writes ml_signals.pine.

Known limitation: IV/IVR uses VIX as a proxy for all ETFs.
VIX is SPX-derived; QQQ/IWM typically have higher actual IV.
"""

import yfinance as yf
import numpy as np
from datetime import datetime, date, timedelta
import pytz
import json
import os

TICKERS = {
    "SPY":  {"label": "SPY",        "options_root": "SPY"},
    "QQQ":  {"label": "QQQ",        "options_root": "QQQ"},
    "DIA":  {"label": "DJX (DIA)",  "options_root": "DIA"},
    "IWM":  {"label": "MRUT (IWM)", "options_root": "IWM"},
}

VIX_TICKER     = "^VIX"
EXPIRY_DAYS_TARGET = 15


def fetch_data(ticker, period="1y"):
    """1y = ~252 trading days — needed for EMA50 convergence AND 52-week IVR range."""
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df


def compute_bias(df):
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()

    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    last  = float(close.iloc[-1])
    e20   = float(ema20.iloc[-1])
    e50   = float(ema50.iloc[-1])

    # 5-bar momentum
    mom5 = float(close.iloc[-1]) - float(close.iloc[-6])

    # Volume: last confirmed day vs 20-day avg
    vol_last  = float(volume.iloc[-1])
    vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = vol_last / vol_avg20 if vol_avg20 > 0 else 1.0

    # ATR (14-day simple average true range as % of price)
    prev_close = close.shift(1)
    tr = np.maximum(
        (high - low).values,
        np.maximum(
            abs(high.values - prev_close.values),
            abs(low.values  - prev_close.values)
        )
    )
    atr14   = float(np.nanmean(tr[-14:]))
    atr_pct = (atr14 / last) * 100

    # --- Scoring: max +7, min -7 ---
    points = 0
    points += 2 if last > e20 else -2       # short-term trend (weighted 2x)
    points += 1 if last > e50 else -1       # medium-term trend
    points += 1 if e20 > e50 else -1        # MA alignment (golden/death cross)
    points += 1 if mom5 > 0 else -1         # 5-bar price momentum
    points += 1 if vol_ratio >= 1.0 else -1 # volume conviction

    # ATR: low volatility = cleaner signal; high = noisy
    if atr_pct < 1.5:
        points += 1    # calm market, higher confidence
    elif atr_pct > 3.0:
        points -= 1    # too volatile, lower confidence
    # 1.5–3.0%: neutral, no change

    # Fix: ties (0 pts) default to BULL — don't fight the broader trend
    bias = "BULL" if points >= 0 else "BEAR"

    # Cap at 82% — no rules-based signal is ever >82% reliable
    conf = min(82, max(35, int(50 + abs(points) / 7 * 40)))

    chg_pct = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100

    meta = {
        "vol_ratio": round(vol_ratio, 2),
        "atr_pct":   round(atr_pct, 2),
        "points":    points,
        "breakdown": {
            "price_vs_ema20":   last > e20,
            "price_vs_ema50":   last > e50,
            "ema20_vs_ema50":   e20 > e50,
            "momentum_5bar":    mom5 > 0,
            "volume_above_avg": vol_ratio >= 1.0,
            "atr_calm":         atr_pct < 1.5,
        }
    }

    return bias, conf, last, chg_pct, meta


def compute_hv_ivr(df):
    """
    Per-ETF historical volatility (HV20) and its 52-week IV Rank.

    HV20 = annualized 20-day realized volatility of log returns.
    Each ETF gets its own IV and IVR — QQQ/IWM naturally show higher
    volatility than SPY/DIA, so strategies can differ across instruments.
    Requires 1-year of data for a meaningful 52-week range.
    """
    close = df["Close"].squeeze()
    log_ret = np.log(close / close.shift(1))
    hv20_series = log_ret.rolling(20).std() * np.sqrt(252) * 100

    current_hv = float(hv20_series.iloc[-1])
    high_52    = float(hv20_series.max())
    low_52     = float(hv20_series.min())
    ivr = int((current_hv - low_52) / (high_52 - low_52) * 100) if high_52 != low_52 else 50

    return round(current_hv, 1), max(0, min(100, ivr))


def compute_vix(vix_df):
    """Return latest VIX close — shown in output for reference only."""
    vix = vix_df["Close"].squeeze()
    return round(float(vix.iloc[-1]), 1)


def select_strategy(bias, ivr, iv_pct):
    """
    IVR >= 50 → IV is elevated → sell premium (credit spreads)
    IVR <  50 → IV is cheap   → buy premium (debit spreads)
    """
    if bias == "BULL":
        return "Bull Put Spread"  if ivr >= 50 else "Bull Call Spread"
    else:
        return "Bear Call Spread" if ivr >= 50 else "Bear Put Spread"


def select_strikes(price, strategy, hv_pct):
    """
    Strike selection scaled to each ETF's own volatility over DTE.

    Short strike: ~0.5 SD OTM (approx 30-delta)
    Spread width: ~0.4 SD, rounded to nearest $5, capped at 5% of price
    This keeps QQQ/IWM strikes proportional to their higher HV vs SPY/DIA.
    """
    sd_move = price * (hv_pct / 100) * np.sqrt(EXPIRY_DAYS_TARGET / 252)

    # Width: 0.4 SD, rounded to $5 increments, min $5, max 5% of price
    raw_width = sd_move * 0.4
    width = max(5.0, min(price * 0.05, raw_width))
    width = round(width / 5) * 5
    width = max(5, width)

    # Short strike distance: 0.5 SD OTM (≈30 delta), min 1% of price
    otm = max(round(sd_move * 0.5), round(price * 0.01))

    if strategy == "Bull Put Spread":
        sell = round(price - otm)
        buy  = round(sell - width)
        return f"S${int(sell)}P / B${int(buy)}P"

    elif strategy == "Bull Call Spread":
        buy  = round(price)
        sell = round(buy + otm + width * 0.5)
        return f"B${int(buy)}C / S${int(sell)}C"

    elif strategy == "Bear Put Spread":
        buy  = round(price)
        sell = round(buy - otm - width * 0.5)
        return f"B${int(buy)}P / S${int(sell)}P"

    else:  # Bear Call Spread
        sell = round(price + otm)
        buy  = round(sell + width)
        return f"S${int(sell)}C / B${int(buy)}C"


def next_expiry():
    exp = date.today() + timedelta(days=EXPIRY_DAYS_TARGET)
    while exp.weekday() != 4:   # roll to nearest Friday
        exp += timedelta(days=1)
    return exp.strftime("%b %-d")


def run_label(dt_eastern):
    """Tag the signal with which daily run generated it."""
    h = dt_eastern.hour
    if h < 13:
        return "Open"
    elif h < 19:
        return "Midday"
    else:
        return "Close"


def generate_pine(signals, run_date, run_tag):
    rows = len(signals)
    lines = [
        "//@version=6",
        f'indicator("ML Signals — {run_date} ({run_tag})", overlay=true)',
        "",
        "if barstate.islast",
        f"    t = table.new(position.top_right, 9, {rows + 1},",
        "         border_width=1, border_color=color.new(color.white,70),",
        "         frame_width=1, frame_color=color.new(color.white,50))",
        "",
    ]

    headers = ["Instrument", "Price", "Chg%", "Bias", "Conf (pts)", "HV / IVR", "Strategy", "Strikes (B/S)", "Expiry"]
    for col, h in enumerate(headers):
        lines.append(f'    table.cell(t, {col}, 0, "{h}", text_color=color.yellow, bgcolor=color.new(color.black,20), text_size=size.small)')
    lines.append("")

    for row_idx, s in enumerate(signals, start=1):
        r = row_idx
        chg_color = "color.green" if s["chg"] >= 0 else "color.red"
        chg_str   = f"+{s['chg']:.2f}%" if s["chg"] >= 0 else f"{s['chg']:.2f}%"
        bias_bg   = "color.new(color.green,20)" if s["bias"] == "BULL" else "color.new(color.red,20)"
        ivr_color = "color.red" if s["ivr"] >= 50 else "color.green"
        conf_color = "color.green" if s["conf"] >= 65 else "color.orange" if s["conf"] >= 55 else "color.red"
        price_str = f"${s['price']:,.2f}"
        conf_str  = f"{s['conf']}% ({s['points']}/7)"

        lines += [
            f'    table.cell(t, 0, {r}, "{s["label"]}", text_color=color.white, bgcolor=color.new(color.navy,30))',
            f'    table.cell(t, 1, {r}, "{price_str}", text_color=color.white)',
            f'    table.cell(t, 2, {r}, "{chg_str}", text_color={chg_color})',
            f'    table.cell(t, 3, {r}, "{s["bias"]}", text_color=color.white, bgcolor={bias_bg})',
            f'    table.cell(t, 4, {r}, "{conf_str}", text_color={conf_color})',
            f'    table.cell(t, 5, {r}, "{s["iv"]}%  IVR {s["ivr"]}", text_color={ivr_color})',
            f'    table.cell(t, 6, {r}, "{s["strategy"]}", text_color=color.lime)',
            f'    table.cell(t, 7, {r}, "{s["strikes"]}", text_color=color.white)',
            f'    table.cell(t, 8, {r}, "{s["expiry"]}", text_color=color.yellow)',
            "",
        ]

    return "\n".join(lines)


def main():
    eastern  = pytz.timezone("America/Toronto")
    now_east = datetime.now(eastern)
    run_date = now_east.strftime("%Y-%m-%d")
    run_tag  = run_label(now_east)
    expiry   = next_expiry()

    print(f"ML Signals — {run_date} ({run_tag}) | Expiry: {expiry}")

    vix_df  = fetch_data(VIX_TICKER, period="1y")
    vix_now = compute_vix(vix_df)
    print(f"VIX: {vix_now}% (reference only — each ETF uses its own HV/IVR)")

    signals = []
    for ticker, meta in TICKERS.items():
        try:
            df = fetch_data(ticker, period="1y")
            bias, conf, price, chg, signal_meta = compute_bias(df)
            iv_pct, ivr = compute_hv_ivr(df)          # per-ETF HV and IVR
            strategy = select_strategy(bias, ivr, iv_pct)
            strikes  = select_strikes(price, strategy, iv_pct)

            signals.append({
                "label":     meta["label"],
                "price":     price,
                "chg":       chg,
                "bias":      bias,
                "conf":      conf,
                "iv":        iv_pct,
                "ivr":       ivr,
                "strategy":  strategy,
                "strikes":   strikes,
                "expiry":    expiry,
                "vol_ratio": signal_meta["vol_ratio"],
                "atr_pct":   signal_meta["atr_pct"],
                "points":    signal_meta["points"],
                "breakdown": signal_meta["breakdown"],
            })
            print(f"  {meta['label']:12s} {bias:4s} {conf}%  pts={signal_meta['points']}/7  "
                  f"HV={iv_pct}% IVR={ivr}  vol={signal_meta['vol_ratio']:.1f}x  atr={signal_meta['atr_pct']:.1f}%  {strikes}")
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    pine_code = generate_pine(signals, run_date, run_tag)

    os.makedirs("output", exist_ok=True)
    with open("output/ml_signals.pine", "w") as f:
        f.write(pine_code)
    with open("output/signals.json", "w") as f:
        json.dump({"date": run_date, "run": run_tag, "signals": signals}, f, indent=2)

    print(f"\nWrote output/ml_signals.pine and output/signals.json")


if __name__ == "__main__":
    main()
