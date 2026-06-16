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


def fetch_data(ticker, period="6mo"):
    """6mo = ~125 trading days — enough for EMA50 to converge properly."""
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


def compute_ivr(vix_df):
    """
    IVR (IV Rank) = where current VIX sits within its 52-week high/low range.
    Must use 1-year data — 3-month data gives a compressed, inaccurate rank.
    VIX is used as IV proxy for all ETFs (SPX-derived; QQQ/IWM have higher actual IV).
    """
    vix = vix_df["Close"].squeeze()
    current_vix = float(vix.iloc[-1])
    high_52 = float(vix.max())
    low_52  = float(vix.min())
    ivr = int((current_vix - low_52) / (high_52 - low_52) * 100) if high_52 != low_52 else 50
    return round(current_vix, 1), ivr


def select_strategy(bias, ivr, iv_pct):
    """
    IVR >= 50 → IV is elevated → sell premium (credit spreads)
    IVR <  50 → IV is cheap   → buy premium (debit spreads)
    """
    if bias == "BULL":
        return "Bull Put Spread"  if ivr >= 50 else "Bull Call Spread"
    else:
        return "Bear Call Spread" if ivr >= 50 else "Bear Put Spread"


def select_strikes(price, strategy, iv_pct):
    """
    Strike width based on 1 SD move over DTE.
    Labels show B (Buy leg) / S (Sell leg) for clarity.
    """
    daily_sd = price * (iv_pct / 100) / np.sqrt(252)
    width = round(daily_sd * np.sqrt(EXPIRY_DAYS_TARGET) * 1.5, 0)
    width = max(width, price * 0.02)   # minimum 2% width

    if strategy == "Bull Call Spread":
        # Buy lower ATM call, Sell higher OTM call
        buy_strike  = round(price * 1.001, 0)
        sell_strike = round(buy_strike + width, 0)
        return f"B${int(buy_strike)}C / S${int(sell_strike)}C"

    elif strategy == "Bull Put Spread":
        # Sell higher OTM put, Buy lower put for protection
        sell_strike = round(price * 0.985, 0)
        buy_strike  = round(sell_strike - width, 0)
        return f"S${int(sell_strike)}P / B${int(buy_strike)}P"

    elif strategy == "Bear Put Spread":
        # Buy higher ATM put, Sell lower OTM put
        buy_strike  = round(price * 0.999, 0)
        sell_strike = round(buy_strike - width, 0)
        return f"B${int(buy_strike)}P / S${int(sell_strike)}P"

    else:  # Bear Call Spread
        # Sell lower OTM call, Buy higher call for protection
        sell_strike = round(price * 1.015, 0)
        buy_strike  = round(sell_strike + width, 0)
        return f"S${int(sell_strike)}C / B${int(buy_strike)}C"


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

    headers = ["Instrument", "Price", "Chg%", "Bias", "Conf (pts)", "VIX / IVR", "Strategy", "Strikes (B/S)", "Expiry"]
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

    # IVR requires 1-year VIX data for a proper 52-week range
    vix_df = fetch_data(VIX_TICKER, period="1y")
    iv_pct, ivr_global = compute_ivr(vix_df)
    print(f"VIX: {iv_pct}%  IVR: {ivr_global}")

    signals = []
    for ticker, meta in TICKERS.items():
        try:
            df = fetch_data(ticker, period="6mo")
            bias, conf, price, chg, signal_meta = compute_bias(df)
            strategy = select_strategy(bias, ivr_global, iv_pct)
            strikes  = select_strikes(price, strategy, iv_pct)

            signals.append({
                "label":     meta["label"],
                "price":     price,
                "chg":       chg,
                "bias":      bias,
                "conf":      conf,
                "iv":        iv_pct,
                "ivr":       ivr_global,
                "strategy":  strategy,
                "strikes":   strikes,
                "expiry":    expiry,
                "vol_ratio": signal_meta["vol_ratio"],
                "atr_pct":   signal_meta["atr_pct"],
                "points":    signal_meta["points"],
                "breakdown": signal_meta["breakdown"],
            })
            print(f"  {meta['label']:12s} {bias:4s} {conf}%  pts={signal_meta['points']}/7  "
                  f"vol={signal_meta['vol_ratio']:.1f}x  atr={signal_meta['atr_pct']:.1f}%  {strikes}")
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
