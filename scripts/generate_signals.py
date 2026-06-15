"""
ML Signals Generator
Runs via GitHub Actions daily at 9:30 AM EDT (weekdays).
Fetches live prices, computes bias + strategy, writes ml_signals.pine.
"""

import yfinance as yf
import numpy as np
from datetime import datetime, date
import pytz
import json
import os

TICKERS = {
    "SPY":        {"label": "SPY",        "options_root": "SPY"},
    "QQQ":        {"label": "QQQ",        "options_root": "QQQ"},
    "DIA":        {"label": "DJX (DIA)",  "options_root": "DIA"},
    "IWM":        {"label": "MRUT (IWM)", "options_root": "IWM"},
}

VIX_TICKER = "^VIX"
EXPIRY_DAYS_TARGET = 15   # target DTE for options


def fetch_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df


def compute_bias(df):
    close = df["Close"].squeeze()
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    last = float(close.iloc[-1])
    e20  = float(ema20.iloc[-1])
    e50  = float(ema50.iloc[-1])

    # momentum: last 5 bars
    mom5 = float(close.iloc[-1]) - float(close.iloc[-6])

    bull_points = 0
    bull_points += 2 if last > e20 else -2
    bull_points += 1 if last > e50 else -1
    bull_points += 1 if e20 > e50 else -1
    bull_points += 1 if mom5 > 0 else -1

    bias = "BULL" if bull_points > 0 else "BEAR"
    conf = min(99, int(50 + abs(bull_points) / 5 * 49))
    chg_pct = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100
    return bias, conf, last, chg_pct, e20, e50


def compute_ivr(vix_df):
    vix = vix_df["Close"].squeeze()
    current_vix = float(vix.iloc[-1])
    high_52 = float(vix.max())
    low_52  = float(vix.min())
    ivr = int((current_vix - low_52) / (high_52 - low_52) * 100) if high_52 != low_52 else 50
    return round(current_vix, 1), ivr


def select_strategy(bias, ivr, iv_pct):
    """
    High IVR (>50) → sell premium (credit spreads)
    Low IVR (<50)  → buy premium (debit spreads)
    """
    if bias == "BULL":
        if ivr >= 50:
            return "Bull Put Spread"
        else:
            return "Bull Call Spread"
    else:
        if ivr >= 50:
            return "Bear Call Spread"
        else:
            return "Bear Put Spread"


def select_strikes(price, strategy, iv_pct):
    """Rough strike selection based on ~1 SD move."""
    daily_sd = price * (iv_pct / 100) / np.sqrt(252)
    width = round(daily_sd * np.sqrt(EXPIRY_DAYS_TARGET) * 1.5, 0)
    width = max(width, price * 0.02)  # min 2% width

    if strategy == "Bull Call Spread":
        short = round(price * 1.001, 0)   # just ATM
        long  = round(short + width, 0)
        return f"${int(short)}C / ${int(long)}C"
    elif strategy == "Bull Put Spread":
        short = round(price * 0.985, 0)   # ~1.5% OTM put
        long  = round(short - width, 0)
        return f"${int(short)}P / ${int(long)}P"
    elif strategy == "Bear Put Spread":
        short = round(price * 0.999, 0)   # just ATM
        long  = round(short - width, 0)
        return f"${int(short)}P / ${int(long)}P"
    else:  # Bear Call Spread
        short = round(price * 1.015, 0)
        long  = round(short + width, 0)
        return f"${int(short)}C / ${int(long)}C"


def next_expiry():
    today = date.today()
    days_ahead = EXPIRY_DAYS_TARGET
    exp = today
    # advance to target DTE, land on a Friday
    from datetime import timedelta
    exp = today + timedelta(days=days_ahead)
    # roll to nearest Friday
    while exp.weekday() != 4:
        exp += timedelta(days=1)
    return exp.strftime("%b %-d")


def generate_pine(signals, run_date):
    rows = len(signals)
    lines = []
    lines.append(f'//@version=6')
    lines.append(f'indicator("ML Signals — {run_date}", overlay=true)')
    lines.append('')
    lines.append('if barstate.islast')
    lines.append(f'    t = table.new(position.top_right, 9, {rows + 1},')
    lines.append('         border_width=1, border_color=color.new(color.white,70),')
    lines.append('         frame_width=1, frame_color=color.new(color.white,50))')
    lines.append('')

    headers = ["Instrument", "Price", "Chg%", "Bias", "Conf", "IV / IVR", "Strategy", "Strikes", "Expiry"]
    for col, h in enumerate(headers):
        lines.append(f'    table.cell(t, {col}, 0, "{h}", text_color=color.yellow, bgcolor=color.new(color.black,20), text_size=size.small)')
    lines.append('')

    for row_idx, s in enumerate(signals, start=1):
        r = row_idx
        chg_color = "color.green" if s["chg"] >= 0 else "color.red"
        chg_str = f"+{s['chg']:.2f}%" if s["chg"] >= 0 else f"{s['chg']:.2f}%"
        bias_bg = "color.new(color.green,20)" if s["bias"] == "BULL" else "color.new(color.red,20)"
        ivr_color = "color.red" if s["ivr"] >= 50 else "color.green"
        price_str = f"${s['price']:,.2f}"

        lines.append(f'    table.cell(t, 0, {r}, "{s["label"]}", text_color=color.white, bgcolor=color.new(color.navy,30))')
        lines.append(f'    table.cell(t, 1, {r}, "{price_str}", text_color=color.white)')
        lines.append(f'    table.cell(t, 2, {r}, "{chg_str}", text_color={chg_color})')
        lines.append(f'    table.cell(t, 3, {r}, "{s["bias"]}", text_color=color.white, bgcolor={bias_bg})')
        lines.append(f'    table.cell(t, 4, {r}, "{s["conf"]}%", text_color=color.white)')
        lines.append(f'    table.cell(t, 5, {r}, "{s["iv"]}%  IVR {s["ivr"]}", text_color={ivr_color})')
        lines.append(f'    table.cell(t, 6, {r}, "{s["strategy"]}", text_color=color.lime)')
        lines.append(f'    table.cell(t, 7, {r}, "{s["strikes"]}", text_color=color.white)')
        lines.append(f'    table.cell(t, 8, {r}, "{s["expiry"]}", text_color=color.yellow)')
        lines.append('')

    return '\n'.join(lines)


def main():
    eastern = pytz.timezone("America/Toronto")
    run_date = datetime.now(eastern).strftime("%Y-%m-%d")
    expiry = next_expiry()

    print(f"Generating ML signals for {run_date} | Expiry target: {expiry}")

    vix_df = fetch_data(VIX_TICKER)
    iv_pct, ivr_global = compute_ivr(vix_df)

    signals = []
    for ticker, meta in TICKERS.items():
        try:
            df = fetch_data(ticker)
            bias, conf, price, chg, e20, e50 = compute_bias(df)
            strategy = select_strategy(bias, ivr_global, iv_pct)
            strikes = select_strikes(price, strategy, iv_pct)

            signals.append({
                "label":    meta["label"],
                "price":    price,
                "chg":      chg,
                "bias":     bias,
                "conf":     conf,
                "iv":       iv_pct,
                "ivr":      ivr_global,
                "strategy": strategy,
                "strikes":  strikes,
                "expiry":   expiry,
            })
            print(f"  {meta['label']:12s} {bias:4s} {conf}%  {strategy:20s} {strikes}")
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    pine_code = generate_pine(signals, run_date)

    os.makedirs("output", exist_ok=True)
    with open("output/ml_signals.pine", "w") as f:
        f.write(pine_code)

    with open("output/signals.json", "w") as f:
        json.dump({"date": run_date, "signals": signals}, f, indent=2)

    print(f"\nWrote output/ml_signals.pine and output/signals.json")


if __name__ == "__main__":
    main()
