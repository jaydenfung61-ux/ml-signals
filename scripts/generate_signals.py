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

    # Volume: today vs 20-day avg
    vol_today  = float(volume.iloc[-1])
    vol_avg20  = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio  = vol_today / vol_avg20 if vol_avg20 > 0 else 1.0

    # ATR (14-day true range as % of price)
    prev_close = close.shift(1)
    tr = np.maximum(
        (high - low).values,
        np.maximum(
            abs(high.values - prev_close.values),
            abs(low.values  - prev_close.values)
        )
    )
    atr14    = float(np.nanmean(tr[-14:]))
    atr_pct  = (atr14 / last) * 100

    # --- Scoring (max 7 pts, min -7 pts) ---
    points = 0
    points += 2 if last > e20 else -2          # trend (weighted)
    points += 1 if last > e50 else -1          # medium-term trend
    points += 1 if e20 > e50 else -1           # MA alignment
    points += 1 if mom5 > 0 else -1            # 5-bar momentum
    points += 1 if vol_ratio >= 1.0 else -1    # volume conviction
    # ATR: calm market = confident signal, volatile = noisy
    if atr_pct < 1.5:
        points += 1
    elif atr_pct > 3.0:
        points -= 1
    # else neutral (0) between 1.5–3%

    bias = "BULL" if points > 0 else "BEAR"
    # Cap at 82% — no signal is ever >82% reliable
    conf = min(82, max(35, int(50 + abs(points) / 7 * 40)))

    chg_pct = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100

    meta = {
        "vol_ratio": round(vol_ratio, 2),
        "atr_pct":   round(atr_pct, 2),
        "points":    points,
        "breakdown": {
            "price_vs_ema20": last > e20,
            "price_vs_ema50": last > e50,
            "ema20_vs_ema50": e20 > e50,
            "momentum_5bar":  mom5 > 0,
            "volume_above_avg": vol_ratio >= 1.0,
            "atr_calm":       atr_pct < 1.5,
        }
    }

    return bias, conf, last, chg_pct, meta


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
        conf_color = "color.green" if s["conf"] >= 65 else "color.orange" if s["conf"] >= 55 else "color.red"
        lines.append(f'    table.cell(t, 4, {r}, "{s["conf"]}% ({s["points"]}/7)", text_color={conf_color})')
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
            bias, conf, price, chg, signal_meta = compute_bias(df)
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
                "vol_ratio":  signal_meta["vol_ratio"],
                "atr_pct":    signal_meta["atr_pct"],
                "points":     signal_meta["points"],
                "breakdown":  signal_meta["breakdown"],
            })
            bd = signal_meta["breakdown"]
            checks = sum(1 for v in bd.values() if v)
            print(f"  {meta['label']:12s} {bias:4s} {conf}%  pts={signal_meta['points']}/7  vol={signal_meta['vol_ratio']:.1f}x  atr={signal_meta['atr_pct']:.1f}%  {strategy}")
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
