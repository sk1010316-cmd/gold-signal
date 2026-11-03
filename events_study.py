# -*- coding: utf-8 -*-
"""
事件反應研究：某類市場事件發生後，金價接下來怎麼走（用歷史資料統計，不靠感覺）

日線事件（2006 至今）：原油暴跌（分衰退恐慌型/供給型）、原油暴漲、美股大跌、VIX 暴衝、
                    美元急升急跌、美債殖利率急升急降、金價自己暴漲暴跌 → 看金價 1/3/5 天後
小時事件（近 2 年）：原油、美元、金價 1 小時急動 → 看金價 1/4 小時後

study()        → 統計結果（存成 events.json）
detect_live()  → 現在有沒有事件正在發生，附上歷史統計與建議方向
"""
import datetime as dt
import numpy as np
import pandas as pd

TICKERS = {"gold": "GC=F", "oil": "CL=F", "spx": "^GSPC", "vix": "^VIX", "dxy": "DX-Y.NYB", "tnx": "^TNX"}
HOURLY_TICKERS = {"gold": "GC=F", "oil": "CL=F", "dxy": "DX-Y.NYB"}
MIN_N = 12          # 少於 12 次不給方向
BULL_P, BEAR_P = 0.60, 0.40

# key, 名稱, 條件, 理論上（經濟邏輯）對金價的方向
DAILY_EVENTS = [
    ("oil_crash_panic", "原油暴跌 + 美股大跌（衰退恐慌型）", lambda r: (r.oil <= -0.05) & (r.spx <= -0.02),
     "恐慌時先賣金補保證金，之後降息預期才拉金價"),
    ("oil_crash_supply", "原油暴跌、美股沒大跌（供給過剩型）", lambda r: (r.oil <= -0.05) & (r.spx > -0.02),
     "通膨預期下降，略偏空"),
    ("oil_spike", "原油暴漲 ≥5%", lambda r: r.oil >= 0.05, "通膨＋地緣風險，偏多"),
    ("spx_crash", "美股大跌 ≥3%", lambda r: r.spx <= -0.03, "避險 vs 流動性賣壓，方向看歷史"),
    ("vix_spike", "VIX 單日暴衝 ≥25%", lambda r: r.vix >= 0.25, "恐慌升溫，常先跌後漲"),
    ("dxy_up", "美元急升 ≥1%", lambda r: r.dxy >= 0.01, "美元強，偏空"),
    ("dxy_down", "美元急跌 ≥1%", lambda r: r.dxy <= -0.01, "美元弱，偏多"),
    ("yield_up", "10 年美債殖利率急升 ≥15bp", lambda r: r.tnx_bp >= 15, "持有黃金成本上升，偏空"),
    ("yield_down", "10 年美債殖利率急降 ≥15bp", lambda r: r.tnx_bp <= -15, "降息預期，偏多"),
    ("gold_crash", "金價單日暴跌 ≥3%", lambda r: r.gold <= -0.03, "看是續跌還是反彈"),
    ("gold_spike", "金價單日暴漲 ≥3%", lambda r: r.gold >= 0.03, "看是續漲還是回吐"),
]
HOURLY_EVENTS = [
    ("oil_1h_drop", "原油 1 小時急跌 ≥2%", lambda r: r.oil <= -0.02, ""),
    ("oil_1h_jump", "原油 1 小時急漲 ≥2%", lambda r: r.oil >= 0.02, ""),
    ("dxy_1h_up", "美元 1 小時急升 ≥0.4%", lambda r: r.dxy >= 0.004, ""),
    ("dxy_1h_down", "美元 1 小時急跌 ≥0.4%", lambda r: r.dxy <= -0.004, ""),
    ("gold_1h_drop", "金價 1 小時急跌 ≥1%", lambda r: r.gold <= -0.01, ""),
    ("gold_1h_jump", "金價 1 小時急漲 ≥1%", lambda r: r.gold >= 0.01, ""),
]


# ====== 資料 ======
def _hist(ticker, period, interval):
    import yfinance as yf
    s = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)["Close"]
    s.index = pd.to_datetime(s.index)
    s.index = s.index.tz_convert("UTC") if s.index.tz is not None else s.index.tz_localize("UTC")
    if interval == "1d":
        s.index = s.index.tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].dropna()


def _changes(closes, base_index):
    """各商品用自己的交易日算漲跌，再對齊到金價的日期（不補值，避免假事件）。"""
    out = pd.DataFrame(index=base_index)
    for k, s in closes.items():
        prev = s.shift(1)
        if k == "tnx":
            scale = 10 if s.median() > 20 else 1          # ^TNX 有時是殖利率×10
            out["tnx_bp"] = ((s - prev) / scale * 100).reindex(base_index)
        else:
            out[k] = (s / prev - 1).where(prev > 0).reindex(base_index)
    return out


def load_daily(period="max"):
    closes = {}
    for k, t in TICKERS.items():
        try:
            closes[k] = _hist(t, period, "1d")
        except Exception as e:
            print(f"[警告] {t}: {e}")
    gold = closes["gold"]
    return gold, _changes(closes, gold.index)


def load_hourly(period="730d"):
    closes = {}
    for k, t in HOURLY_TICKERS.items():
        try:
            closes[k] = _hist(t, period, "60m")
        except Exception as e:
            print(f"[警告] {t} 小時線: {e}")
    gold = closes["gold"]
    return gold, _changes(closes, gold.index)


# ====== 統計 ======
def _dedupe(mask, gap):
    idx = np.flatnonzero(mask.values)
    keep, last = [], -10**9
    for i in idx:                      # 連續好幾天/好幾根都觸發，只算第一次
        if i - last > gap:
            keep.append(i)
        last = i
    m = np.zeros(len(mask), bool)
    m[keep] = True
    return pd.Series(m, index=mask.index)


def _summarize(key, name, logic, mask, fwd, labels, main, chg, fmt_time):
    ev = mask[mask].index
    res = {"key": key, "name": name, "logic": logic, "horizons": labels, "n": int(len(ev))}
    up, mean, base_up, base_mean = [], [], [], []
    for h in fwd:
        r = fwd[h].reindex(ev).dropna()
        allr = fwd[h].dropna()
        up.append(round(float((r > 0).mean() * 100), 1) if len(r) else None)
        mean.append(round(float(r.mean() * 100), 2) if len(r) else None)
        base_up.append(round(float((allr > 0).mean() * 100), 1))
        base_mean.append(round(float(allr.mean() * 100), 3))
    res.update(up=up, mean=mean, base_up=base_up, base_mean=base_mean)

    r = fwd[main].reindex(ev).dropna()
    half = len(r) // 2
    p1 = (r.iloc[:half] > 0).mean() if half else np.nan
    p2 = (r.iloc[half:] > 0).mean() if half else np.nan
    stable = bool(half >= 5 and ((p1 > .5 and p2 > .5) or (p1 < .5 and p2 < .5)))
    p, m = (r > 0).mean() if len(r) else np.nan, r.mean() if len(r) else np.nan
    if len(r) < MIN_N:
        rec = "樣本太少"
    elif p >= BULL_P and m > 0 and stable:
        rec = "偏多"
    elif p <= BEAR_P and m < 0 and stable:
        rec = "偏空"
    else:
        rec = "無明顯方向"
    res.update(rec=rec, stable=stable, main=list(fwd).index(main),
               halves=[None if np.isnan(p1) else round(p1 * 100), None if np.isnan(p2) else round(p2 * 100)])
    res["recent"] = [{"time": fmt_time(t),
                      "trigger": {k: (None if pd.isna(chg.at[t, k]) else round(float(chg.at[t, k]) * (1 if k == "tnx_bp" else 100), 2))
                                  for k in chg.columns},
                      "fwd": [None if pd.isna(fwd[h].get(t)) else round(float(fwd[h][t] * 100), 2) for h in fwd]}
                     for t in ev[-6:][::-1]]
    return res


def study():
    out = {"generated": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")}
    gold, chg = load_daily("max")
    chg = chg[chg.index >= "2006-01-01"]
    gold = gold.reindex(chg.index)
    fwd = {1: gold.shift(-1) / gold - 1, 3: gold.shift(-3) / gold - 1, 5: gold.shift(-5) / gold - 1}
    out["daily"] = []
    for key, name, cond, logic in DAILY_EVENTS:
        try:
            mask = _dedupe(cond(chg).fillna(False), 3)
        except AttributeError:
            continue
        out["daily"].append(_summarize(key, name, logic, mask, fwd, ["1天", "3天", "5天"], 3, chg,
                                       lambda t: t.strftime("%Y-%m-%d")))
    out["daily_range"] = f"{chg.index[0]:%Y-%m-%d} ~ {chg.index[-1]:%Y-%m-%d}"

    out["hourly"] = []
    try:
        hg, hchg = load_hourly()
        hfwd = {1: hg.shift(-1) / hg - 1, 4: hg.shift(-4) / hg - 1}
        for key, name, cond, logic in HOURLY_EVENTS:
            mask = _dedupe(cond(hchg).fillna(False), 4)
            out["hourly"].append(_summarize(key, name, logic, mask, hfwd, ["1小時", "4小時"], 4, hchg,
                                            lambda t: (t + pd.Timedelta(hours=9)).strftime("%m/%d %H:%M")))  # K 收盤時間(台灣)
        out["hourly_range"] = f"{hchg.index[0]:%Y-%m-%d} ~ {hchg.index[-1]:%Y-%m-%d}"
    except Exception as e:
        print("[警告] 小時線研究失敗：", e)
    return out


# ====== 即時偵測 ======
def _trigger_text(row):
    names = {"gold": "金價", "oil": "原油", "spx": "美股", "vix": "VIX", "dxy": "美元"}
    parts = [f"{names[k]} {row[k]*100:+.1f}%" for k in names if k in row and pd.notna(row[k])]
    if "tnx_bp" in row and pd.notna(row["tnx_bp"]):
        parts.append(f"10Y殖利率 {row['tnx_bp']:+.0f}bp")
    return "、".join(parts)


def detect_live(stats):
    now = pd.Timestamp.now(tz="UTC")
    found = []
    by_key = {s["key"]: s for s in stats.get("daily", []) + stats.get("hourly", [])}
    try:
        closes = {k: _hist(t, "10d", "1d") for k, t in TICKERS.items()}
        chg = _changes(closes, closes["gold"].index)
        row = chg.iloc[-1]
        day = chg.index[-1]
        if (now.tz_localize(None).normalize() - day).days <= 3:
            for key, name, cond, _ in DAILY_EVENTS:
                try:
                    hit = bool(cond(chg.iloc[[-1]]).fillna(False).iloc[0])
                except Exception:
                    hit = False
                if hit and key in by_key:
                    found.append({"id": f"{key}|{day:%Y-%m-%d}", "freq": "日線", "trigger": _trigger_text(row),
                                  "date": f"{day:%Y-%m-%d}（盤中為即時漲跌，收盤前可能變動）", **by_key[key]})
    except Exception as e:
        print("[警告] 日線事件偵測失敗：", e)
    try:
        closes = {k: _hist(t, "5d", "60m") for k, t in HOURLY_TICKERS.items()}
        chg = _changes(closes, closes["gold"].index)
        chg = chg[chg.index + pd.Timedelta(hours=1) <= now]          # 只看已收完的小時 K
        if len(chg) and now - chg.index[-1] <= pd.Timedelta(hours=2):
            t = chg.index[-1]
            for key, name, cond, _ in HOURLY_EVENTS:
                try:
                    hit = bool(cond(chg.iloc[[-1]]).fillna(False).iloc[0])
                except Exception:
                    hit = False
                if hit and key in by_key:
                    found.append({"id": f"{key}|{t:%Y-%m-%d %H}", "freq": "小時線", "trigger": _trigger_text(chg.iloc[-1]),
                                  "date": (t + pd.Timedelta(hours=9)).strftime("%m/%d %H:%M"), **by_key[key]})
    except Exception as e:
        print("[警告] 小時事件偵測失敗：", e)
    return found


def event_message(e):
    i = e["main"]
    h = e["horizons"][i]
    icon = {"偏多": "🟢", "偏空": "🔴"}.get(e["rec"], "⚪")
    lines = [f"⚡ 事件：{e['name']}",
             f"觸發：{e['trigger']}（{e['date']}）",
             f"歷史 {e['n']} 次，金價 {h}後上漲機率 {e['up'][i]}%（平常 {e['base_up'][i]}%），平均 {e['mean'][i]:+.2f}%"]
    lines.append("　".join(f"{lab}：漲 {u}%／均 {m:+.2f}%" for lab, u, m in zip(e["horizons"], e["up"], e["mean"]) if u is not None))
    note = "（前後期結果不一致，參考性低）" if (not e["stable"] and e["rec"] != "樣本太少") else ""
    lines.append(f"{icon} 歷史傾向：{e['rec']}{note}")
    if e.get("logic"):
        lines.append(f"理論：{e['logic']}")
    if e.get("recent"):
        r = e["recent"][0]
        lines.append(f"上一次：{r['time']}，之後金價 " + "、".join(
            f"{lab} {v:+.2f}%" for lab, v in zip(e["horizons"], r["fwd"]) if v is not None))
    return "\n".join(lines)
