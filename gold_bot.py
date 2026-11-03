# -*- coding: utf-8 -*-
"""
幣安 XAUUSDT 短線多空訊號 + 止損止盈 + Telegram 推播（在自己電腦執行）
-----------------------------------------------------------------------
週期：15 分鐘 K，持倉幾分鐘到幾小時
整合：15m 趨勢/動能/VWAP/突破、主動買賣盤、持倉量變化、資金費率、大戶多空比、
      宏觀方向（實質利率、美元、原油、VIX、新聞 → 來自 GitHub 網頁的 data.json）、
      美國重大數據（公布前後暫停進場）

用法（Windows 工作排程器每 5 分鐘執行一次 run_bot.bat）：
  python gold_bot.py             # 檢查訊號、追蹤虛擬單、需要時推播
  python gold_bot.py --status    # 強制推播目前狀態
  python gold_bot.py --backtest  # 用幣安歷史 15m K 回測止損止盈規則
  python gold_bot.py --scan      # 條件掃描：140+ 種條件樣本外驗證，通過的條件之後會即時推播（建議每週跑一次）
  python gold_bot.py --events    # 事件反應統計：原油暴跌/美股大跌/美元急動…後金價怎麼走、數據公布後延續或反轉
  python gold_bot.py --dry       # 只印不推播

安裝：pip install pandas numpy requests yfinance
"""
import os
import sys
import json
import time
import argparse
import datetime as dt

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")
STATE_PATH = os.path.join(BASE, "bot_state.json")
TRADES_CSV = os.path.join(BASE, "signal_trades.csv")
TW = dt.timezone(dt.timedelta(hours=8))
FAPI = "https://fapi.binance.com"
SYMBOL = "XAUUSDT"

# ====== 策略參數（網頁 index.html 裡有同一組，改的話兩邊一起改）======
W = {"trend": 25, "momentum": 15, "vwap": 10, "breakout": 10,   # 15m 技術面
     "flow": 10,                                                 # 主動買賣盤
     "oi": 5, "crowd": 5,                                        # 持倉量、擁擠度（資金費率+大戶多空比，反向）
     "macro": 20}                                                # 宏觀方向（data.json）
NAMES = {"trend": "15分趨勢", "momentum": "短線動能", "vwap": "VWAP", "breakout": "4h突破",
         "flow": "主動買盤", "oi": "持倉量", "crowd": "擁擠度(反向)", "macro": "宏觀方向"}
ENTRY_TH = 40          # 分數 ≥ +40 做多、≤ -40 做空
SL_ATR = 1.5           # 止損 = 1.5 × ATR(15m)
TP_R = 1.5             # 止盈 = 1.5 倍止損距離（1.5R）
MAX_HOLD_BARS = 16     # 最長持有 16 根 15m = 4 小時
EVENT_BLOCK_MIN = 30   # 重大數據前後 30 分鐘不進場
FEE_ROUND = 0.0012     # 回測來回成本：吃單 0.05%×2 + 滑價 0.02%
MACRO_MAX_AGE_H = 4    # data.json 超過 4 小時沒更新就不採用宏觀分數
LABEL = {"LONG": "🟢 做多", "SHORT": "🔴 做空", "WAIT": "⚪ 觀望"}


# ====== 工具 ======
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)


CFG = load_json(CONFIG_PATH, {})


def cfg(key, default):
    v = CFG.get(key, default)
    return default if v in ("", None) else v


def bget(path, **params):
    r = requests.get(FAPI + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


# ====== 幣安資料 ======
def klines(interval="15m", limit=500, start=None, end=None):
    p = {"symbol": SYMBOL, "interval": interval, "limit": limit}
    if start:
        p["startTime"] = int(start)
    if end:
        p["endTime"] = int(end)
    raw = bget("/fapi/v1/klines", **p)
    df = pd.DataFrame(raw, columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tbq", "x"])
    for c in ["o", "h", "l", "c", "v", "tb"]:
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.set_index("time")[["o", "h", "l", "c", "v", "tb", "ct"]]


def klines_history(days):
    end = int(time.time() * 1000)
    start = end - days * 86400 * 1000
    out = []
    while start < end:
        df = klines("15m", 1500, start=start)
        if df.empty:
            break
        out.append(df)
        start = int(df["ct"].iloc[-1]) + 1
        time.sleep(0.2)
        if len(df) < 1500:
            break
    return pd.concat(out)[lambda d: ~d.index.duplicated()]


def live_extras():
    """只能即時取得的資料：資金費率、持倉量變化、大戶多空比。抓不到就回傳 None（該因子不計分）。"""
    ex = {"funding": None, "oi_chg": None, "top_ls": None, "mark": None, "index": None, "oi": None}
    try:
        p = bget("/fapi/v1/premiumIndex", symbol=SYMBOL)
        ex["funding"], ex["mark"] = float(p["lastFundingRate"]), float(p["markPrice"])
        ex["index"] = float(p.get("indexPrice") or 0) or None
    except Exception as e:
        print("[警告] 資金費率：", e)
    try:
        oi = bget("/futures/data/openInterestHist", symbol=SYMBOL, period="5m", limit=13)
        ex["oi_chg"] = float(oi[-1]["sumOpenInterest"]) / float(oi[0]["sumOpenInterest"]) - 1
        ex["oi"] = float(oi[-1]["sumOpenInterest"])
    except Exception as e:
        print("[警告] 持倉量：", e)
    try:
        ls = bget("/futures/data/topLongShortPositionRatio", symbol=SYMBOL, period="5m", limit=1)
        ex["top_ls"] = float(ls[-1]["longShortRatio"])
    except Exception as e:
        print("[警告] 大戶多空比：", e)
    return ex


# ====== 指標與分數（回測、即時、網頁同一套）======
def indicators(df):
    d = df.copy()
    c = d["c"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    tr = pd.concat([d["h"] - d["l"], (d["h"] - c.shift()).abs(), (d["l"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    tp = (d["h"] + d["l"] + c) / 3
    day = d.index.floor("D")                               # VWAP 每天 00:00 UTC（台灣 08:00）重算
    d["vwap"] = (tp * d["v"]).groupby(day).cumsum() / d["v"].groupby(day).cumsum().replace(0, np.nan)
    d["hh16"] = d["h"].rolling(16).max().shift(1)
    d["ll16"] = d["l"].rolling(16).min().shift(1)
    buy = d["tb"].rolling(4).sum()
    sell = (d["v"] - d["tb"]).rolling(4).sum()
    d["taker_ratio"] = buy / sell.replace(0, np.nan)
    return d


def factor_frame(d):
    f = pd.DataFrame(index=d.index)
    a = d["atr"].replace(0, np.nan)
    f["trend"] = 0.5 * np.sign(d["ema20"] - d["ema50"]) + 0.5 * np.sign(d["c"] - d["ema200"])
    f["momentum"] = ((d["c"] - d["c"].shift(8)) / a / 2).clip(-1, 1)
    f["vwap"] = ((d["c"] - d["vwap"]) / a / 2).clip(-1, 1)
    f["breakout"] = np.where(d["c"] > d["hh16"], 1.0, np.where(d["c"] < d["ll16"], -1.0, 0.0))
    f["flow"] = ((d["taker_ratio"] - 1) / 0.3).clip(-1, 1)
    return f.fillna(0.0)


def extras_factors(ex, price_chg_1h):
    out = {}
    if ex.get("oi_chg") is not None:
        # 持倉增加 + 價格上漲 = 新多單進場（偏多）；持倉增加 + 價格下跌 = 新空單（偏空）；持倉減少 = 平倉行情，不計
        out["oi"] = float(np.sign(price_chg_1h) * np.clip(ex["oi_chg"] / 0.02, 0, 1))
    parts = []
    if ex.get("funding") is not None:
        parts.append(-np.clip(ex["funding"] / 0.0005, -1, 1))     # 費率太正 = 多頭擁擠 → 偏空
    if ex.get("top_ls") is not None:
        parts.append(-np.clip((ex["top_ls"] - 1) / 1.0, -1, 1))    # 大戶多空比太高 → 偏空
    if parts:
        out["crowd"] = float(np.mean(parts))
    return out


def total_score(fvals):
    keys = [k for k in W if k in fvals and fvals[k] is not None]
    tot = sum(W[k] for k in keys)
    return sum(fvals[k] * W[k] for k in keys) / tot * 100 if tot else 0.0


def sl_tp(direction, price, atr):
    dist = SL_ATR * atr
    if direction == "LONG":
        return price - dist, price + dist * TP_R, dist
    return price + dist, price - dist * TP_R, dist


# ====== 宏觀 / 事件 / 休市 ======
def get_macro():
    url = cfg("macro_url", "")
    if not url:
        return None, None
    try:
        d = requests.get(url, params={"t": int(time.time())}, timeout=15).json()
        upd = dt.datetime.strptime(d["updated"], "%Y-%m-%d %H:%M").replace(tzinfo=TW)
        age = (dt.datetime.now(TW) - upd).total_seconds() / 3600
        return (None if age > MACRO_MAX_AGE_H else d["score"] / 100), d
    except Exception as e:
        print("[警告] 宏觀資料：", e)
        return None, None


def blocking_event(macro_data, now):
    if not macro_data:
        return None
    for e in macro_data.get("events", []):
        try:
            t = dt.datetime.strptime(f"{now.year}/{e['time']}", "%Y/%m/%d %H:%M").replace(tzinfo=TW)
        except Exception:
            continue
        if abs((t - now).total_seconds()) <= EVENT_BLOCK_MIN * 60:
            return e
    return None


def market_closed(now_utc):
    """國際金市（COMEX）大約 週五 21:00 UTC ~ 週日 22:00 UTC 休市，每天 21:00~22:00 UTC 也休一小時。"""
    wd, h = now_utc.weekday(), now_utc.hour
    if (wd == 4 and h >= 21) or wd == 5 or (wd == 6 and h < 22):
        return "週末國際金市休市"
    if h == 21:
        return "國際金市每日休市時段"
    return None


# ====== Telegram ======
def send(text, dry=False):
    print(text)
    tok, chat = cfg("telegram_token", ""), cfg("chat_id", "")
    if dry or not tok or not chat or "填入" in tok:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": text, "disable_web_page_preview": True}, timeout=15)
    except Exception as e:
        print("Telegram 失敗：", e)


def risk_lines(direction, price, sl, dist):
    pct = dist / price
    acct = float(cfg("account_usdt", 1000))
    risk = float(cfg("risk_pct", 2)) / 100
    lev_list = cfg("leverages", [20, 30, 50])
    lines = ["— 槓桿換算 —"]
    for lev in lev_list:
        liq = 1 / lev - 0.005
        warn = "  ⚠️止損比爆倉還遠" if pct >= liq else ""
        lines.append(f"{lev}x：打到止損 = 保證金 -{pct*lev*100:.1f}%｜約 {liq*100:.2f}% 反向就爆倉{warn}")
    notional = acct * risk / pct
    lines.append(f"本金 {acct:,.0f}U、每筆最多虧 {risk*100:.0f}% → 部位名目 {notional:,.0f}U"
                 f"（= 本金 {notional/acct:.1f} 倍）")
    return lines


# ====== 即時 ======
def run(status=False, dry=False):
    now = dt.datetime.now(TW)
    now_utc = now.astimezone(dt.timezone.utc)
    df = klines("15m", 500)
    df = df[df.index + pd.Timedelta(minutes=15) <= pd.Timestamp(now_utc)]   # 只用已收盤的 K
    d = indicators(df)
    f = factor_frame(d)
    last, row = f.iloc[-1].to_dict(), d.iloc[-1]
    price_now = float(bget("/fapi/v1/ticker/price", symbol=SYMBOL)["price"])
    ex = live_extras()
    last.update(extras_factors(ex, row["c"] / d["c"].iloc[-5] - 1))
    macro, mdata = get_macro()
    last["macro"] = macro
    score = total_score(last)
    raw_dir = "LONG" if score >= ENTRY_TH else "SHORT" if score <= -ENTRY_TH else "WAIT"
    ev = blocking_event(mdata, now)
    closed = market_closed(now_utc)
    block = (f"{ev['time']} {ev['title']} 前後 {EVENT_BLOCK_MIN} 分鐘" if ev else None) or closed
    direction = "WAIT" if block else raw_dir

    state = load_json(STATE_PATH, {"open": None, "last_dir": "WAIT", "last_bar": "", "warned": []})
    msgs = []

    # 1) 追蹤虛擬單（統計實盤表現用）
    op = state.get("open")
    if op:
        m1 = klines("1m", 300, start=op["checked_ms"])
        hit = None
        for t, k in m1.iterrows():
            if op["dir"] == "LONG":
                if k["l"] <= op["sl"]: hit = ("止損", op["sl"]); break
                if k["h"] >= op["tp"]: hit = ("止盈", op["tp"]); break
            else:
                if k["h"] >= op["sl"]: hit = ("止損", op["sl"]); break
                if k["l"] <= op["tp"]: hit = ("止盈", op["tp"]); break
        if not hit and now.timestamp() * 1000 - op["entry_ms"] >= MAX_HOLD_BARS * 15 * 60000:
            hit = ("時間到出場", price_now)
        if hit:
            px = hit[1]
            r = ((px - op["entry"]) if op["dir"] == "LONG" else (op["entry"] - px)) / op["dist"]
            msgs.append(f"📌 XAUUSDT {LABEL[op['dir']]} 訊號結束：{hit[0]} @ {px:,.2f}（{r:+.2f}R，未扣手續費）")
            new = not os.path.exists(TRADES_CSV)
            with open(TRADES_CSV, "a", encoding="utf-8-sig") as fcsv:
                if new:
                    fcsv.write("進場時間,方向,進場,止損,止盈,出場時間,出場,結果,R\n")
                fcsv.write(f"{op['time']},{op['dir']},{op['entry']:.2f},{op['sl']:.2f},{op['tp']:.2f},"
                           f"{now:%Y-%m-%d %H:%M},{px:.2f},{hit[0]},{r:.2f}\n")
            state["open"] = None
        elif len(m1):
            op["checked_ms"] = int(m1["ct"].iloc[-1]) + 1

    # 2) 新訊號：方向剛轉成多/空、且這根 K 還沒發過、且沒有同向的單在跑
    bar_id = str(d.index[-1])
    new_signal = bool(cfg("push_15m", False)) and (direction != "WAIT" and direction != state.get("last_dir") and bar_id != state.get("last_bar")
                  and not (state.get("open") and state["open"]["dir"] == direction))
    sl, tp, dist = sl_tp(direction if direction != "WAIT" else ("LONG" if score >= 0 else "SHORT"),
                         price_now, float(row["atr"]))

    def body(title):
        L = [f"🥇 XAUUSDT {title}  {now:%m/%d %H:%M}",
             f"現價 {price_now:,.2f}｜分數 {score:+.0f}（門檻 ±{ENTRY_TH}）"]
        if block:
            L.append(f"⛔ 暫停進場：{block}")
        if direction != "WAIT":
            L += [f"進場參考：{price_now:,.2f}",
                  f"止損：{sl:,.2f}（{'-' if direction=='LONG' else '+'}{dist/price_now*100:.2f}%）",
                  f"止盈：{tp:,.2f}（{TP_R}R，{'+' if direction=='LONG' else '-'}{dist*TP_R/price_now*100:.2f}%）",
                  f"最長持有 {MAX_HOLD_BARS*15//60} 小時，時間到沒打到就出場", ""]
            L += risk_lines(direction, price_now, sl, dist)
        L += ["", "— 理由（-1 空 ~ +1 多）—"]
        L += [f"{NAMES[k]}：{last[k]:+.2f}" for k in W if last.get(k) is not None]
        extra = []
        if ex["funding"] is not None: extra.append(f"資金費率 {ex['funding']*100:.4f}%")
        if ex["top_ls"] is not None: extra.append(f"大戶多空比 {ex['top_ls']:.2f}")
        if ex["oi_chg"] is not None: extra.append(f"1h 持倉 {ex['oi_chg']*100:+.1f}%")
        if extra: L.append("｜".join(extra))
        if macro is None: L.append("（宏觀資料未取得或過舊，本次不計）")
        if cfg("site_url", ""): L += ["", cfg("site_url", "")]
        L.append("※ 統計訊號，不保證獲利；請先看回測與虛擬單紀錄")
        return "\n".join(L)

    if new_signal:
        msgs.append(body(f"{LABEL[direction]}訊號"))
        state["open"] = {"dir": direction, "entry": price_now, "sl": sl, "tp": tp, "dist": dist,
                         "time": f"{now:%Y-%m-%d %H:%M}", "entry_ms": int(now.timestamp() * 1000),
                         "checked_ms": int(now.timestamp() * 1000)}
        state["last_bar"] = bar_id
    elif status:
        msgs.append(body(f"目前狀態：{LABEL[direction]}"))

    # 3) 重大數據預告（公布前 60 分鐘提醒一次）
    for e in (mdata or {}).get("events", []):
        try:
            t = dt.datetime.strptime(f"{now.year}/{e['time']}", "%Y/%m/%d %H:%M").replace(tzinfo=TW)
        except Exception:
            continue
        key = e["time"] + e["title"]
        if 0 <= (t - now).total_seconds() <= 3600 and key not in state["warned"]:
            msgs.append(f"⚠️ {e['time']} 美國 {e['title']} 即將公布，前後 {EVENT_BLOCK_MIN} 分鐘不要進場，"
                        f"持倉請考慮先減碼或拉好止損")
            state["warned"] = (state["warned"] + [key])[-30:]

    # 4) 美國數據公布衝擊：剛收完的這根 15m K 是不是 08:30/10:00/14:00(美東) 的大波動
    try:
        shock = release_shocks(d).iloc[-1]
        bar_key = "shock|" + bar_id
        if shock["is_shock"] and bar_key not in state.get("event_ids", []):
            msgs.append(shock_message(shock, d, price_now))
            state["event_ids"] = (state.get("event_ids", []) + [bar_key])[-100:]
    except Exception as e:
        print("[警告] 數據衝擊偵測：", e)

    # 5) 市場事件（原油暴跌、美股大跌、美元急動…）每 15 分鐘偵測一次
    if now.minute % 15 < 5 or status:
        try:
            import events_study as es
            stats = load_event_stats()
            for e in es.detect_live(stats):
                if e["id"] not in state.get("event_ids", []):
                    msgs.append(es.event_message(e) + ("\n" + cfg("site_url", "") if cfg("site_url", "") else ""))
                    state["event_ids"] = (state.get("event_ids", []) + [e["id"]])[-100:]
        except Exception as e:
            print("[警告] 市場事件偵測：", e)

    # 波段主訊號（6 個月動能）：每小時更新一次
    sw = state.get("swing_cache")
    if not sw or state.get("swing_hour") != now.strftime("%Y-%m-%d %H") or status:
        try:
            sw = get_swing(now)
            state["swing_cache"], state["swing_hour"] = sw, now.strftime("%Y-%m-%d %H")
        except Exception as e:
            print("[警告] 波段訊號：", e)
    if sw:
        prev_sw = state.get("swing_dir")
        if prev_sw and prev_sw != sw["dir"]:
            msgs.append(swing_flip_message(sw, prev_sw, price_now))
        state["swing_dir"] = sw["dir"]
        days_left = (pd.Timestamp(now.strftime("%Y-%m-01")) + pd.offsets.MonthEnd(0) - pd.Timestamp(now.strftime("%Y-%m-%d"))).days
        warn_key = now.strftime("%Y-%m-%d")
        if days_left <= 4 and sw["preview"] != sw["dir"] and state.get("swing_warned") != warn_key:
            msgs.append(f"⏳ 波段可能在月底翻向：目前 {LABEL[sw['dir']]}，但金價已比 6 個月前 {sw['mom_now']:+.1f}%"
                        f"（翻向價約 {sw['flip_price']:,.0f}）。{sw['month_end']} 收盤若維持，下個月改 {LABEL[sw['preview']]}")
            state["swing_warned"] = warn_key
        if status:
            msgs.append(swing_status_message(sw, price_now))

    # 每日方向 + 心跳（每天 DAILY_HOUR 點推一次；沒收到 = 程式或電腦有問題）
    today = now.strftime("%Y-%m-%d")
    state["runs"] = state.get("runs", {})
    state["runs"][today] = state["runs"].get(today, 0) + 1
    state["runs"] = {k: v for k, v in state["runs"].items() if k >= (now - dt.timedelta(days=7)).strftime("%Y-%m-%d")}
    if now.hour == int(cfg("daily_hour", 8)) and state.get("last_daily") != today:
        msgs.append(daily_message(now, price_now, score, last, macro, mdata, row, ex, state, sw))
        state["last_daily"] = today

    # 6) 條件掃描通過的條件：每根新 15m K 檢查一次
    if bar_id != state.get("last_cond_bar") and os.path.exists(os.path.join(BASE, "conditions.json")):
        try:
            import conditions_scan as cs
            recent = indicators(klines_history(32))
            recent = recent[recent.index + pd.Timedelta(minutes=15) <= pd.Timestamp(now_utc)]
            for h in cs.live_hits(recent, macro, cs.load_mlog()):
                msgs.append(cs.hit_message(h, price_now, float(recent["atr"].iloc[-1])))
            state["last_cond_bar"] = bar_id
        except Exception as e:
            print("[警告] 條件訊號：", e)

    # 7) 收集資料（之後自動納入條件掃描）
    if not dry:
        try:
            log_market(now_utc, price_now, ex, d)
            log_event_reactions(state, mdata, now, d)
        except Exception as e:
            print("[警告] 資料收集：", e)

    state["last_dir"] = direction
    if not msgs:
        print(body(f"目前 {LABEL[direction]}（無需推播）"))
    for m in msgs:
        send(m, dry)
    if not dry:
        save_json(STATE_PATH, state)


# ====== 每日方向 ======
def get_swing(now):
    import yfinance as yf
    sys.path.insert(0, BASE)
    from gold_signal import swing_signal
    g = yf.Ticker("GC=F").history(period="2y", auto_adjust=False)["Close"]
    g.index = pd.to_datetime(g.index).tz_localize(None).normalize()
    return swing_signal(g[~g.index.duplicated(keep="last")], now)


def swing_lines(sw, price):
    levs = cfg("swing_leverage", [1, 1.5])
    acct = float(cfg("account_usdt", 1000))
    return [f"波段方向：{LABEL[sw['dir']]}（自 {sw['since']} 起，COMEX 進場 {sw['entry']:,.0f}，目前 {sw['pnl']:+.1f}%）",
            f"6 個月動能：上月底 {sw['mom']:+.1f}%｜現在 {sw['mom_now']:+.1f}%（翻向價約 {sw['flip_price']:,.0f}，{sw['month_end']} 收盤決定）",
            f"倉位：{levs[0]}～{levs[-1]} 倍 → 本金 {acct:,.0f}U 對應部位名目 {acct*levs[0]:,.0f}～{acct*levs[-1]:,.0f}U",
            f"XAUUSDT 現價 {price:,.2f}"]


def swing_flip_message(sw, prev, price):
    return "\n".join([f"📈 黃金波段訊號：{LABEL[prev]} → {LABEL[sw['dir']]}",
                      f"{sw['decided']} 收盤，COMEX 金價比 6 個月前 {sw['mom']:+.1f}%",
                      f"👉 {'平空單、改做多' if sw['dir']=='LONG' else '平多單、改做空'}，持有到下個月底再檢查",
                      *swing_lines(sw, price)[2:],
                      "不設固定止損（回測就是這樣算的）；槓桿請勿超過 1.5 倍，回測最大回撤 -50%",
                      sw["stats"]])


def swing_status_message(sw, price):
    return "\n".join(["📈 黃金波段（6 個月動能）目前狀態"] + swing_lines(sw, price) + [sw["stats"]])


def daily_message(now, price, score, fvals, macro, mdata, row, ex, state, sw=None):
    head = [f"☀️ 每日黃金  {now:%m/%d}"]
    if sw:
        head += ["【主訊號・波段】"] + swing_lines(sw, price)
        if sw["preview"] != sw["dir"]:
            head.append(f"⚠️ 若 {sw['month_end']} 收盤仍在翻向價另一側，下個月改 {LABEL[sw['preview']]}")
    head += ["", "【短線參考・15 分 K，回測無優勢，僅供觀察】"]
    return "\n".join(head + [_daily_body(now, price, score, fvals, macro, mdata, row, ex, state)])


def _daily_body(now, price, score, fvals, macro, mdata, row, ex, state):
    lean = "LONG" if score >= 0 else "SHORT"
    a = abs(score)
    if a >= 40:
        conf, size = "高", "正常倉位（每筆風險 = 設定值）"
    elif a >= 20:
        conf, size = "中", "半倉（每筆風險減半）"
    else:
        conf, size = "低", "多空不明，建議觀望；真要做請 1/4 倉"
    if macro is not None and np.sign(macro) != 0 and np.sign(macro) != np.sign(score) and conf != "低":
        conf, size = "中" if conf == "高" else "低", size + "（短線與宏觀方向相反，降一級）"
    sl, tp, dist = sl_tp(lean, price, float(row["atr"]))
    y = (now - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    L = [f"短線偏向：{LABEL[lean]}｜分數 {score:+.0f}",
         "", "— 短線理由 —"]
    top = sorted([(k, v) for k, v in fvals.items() if v is not None and k in W], key=lambda kv: -abs(kv[1] * W[kv[0]]))[:4]
    L += [f"{NAMES[k]} {v:+.2f}" for k, v in top]
    if mdata:
        L.append(f"宏觀分數 {mdata.get('score', 0):+.0f}（{LABEL.get(mdata.get('direction'), mdata.get('direction'))}）")
        news = mdata.get("news", [])[:3]
        if news:
            L += ["", "— 新聞 —"] + [f"{'▲' if n['score'] > 0 else '▼'} {n['title'][:55]}" for n in news]
        ev = [e for e in mdata.get("events", [])]
        if ev:
            L += ["", "— 今明兩天美國重大數據 —"] + [f"⚠️ {e['time']} {e['title']}" for e in ev[:5]]
            L.append(f"公布前後 {EVENT_BLOCK_MIN} 分鐘不要進場")
        for e in mdata.get("active_events", [])[:2]:
            L.append(f"⚡ 事件：{e['name']} → 歷史傾向 {e['rec']}")
    if ex.get("funding") is not None:
        L.append(f"資金費率 {ex['funding']*100:.4f}%｜大戶多空比 {ex.get('top_ls') or '—'}")
    L += ["", f"✅ 程式正常（昨天執行 {state.get('runs', {}).get(y, 0)} 次，正常約 288 次）",
          "※ 依波段主訊號操作；短線資訊僅供參考"]
    return "\n".join(L)


# ====== 資料收集 ======
MLOG = os.path.join(BASE, "market_log.csv")
ELOG = os.path.join(BASE, "event_log.csv")


def _append(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8-sig") as f:
        if new:
            f.write(header + "\n")
        f.write(row + "\n")


def log_market(now_utc, price, ex, d):
    prem = (ex["mark"] / ex["index"] - 1) if ex.get("mark") and ex.get("index") else ""
    tr = d["taker_ratio"].iloc[-1]
    vals = [now_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"), price, ex.get("mark") or "", ex.get("index") or "", prem,
            "" if ex.get("funding") is None else ex["funding"], ex.get("oi") or "",
            "" if ex.get("top_ls") is None else ex["top_ls"], "" if pd.isna(tr) else round(float(tr), 4)]
    _append(MLOG, "time,price,mark,index,premium,funding,oi,top_ls,taker_ratio", ",".join(str(v) for v in vals))


def log_event_reactions(state, mdata, now, d):
    """記下每個美國重大數據：預期值、前值、公布後 15 分 / 1 小時 / 4 小時金價反應（之後拿來統計預期差）"""
    pend = state.setdefault("pending_ev", {})
    for e in (mdata or {}).get("events", []):
        try:
            t = dt.datetime.strptime(f"{now.year}/{e['time']}", "%Y/%m/%d %H:%M").replace(tzinfo=TW)
        except Exception:
            continue
        key = f"{t:%Y-%m-%d %H:%M}|{e['title']}"
        if key not in state.get("logged_ev", []):
            pend.setdefault(key, {"forecast": e.get("forecast", ""), "previous": e.get("previous", "")})
    for key in list(pend):
        tstr, title = key.split("|", 1)
        t = pd.Timestamp(tstr, tz=TW).tz_convert("UTC")
        if pd.Timestamp(now) < t + pd.Timedelta(hours=4, minutes=20):
            continue
        try:
            base = d["o"].loc[t]
            mv = {n: d["c"].loc[t + pd.Timedelta(minutes=15 * (n - 1))] / base - 1 for n in (1, 4, 16)}
            row = [tstr, title.replace(",", " "), str(pend[key]["forecast"]).replace(",", ""),
                   str(pend[key]["previous"]).replace(",", ""), *(round(mv[n] * 100, 3) for n in (1, 4, 16))]
            _append(ELOG, "time_tw,event,forecast,previous,move15m_pct,move1h_pct,move4h_pct", ",".join(map(str, row)))
        except Exception:
            pass
        del pend[key]
        state["logged_ev"] = (state.get("logged_ev", []) + [key])[-200:]


# ====== 事件：市場衝擊統計（來自 GitHub 每天更新的 events.json）======
EV_LOCAL = os.path.join(BASE, "events_local.json")
REL_PATH = os.path.join(BASE, "release_stats.json")


def load_event_stats():
    url = cfg("macro_url", "").replace("data.json", "events.json")
    if url:
        try:
            return requests.get(url, params={"t": int(time.time())}, timeout=15).json()
        except Exception as e:
            print("[警告] 網站 events.json 取得失敗，改用本機計算：", e)
    st = load_json(EV_LOCAL, {})
    if not st or st.get("generated", "")[:10] < (dt.datetime.now(TW) - dt.timedelta(days=7)).strftime("%Y-%m-%d"):
        import events_study as es
        st = es.study()
        save_json(EV_LOCAL, st)
    return st


# ====== 事件：美國數據公布衝擊（幣安 15m）======
# 美東 08:30 = CPI/非農/PPI/零售/初領失業金；10:00 = ISM/JOLTS/消費者信心；14:00 = FOMC 利率決議
SLOTS = {"08:30": "08:30 數據（CPI/非農/PPI/零售）", "10:00": "10:00 數據（ISM/JOLTS 等）", "14:00": "14:00 FOMC/聯準會"}
SHOCK_MULT = 3.0       # 這根 K 的振幅 ≥ 前 24 小時中位數 3 倍 才算「市場有反應」


def release_shocks(d):
    ny = d.index.tz_convert("America/New_York")
    slot = pd.Series(ny.strftime("%H:%M"), index=d.index)
    rng = d["h"] - d["l"]
    med = rng.rolling(96).median().shift(1)
    out = pd.DataFrame(index=d.index)
    out["slot"] = slot
    out["ratio"] = rng / med
    out["is_shock"] = slot.isin(list(SLOTS)) & (ny.weekday < 5) & (out["ratio"] >= SHOCK_MULT)
    out["dir"] = np.sign(d["c"] - d["o"])
    out["move"] = (d["c"] - d["o"]) / d["o"]
    for n, lab in [(4, "f1h"), (16, "f4h")]:
        out[lab] = (d["c"].shift(-n) / d["c"] - 1) * out["dir"]        # >0 = 第一波方向延續
    return out


def release_study(days=300, save=True):
    raw = klines_history(days)
    raw = raw[(raw["v"] > 0) & (raw["h"] > raw["l"])]
    s = release_shocks(indicators(raw))
    ev = s[s["is_shock"]].dropna(subset=["f4h"])
    groups = [("全部", ev)] + [(SLOTS[k], ev[ev.slot == k]) for k in SLOTS]
    res = {"generated": dt.datetime.now(TW).strftime("%Y-%m-%d %H:%M"),
           "range": f"{raw.index[0]:%Y-%m-%d} ~ {raw.index[-1]:%Y-%m-%d}", "groups": []}
    for name, g in groups:
        n = len(g)
        c1, c4 = ((g.f1h > 0).mean() * 100 if n else None), ((g.f4h > 0).mean() * 100 if n else None)
        m4 = g.f4h.mean() * 100 if n else None
        if n < 8:
            rec = "樣本太少"
        elif c4 >= 60 and m4 > 0:
            rec = "順勢"
        elif c4 <= 40 and m4 < 0:
            rec = "反向"
        else:
            rec = "不建議交易"
        res["groups"].append({"name": name, "n": n, "cont1h": None if c1 is None else round(c1, 1),
                              "cont4h": None if c4 is None else round(c4, 1),
                              "mean1h": None if not n else round(g.f1h.mean() * 100, 3),
                              "mean4h": None if m4 is None else round(m4, 3), "rec": rec,
                              "recent": [{"time": (t.tz_convert(TW)).strftime("%Y-%m-%d %H:%M"),
                                          "move": round(r.move * 100, 2), "f4h": round(r.f4h * 100, 2)}
                                         for t, r in g.tail(5)[::-1].iterrows()]})
    if save:
        save_json(REL_PATH, res)
    return res


def get_release_stats():
    st = load_json(REL_PATH, {})
    if not st or st.get("generated", "")[:10] < (dt.datetime.now(TW) - dt.timedelta(days=7)).strftime("%Y-%m-%d"):
        try:
            st = release_study()
        except Exception as e:
            print("[警告] 數據衝擊統計失敗：", e)
    return st


def shock_message(shock, d, price_now):
    st = get_release_stats()
    g = next((x for x in st.get("groups", []) if x["name"] == SLOTS[shock.slot]), None)
    if g and g["rec"] == "樣本太少":
        g = next((x for x in st.get("groups", []) if x["name"] == "全部"), g)
    first = "LONG" if shock.dir > 0 else "SHORT"
    L = [f"💥 美國數據公布衝擊：{SLOTS[shock.slot]}",
         f"這根 15 分 K {shock.move*100:+.2f}%（平常的 {shock.ratio:.1f} 倍），第一波 {LABEL[first]}"]
    if g:
        L.append(f"歷史（{g['name']}，{g['n']} 次）：4 小時後延續第一波方向 {g['cont4h']}%，"
                 f"順勢平均 {g['mean4h']:+.2f}%；1 小時延續 {g['cont1h']}%")
        trade = first if g["rec"] == "順勢" else ("SHORT" if first == "LONG" else "LONG") if g["rec"] == "反向" else None
        if trade:
            atr = float(d["atr"].iloc[-2])                      # 用衝擊前的 ATR，避免止損被放太寬
            sl, tp, dist = sl_tp(trade, price_now, atr * 1.3)
            L += [f"👉 歷史傾向：{g['rec']} → {LABEL[trade]}",
                  f"參考進場 {price_now:,.2f}｜止損 {sl:,.2f}（{dist/price_now*100:.2f}%）｜止盈 {tp:,.2f}",
                  "數據剛出來點差大，建議等 1~2 分鐘、用掛單"]
        else:
            L.append(f"👉 歷史傾向：{g['rec']}，這次建議不追")
    else:
        L.append("（尚無歷史統計，等第一次 --events 跑完）")
    return "\n".join(L)


def print_events():
    import events_study as es
    st = load_event_stats()
    print(f"\n===== 市場事件後金價（日線 {st.get('daily_range','')}）=====")
    for e in st.get("daily", []) + st.get("hourly", []):
        if e is st.get("hourly", [None])[0]:
            print(f"\n===== 小時事件（{st.get('hourly_range','')}）=====")
        cells = "  ".join(f"{h}:漲{u}%/均{m:+.2f}%" for h, u, m in zip(e["horizons"], e["up"], e["mean"]) if u is not None)
        print(f"{e['name']:<26} {e['n']:4d}次  {cells}  → {e['rec']}{'' if e['stable'] else '(不穩)'}")
    print("（平常任一天：1天上漲 " + str(st["daily"][0]["base_up"][0]) + "%）" if st.get("daily") else "")
    rs = release_study()
    print(f"\n===== 美國數據公布衝擊（幣安 15m，{rs['range']}）=====")
    for g in rs["groups"]:
        print(f"{g['name']:<24} {g['n']:3d}次  1h延續 {g['cont1h']}%  4h延續 {g['cont4h']}%  "
              f"4h順勢平均 {g['mean4h']}% → {g['rec']}")
    if not cfg("telegram_token", "") or "填入" in cfg("telegram_token", ""):
        return
    lines = ["🧪 事件反應統計"] + [f"{e['name']}：{e['n']}次，{e['horizons'][e['main']]}漲{e['up'][e['main']]}% → {e['rec']}"
                                for e in st.get("daily", [])]
    lines += [f"{g['name']}：{g['n']}次，4h延續{g['cont4h']}% → {g['rec']}" for g in rs["groups"]]
    send("\n".join(lines))


# ====== 回測 ======
def macro_daily():
    """每日宏觀分數（不含新聞）。用前一天收盤算的分數，避免偷看。"""
    try:
        sys.path.insert(0, BASE)
        from gold_signal import build_frame, compute_factors, combine
        df = build_frame("2y")
        s = combine(compute_factors(df), use_news=False) / 100
        s.index = pd.to_datetime(s.index).tz_localize("UTC") + pd.Timedelta(days=1)
        return s
    except Exception as e:
        print("[警告] 宏觀歷史失敗，回測不含宏觀：", e)
        return None


def simulate(d, score, sl_atr, tp_r, th=ENTRY_TH, max_hold=MAX_HOLD_BARS, fee=FEE_ROUND):
    dirs = np.where(score >= th, 1, np.where(score <= -th, -1, 0))
    o, h, l, c, a = (d[x].values for x in ["o", "h", "l", "c", "atr"])
    times = d.index
    trades, i, n, prev = [], 1, len(d), 0
    while i < n - 1:
        sig = dirs[i]
        if sig != 0 and sig != prev and not market_closed(times[i + 1].to_pydatetime()):
            entry, dist = o[i + 1], sl_atr * a[i]
            if dist <= 0 or np.isnan(dist):
                prev, i = sig, i + 1
                continue
            sl = entry - sig * dist
            tp = entry + sig * dist * tp_r
            j, exit_px, why = i + 1, None, ""
            while j < n and j <= i + max_hold:
                if (sig == 1 and l[j] <= sl) or (sig == -1 and h[j] >= sl):
                    exit_px, why = sl, "SL"; break           # 同一根同時碰到，保守算止損
                if (sig == 1 and h[j] >= tp) or (sig == -1 and l[j] <= tp):
                    exit_px, why = tp, "TP"; break
                j += 1
            if exit_px is None:
                j = min(j, n - 1); exit_px, why = c[j], "TIME"
            r = sig * (exit_px - entry) / dist - fee * entry / dist
            trades.append({"time": times[i + 1], "dir": sig, "r": r, "why": why,
                           "pct": sig * (exit_px - entry) / entry - fee, "fee_r": fee * entry / dist})
            prev, i = sig, j + 1
            continue
        prev, i = sig, i + 1
    return pd.DataFrame(trades)


def summarize(t):
    if t.empty:
        return "無交易"
    wins = t[t.r > 0]
    pf = wins.r.sum() / -t[t.r <= 0].r.sum() if (t.r <= 0).any() else float("inf")
    streak = max((len(list(g)) for k, g in __import__("itertools").groupby(t.r <= 0) if k), default=0)
    eq = t.r.cumsum()
    mdd = (eq - eq.cummax()).min()
    return (f"{len(t):4d} 筆｜勝率 {len(wins)/len(t)*100:4.1f}%｜平均 {t.r.mean():+.3f}R｜總計 {t.r.sum():+6.1f}R｜"
            f"獲利因子 {pf:4.2f}｜最大連虧 {streak}｜最大回撤 {mdd:.1f}R")


def backtest(days=300):
    print(f"下載幣安 {SYMBOL} 15m K 線（最多 {days} 天，XAUUSDT 2026/1 才上線）...")
    raw = klines_history(days)
    raw = raw[(raw["v"] > 0) & (raw["h"] > raw["l"])]
    d = indicators(raw)
    f = factor_frame(d)
    md = macro_daily()
    if md is not None:
        f["macro"] = md.reindex(f.index, method="ffill").fillna(0)
    keys = [k for k in W if k in f.columns]
    score = sum(f[k] * W[k] for k in keys) / sum(W[k] for k in keys) * 100
    print(f"期間 {d.index[0]:%Y-%m-%d} ~ {d.index[-1]:%Y-%m-%d}，{len(d)} 根 K；"
          f"使用因子：{'、'.join(NAMES[k] for k in keys)}（持倉量/擁擠度/新聞沒有歷史，未納入）")
    print(f"平均 ATR(15m) = {(d.atr / d.c).mean()*100:.3f}%，止損 {SL_ATR}ATR ≈ {(d.atr/d.c).mean()*SL_ATR*100:.3f}%")

    base = simulate(d, score, SL_ATR, TP_R)
    print(f"\n=== 目前設定：止損 {SL_ATR}ATR、止盈 {TP_R}R、門檻 ±{ENTRY_TH}、最長 {MAX_HOLD_BARS*15//60}h ===")
    print("全部　" + summarize(base))
    if not base.empty:
        print("只做多 " + summarize(base[base.dir == 1]))
        print("只做空 " + summarize(base[base.dir == -1]))
        print(f"手續費平均吃掉每筆 {base.fee_r.mean():.2f}R（吃單）→ 用掛單可大幅降低")
        half = len(base) // 2
        print("前半段 " + summarize(base.iloc[:half]))
        print("後半段 " + summarize(base.iloc[half:]))
        print(f"出場原因：止盈 {(base.why=='TP').mean()*100:.0f}%｜止損 {(base.why=='SL').mean()*100:.0f}%｜"
              f"時間到 {(base.why=='TIME').mean()*100:.0f}%")
        print("\n=== 每月 ===")
        for m, g in base.groupby(base.time.dt.strftime("%Y-%m")):
            print(f"{m}  {summarize(g)}")

    print("\n=== 參數敏感度（平均 R／筆數）— 只用來看穩不穩，別挑最好的那格直接用 ===")
    rows = []
    for th in [30, 40, 50]:
        for k in [1.0, 1.5, 2.0]:
            for m in [1.0, 1.5, 2.0]:
                t = simulate(d, score, k, m, th=th)
                rows.append({"門檻": th, "止損ATR": k, "止盈R": m,
                             "平均R": round(t.r.mean(), 3) if len(t) else None, "筆數": len(t),
                             "勝率%": round((t.r > 0).mean() * 100, 1) if len(t) else None})
    print(pd.DataFrame(rows).to_string(index=False))

    t0 = simulate(d, score, SL_ATR, TP_R, fee=0)
    print("\n不扣手續費：" + summarize(t0))
    print("若平均 R 扣費後 ≤ 0，代表這套規則沒有優勢，不建議實單。")
    if not base.empty:
        send(f"🧪 XAUUSDT 回測（{d.index[0]:%m/%d}~{d.index[-1]:%m/%d}）\n" + summarize(base)
             + f"\n每筆手續費約 {base.fee_r.mean():.2f}R", dry=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--events", action="store_true")
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if a.scan:
        import conditions_scan
        conditions_scan.scan()
    elif a.events:
        print_events()
    elif a.backtest:
        backtest(a.days)
    else:
        run(status=a.status, dry=a.dry)
