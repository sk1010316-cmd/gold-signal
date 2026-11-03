# -*- coding: utf-8 -*-
"""
黃金多空訊號（GitHub Actions 每小時執行）
整合：金價趨勢/動能、美國實質利率、美元指數、原油、通膨預期、VIX、全球新聞、美國重大數據日曆
輸出：data.json（網頁讀取）、backtest.json（每週更新）、Telegram 推播

  python gold_signal.py             # 一般執行
  python gold_signal.py --report    # 強制推播完整報告
  python gold_signal.py --backtest  # 只跑回測
  python gold_signal.py --dry       # 不推播、不存狀態
"""
import os
import io
import json
import argparse
import datetime as dt
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data.json")
BT_PATH = os.path.join(BASE, "backtest.json")
EV_PATH = os.path.join(BASE, "events.json")
STATE_PATH = os.path.join(BASE, "gold_state.json")
TW = dt.timezone(dt.timedelta(hours=8))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SITE_URL = os.environ.get("SITE_URL", "")

# ====== 參數 ======
WEIGHTS = {
    "trend": 25,       # 收盤 vs 50MA、50MA vs 200MA
    "momentum": 15,    # 20 日動能（波動標準化）
    "real_yield": 20,  # 10Y 實質利率 20 日變化（反向）—— 黃金最核心的定價因子
    "dollar": 15,      # 美元指數 20 日變化（反向）
    "oil": 5,          # 原油 20 日變化（油價漲 → 通膨/地緣風險 → 偏多）
    "inflation": 5,    # 10Y 通膨預期（breakeven）20 日變化
    "vix": 5,          # 恐慌指數相對 60 日均值
    "news": 10,        # 近 24h 新聞關鍵字情緒（無法回測，權重壓低）
}
NAMES = {"trend": "金價趨勢", "momentum": "金價動能", "real_yield": "實質利率", "dollar": "美元指數",
         "oil": "原油", "inflation": "通膨預期", "vix": "VIX 避險", "news": "全球新聞"}
LONG_TH, SHORT_TH = 30, -30
EVENT_WARN_HOURS = 3
REPORT_HOUR_TW = 8          # 每天台灣時間 8 點推完整報告
COST_PER_TRADE = 0.0002
LABEL = {"LONG": "🟢 做多", "SHORT": "🔴 做空", "FLAT": "⚪ 觀望"}


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)


# ====== 資料 ======
def fetch_yf(ticker, period):
    import yfinance as yf
    s = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].dropna()


def fetch_fred(series_id):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", headers=UA, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def build_frame(period="3y"):
    df = pd.DataFrame({"gold": fetch_yf("GC=F", period)})
    sources = [("dxy", lambda: fetch_yf("DX-Y.NYB", period)),
               ("oil", lambda: fetch_yf("CL=F", period)),
               ("vix", lambda: fetch_yf("^VIX", period)),
               ("real", lambda: fetch_fred("DFII10").shift(1)),     # FRED 晚一天公布 → 延一天防偷看
               ("bei", lambda: fetch_fred("T10YIE").shift(1))]
    for col, fn in sources:
        try:
            df[col] = fn().reindex(df.index, method="ffill")
        except Exception as e:
            print(f"[警告] {col} 抓取失敗：{e}")
            df[col] = np.nan
    return df


def compute_factors(df):
    f = pd.DataFrame(index=df.index)
    c = df["gold"]
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["trend"] = 0.5 * np.sign(c - sma50) + 0.5 * np.sign(sma50 - sma200)
    vol20 = c.pct_change().rolling(60).std() * np.sqrt(20)
    f["momentum"] = ((c / c.shift(20) - 1) / vol20 / 1.5).clip(-1, 1)
    f["real_yield"] = (-(df["real"] - df["real"].shift(20)) / 0.25).clip(-1, 1)
    f["dollar"] = (-(df["dxy"] / df["dxy"].shift(20) - 1) / 0.02).clip(-1, 1)
    f["oil"] = ((df["oil"] / df["oil"].shift(20) - 1) / 0.10).clip(-1, 1)
    f["inflation"] = ((df["bei"] - df["bei"].shift(20)) / 0.15).clip(-1, 1)
    vz = (df["vix"] - df["vix"].rolling(60).mean()) / df["vix"].rolling(60).std()
    f["vix"] = (vz / 2).clip(-1, 1)
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def combine(factors, use_news):
    keys = [k for k in WEIGHTS if k in factors.columns and (use_news or k != "news")]
    return sum(factors[k] * WEIGHTS[k] for k in keys) / sum(WEIGHTS[k] for k in keys) * 100


def direction(score):
    return "LONG" if score >= LONG_TH else "SHORT" if score <= SHORT_TH else "FLAT"


# ====== 新聞 ======
BULL = ["rate cut", "dovish", "safe haven", "safe-haven", "war ", "escalat", "attack", "missile", "airstrike",
        "sanction", "central bank buying", "central banks buy", "record high", "all-time high", "weak jobs",
        "recession", "tariff", "geopolitical", "weaker dollar", "dollar falls", "dollar slides", "crisis",
        "oil surges", "oil jumps", "opec cut", "supply disruption", "inflation rises", "hotter inflation",
        "降息", "鴿派", "避險", "戰爭", "衝突", "飛彈", "空襲", "制裁", "央行購金", "創新高", "歷史新高", "衰退",
        "關稅", "美元走弱", "美元下跌", "危機", "油價大漲", "油價飆", "減產", "通膨升溫"]
BEAR = ["rate hike", "hawkish", "strong dollar", "dollar rises", "dollar jumps", "dollar gains", "ceasefire",
        "peace deal", "truce", "profit-taking", "profit taking", "selloff", "sell-off", "yields rise",
        "yields jump", "strong jobs", "risk-on", "gold falls", "gold slips", "gold drops", "oil slumps",
        "oil plunges", "inflation cools", "cooler inflation",
        "升息", "鷹派", "美元走強", "美元上漲", "停火", "和平協議", "獲利了結", "殖利率上升", "金價下跌", "金價回落",
        "油價重挫", "油價大跌", "通膨降溫"]
FEEDS = [
    ("en", "gold price"), ("en", "Federal Reserve"), ("en", "crude oil"), ("en", "geopolitical tensions"),
    ("zh", "黃金 金價"), ("zh", "原油 油價"),
]


def feed_url(lang, q):
    q = requests.utils.quote(q + " when:1d")
    if lang == "zh":
        return f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def news_sentiment():
    now = dt.datetime.now(dt.timezone.utc)
    seen, items = set(), []
    for lang, q in FEEDS:
        try:
            root = ET.fromstring(requests.get(feed_url(lang, q), headers=UA, timeout=20).content)
        except Exception as e:
            print(f"[警告] 新聞 {q} 失敗：{e}")
            continue
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            try:
                pub = parsedate_to_datetime(it.findtext("pubDate"))
            except Exception:
                continue
            if not title or title in seen or (now - pub).total_seconds() > 86400:
                continue
            seen.add(title)
            t = " " + title.lower() + " "
            s = sum(k in t for k in BULL) - sum(k in t for k in BEAR)
            items.append({"score": int(s), "title": title, "link": it.findtext("link") or "",
                          "time": pub.astimezone(TW).strftime("%m/%d %H:%M"), "topic": q})
    if not items:
        return 0.0, [], 0
    net = sum(i["score"] for i in items)
    score = float(np.clip(net / max(5, len(items) * 0.3), -1, 1))
    top = sorted([i for i in items if i["score"] != 0], key=lambda i: -abs(i["score"]))[:12]
    return score, top, len(items)


# ====== 經濟日曆 ======
def upcoming_events(hours=48):
    try:
        data = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", headers=UA, timeout=20).json()
    except Exception as e:
        print(f"[警告] 經濟日曆失敗：{e}")
        return []
    now = dt.datetime.now(TW)
    out = []
    for ev in data:
        if ev.get("country") != "USD" or ev.get("impact") != "High":
            continue
        try:
            t = dt.datetime.fromisoformat(ev["date"]).astimezone(TW)
        except Exception:
            continue
        if now - dt.timedelta(minutes=30) <= t <= now + dt.timedelta(hours=hours):
            out.append({"id": f'{ev["title"]}|{ev["date"]}', "title": ev["title"], "time": t,
                        "forecast": ev.get("forecast", ""), "previous": ev.get("previous", "")})
    return sorted(out, key=lambda e: e["time"])


# ====== Telegram ======
def send_telegram(text, dry=False):
    token, chat = os.environ.get("TG_TOKEN", ""), os.environ.get("TG_CHAT_ID", "")
    print(text)
    if dry or not token or not chat:
        return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text, "disable_web_page_preview": True}, timeout=20)
    if not r.ok:
        print("Telegram 失敗：", r.text)


# ====== 即時 ======
def live(report=False, dry=False):
    df = build_frame("3y")
    f = compute_factors(df)
    last = f.iloc[-1].copy()
    nscore, headlines, ncount = news_sentiment()
    last["news"] = nscore
    score = float(combine(last.to_frame().T, use_news=True).iloc[0])
    d = direction(score)
    now = dt.datetime.now(TW)
    events = upcoming_events(48)

    state = load_json(STATE_PATH, {"dir": None, "warned": [], "last_report": "", "history": []})
    prev = state.get("dir")
    changed = prev is not None and prev != d
    soon = [e for e in events if e["time"] <= now + dt.timedelta(hours=EVENT_WARN_HOURS)
            and e["id"] not in state.get("warned", [])]
    daily = now.hour == REPORT_HOUR_TW and state.get("last_report") != now.strftime("%Y-%m-%d")

    hist_score = combine(f, use_news=False).iloc[-180:]
    tail = df.iloc[-180:]
    data = {
        "updated": now.strftime("%Y-%m-%d %H:%M"),
        "direction": d, "score": round(score, 1), "long_th": LONG_TH, "short_th": SHORT_TH,
        "price": round(float(df["gold"].iloc[-1]), 1),
        "change_pct": round(float(df["gold"].pct_change().iloc[-1] * 100), 2),
        "data_date": df.index[-1].strftime("%Y-%m-%d"),
        "markets": {k: (None if pd.isna(df[k].iloc[-1]) else round(float(df[k].iloc[-1]), 2))
                    for k in ["dxy", "oil", "vix", "real", "bei"]},
        "factors": [{"key": k, "name": NAMES[k], "value": round(float(last[k]), 2), "weight": WEIGHTS[k]}
                    for k in WEIGHTS],
        "news": headlines, "news_count": ncount,
        "events": [{**e, "time": e["time"].strftime("%m/%d %H:%M")} for e in events],
        "chart": {"dates": [x.strftime("%Y-%m-%d") for x in tail.index],
                  "gold": [round(float(x), 1) for x in tail["gold"]],
                  "score": [round(float(x), 1) for x in hist_score]},
        "signal_log": state.get("history", [])[-20:],
    }

    # 事件反應：統計每天重算一次，即時偵測每小時
    try:
        import events_study as es
        ev_stats = load_json(EV_PATH, {})
        if not ev_stats or ev_stats.get("generated", "")[:10] != now.strftime("%Y-%m-%d"):
            ev_stats = es.study()
            if not dry:
                save_json(EV_PATH, ev_stats)
        data["active_events"] = es.detect_live(ev_stats)
    except Exception as e:
        print("[警告] 事件研究失敗：", e)
        data["active_events"] = []

    lines = [f"🥇 黃金多空訊號  {now:%m/%d %H:%M}"]
    if changed:
        lines.append(f"🔔 方向改變：{LABEL[prev]} → {LABEL[d]}")
    lines += [f"方向：{LABEL[d]}（分數 {score:+.0f}，門檻 ±{LONG_TH}）",
              f"金價：{data['price']:,.1f}（{data['change_pct']:+.2f}%）", "", "— 因子（-1 空 ~ +1 多）—"]
    lines += [f"{NAMES[k]}：{last[k]:+.2f}" for k in WEIGHTS]
    if headlines:
        lines += ["", "— 重點新聞 —"] + [f"{'▲' if h['score'] > 0 else '▼'} {h['title'][:60]}" for h in headlines[:4]]
    near = [e for e in events if e["time"] <= now + dt.timedelta(hours=24)]
    if near:
        lines += ["", "— 24h 內美國重大數據 —"] + [f"⚠️ {e['time']:%m/%d %H:%M} {e['title']}" for e in near]
        lines.append("公布前後 30 分鐘避免進場")
    if SITE_URL:
        lines += ["", SITE_URL]
    lines.append("※ 統計訊號僅供參考")
    msg = "\n".join(lines)

    if report or changed or soon or daily:
        send_telegram(msg, dry)
    else:
        print("無需推播\n" + msg)

    if not dry:
        if prev != d:
            state.setdefault("history", []).append({"time": data["updated"], "dir": d, "price": data["price"],
                                                    "score": data["score"]})
            data["signal_log"] = state["history"][-20:]
        state["dir"] = d
        state["warned"] = (state.get("warned", []) + [e["id"] for e in soon])[-50:]
        if daily:
            state["last_report"] = now.strftime("%Y-%m-%d")
        save_json(STATE_PATH, state)
        save_json(DATA_PATH, data)

    bt = load_json(BT_PATH, {})
    if not bt or (now - dt.datetime.fromisoformat(bt.get("generated_iso", "2000-01-01T00:00:00+08:00"))).days >= 7:
        backtest()


# ====== 回測 ======
def stats(r):
    r = r.dropna()
    if len(r) < 20:
        return None
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return {"cagr": round(cagr * 100, 1), "sharpe": round(float(sharpe), 2),
            "mdd": round(float((eq / eq.cummax() - 1).min() * 100), 1), "total": round(float(eq.iloc[-1] - 1) * 100, 0)}


def backtest():
    print("回測中...")
    df = build_frame("max")
    df = df[df.index >= "2006-01-01"]
    f = compute_factors(df)
    score = combine(f, use_news=False)
    ret = df["gold"].pct_change()
    pos = score.apply(direction).map({"LONG": 1, "SHORT": -1, "FLAT": 0})

    def strat(p):
        p = p.shift(1)                       # 今天收盤出訊號 → 吃明天報酬
        return p * ret - p.diff().abs().fillna(0) * COST_PER_TRADE

    W = 201
    res = {"多空策略": strat(pos).iloc[W:], "只做多": strat(pos.clip(lower=0)).iloc[W:], "買進持有": ret.iloc[W:]}
    p = pos.iloc[W:]
    active = res["多空策略"][pos.shift(1).iloc[W:] != 0]
    periods = [("2006", "2012"), ("2013", "2018"), ("2019", "2021"), ("2022", "2099")]
    seg = []
    for a, b in periods:
        row = {"period": f"{a}-{'今' if b == '2099' else b}"}
        for k, v in res.items():
            row[k] = stats(v[(v.index >= f"{a}-01-01") & (v.index <= f"{b}-12-31")])
        seg.append(row)
    fwd5 = df["gold"].shift(-5) / df["gold"] - 1
    fac = []
    for k in [x for x in WEIGHTS if x != "news"]:
        sig, fr = np.sign(f[k]).iloc[W:], fwd5.iloc[W:]
        m = (sig != 0) & fr.notna()
        fac.append({"name": NAMES[k], "hit": round(float((np.sign(fr[m]) == sig[m]).mean() * 100), 1) if m.any() else None,
                    "ic": round(float(f[k].iloc[W:].corr(fr)), 3)})
    eq = {k: (1 + v.fillna(0)).cumprod() for k, v in res.items()}
    step = max(1, len(p) // 400)
    out = {
        "generated": dt.datetime.now(TW).strftime("%Y-%m-%d %H:%M"),
        "generated_iso": dt.datetime.now(TW).isoformat(),
        "range": f"{p.index[0]:%Y-%m-%d} ~ {p.index[-1]:%Y-%m-%d}",
        "summary": {k: stats(v) for k, v in res.items()},
        "exposure": {"long": round(float((p == 1).mean() * 100)), "short": round(float((p == -1).mean() * 100)),
                     "flat": round(float((p == 0).mean() * 100))},
        "trades": int((p.diff() != 0).sum()),
        "winrate": round(float((active > 0).mean() * 100), 1),
        "periods": seg, "factors": fac,
        "equity": {"dates": [x.strftime("%Y-%m-%d") for x in p.index[::step]],
                   **{k: [round(float(x), 3) for x in v.iloc[::step]] for k, v in eq.items()}},
    }
    save_json(BT_PATH, out)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.backtest:
        backtest()
    else:
        live(report=a.report, dry=a.dry)
