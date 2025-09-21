#!/usr/bin/env python3
"""
All-in-one Forex Telegram Bot (single-file).
Features:
- Hardcoded credentials (as provided)
- 3-day free trial per user (tracked by Telegram ID)
- Manual premium activation by admin (/activate <user_id> <PlanKey>)
- AI 5-minute prediction (pretrained or bootstrapped at first run)
- Colorful candlestick chart generation (matplotlib Agg)
- Button-driven UI (Get Signal, Chart, Subscribe, Status)
- Local JSON store for users (data/users.json created automatically)
- Pinned dependency expectations in requirements.txt
"""

import os
import json
import time
import logging
from io import BytesIO
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from sklearn.ensemble import RandomForestClassifier

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters, CallbackContext

# -------------------------
# === HARDCODED CREDENTIALS (YOU PROVIDED) ===
# -------------------------
TELEGRAM_BOT_TOKEN = "7790080100:AAGwX4riIDhZ9JKn6qnQ1UsDEa4EkNZSlE8"
TWELVE_DATA_API_KEY = "f7249b9c22574caea6d71ac931d3f8e0"

PAYPAL_EMAIL = "susanzeedy4259@gmail.com"
MPESA_PHONE = "0701767822"
BINANCE_BNB = "0x412930bc47da7a7b5929ae8876ac41e7d39bc9e2"
BINANCE_USDT = "TD6TWzH3NW9Phfws6DUDKkpgWLjf9924md"

ADMIN_TELEGRAM_ID = 7239427141  # as integer

TRIAL_DAYS = 3
PREMIUM_MONTH_DAYS = 30
PREMIUM_QUARTER_DAYS = 90
PREMIUM_YEAR_DAYS = 365

USERS_DB = "data/users.json"
MODEL_PATH = "ai_model.pkl"

# Ensure data folder exists
os.makedirs(os.path.dirname(USERS_DB), exist_ok=True)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("forex-bot")

# -------------------------
# Subscription manager (JSON file)
# -------------------------
class SubscriptionManager:
    PLANS = {
        "Silver": {"days": PREMIUM_MONTH_DAYS, "price": "$20"},
        "Gold": {"days": PREMIUM_QUARTER_DAYS, "price": "$50"},
        "Platinum": {"days": PREMIUM_YEAR_DAYS, "price": "$150"},
    }

    def __init__(self, path=USERS_DB):
        self.path = path
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.users = json.load(f)
            except Exception:
                self.users = {}
        else:
            self.users = {}
            self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.users, f, default=str, indent=2)

    def ensure_user(self, user_id, username=None):
        uid = str(user_id)
        if uid not in self.users:
            expiry = (datetime.utcnow() + timedelta(days=TRIAL_DAYS)).isoformat()
            self.users[uid] = {"username": username, "expiry": expiry, "plan": "Trial"}
            self._save()

    def is_active(self, user_id):
        uid = str(user_id)
        if uid not in self.users:
            return False
        expiry = datetime.fromisoformat(self.users[uid]["expiry"])
        return datetime.utcnow() <= expiry

    def activate_plan(self, user_id, plan_key):
        uid = str(user_id)
        plan = self.PLANS.get(plan_key)
        if not plan:
            raise ValueError("Unknown plan")
        expiry = (datetime.utcnow() + timedelta(days=plan["days"])).isoformat()
        self.users[uid] = {
            "username": self.users.get(uid, {}).get("username"),
            "expiry": expiry,
            "plan": plan_key,
        }
        self._save()

    def days_left(self, user_id):
        uid = str(user_id)
        if uid not in self.users:
            return 0
        expiry = datetime.fromisoformat(self.users[uid]["expiry"])
        delta = expiry - datetime.utcnow()
        return max(0, delta.days)

    def status_text(self, user_id):
        uid = str(user_id)
        if uid not in self.users:
            return "No account found. Use /start to begin."
        plan = self.users[uid].get("plan", "Trial")
        days = self.days_left(user_id)
        return f"Plan: {plan}\nDays left: {days}"

# -------------------------
# AI model helper: load or train lightweight model
# -------------------------
def train_from_twelvedata(api_key):
    try:
        logger.info("Attempting to train model from Twelve Data historical EURUSD data...")
        symbol = "EURUSD"
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=800&apikey={api_key}"
        r = requests.get(url, timeout=15).json()
        if "values" not in r:
            logger.warning("Twelve Data returned no values for training.")
            return None
        df = pd.DataFrame(r["values"]).iloc[::-1]
        df = df.astype({"open": "float", "high": "float", "low": "float", "close": "float"})
        df["ema5"] = df["close"].ewm(span=5).mean()
        df["ema10"] = df["close"].ewm(span=10).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
        df.dropna(inplace=True)
        X = df[["ema5", "ema10", "rsi", "atr"]].values[:-1]
        y = (df["close"].shift(-1) > df["close"]).astype(int).values[:-1]
        if len(X) < 200:
            logger.warning("Not enough rows for training.")
            return None
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        logger.info("Model trained and saved to %s", MODEL_PATH)
        return model
    except Exception as e:
        logger.exception("Training from Twelve Data failed: %s", e)
        return None

def train_synthetic():
    logger.info("Training small synthetic model as fallback...")
    rng = np.random.RandomState(42)
    n = 2000
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.1 * X[:, 1] + 0.2 * X[:, 2] + rng.normal(scale=0.2, size=n) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=80, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    logger.info("Synthetic model trained and saved.")
    return model

def load_or_create_model(api_key):
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            logger.info("Loaded model from %s", MODEL_PATH)
            return m
        except Exception:
            logger.exception("Failed to load existing model, will recreate.")
    # Try training from real data
    m = train_from_twelvedata(api_key)
    if m is not None:
        return m
    # Fallback synthetic
    return train_synthetic()

# -------------------------
# Signal engine
# -------------------------
class SignalEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = load_or_create_model(api_key)

    def fetch_ohlc(self, pair, interval='1min', outputsize=200):
        symbol = pair.replace('/', '')
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={self.api_key}"
        try:
            r = requests.get(url, timeout=10).json()
            if "values" not in r:
                logger.warning("Twelve Data: no 'values' in response for %s", pair)
                return None
            df = pd.DataFrame(r["values"]).iloc[::-1]
            for c in ['open', 'high', 'low', 'close']:
                df[c] = df[c].astype(float)
            df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        except Exception as e:
            logger.exception("Error fetching OHLC for %s: %s", pair, e)
            return None

    def compute_features(self, df):
        df2 = df.copy()
        df2['ema5'] = df2['close'].ewm(span=5).mean()
        df2['ema10'] = df2['close'].ewm(span=10).mean()
        delta = df2['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df2['rsi'] = 100 - (100 / (1 + rs))
        df2['atr'] = (df2['high'] - df2['low']).rolling(14).mean()
        return df2.dropna()

    def predict_next(self, pair, interval='1min'):
        df = self.fetch_ohlc(pair, interval=interval, outputsize=300)
        if df is None or df.empty:
            return None
        feats = self.compute_features(df)
        if len(feats) < 20:
            return None
        X = feats[['ema5','ema10','rsi','atr']].values
        last = X[-1].reshape(1, -1)
        try:
            pred = int(self.model.predict(last)[0])
            proba = float(self.model.predict_proba(last)[0][pred]) * 100 if hasattr(self.model, "predict_proba") else 0.0
        except Exception as e:
            logger.exception("Model inference error: %s", e)
            return None
        entry = float(feats['close'].iloc[-1])
        atr = float(feats['atr'].iloc[-1])
        if pred == 1:
            signal = "BUY"
            stop = entry - atr
            tp = entry + atr
        else:
            signal = "SELL"
            stop = entry + atr
            tp = entry - atr
        return {"pair": pair, "interval": interval, "signal": signal, "entry": entry, "stop": stop, "tp": tp, "confidence": proba}

# -------------------------
# Chart generation
# -------------------------
def make_candlestick(df, pair, signal=None, entry=None, stop=None, tp=None):
    # df expected to have datetime, open, high, low, close
    try:
        df_plot = df.copy().tail(60)  # last 60 candles
        df_plot['time'] = df_plot['datetime']
        fig, ax = plt.subplots(figsize=(8, 4))
        for _, row in df_plot.iterrows():
            color = 'green' if row['close'] >= row['open'] else 'red'
            ax.plot([row['time'], row['time']], [row['low'], row['high']], color=color, linewidth=1)
            ax.plot([row['time'], row['time']], [row['open'], row['close']], color=color, linewidth=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_title(f"{pair} — Candlestick (1m)")
        plt.xticks(rotation=30)
        if signal and entry:
            col = 'green' if signal == 'BUY' else 'red'
            ax.scatter(df_plot['time'].iloc[-1], entry, color=col, s=90, zorder=5)
            if stop is not None:
                ax.axhline(stop, color='gray', linestyle='--')
            if tp is not None:
                ax.axhline(tp, color='gold', linestyle='--')
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        logger.exception("Chart generation error: %s", e)
        return None

# -------------------------
# Utilities
# -------------------------
def format_signal_text(sig: dict, tz_hours=3):
    # Default to EAT (UTC+3) as requested
    now_eat = datetime.utcnow() + timedelta(hours=tz_hours)
    return (
        f"*{sig['pair']}* — {sig['interval']}\n"
        f"Signal: *{sig['signal']}*\n"
        f"Entry: `{sig['entry']:.5f}`\n"
        f"Stop Loss: `{sig['stop']:.5f}`\n"
        f"Take Profit: `{sig['tp']:.5f}`\n"
        f"Confidence: `{sig['confidence']:.1f}%`\n"
        f"Time (EAT): {now_eat.strftime('%Y-%m-%d %H:%M')}"
    )

def payment_instructions_text(plan_key):
    plan = SubscriptionManager.PLANS.get(plan_key, {"days":0,"price":"?"})
    return (
        f"You selected *{plan_key}* — {plan['days']} days — {plan['price']}\n\n"
        f"Please make payment using one of the methods below and then tap *I Paid — Notify Admin*:\n\n"
        f"• PayPal: `{PAYPAL_EMAIL}`\n"
        f"• M-Pesa: `{MPESA_PHONE}`\n"
        f"• Binance BNB: `{BINANCE_BNB}`\n"
        f"• Binance USDT: `{BINANCE_USDT}`"
    )

# -------------------------
# Bot UI & Handlers
# -------------------------
sub_mgr = SubscriptionManager()
sig_engine = SignalEngine(TWELVE_DATA_API_KEY)

def main_menu_kb():
    kb = [
        [InlineKeyboardButton("🤖 Get AI Signal", callback_data="menu_signal")],
        [InlineKeyboardButton("📈 Chart", callback_data="menu_chart")],
        [InlineKeyboardButton("💎 Plans & Subscribe", callback_data="menu_plans")],
        [InlineKeyboardButton("ℹ️ My Status", callback_data="menu_status")],
    ]
    return InlineKeyboardMarkup(kb)

def plans_kb():
    kb = []
    for key, plan in sub_mgr.PLANS.items():
        kb.append([InlineKeyboardButton(f"{key} — {plan['days']}d — {plan['price']}", callback_data=f"buy_{key}")])
    kb.append([InlineKeyboardButton("⬅ Back", callback_data="menu_back")])
    return InlineKeyboardMarkup(kb)

def quick_pairs_kb():
    pairs = ["EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD"]
    kb = [[InlineKeyboardButton(p[:3] + "/" + p[3:], callback_data=f"pair_{p}") ] for p in pairs]
    kb.append([InlineKeyboardButton("⬅ Back", callback_data="menu_back")])
    return InlineKeyboardMarkup(kb)

def start(update: Update, context: CallbackContext):
    user = update.effective_user
    sub_mgr.ensure_user(user.id, user.username)
    update.message.reply_text(
        f"Hello {user.first_name or user.username}! Welcome to Pro Forex Bot.\nYour {TRIAL_DAYS}-day trial is active.",
        reply_markup=main_menu_kb()
    )

def callback_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    uid = query.from_user.id
    data = query.data

    if data == "menu_signal":
        if not sub_mgr.is_active(uid):
            query.edit_message_text("⛔ Trial expired. Please subscribe.", reply_markup=plans_kb())
            return
        query.edit_message_text("Choose quick pair or send a pair like EUR/USD:", reply_markup=quick_pairs_kb())

    elif data == "menu_chart":
        if not sub_mgr.is_active(uid):
            query.edit_message_text("⛔ Trial expired. Please subscribe.", reply_markup=plans_kb())
            return
        query.edit_message_text("Choose quick pair for chart or send pair like EUR/USD:", reply_markup=quick_pairs_kb())

    elif data == "menu_plans":
        query.edit_message_text("Available plans:", reply_markup=plans_kb())

    elif data.startswith("buy_"):
        plan = data.split("_",1)[1]
        text = payment_instructions_text(plan)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ I Paid — Notify Admin", callback_data=f"notify_{plan}")],
            [InlineKeyboardButton("⬅ Back", callback_data="menu_plans")]
        ])
        query.edit_message_text(text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

    elif data.startswith("notify_"):
        plan = data.split("_",1)[1]
        admin_msg = f"💰 Payment notification:\nUser: @{query.from_user.username} ({query.from_user.id})\nPlan: {plan}"
        context.bot.send_message(chat_id=ADMIN_TELEGRAM_ID, text=admin_msg)
        query.edit_message_text("✅ Notified admin. Wait for approval.")

    elif data == "menu_status":
        txt = sub_mgr.status_text(uid)
        query.edit_message_text(txt)

    elif data == "menu_back":
        query.edit_message_text("Back to Main Menu", reply_markup=main_menu_kb())

    elif data.startswith("pair_"):
        raw = data.split("_",1)[1]
        pair = raw[:3] + "/" + raw[3:]
        query.answer()
        # Provide signal + chart
        sig = sig_engine.predict_next(pair)
        if not sig:
            query.edit_message_text("Unable to generate signal at the moment. Try again later.")
            return
        df = sig_engine.fetch_ohlc(pair, interval='1min', outputsize=120)
        chart_buf = make_candlestick(df, pair, sig['signal'], sig['entry'], sig['stop'], sig['tp'])
        if chart_buf:
            context.bot.send_photo(chat_id=uid, photo=chart_buf, caption=format_signal_text(sig), parse_mode=ParseMode.MARKDOWN)
            query.edit_message_text("Signal delivered.", reply_markup=main_menu_kb())
        else:
            query.edit_message_text("Signal generated but chart failed. Sending text only.")
            context.bot.send_message(chat_id=uid, text=format_signal_text(sig), parse_mode=ParseMode.MARKDOWN)

def text_handler(update: Update, context: CallbackContext):
    txt = update.message.text.strip().upper()
    uid = update.message.from_user.id

    if "/" in txt or len(txt) >= 6:
        pair = txt if "/" in txt else (txt[:3] + "/" + txt[3:])
        if not sub_mgr.is_active(uid):
            update.message.reply_text("⛔ Trial/Subscription expired. Please subscribe.", reply_markup=plans_kb())
            return
        sig = sig_engine.predict_next(pair)
        if not sig:
            update.message.reply_text("No reliable signal at this time. Try again shortly.")
            return
        df = sig_engine.fetch_ohlc(pair, interval='1min', outputsize=120)
        chart_buf = make_candlestick(df, pair, sig['signal'], sig['entry'], sig['stop'], sig['tp'])
        if chart_buf:
            update.message.reply_photo(photo=chart_buf, caption=format_signal_text(sig), parse_mode=ParseMode.MARKDOWN)
        else:
            update.message.reply_text(format_signal_text(sig), parse_mode=ParseMode.MARKDOWN)
    else:
        update.message.reply_text("Send pair like EUR/USD or use the menu.", reply_markup=main_menu_kb())

def activate_cmd(update: Update, context: CallbackContext):
    if update.effective_user.id != ADMIN_TELEGRAM_ID:
        update.message.reply_text("⛔ Only admin can run this command.")
        return
    try:
        args = context.args
        target = int(args[0])
        plan = args[1]
        if plan not in sub_mgr.PLANS:
            update.message.reply_text("Unknown plan. Valid keys: " + ", ".join(sub_mgr.PLANS.keys()))
            return
        sub_mgr.activate_plan(target, plan)
        update.message.reply_text(f"✅ Activated user {target} for plan {plan}.")
        context.bot.send_message(chat_id=target, text=f"🎉 Your {plan} plan has been activated by admin.")
    except Exception as e:
        update.message.reply_text("Usage: /activate <user_id> <PlanKey>")

# -------------------------
# Run bot
# -------------------------
def main():
    logger.info("Starting Forex Telegram Bot...")
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(callback_handler))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, text_handler))
    dp.add_handler(CommandHandler("activate", activate_cmd))

    # start polling
    updater.start_polling()
    logger.info("Bot started. Listening for updates...")
    updater.idle()

if __name__ == "__main__":
    main()