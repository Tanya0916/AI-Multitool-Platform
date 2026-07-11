"""
Usage analytics: logs each module interaction to SQLite and provides
aggregate views for the dashboard (per-module counts, per-user counts,
activity over time).
"""
import pandas as pd
from utils.auth import get_connection


def log_usage(username: str, module: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO usage_log (username, module) VALUES (?, ?)",
        (username, module),
    )
    conn.commit()
    conn.close()


def get_usage_df() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT username, module, timestamp FROM usage_log ORDER BY timestamp DESC",
        conn,
    )
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
    return df


def get_module_counts() -> pd.Series:
    df = get_usage_df()
    if df.empty:
        return pd.Series(dtype=int)
    return df["module"].value_counts()


def get_daily_activity() -> pd.Series:
    df = get_usage_df()
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby("date").size()


def get_user_counts() -> pd.Series:
    conn = get_connection()
    df = pd.read_sql_query("SELECT username FROM users", conn)
    conn.close()
    return df


def get_total_stats():
    df = get_usage_df()
    total_events = len(df)
    active_users = df["username"].nunique() if not df.empty else 0
    return total_events, active_users
