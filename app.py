"""
AI Multi-Tool Platform

"""
import streamlit as st
import pandas as pd

from utils.auth import init_db, create_user, verify_user
from utils.analytics import (
    log_usage, get_usage_df, get_module_counts,
    get_daily_activity, get_total_stats,
)
from modules import chat_assistant, summarizer, sentiment

st.set_page_config(
    page_title="AI Multi-Tool Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# styling
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    div[data-testid="stMetric"] {
        background: #f7f8fa; border-radius: 12px; padding: 1rem;
        border: 1px solid #e5e7eb;
    }
    .app-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .app-subtitle { color: #6b7280; margin-top: 0; }
    @media (max-width: 640px) {
        .main .block-container { padding-left: 0.5rem; padding-right: 0.5rem; }
    }
</style>
""", unsafe_allow_html=True)

#  session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None


def login_view():
    st.markdown('<p class="app-title">🧠 AI Multi-Tool Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Chat Assistant · Text Summarizer · Sentiment Analysis</p>', unsafe_allow_html=True)
    st.write("")

    tab_login, tab_register = st.tabs(["Log In", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)
            if submitted:
                if verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with st.expander("Just want to explore? Use the demo account"):
            st.code("username: demo\npassword: demo123")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    ok, message = create_user(new_username, new_password)
                    if ok:
                        st.success(message + " You can now log in.")
                    else:
                        st.error(message)


def ensure_demo_account():
    from utils.auth import user_exists
    if not user_exists("demo"):
        create_user("demo", "demo123")


def dashboard_view():
    st.markdown('<p class="app-title">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="app-subtitle">Welcome back, <b>{st.session_state.username}</b></p>', unsafe_allow_html=True)

    total_events, active_users = get_total_stats()
    module_counts = get_module_counts()
    daily = get_daily_activity()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total interactions", total_events)
    c2.metric("Active users", active_users)
    c3.metric("Modules used", module_counts.shape[0] if not module_counts.empty else 0)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Usage by module")
        if module_counts.empty:
            st.info("No usage yet — try out a module and come back!")
        else:
            st.bar_chart(module_counts)
    with col2:
        st.subheader("Activity over time")
        if daily.empty:
            st.info("No activity logged yet.")
        else:
            st.line_chart(daily)

    st.write("")
    st.subheader("Recent activity")
    df = get_usage_df()
    if df.empty:
        st.info("Nothing logged yet.")
    else:
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)


def main_app():
    with st.sidebar:
        st.markdown(f"### Hii! {st.session_state.username}")
        page = st.radio(
            "Navigate",
            ["Dashboard", "AI Chat Assistant", "Text Summarizer", "Sentiment Analysis"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        if st.button("Log Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()

    if page == "Dashboard":
        dashboard_view()
    elif page == "AI Chat Assistant":
        chat_assistant.render(log_usage_fn=log_usage, username=st.session_state.username)
    elif page == "Text Summarizer":
        summarizer.render(log_usage_fn=log_usage, username=st.session_state.username)
    elif page == "Sentiment Analysis":
        sentiment.render(log_usage_fn=log_usage, username=st.session_state.username)


ensure_demo_account()

if st.session_state.authenticated:
    main_app()
else:
    login_view()
