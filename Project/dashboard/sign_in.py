"""The sign-in page."""
import streamlit as st

from config import config
from dashboard.api_client import SESSION_ENDED_KEY, SIGNED_IN_KEY, call_service


def render_sign_in_page():
    """Let a person sign in."""
    _left, centre, _right = st.columns([1, 2, 1])
    with centre:
        with st.container(border=True):
            st.title(config.DASHBOARD_TITLE)
            st.caption(config.DASHBOARD_SUBTITLE)
            if st.session_state.pop(SESSION_ENDED_KEY, False):
                st.info(config.SESSION_ENDED_MESSAGE, icon=":material/logout:")
            with st.form("sign_in_form", border=False):
                username = st.text_input("Username", autocomplete="username")
                password = st.text_input("Password", type="password",
                                         autocomplete="current-password")
                submitted = st.form_submit_button("Sign in", type="primary",
                                                  icon=":material/login:", width="stretch")
            if submitted:
                reply = call_service("POST", "/auth/signin",
                                     payload={"username": username.strip(),
                                              "password": password})
                st.session_state[SIGNED_IN_KEY] = reply
                st.rerun()
