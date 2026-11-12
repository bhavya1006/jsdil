"""Every request the dashboard sends to the service passes through here, so no page holds an
address, a timeout or a session token."""
from contextlib import nullcontext

import requests
import streamlit as st

from config import config

SIGNED_IN_KEY = "signed_in_user"
SESSION_ENDED_KEY = "session_ended"


def signed_in_user():
    """The sign-in reply kept in the browser session, or None."""
    return st.session_state.get(SIGNED_IN_KEY)


def end_browser_session(session_ended=False):
    """Forget the signed-in person in this browser."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if session_ended:
        st.session_state[SESSION_ENDED_KEY] = True


def _detail_of(reply):
    try:
        body = reply.json()
    except ValueError:
        return f"{reply.status_code} {reply.text}"
    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                         for item in detail)
    return str(detail)


def call_service(method, path, query=None, payload=None, waiting_message=None,
                 failure_message=None):
    """Make one call to the service and give the page back the reply read from JSON."""
    user = signed_in_user()
    headers = {config.SESSION_TOKEN_HEADER: user["session_token"]} if user else {}
    waiting = (st.spinner(waiting_message, show_time=True) if waiting_message
               else nullcontext())
    with waiting:
        try:
            reply = requests.request(method, config.API_BASE_URL + path, params=query,
                                     json=payload, headers=headers,
                                     timeout=config.API_REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as error:
            st.error(f"The service at {config.API_BASE_URL} could not be reached: {error}")
            st.stop()
    if reply.status_code == 401 and user:
        end_browser_session(session_ended=True)
        st.rerun()
    if reply.status_code >= 400:
        detail = _detail_of(reply)
        if reply.status_code == 502 and failure_message:
            st.error(failure_message.format(reason=detail.rstrip(".")))
        else:
            st.error(detail)
        st.stop()
    return reply.json()
