"""The dashboard's front door: the service check, signing in, and the navigation."""
import requests
import streamlit as st

from config import config
from dashboard.api_client import call_service, end_browser_session, signed_in_user
from dashboard.audit_trail import render_audit_trail_page
from dashboard.decision_queue import render_decision_queue_page
from dashboard.display import NAVIGATION_PAGES_KEY
from dashboard.permit_chat import render_permit_chat_page
from dashboard.request_detail import render_request_detail_page
from dashboard.request_queue import render_request_queue_page
from dashboard.sign_in import render_sign_in_page

PAGE_RENDERERS = {
    config.PAGE_QUEUE: render_request_queue_page,
    config.PAGE_DETAIL: render_request_detail_page,
    config.PAGE_DECISION: render_decision_queue_page,
    config.PAGE_AUDIT: render_audit_trail_page,
    config.PAGE_CHAT: render_permit_chat_page,
}


def _service_answers():
    try:
        requests.get(config.API_BASE_URL + "/openapi.json", timeout=5)
        return True
    except requests.RequestException:
        return False


def _sidebar(user):
    with st.sidebar:
        st.markdown(f"### :material/construction: {config.DASHBOARD_TITLE}")
        with st.container(border=True):
            st.markdown(f"**{user['display_name']}**")
            st.caption(user["role"] + (f", {user['area_group']}" if user["area_group"] else ""))
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            call_service("POST", "/auth/signout")
            end_browser_session()
            st.rerun()


def main():
    """One screen to work from, with only the pages the signed-in person's role is offered."""
    st.set_page_config(page_title=config.DASHBOARD_TITLE, page_icon=config.DASHBOARD_ICON,
                       layout="wide")
    st.markdown(config.DASHBOARD_PAGE_STYLE, unsafe_allow_html=True)
    if not _service_answers():
        st.error(f"The {config.API_TITLE} is not answering at {config.API_BASE_URL}. Start it "
                 "from the Project folder with: python3 run_api_service.py",
                 icon=":material/cloud_off:")
        return
    user = signed_in_user()
    if user is None:
        st.navigation([st.Page(render_sign_in_page, title="Sign in", url_path="sign-in",
                               icon=":material/login:")], position="hidden").run()
        return
    _sidebar(user)
    pages = {name: st.Page(PAGE_RENDERERS[name], title=name, icon=config.PAGE_ICONS[name],
                           url_path=config.PAGE_URL_PATHS[name])
             for name in user["pages"] if name in PAGE_RENDERERS}
    st.session_state[NAVIGATION_PAGES_KEY] = pages
    st.navigation(list(pages.values())).run()


if __name__ == "__main__":
    main()
