"""The permit assistant: questions answered from Hexley's documents and the records."""
import streamlit as st

from config import config
from dashboard.api_client import call_service
from dashboard.display import chat_type_badge


def _show_turn(turn):
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(turn["question_text"])
    with st.chat_message("assistant", avatar=":material/construction:"):
        st.markdown(chat_type_badge(turn["question_type"]))
        st.markdown(turn["answer_text"])
        if turn["source_documents"]:
            with st.expander("Sources"):
                for name in turn["source_documents"].split(", "):
                    st.markdown(f"- {name}")


def render_permit_chat_page():
    """Let anyone signed in ask the permit assistant, and show their own conversation."""
    st.title(config.PAGE_CHAT)
    st.caption("Ask about a permit standard, the Control of Work Policy, the Permit Rules, a "
               "statement guide, a permit request by its reference, or a plant area.")
    turns = call_service("GET", "/chat/turns", query={"turn_limit": config.CHAT_PAGE_TURN_LIMIT})
    if not turns:
        st.info("Your conversation starts with your first question.",
                icon=":material/forum:")
    for turn in reversed(turns):
        _show_turn(turn)
    question = st.chat_input("Ask about a permit standard, the policy, the Permit Rules, a "
                             "statement guide, a permit request or a plant area",
                             max_chars=config.CHAT_QUESTION_MAXIMUM_CHARACTERS)
    if question is None:
        return
    if len(question.strip()) < config.CHAT_QUESTION_MINIMUM_CHARACTERS:
        st.warning(f"A question needs at least {config.CHAT_QUESTION_MINIMUM_CHARACTERS} "
                   "characters.")
        return
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)
    call_service("POST", "/chat/ask", payload={"question": question},
                 waiting_message="The permit assistant is reading the documents",
                 failure_message=config.CHAT_FAILED_MESSAGE)
    st.rerun()
