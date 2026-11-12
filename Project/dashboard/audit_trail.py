"""What has been reviewed and decided, the record behind each review, and every chat turn."""
import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service
from dashboard.display import (answer_label, band_badge, decision_badge, extension_text,
                               hours_text, likelihood_text, live_permits_text, markdown_table,
                               requirements_by_standard, result_badge, status_badge)
from dashboard.decision_queue import _WRITE_UPS, _write_up


def _simops_brief(brief):
    st.markdown("**SIMOPS brief**")
    st.dataframe([{"Detail": answer_label(column), "Answer": brief.get(column)}
                  for column in config.SIMOPS_BRIEF_DETAIL_COLUMNS],
                 hide_index=True, width="stretch")
    markdown_table(["Permit type", "Requirement", "Result", "Answers needed"], [
        [item["permit_type"], item["requirement_name"], result_badge(item["result"]),
         ", ".join(answer_label(column) for column in item["blank_answers"]) or "—"]
        for item in brief["requirements"]])
    st.markdown("**SIMOPS triggers:** "
                + ("; ".join(brief["simops_triggers"]) or config.CHAT_NONE_TEXT))
    st.markdown("**Permits live in the plant area:** "
                + live_permits_text(brief["live_permits"], brief["conflicting_permits"]))
    st.markdown(f"**Flagged statements:** {len(brief['flagged_statements'])}")
    st.markdown(f"**Figures:** suspension likelihood "
                f"{likelihood_text(brief['suspension_likelihood'])} "
                f"{band_badge(brief['likelihood_band'])}, expected work hours "
                f"{hours_text(brief['expected_work_hours'])}, extension expected "
                f"{extension_text(brief['extension_expected'])}")


def _call_budget(review_id):
    budget = call_service("GET", f"/reviews/{review_id}/calls")
    st.markdown("**Call budget**")
    figures = st.columns(3)
    figures[0].metric("Model calls", f"{budget['call_count']} of {budget['expected_call_count']}",
                      border=True, icon=":material/smart_toy:")
    figures[1].metric("Total tokens", f"{budget['total_tokens']:,}", border=True,
                      icon=":material/token:")
    seconds = budget["total_latency_milliseconds"] / config.MILLISECONDS_PER_SECOND
    figures[2].metric("Total latency",
                      f"{seconds:.{config.LATENCY_SECONDS_DECIMALS}f} s", border=True,
                      icon=":material/timer:")
    st.dataframe(pd.DataFrame(budget["calls"]).rename(columns={
        "agent_name": "Agent", "model_name": "Model", "prompt_tokens": "Prompt tokens",
        "response_tokens": "Response tokens", "total_tokens": "Total tokens",
        "latency_milliseconds": "Latency (ms)"}), hide_index=True, width="stretch")


def _review_record(review):
    with st.container(border=True):
        st.markdown(f"**{review['review_id']}** · {review['permit_reference']} revision "
                    f"{review['revision_number']} " + status_badge(review["status"]))
        st.markdown(f"Assessment {decision_badge(review['assessment'])} · recommended "
                    f"{decision_badge(review['recommended_decision'])}")
        st.markdown(review["recommendation_text"])
        st.markdown(f"**Expected work hours:** {review['work_hours_text']}")
        if review["decision"]:
            st.markdown(f"**Decision:** {decision_badge(review['decision'])} by "
                        f"{review['decided_by']}, {review['decided_by_role']} for "
                        f"{review['decided_by_area_group']}")
            st.markdown(f"> {review['decision_reason']}")
        requirements_by_standard(review["requirements"], with_sentences=False)
        with st.expander("Each agent's write-up"):
            st.markdown("**What was asked**")
            st.markdown(review["coordinator_request"])
            for title, key in _WRITE_UPS:
                st.markdown(f"**{title}**")
                _write_up(review[key])
        if review["permit_document_text"] and review["decision"] in config.DOCUMENT_KINDS:
            kind = config.DOCUMENT_KINDS[review["decision"]]
            with st.expander(kind[:1].upper() + kind[1:], expanded=True):
                _write_up(review["permit_document_text"])
        if review["work_party_briefing_text"]:
            with st.expander("Work party briefing", expanded=True):
                _write_up(review["work_party_briefing_text"])
        if review["simops_brief"]:
            _simops_brief(review["simops_brief"])
    _call_budget(review["review_id"])


def _reviews_tab():
    chosen = st.pills("Status", config.REVIEW_STATUS_OPTIONS, selection_mode="multi",
                      default=config.DECIDED_REVIEW_STATUSES, key="audit_statuses",
                      format_func=lambda status: config.REQUEST_STATUS_LABELS[status])
    if not chosen:
        st.info("Choose at least one status.")
        return
    reviews = call_service("GET", "/reviews", query={"status_value": chosen})
    if not reviews:
        st.info("No review has these statuses yet.")
        return
    table = pd.DataFrame([{
        "Request": review["permit_reference"], "Revision": review["revision_number"],
        "Permit type": review["permit_type"], "Linked permit type": review["linked_permit_type"],
        "Plant area": review["plant_area"],
        "Status": config.REQUEST_STATUS_LABELS.get(review["status"], review["status"]),
        "Opened by": review["coordinator_id"], "Decided by": review["decided_by"] or "",
        "Model calls": review["llm_call_count"]} for review in reviews])
    selection = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key="audit_reviews")
    picked = selection.selection.rows if selection and selection.selection else []
    if picked:
        _review_record(reviews[picked[0]])
    else:
        st.caption("Select a review to read the record behind it.")


def _chat_tab():
    turns = call_service("GET", "/audit/chat-turns",
                         query={"turn_limit": config.CHAT_AUDIT_MAX_ROWS})
    if not turns:
        st.info("Nobody has asked the permit assistant anything yet.")
        return
    st.dataframe(pd.DataFrame([{
        "Turn": turn["turn_id"], "Asked by": turn["username"], "Role": turn["user_role"],
        "Type": turn["question_type"], "Request": turn["permit_reference"] or "",
        "Plant area": turn["plant_area"] or "", "Permit type": turn["permit_type"] or "",
        "Refused": config.BOOLEAN_YES if turn["was_refused"] else config.BOOLEAN_NO,
        "Question": turn["question_text"], "Answer": turn["answer_text"],
        "Sources": turn["source_documents"]} for turn in turns]),
        hide_index=True, width="stretch")


def render_audit_trail_page():
    """Show what has been reviewed and decided, and every question the chat replied to."""
    st.title(config.PAGE_AUDIT)
    reviews_tab, chat_tab = st.tabs(["Permit reviews", "Chat log"])
    with reviews_tab:
        _reviews_tab()
    with chat_tab:
        _chat_tab()
