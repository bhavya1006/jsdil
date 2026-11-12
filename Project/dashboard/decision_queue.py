"""The reviews of one area group waiting at the gate, and the Area Authority's decision."""
import streamlit as st

from config import config
from dashboard.api_client import call_service, signed_in_user
from dashboard.display import (band_badge, decision_badge, flagged_sentence, hours_text,
                               likelihood_text, live_permits_text, neutral_badge,
                               permit_types_text, requirements_by_standard)

_OPEN_REVIEWS_KEY = "decision_queue_open_reviews"
_REPORT_KEY = "decision_queue_report"
_WRITE_UPS = [("Plan", "review_plan_text"), ("Requirements", "requirements_analysis_text"),
              ("Statement evidence", "evidence_review_text"),
              ("Figures", "prediction_analysis_text"), ("Recommendation", "recommendation_text")]


def _write_up(text):
    for block in text.split("\n\n"):
        label, _, body = block.partition(": ")
        if body and label.isupper():
            st.caption(label)
            st.markdown(body)
        else:
            st.markdown(block)


def _review_summary(review):
    st.markdown(f"Recommended: {decision_badge(review['recommended_decision'])} "
                + neutral_badge(f"Opened by {review['coordinator_id']}"))
    figures = st.columns(2)
    with figures[0].container(border=True):
        st.metric("Suspension likelihood", likelihood_text(review["suspension_likelihood"]),
                  icon=":material/percent:")
        st.markdown(band_badge(review["likelihood_band"]))
    with figures[1].container(border=True):
        st.metric("Expected work hours", hours_text(review["expected_work_hours"]),
                  icon=":material/schedule:")
        planned = review["planned_work_hours"]
        marks = [neutral_badge(f"{hours_text(planned)} planned" if planned is not None
                               else "no planned hours")]
        if review["extension_expected"]:
            marks.append(f":{config.EXTENSION_COLOUR}-badge[extension expected]")
        st.markdown(" ".join(marks))
    st.markdown("**Recommendation**")
    st.markdown(review["recommendation_text"])
    requirements_by_standard(review["requirements"], with_sentences=False)
    st.markdown("**SIMOPS triggers:** "
                + ("; ".join(review["simops_triggers"]) or config.CHAT_NONE_TEXT))
    st.markdown("**Permits live in the plant area:** "
                + live_permits_text(review["live_permits"], review["conflicting_permits"]))
    for flag in review["disagreements"]:
        flagged_sentence(flag)
    with st.expander("The review, agent by agent"):
        st.markdown("**What was asked**")
        st.markdown(review["coordinator_request"])
        for title, key in _WRITE_UPS:
            st.markdown(f"**{title}**")
            _write_up(review[key])


def _decision_form(review, user, group):
    with st.container(border=True):
        st.markdown("**Your decision**")
        if review["coordinator_id"] == user["username"]:
            st.info(f"You opened this review, so you may not decide it: the other Area "
                    f"Authority appointed for {group} must decide it.", icon=":material/block:")
            return
        review_id = review["review_id"]
        decision = st.radio("Decision", review["decisions_open"], key=f"decision_{review_id}",
                            captions=[config.DECISION_DESCRIPTIONS[decision]
                                      for decision in review["decisions_open"]])
        reason = st.text_area("Reason", max_chars=config.DECISION_REASON_MAXIMUM_CHARACTERS,
                              key=f"reason_{review_id}")
        if st.button("Record the decision", type="primary", icon=":material/gavel:",
                     width="stretch", key=f"record_{review_id}"):
            if len(reason.strip()) < config.DECISION_REASON_MINIMUM_CHARACTERS:
                st.warning(f"Give a reason of at least {config.DECISION_REASON_MINIMUM_CHARACTERS}"
                           " characters.")
                return
            calls = config.LLM_CALLS_BY_DECISION[decision] - config.LLM_CALLS_BEFORE_DECISION
            decided = call_service(
                "POST", f"/reviews/{review_id}/decision",
                payload={"decision": decision, "decision_reason": reason},
                waiting_message=(f"Recording {decision} for {review['permit_reference']}"
                                 + (f": {calls} language model calls are being made"
                                    if calls else "")),
                failure_message=config.DECISION_FAILED_MESSAGE)
            st.session_state[_REPORT_KEY] = (
                f"{decided['permit_reference']} is "
                f"{config.REQUEST_STATUS_PHRASES[decided['status']]}: "
                f"{config.DECISION_DESCRIPTIONS[decided['decision']]}.")
            st.rerun()


def render_decision_queue_page():
    """Show the reviews waiting in the signed-in Area Authority's group, and let them decide."""
    st.title(config.PAGE_DECISION)
    user = signed_in_user()
    group = user["area_group"]
    plant_areas = config.AREA_GROUPS.get(group, [])
    st.caption(f"Appointed for {group}: {', '.join(plant_areas)}")
    report = st.session_state.pop(_REPORT_KEY, None)
    if report:
        st.success(report, icon=":material/task_alt:")
    waiting = [review for review in call_service(
        "GET", "/reviews", query={"status_value": [config.REVIEW_STATUS_AWAITING]})
        if review["plant_area"] in plant_areas]
    previously_open = st.session_state.get(_OPEN_REVIEWS_KEY, set())
    now_open = {review["review_id"] for review in waiting}
    gone = previously_open - now_open
    if gone and not report:
        st.info("Decided since this page was last shown: " + ", ".join(sorted(gone)),
                icon=":material/update:")
    st.session_state[_OPEN_REVIEWS_KEY] = now_open
    if not waiting:
        st.info(f"No review in {group} is waiting for a decision.", icon=":material/inbox:")
        return
    labels = {review["review_id"]: " | ".join([
        review["permit_reference"],
        permit_types_text(review["permit_type"], review["linked_permit_type"]),
        review["plant_area"]]) for review in waiting}
    chosen = st.selectbox(f"Review to decide ({len(waiting)} waiting)", list(labels),
                          format_func=labels.get)
    review = next(review for review in waiting if review["review_id"] == chosen)
    left, right = st.columns([3, 2], gap="large")
    with left:
        _review_summary(review)
    with right:
        _decision_form(review, user, group)
