"""Everything established about one request, and the place to open its review."""
import streamlit as st

from config import config
from dashboard.api_client import call_service, signed_in_user
from dashboard.display import (REMEMBERED_REQUEST_KEY, answer_label, badge_markdown, band_badge,
                               decision_badge, flagged_sentence, hours_text, likelihood_text,
                               live_permits_text, neutral_badge, permit_types_text,
                               requirements_by_standard, sentence_type_badge, status_badge)


def _headline(detail, request, screening):
    st.markdown(" ".join([
        badge_markdown("Assessment: " + screening["assessment"],
                       config.DECISION_BADGE_COLOURS.get(screening["assessment"],
                                                         config.NEUTRAL_BADGE_COLOUR)),
        status_badge(detail["request_status"]),
        neutral_badge(f"Queue position {detail['queue_position']}")]))
    st.caption(" · ".join([
        f"Area classification: {request['area_classification']}",
        f"Process unit: {request[config.PROCESS_UNIT_STATUS_COLUMN]}",
        f"Channel: {request['request_channel']}",
        f"Revision {request[config.REVISION_NUMBER_COLUMN]}"]))


def _metrics(detail, screening):
    total = sum(screening[column] for column in config.REQUIREMENT_COUNT_RESULTS)
    likelihood, hours = detail["likelihood"], detail["work_hours"]
    first = st.columns(4)
    first[0].metric("Requirements met",
                    f"{screening[config.REQUIREMENTS_MET_COUNT_COLUMN]} of {total}",
                    border=True, icon=":material/check_circle:")
    first[1].metric("Not met", screening[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN],
                    border=True, icon=":material/cancel:")
    first[2].metric("Not documented", screening[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN],
                    border=True, icon=":material/help:")
    first[3].metric("SIMOPS triggers", len(screening["simops_triggers"]), border=True,
                    icon=":material/groups:")
    second = st.columns(4)
    count = detail["register_count"]
    second[0].metric("Permits live in the area", count[config.CONCURRENT_LIVE_PERMITS_COLUMN],
                     border=True, icon=":material/badge:")
    second[1].metric("Conflicting live permits", count[config.CONFLICTING_LIVE_PERMITS_COLUMN],
                     border=True, icon=":material/warning:")
    with second[2].container(border=True):
        st.metric("Suspension likelihood", likelihood_text(likelihood["suspension_likelihood"]),
                  icon=":material/percent:")
        st.markdown(band_badge(likelihood["likelihood_band"]))
    with second[3].container(border=True):
        st.metric("Expected work hours", hours_text(hours["expected_work_hours"]),
                  icon=":material/schedule:")
        planned = hours["planned_work_hours"]
        marks = [neutral_badge(f"{hours_text(planned)} planned") if planned is not None
                 else neutral_badge("no planned hours")]
        if hours["extension_expected"]:
            marks.append(f":{config.EXTENSION_COLOUR}-badge[extension expected]")
        st.markdown(" ".join(marks))


def _screening_tab(detail, screening):
    requirements_by_standard(screening["requirements"])
    st.markdown("**Decisions open:** " + " ".join(decision_badge(decision)
                                                  for decision in screening["decisions_open"]))
    st.markdown("**SIMOPS triggers:** "
                + ("; ".join(screening["simops_triggers"]) or config.CHAT_NONE_TEXT))
    count = detail["register_count"]
    st.markdown("**Permits live in the plant area:** "
                + live_permits_text(count["live_permits"], count["conflicting_permits"]))


def _evidence_tab(detail):
    documents = detail["request_documents"]
    if not documents:
        st.info(config.CHAT_NO_DOCUMENTS_TEXT[:1].upper() + config.CHAT_NO_DOCUMENTS_TEXT[1:]
                + " to this request.")
        return
    flags = {(flag["document_id"], flag["sentence"]): flag for flag in detail["disagreements"]}
    for document in documents:
        document_id = document[config.DOCUMENT_ID_COLUMN]
        with st.container(border=True):
            st.markdown(f"**{document[config.DOCUMENT_TITLE_COLUMN]}** · {document_id}")
            for finding in detail["evidence_findings"]:
                if finding["document_id"] != document_id:
                    continue
                st.markdown(sentence_type_badge(finding["sentence_type"]) + " "
                            + finding["sentence"])
                borne = finding.get("requirements_borne_on", [])
                if borne:
                    grouped = {}
                    for item in borne:
                        grouped.setdefault(item["permit_type"], []).append(
                            item["requirement_name"])
                    st.caption("Bears on " + "; ".join(
                        f"{permit_type}: {', '.join(names)}"
                        for permit_type, names in grouped.items()))
                flag = flags.get((document_id, finding["sentence"]))
                if flag:
                    flagged_sentence(flag)


def _request_tab(request, screening):
    uses = {}
    for column in config.PERMIT_DETAIL_COLUMNS:
        uses.setdefault(column, []).append("Permit document")
    for requirement in screening["requirements"]:
        for column in config.REQUIREMENT_ANSWER_COLUMNS.get(requirement["requirement_kind"], []):
            uses.setdefault(column, []).append(
                f"{requirement['requirement_name']} ({requirement['permit_type']})")
    for column in config.TRIGGER_ANSWER_COLUMNS:
        uses.setdefault(column, []).append("SIMOPS triggers")
    rows = [{"Question": answer_label(column), "Answer": request.get(column, config.NOT_GIVEN_TEXT),
             "Used for": ", ".join(dict.fromkeys(purposes))} for column, purposes in uses.items()]
    st.dataframe(rows, hide_index=True, width="stretch")


def _start_review_form(request, reference):
    user = signed_in_user()
    plant_area = request[config.PLANT_AREA_COLUMN]
    if user["role"] == config.ROLE_AREA_AUTHORITY and plant_area not in config.AREA_GROUPS.get(
            user["area_group"], []):
        group = next((name for name, areas in config.AREA_GROUPS.items() if plant_area in areas),
                     config.NOT_GIVEN_TEXT)
        st.info(f"{plant_area} is in {group}, not in {user['area_group']}: its review is opened "
                f"by a Permit Coordinator or by an Area Authority appointed for {group}.",
                icon=":material/block:")
        return
    st.markdown("No review of this revision has been started. Say what the review should look "
                "at, and the reviewing agents will prepare it for an Area Authority.")
    asked = st.text_area("What should the review look at?",
                         max_chars=config.COORDINATOR_REQUEST_MAXIMUM_CHARACTERS,
                         key=f"coordinator_request_{reference}")
    if st.button("Start the review", type="primary", icon=":material/play_arrow:"):
        if len(asked.strip()) < config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS:
            st.warning(f"Write at least {config.COORDINATOR_REQUEST_MINIMUM_CHARACTERS} "
                       "characters.")
            return
        call_service("POST", "/reviews",
                     payload={"permit_reference": reference, "coordinator_request": asked},
                     waiting_message=(f"Reviewing {reference}: {config.LLM_CALLS_BEFORE_DECISION} "
                                      "language model calls are being made"),
                     failure_message=config.REVIEW_FAILED_MESSAGE)
        st.rerun()


def _review_tab(request, reference, reviews):
    if not reviews:
        _start_review_form(request, reference)
        return
    for review in reviews:
        with st.container(border=True):
            st.markdown(f"**{review['review_id']}** · opened by {review['coordinator_id']} "
                        + status_badge(review["status"]) + " Recommended: "
                        + decision_badge(review["recommended_decision"]))
            st.markdown(review["recommendation_text"])
            if review["decision"]:
                st.markdown(f"**Decision:** {decision_badge(review['decision'])} by "
                            f"{review['decided_by']}, {review['decided_by_role']} for "
                            f"{review['decided_by_area_group']}")
                st.markdown(f"> {review['decision_reason']}")
            else:
                st.caption("Waiting for an Area Authority's decision on the decision queue.")


def render_request_detail_page():
    """Show everything established about one request, and let a person open its review."""
    st.title(config.PAGE_DETAIL)
    queue = call_service("GET", "/queue", query={"permit_type": config.QUEUE_PERMIT_TYPE_ALL})
    if not queue:
        st.info("The queue is empty.")
        return
    labels = {row["permit_reference"]: " | ".join([
        row["permit_reference"], permit_types_text(row["permit_type"], row["linked_permit_type"]),
        row["plant_area"]]) for row in queue}
    references = list(labels)
    remembered = st.session_state.get(REMEMBERED_REQUEST_KEY)
    reference = st.selectbox("Request", references, format_func=labels.get,
                             index=references.index(remembered) if remembered in references
                             else 0)
    st.session_state[REMEMBERED_REQUEST_KEY] = reference
    detail = call_service("GET", "/queue/" + reference)
    request, screening = detail["request"], detail["screening"]
    reviews = [review for review in call_service(
        "GET", "/reviews", query={"status_value": config.REVIEW_STATUS_OPTIONS})
        if review["permit_reference"] == reference
        and str(review["revision_number"]) == request[config.REVISION_NUMBER_COLUMN]]
    _headline(detail, request, screening)
    _metrics(detail, screening)
    screening_tab, evidence_tab, request_tab, review_tab = st.tabs(
        ["Screening", "Statement evidence", "Request", "Review"])
    with screening_tab:
        _screening_tab(detail, screening)
    with evidence_tab:
        _evidence_tab(detail)
    with request_tab:
        _request_tab(request, screening)
    with review_tab:
        _review_tab(request, reference, reviews)
