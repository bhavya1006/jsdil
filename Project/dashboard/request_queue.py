"""The open queue in the order the permit office works it."""
import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service
from dashboard.display import REMEMBERED_REQUEST_KEY, extension_text, go_to_page


def _metrics(queue):
    first = config.QUEUE_FIRST_PERMIT_TYPE
    statuses = [row["request_status"] for row in queue]
    figures = [("Open requests", len(queue), ":material/inbox:"),
               (f"{first} first", sum(1 for row in queue if first in (
                   row["permit_type"], row["linked_permit_type"])), ":material/priority_high:"),
               (config.REQUEST_STATUS_LABELS[config.REQUEST_STATUS_AWAITING_REVIEW],
                statuses.count(config.REQUEST_STATUS_AWAITING_REVIEW), ":material/hourglass_empty:"),
               (config.REQUEST_STATUS_LABELS[config.REVIEW_STATUS_AWAITING],
                statuses.count(config.REVIEW_STATUS_AWAITING), ":material/gavel:")]
    columns = st.columns(len(figures))
    for column, (label, value, icon) in zip(columns, figures):
        column.metric(label, value, border=True, icon=icon)
    columns = st.columns(len(config.DECISIONS))
    for column, decision in zip(columns, config.DECISIONS):
        column.metric(f"Assessed {decision}",
                      sum(1 for row in queue if row["assessment"] == decision), border=True)


def _table(rows):
    return pd.DataFrame([{
        "Position": row["queue_position"],
        "Reference": row["permit_reference"],
        "Revision": row["revision_number"],
        "Permit type": row["permit_type"],
        "Linked permit type": row["linked_permit_type"],
        "Plant area": row["plant_area"],
        "Channel": row["request_channel"],
        "Status": config.REQUEST_STATUS_LABELS.get(row["request_status"], row["request_status"]),
        "Assessment": row["assessment"],
        "Met": row["requirements_met_count"],
        "Not met": row["requirements_not_met_count"],
        "Not documented": row["requirements_not_documented_count"],
        "Live permits": row["live_permit_count"],
        "Conflicting": row["conflicting_permit_count"],
        "SIMOPS triggers": row["simops_trigger_count"],
        "Likelihood": row["suspension_likelihood"],
        "Band": row["likelihood_band"],
        "Expected hours": row["expected_work_hours"],
        "Extension": extension_text(row["extension_expected"]),
        "Flags": row["disagreement_count"],
        "Documents": row["document_count"],
    } for row in rows])


def render_request_queue_page():
    """Show the open queue and let a person pick one request."""
    st.title(config.PAGE_QUEUE)
    st.caption(f"Requests whose permit type or linked permit type is "
               f"{config.QUEUE_FIRST_PERMIT_TYPE.lower()} come first, because the rescue cover "
               "must be arranged before anyone enters; each group runs by permit reference.")
    everything = call_service("GET", "/queue",
                              query={"permit_type": config.QUEUE_PERMIT_TYPE_ALL})
    _metrics(everything)
    chosen = st.segmented_control("Permit type", config.QUEUE_PERMIT_TYPE_OPTIONS,
                                  default=config.QUEUE_PERMIT_TYPE_ALL, required=True,
                                  key="queue_permit_type")
    rows = everything if chosen == config.QUEUE_PERMIT_TYPE_ALL else call_service(
        "GET", "/queue", query={"permit_type": chosen})
    if not rows:
        st.info("No open request matches this permit type.")
        return
    likelihood_format = "%." + str(config.LIKELIHOOD_DECIMALS) + "f"
    hours_format = "%." + str(config.WORK_HOURS_DECIMALS) + "f"
    selection = st.dataframe(
        _table(rows), hide_index=True, width="stretch", on_select="rerun",
        selection_mode="single-row", key="queue_table",
        column_config={
            "Position": st.column_config.NumberColumn(width="small"),
            "Revision": st.column_config.NumberColumn(width="small"),
            "Likelihood": st.column_config.ProgressColumn(
                format=likelihood_format, min_value=0.0, max_value=1.0, width="small"),
            "Expected hours": st.column_config.NumberColumn(format=hours_format),
        })
    picked = selection.selection.rows if selection and selection.selection else []
    if picked:
        reference = rows[picked[0]]["permit_reference"]
        st.session_state[REMEMBERED_REQUEST_KEY] = reference
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{reference}** is selected; the request detail page opens on it.")
            if st.button("Open request detail", icon=":material/arrow_forward:", type="primary"):
                go_to_page(config.PAGE_DETAIL)
    else:
        st.caption("Select a row to open that request on the request detail page.")
