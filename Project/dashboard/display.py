"""Shared presentation helpers: badges, tables and labels. No business rule lives here; every
value shown was produced by the service."""
import streamlit as st

from config import config

NAVIGATION_PAGES_KEY = "navigation_pages"
REMEMBERED_REQUEST_KEY = "remembered_request"


def badge_markdown(text, colour):
    """A coloured badge written inline in markdown."""
    safe = str(text).replace("[", "(").replace("]", ")")
    return f":{colour}-badge[{safe}]"


def decision_badge(decision):
    return badge_markdown(decision, config.DECISION_BADGE_COLOURS.get(
        decision, config.NEUTRAL_BADGE_COLOUR))


def status_badge(status):
    return badge_markdown(config.REQUEST_STATUS_LABELS.get(status, status),
                          config.STATUS_BADGE_COLOURS.get(status, config.NEUTRAL_BADGE_COLOUR))


def band_badge(band):
    return badge_markdown(band, config.LIKELIHOOD_BAND_BADGE_COLOURS.get(
        band, config.NEUTRAL_BADGE_COLOUR))


def result_badge(result):
    return badge_markdown(result, config.REQUIREMENT_RESULT_BADGE_COLOURS.get(
        result, config.NEUTRAL_BADGE_COLOUR))


def chat_type_badge(question_type):
    return badge_markdown(question_type, config.CHAT_TYPE_BADGE_COLOURS.get(
        question_type, config.NEUTRAL_BADGE_COLOUR))


def sentence_type_badge(sentence_type):
    return badge_markdown(sentence_type, config.SENTENCE_TYPE_BADGE_COLOUR)


def neutral_badge(text):
    return badge_markdown(text, config.NEUTRAL_BADGE_COLOUR)


def extension_text(extension_expected):
    """Whether an extension is expected, in the configuration's words."""
    if extension_expected is None:
        return config.NOT_GIVEN_TEXT
    return config.BOOLEAN_YES if extension_expected else config.BOOLEAN_NO


def answer_label(column_name):
    label = config.FORM_ANSWER_LABELS.get(column_name, column_name.replace("_", " "))
    return label[:1].upper() + label[1:]


def permit_types_text(permit_type, linked_permit_type):
    if linked_permit_type and linked_permit_type != config.NO_LINKED_PERMIT:
        return config.CHAT_LINKED_PERMIT_TEMPLATE.format(permit_type=permit_type,
                                                         linked_permit_type=linked_permit_type)
    return permit_type


def likelihood_text(likelihood):
    return format(likelihood, "." + str(config.LIKELIHOOD_DECIMALS) + "f")


def hours_text(hours):
    return format(hours, "." + str(config.WORK_HOURS_DECIMALS) + "f")


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers, rows):
    """A compact table that can carry badges, which a dataframe cannot."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    st.markdown("\n".join(lines))


def requirements_by_standard(requirements, with_sentences=True):
    """Each standard's requirements with their results and any answers left blank."""
    standards = {}
    for requirement in requirements:
        standards.setdefault((requirement["permit_type"], requirement.get("standard_document")),
                             []).append(requirement)
    for (permit_type, document_name), items in standards.items():
        heading = f"**{permit_type} standard**"
        if document_name:
            heading += f" · {document_name}"
        st.markdown(heading)
        headers = ["Requirement", "Result"]
        if with_sentences:
            headers.append("The standard's requirement")
        headers.append("Answers needed")
        rows = []
        for requirement in items:
            needed = ", ".join(answer_label(column) for column in requirement["blank_answers"])
            row = [requirement["requirement_name"], result_badge(requirement["result"])]
            if with_sentences:
                row.append(requirement.get("requirement_sentence", ""))
            row.append(needed or "—")
            rows.append(row)
        markdown_table(headers, rows)


def live_permits_text(live_permits, conflicting_permits):
    conflicting = set(conflicting_permits)
    shown = [config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference=reference)
             if reference in conflicting else reference for reference in live_permits]
    return ", ".join(shown) if shown else config.CHAT_NONE_TEXT


def flagged_sentence(flag):
    """A flagged statement sentence with the readings it states beside the request's."""
    readings = []
    for column_name, stated in flag["stated_readings"].items():
        request_value = flag["request_readings"].get(column_name)
        readings.append(f"{answer_label(column_name)}: the document states "
                        f"{', '.join(str(value) for value in stated)}, the request gives "
                        f"{config.NOT_GIVEN_TEXT if request_value is None else request_value}")
    st.warning(f"**{flag['flag']}** · {flag['document_id']}  \n“{flag['sentence']}”  \n"
               + "  \n".join(readings), icon=":material/flag:")


def go_to_page(page_name):
    """Open another page of the dashboard."""
    pages = st.session_state.get(NAVIGATION_PAGES_KEY, {})
    if page_name in pages:
        st.switch_page(pages[page_name])
