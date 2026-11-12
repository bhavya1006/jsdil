"""Answers from the records, never from a model, so a model cannot give away what the rules
withhold."""
from collections import Counter

from sqlalchemy import and_

from config import config
from persistence.models import PermitReview
from persistence.repository import fetch_review_records
from tools.request_data_tools import describe_record, fetch_request


def _permit_types_text(described):
    if described[config.LINKED_PERMIT_TYPE_COLUMN] != config.NO_LINKED_PERMIT:
        return config.CHAT_LINKED_PERMIT_TEMPLATE.format(
            permit_type=described[config.PERMIT_TYPE_COLUMN],
            linked_permit_type=described[config.LINKED_PERMIT_TYPE_COLUMN])
    return described[config.PERMIT_TYPE_COLUMN]


def _joined(items, separator=", ", empty=config.CHAT_NONE_TEXT):
    return separator.join(items) if items else empty


def _review_lines(review, permit_types):
    status = config.REQUEST_STATUS_PHRASES.get(review["status"], review["status"])
    by_standard = {}
    for requirement in review.get("requirements") or []:
        by_standard.setdefault(requirement["permit_type"], []).append(
            f"{requirement['requirement_name']} {requirement['result']}")
    requirements = _joined([f"{permit_type} standard: " + ", ".join(items)
                            for permit_type, items in by_standard.items()], "; ")
    conflicting = set(review.get("conflicting_permits") or [])
    live = [config.CONFLICTING_PERMIT_TEMPLATE.format(permit_reference=reference)
            if reference in conflicting else reference
            for reference in review.get("live_permits") or []]
    evidence = review.get("evidence") or []
    counted = Counter(finding["sentence_type"] for finding in evidence)
    sentence_types = (", ".join(f"{name} {counted[name]}" for name in sorted(counted))
                      if evidence else config.CHAT_NO_DOCUMENTS_TEXT)
    lines = [
        config.CHAT_REVIEW_TEMPLATE.format(
            review_id=review["review_id"], permit_reference=review["permit_reference"],
            revision_number=review["revision_number"], permit_types=permit_types,
            plant_area=review["plant_area"], coordinator_id=review["coordinator_id"],
            status=status),
        config.CHAT_SCREENING_TEMPLATE.format(
            assessment=review["assessment"], requirements=requirements,
            triggers=_joined(review.get("simops_triggers") or [], "; "),
            decisions_open=_joined(review.get("decisions_open") or [])),
        config.CHAT_PERMITS_TEMPLATE.format(live_permits=_joined(live)),
        config.CHAT_FIGURES_TEMPLATE.format(
            suspension_likelihood=format(review["suspension_likelihood"],
                                         "." + str(config.LIKELIHOOD_DECIMALS) + "f"),
            likelihood_band=review["likelihood_band"], work_hours_text=review["work_hours_text"]),
        config.CHAT_EVIDENCE_TEMPLATE.format(sentence_count=len(evidence),
                                             sentence_types=sentence_types,
                                             flag_count=len(review.get("disagreements") or [])),
    ]
    if review.get("decision") is None:
        lines.append(config.CHAT_AWAITING_DECISION_TEMPLATE.format(
            recommended_decision=review["recommended_decision"]))
    else:
        lines.append(config.CHAT_DECIDED_TEMPLATE.format(
            decided_by=review["decided_by"], decided_by_role=review["decided_by_role"],
            decided_by_area_group=review["decided_by_area_group"], decision=review["decision"],
            decision_reason=review["decision_reason"]))
    return lines


def answer_request_record(session, signed_in_user, open_requests, question_type,
                          permit_reference):
    """Answer a question about one permit request from the review records, by the asker's role."""
    role = signed_in_user.get("role")
    operator = role == config.ROLE_CONTROL_ROOM_OPERATOR
    if not operator and role not in config.ROLES_SEEING_EVERY_REVIEW:
        raise ValueError(f"The role {role} hears nothing about a permit request")
    if operator and question_type == config.CHAT_TYPE_OUTCOME_PREDICTION:
        return config.OUTCOME_NOT_PREDICTED_LINE
    if not permit_reference:
        return config.CHAT_REQUEST_NOT_NAMED_MESSAGE
    if permit_reference not in set(open_requests[config.PERMIT_REFERENCE_COLUMN]):
        return config.CHAT_NO_REQUEST_TEMPLATE.format(permit_reference=permit_reference)
    request = fetch_request(open_requests, permit_reference)
    described = describe_record(request)
    permit_types = _permit_types_text(described)
    reviews = fetch_review_records(session, and_(
        PermitReview.permit_reference == permit_reference,
        PermitReview.revision_number == int(request[config.REVISION_NUMBER_COLUMN])))
    if operator:
        status = reviews[-1]["status"] if reviews else config.REQUEST_STATUS_AWAITING_REVIEW
        answer = config.CHAT_OPERATOR_STATUS_TEMPLATE.format(
            permit_reference=permit_reference, permit_types=permit_types,
            plant_area=described[config.PLANT_AREA_COLUMN],
            status=config.REQUEST_STATUS_PHRASES[status])
        if status == config.REVIEW_STATUS_ISSUED:
            details = describe_record(reviews[-1].get("document_details") or {})
            answer += " " + config.CHAT_ISSUED_WORK_TEMPLATE.format(
                equipment_tag=details.get(config.EQUIPMENT_TAG_COLUMN, config.NOT_GIVEN_TEXT),
                work_party_size=details.get("work_party_size", config.NOT_GIVEN_TEXT))
        return answer
    if not reviews:
        return config.CHAT_NO_REVIEW_TEMPLATE.format(permit_reference=permit_reference)
    lines = []
    for review in reviews:
        lines += _review_lines(review, permit_types)
    return "\n".join(lines)


def answer_plant_area_figure(permit_history, plant_area, permit_type):
    """A plant area figure only where at least five permits make it up."""
    if not plant_area or not permit_type:
        return config.CHAT_PLANT_AREA_NOT_NAMED_MESSAGE
    rows = permit_history[(permit_history[config.PLANT_AREA_COLUMN] == plant_area)
                          & (permit_history[config.PERMIT_TYPE_COLUMN] == permit_type)]
    count = int(rows[config.PERMIT_REFERENCE_COLUMN].nunique())
    if count < config.GROUP_SIZE_MINIMUM:
        return config.GROUP_SIZE_LINE
    return config.CHAT_PLANT_AREA_FIGURE_TEMPLATE.format(permit_count=count,
                                                         permit_type=permit_type.lower(),
                                                         plant_area=plant_area)
