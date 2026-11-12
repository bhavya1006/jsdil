"""Why a decision may not be recorded. Asked before a paused run resumes, so a decision the gate
would refuse never costs a language model call."""
from config import config


def find_review_conflict(review_status, pending_decision, review_decision, permit_reference):
    """Whether the review's state allows a decision at all."""
    if review_status in config.DECIDED_REVIEW_STATUSES:
        return config.ALREADY_DECIDED_MESSAGE.format(permit_reference=permit_reference)
    if pending_decision is not None and any(
            pending_decision.get(field) != review_decision.get(field)
            for field in config.DECISION_RECORD_FIELDS):
        return config.DECISION_IN_PROGRESS_MESSAGE.format(
            pending_decided_by=pending_decision.get("decided_by"),
            pending_decision=pending_decision.get("decision"),
            permit_reference=permit_reference)
    return None


def find_decision_refusal(review_decision, request, screening, opened_by):
    """The first reason this decision may not be recorded on this review, or None."""
    reference = screening.get("permit_reference")
    if reference != request.get(config.PERMIT_REFERENCE_COLUMN):
        raise ValueError(config.SCREENING_OF_ANOTHER_REQUEST_MESSAGE.format(
            screening_reference=reference,
            permit_reference=request.get(config.PERMIT_REFERENCE_COLUMN)))
    missing = [field for field in config.DECISION_RECORD_FIELDS
               if not isinstance(review_decision.get(field), str)
               or not review_decision.get(field).strip()]
    if missing:
        return config.DECISION_FIELDS_MISSING_MESSAGE.format(permit_reference=reference,
                                                             fields=", ".join(missing))
    decision = review_decision["decision"]
    if decision not in config.DECISIONS:
        return config.UNKNOWN_DECISION_MESSAGE.format(decision=decision, permit_reference=reference,
                                                      decisions=", ".join(config.DECISIONS))
    if review_decision["decided_by_role"] != config.ROLE_AREA_AUTHORITY:
        return config.DECIDING_ROLE_MESSAGE.format(role=review_decision["decided_by_role"],
                                                   permit_reference=reference,
                                                   authority_role=config.ROLE_AREA_AUTHORITY)
    area_group = review_decision["decided_by_area_group"]
    plant_area = request.get(config.PLANT_AREA_COLUMN)
    if plant_area not in config.AREA_GROUPS.get(area_group, []):
        return config.NOT_APPOINTED_MESSAGE.format(decided_by=review_decision["decided_by"],
                                                   area_group=area_group, plant_area=plant_area,
                                                   permit_reference=reference)
    if review_decision["decided_by"] == opened_by:
        return config.OWN_REVIEW_MESSAGE.format(decided_by=review_decision["decided_by"],
                                                permit_reference=reference)
    if decision not in screening.get("decisions_open", []):
        return config.DECISION_NOT_OPEN_MESSAGE.format(
            decision=decision, permit_reference=reference,
            reason=config.ISSUE_CLOSED_REASONS.get(screening.get("assessment"), ""))
    return None
