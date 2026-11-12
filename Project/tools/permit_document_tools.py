"""The details the permit document, the work party briefing, the rework request and the SIMOPS
brief carry, taken from the request and its screening by code, never by a model."""
from config import config
from tools.request_data_tools import is_blank, plain_value


def _refuse_other_screening(request, screening):
    reference = request.get(config.PERMIT_REFERENCE_COLUMN)
    if screening.get("permit_reference") != reference:
        raise ValueError(config.SCREENING_OF_ANOTHER_REQUEST_MESSAGE.format(
            screening_reference=screening.get("permit_reference"), permit_reference=reference))


def _answer(request, column_name):
    return plain_value(request.get(column_name))


def build_permit_details(request, screening):
    """What a permit document and a work party briefing show, with each standard's controls."""
    _refuse_other_screening(request, screening)
    details = {config.PERMIT_REFERENCE_COLUMN: request.get(config.PERMIT_REFERENCE_COLUMN)}
    for column_name in config.PERMIT_DETAIL_COLUMNS:
        details[column_name] = _answer(request, column_name)
    details["controls"] = {
        permit_type: {column_name: _answer(request, column_name)
                      for column_name in config.PERMIT_CONTROL_COLUMNS.get(permit_type, [])
                      if not is_blank(request.get(column_name))}
        for permit_type in screening.get("permit_types", [])}
    return details


def build_rework_details(request, screening, documentation_to_provide, disagreements):
    """Everything a rework request must name."""
    _refuse_other_screening(request, screening)
    permit_types = list(screening.get("permit_types", []))
    if list(documentation_to_provide) != permit_types:
        raise ValueError(f"Documentation was given for {list(documentation_to_provide)} where the "
                         f"screening screened {permit_types}")
    to_put_right = []
    for requirement in screening.get("requirements", []):
        if requirement.get("result") == config.REQUIREMENT_MET:
            continue
        columns = config.REQUIREMENT_ANSWER_COLUMNS.get(requirement.get("requirement_kind"), [])
        to_put_right.append({
            "permit_type": requirement.get("permit_type"),
            "requirement_name": requirement.get("requirement_name"),
            "requirement_sentence": requirement.get("requirement_sentence"),
            "result": requirement.get("result"),
            "request_answers": {column_name: _answer(request, column_name)
                                for column_name in columns},
            "blank_answers": list(requirement.get("blank_answers", [])),
        })
    return {"permit_reference": request.get(config.PERMIT_REFERENCE_COLUMN),
            "permit_types": permit_types,
            "requirements_to_put_right": to_put_right,
            "documentation_to_provide": dict(documentation_to_provide),
            "flagged_statements": list(disagreements)}


def build_simops_brief(request, screening, register_count, likelihood, work_hours,
                       disagreements):
    """The brief the SIMOPS panel receives with a referred request, from the record alone."""
    _refuse_other_screening(request, screening)
    brief = {"permit_reference": request.get(config.PERMIT_REFERENCE_COLUMN)}
    for column_name in config.SIMOPS_BRIEF_DETAIL_COLUMNS:
        brief[column_name] = _answer(request, column_name)
    brief.update({
        "requirements": [{"permit_type": requirement.get("permit_type"),
                          "requirement_name": requirement.get("requirement_name"),
                          "result": requirement.get("result"),
                          "blank_answers": list(requirement.get("blank_answers", []))}
                         for requirement in screening.get("requirements", [])],
        "simops_triggers": list(screening.get("simops_triggers", [])),
        "live_permits": list(register_count.get("live_permits", [])),
        "conflicting_permits": list(register_count.get("conflicting_permits", [])),
        "flagged_statements": list(disagreements),
        "suspension_likelihood": likelihood.get("suspension_likelihood"),
        "likelihood_band": likelihood.get("likelihood_band"),
        "expected_work_hours": work_hours.get("expected_work_hours"),
        "planned_work_hours": work_hours.get("planned_work_hours"),
        "extension_expected": work_hours.get("extension_expected"),
    })
    return brief
