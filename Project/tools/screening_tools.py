"""Screens a request against its standards and the SIMOPS triggers, and counts the permits live
beside it. Every figure judged against comes from the standard handed over."""
from config import config
from data_pipeline.common_cleaning import to_decimal
from tools.request_data_tools import is_blank

_YES_NO_KINDS = [config.RISK_ASSESSMENT_REQUIREMENT, config.RESCUE_PLAN_REQUIREMENT,
                 config.PROVED_DEAD_REQUIREMENT, config.PERSONAL_LOCKS_REQUIREMENT,
                 config.FALL_PROTECTION_REQUIREMENT, config.BARRIERS_REQUIREMENT,
                 config.SERVICES_LOCATED_REQUIREMENT]
_READING_KINDS = [config.FLAMMABLE_GAS_REQUIREMENT, config.OXYGEN_REQUIREMENT,
                  config.COMBUSTIBLES_CLEARED_REQUIREMENT, config.FIRE_WATCH_REQUIREMENT,
                  config.HYDROGEN_SULPHIDE_REQUIREMENT, config.CARBON_MONOXIDE_REQUIREMENT]


def _permit_types(record):
    types = [record.get(config.PERMIT_TYPE_COLUMN)]
    linked = record.get(config.LINKED_PERMIT_TYPE_COLUMN)
    if not is_blank(linked) and linked != config.NO_LINKED_PERMIT:
        types.append(linked)
    return [permit_type for permit_type in types if not is_blank(permit_type)]


def _conflicts(first_types, second_types):
    return any((first, second) in config.CONFLICTING_PERMIT_TYPES
               or (second, first) in config.CONFLICTING_PERMIT_TYPES
               for first in first_types for second in second_types)


def count_live_permits(register, request):
    """Every permit live in the request's plant area, and those that conflict with it."""
    plant_area = request.get(config.PLANT_AREA_COLUMN)
    if is_blank(plant_area):
        raise ValueError(config.BLANK_PLANT_AREA_MESSAGE.format(
            permit_reference=request.get(config.PERMIT_REFERENCE_COLUMN)))
    beside = register[(register[config.PLANT_AREA_COLUMN] == plant_area)
                      & (register[config.PERMIT_STATUS_COLUMN] == config.PERMIT_STATUS_LIVE)]
    beside = beside.sort_values(config.PERMIT_REFERENCE_COLUMN, kind="stable")
    request_types = _permit_types(request)
    live, conflicting = [], []
    for permit in beside.to_dict("records"):
        live.append(permit[config.PERMIT_REFERENCE_COLUMN])
        if _conflicts(request_types, _permit_types(permit)):
            conflicting.append(permit[config.PERMIT_REFERENCE_COLUMN])
    return {config.CONCURRENT_LIVE_PERMITS_COLUMN: len(live),
            config.CONFLICTING_LIVE_PERMITS_COLUMN: len(conflicting),
            "live_permits": live, "conflicting_permits": conflicting}


def find_simops_triggers(request):
    """The SIMOPS triggers of the Control of Work Policy the request meets, in policy order."""
    triggers = []
    conflicting = request.get(config.CONFLICTING_LIVE_PERMITS_COLUMN)
    if not is_blank(conflicting) and to_decimal(conflicting) > 0:
        triggers.append(config.CONFLICTING_PERMIT_TRIGGER)
    types = _permit_types(request)
    unit_status = request.get(config.PROCESS_UNIT_STATUS_COLUMN)
    if (any(permit_type in config.LIVE_PROCESS_PERMIT_TYPES for permit_type in types)
            and (is_blank(unit_status) or unit_status not in config.NOT_LIVE_UNIT_STATUSES)):
        triggers.append(config.LIVE_PROCESS_TRIGGER)
    voltage = request.get(config.SYSTEM_VOLTAGE_COLUMN)
    if (any(permit_type in config.HIGH_VOLTAGE_PERMIT_TYPES for permit_type in types)
            and (is_blank(voltage) or to_decimal(voltage) > config.HIGH_VOLTAGE_ABOVE_VOLTS)):
        triggers.append(config.HIGH_VOLTAGE_TRIGGER)
    return triggers


def _yes_no(request, column_name):
    answer = request.get(column_name)
    if is_blank(answer):
        return config.REQUIREMENT_NOT_DOCUMENTED, [column_name]
    if answer == config.BOOLEAN_YES:
        return config.REQUIREMENT_MET, []
    if answer == config.BOOLEAN_NO:
        return config.REQUIREMENT_NOT_MET, []
    raise ValueError(f"{column_name}: {answer!r} is neither {config.BOOLEAN_YES} nor "
                     f"{config.BOOLEAN_NO}")


def _within(value, rules):
    number = to_decimal(value)
    if config.MINIMUM_RULE in rules and number < to_decimal(rules[config.MINIMUM_RULE]):
        return False
    if config.MAXIMUM_RULE in rules and number > to_decimal(rules[config.MAXIMUM_RULE]):
        return False
    return True


def _judge(requirement, request, document_name):
    kind = requirement["requirement_kind"]
    rules = requirement.get("rules") or {}
    if kind not in config.REQUIREMENT_ANSWER_COLUMNS:
        raise ValueError(f"{document_name} states a requirement of the kind {kind}, which the "
                         "screening does not know")
    columns = config.REQUIREMENT_ANSWER_COLUMNS[kind]
    if kind in _YES_NO_KINDS:
        return _yes_no(request, columns[0])
    if kind in _READING_KINDS:
        reading = request.get(columns[0])
        if is_blank(reading):
            return config.REQUIREMENT_NOT_DOCUMENTED, [columns[0]]
        met = _within(reading, rules)
        return (config.REQUIREMENT_MET if met else config.REQUIREMENT_NOT_MET), []
    first_column, second_column = columns
    first = request.get(first_column)
    if is_blank(first):
        return config.REQUIREMENT_NOT_DOCUMENTED, [first_column]
    if kind == config.CIRCUIT_EARTHS_REQUIREMENT:
        if to_decimal(first) <= to_decimal(rules[config.THRESHOLD_RULE]):
            return config.REQUIREMENT_MET, []
        return _yes_no(request, second_column)
    if kind == config.SCAFFOLD_INSPECTION_REQUIREMENT:
        if first != config.SCAFFOLD:
            return config.REQUIREMENT_MET, []
        return _yes_no(request, second_column)
    if kind == config.TRENCH_SUPPORT_REQUIREMENT:
        if to_decimal(first) <= to_decimal(rules[config.THRESHOLD_RULE]):
            return config.REQUIREMENT_MET, []
        return _yes_no(request, second_column)
    if kind == config.WIND_SPEED_REQUIREMENT:
        if first == config.INDOORS:
            return config.REQUIREMENT_MET, []
        wind = request.get(second_column)
        if is_blank(wind):
            return config.REQUIREMENT_NOT_DOCUMENTED, [second_column]
        met = _within(wind, rules)
        return (config.REQUIREMENT_MET if met else config.REQUIREMENT_NOT_MET), []
    raise ValueError(f"{document_name} states a requirement of the kind {kind}, which the "
                     "screening does not know")


def screen_request(request, standards):
    """Screen a request under its own standard and then its linked type's, and the triggers."""
    reference = request.get(config.PERMIT_REFERENCE_COLUMN)
    permit_types = _permit_types(request)
    requirements = []
    for permit_type in permit_types:
        if permit_type not in standards:
            raise ValueError(config.NO_STANDARD_READ_MESSAGE.format(permit_reference=reference,
                                                                    permit_type=permit_type))
        standard = standards[permit_type]
        if standard.get("permit_type") != permit_type:
            raise ValueError(config.STANDARD_OF_ANOTHER_PERMIT_TYPE_MESSAGE.format(
                permit_type=permit_type, document_name=standard.get("document_name"),
                stated_permit_type=standard.get("permit_type")))
        for requirement in standard["requirements"]:
            result, blank_answers = _judge(requirement, request, standard.get("document_name"))
            requirements.append({
                "permit_type": permit_type,
                "standard_document": standard.get("document_name"),
                "requirement_kind": requirement["requirement_kind"],
                "requirement_name": requirement["requirement_name"],
                "requirement_sentence": requirement["requirement_sentence"],
                "rules": dict(requirement.get("rules") or {}),
                "result": result,
                "blank_answers": blank_answers,
            })
    counts = {column_name: sum(1 for requirement in requirements if requirement["result"] == result)
              for column_name, result in config.REQUIREMENT_COUNT_RESULTS.items()}
    triggers = find_simops_triggers(request)
    if counts[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN] or counts[
            config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN]:
        assessment = config.DECISION_RETURN
    elif triggers:
        assessment = config.DECISION_REFER
    else:
        assessment = config.DECISION_ISSUE
    screening = {"permit_reference": reference, "permit_types": permit_types,
                 "requirements": requirements}
    screening.update(counts)
    screening.update({"simops_triggers": triggers, "assessment": assessment,
                      "decisions_open": list(config.DECISIONS_OPEN[assessment])})
    return screening
