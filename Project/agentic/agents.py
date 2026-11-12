"""The language model client, the reply parser, and the seven agents.

Each agent hands the model facts the tools already established and asks for prose in a fixed
layout of labelled fields. No agent calculates anything of its own.
"""
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import config
from tools.prediction_tools import describe_expected_work_hours
from tools.request_data_tools import describe_record

load_dotenv(config.ENV_FILE)

generation_settings = types.GenerateContentConfig(
    max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
    temperature=config.GEMINI_TEMPERATURE,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

_ACCEPT_ATTEMPTS = 3
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

_PREAMBLE = (f"You write for the permit office of {config.ORGANISATION_NAME} at {config.SITE_NAME}. "
             "Use only the facts given below. Every result, figure and passage was established "
             "by the permit standards, the live permit register, the models and the documents; "
             "never work anything out for yourself, never change a result and never add a figure "
             "that is not given.")


# ---------------------------------------------------------------- the client and the parser


def call_language_model(prompt_text, agent_name):
    """Make one language model call and measure what it cost."""
    api_key = os.environ.get(config.GEMINI_API_KEY_VARIABLE)
    model_name = os.environ.get(config.GEMINI_MODEL_VARIABLE)
    if not api_key or not model_name:
        raise ValueError(f"{config.GEMINI_API_KEY_VARIABLE} and {config.GEMINI_MODEL_VARIABLE} "
                         f"must both be set in {config.ENV_FILE} before {agent_name} can be called")
    for attempt in range(_ACCEPT_ATTEMPTS):
        client = genai.Client(api_key=api_key)
        try:
            started = time.perf_counter()
            response = client.models.generate_content(model=model_name, contents=prompt_text,
                                                      config=generation_settings)
            elapsed = time.perf_counter() - started
            break
        except genai.errors.APIError as error:
            if getattr(error, "code", None) not in _TRANSIENT_STATUS_CODES or (
                    attempt == _ACCEPT_ATTEMPTS - 1):
                raise
            time.sleep(2 * (attempt + 1))
        finally:
            client.close()
    text = (response.text or "").strip() if response.candidates else ""
    finish = response.candidates[0].finish_reason if response.candidates else None
    usage = response.usage_metadata
    if not text:
        raise ValueError(f"{agent_name} received an empty reply from {model_name}")
    if finish != types.FinishReason.STOP:
        raise ValueError(f"{agent_name}'s reply from {model_name} did not finish normally: "
                         f"{finish}")
    if usage is None or not usage.prompt_token_count or not usage.candidates_token_count \
            or not usage.total_token_count:
        raise ValueError(f"{agent_name}'s reply from {model_name} does not say how many tokens "
                         "it used")
    return {"response_text": text,
            "call_record": {"agent_name": agent_name, "model_name": model_name,
                            "prompt_tokens": int(usage.prompt_token_count),
                            "response_tokens": int(usage.candidates_token_count),
                            "total_tokens": int(usage.total_token_count),
                            "latency_milliseconds": int(round(
                                elapsed * config.MILLISECONDS_PER_SECOND))}}


def _undecorated(text):
    return re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "", text)


def parse_labelled_response(response_text, field_labels, agent_name):
    """Read a labelled reply back into exactly the fields asked for, in order."""
    wanted = {label.upper(): label for label in field_labels}
    fields = {label: [] for label in field_labels}
    current = None
    for raw_line in str(response_text).splitlines():
        line = raw_line.replace("**", "").strip()
        if ":" in line:
            before, after = line.split(":", 1)
            label = _undecorated(before.strip()).replace("-", " ").strip().upper()
            label = re.sub(r"\s+", " ", label)
            if label in wanted:
                current = wanted[label]
                value = after.strip(config.RESPONSE_DECORATION_CHARACTERS)
                if value:
                    fields[current].append(value)
                continue
        if current is None or not _undecorated(line):
            continue
        fields[current].append(line)
    missing = [label for label in field_labels if not "\n".join(fields[label]).strip()]
    if missing:
        raise ValueError(f"{agent_name} replied without " + ", ".join(missing))
    return {label: "\n".join(fields[label]).strip() for label in field_labels}


def join_labelled_response(labelled_fields):
    """An agent's fields written back as one text, a blank line between fields."""
    return "\n\n".join(f"{label}: {text}" for label, text in labelled_fields.items())


def _ask(prompt_text, agent_name, field_labels, accept=None):
    """One accepted call: a reply refused by the parser or the checks is asked for again, and
    only the accepted call is handed back to be recorded."""
    problem = None
    for _attempt in range(_ACCEPT_ATTEMPTS):
        prompt = prompt_text
        if problem:
            prompt = (prompt_text + "\n\nYOUR PREVIOUS REPLY WAS REFUSED: " + problem
                      + "\nWrite the whole reply again in the required layout.")
        called = call_language_model(prompt, agent_name)
        try:
            fields = parse_labelled_response(called["response_text"], field_labels, agent_name)
            if accept:
                accept(fields)
            return fields, called["call_record"]
        except ValueError as error:
            problem = str(error)
    raise ValueError(problem)


def _layout(field_labels, guidance):
    lines = ["Reply with exactly these labelled fields, each label in capitals at the start of "
             "its own line followed by a colon, and nothing else:"]
    lines += [f"{label}: {guidance[label]}" for label in field_labels]
    return "\n".join(lines)


# ---------------------------------------------------------------- facts set out for prompts


def _answer_line(column_name, value):
    return f"- {config.FORM_ANSWER_LABELS.get(column_name, column_name)}: {value}"


def _request_lines(request_profile):
    described = describe_record(request_profile)
    return [f"- the permit reference: {described[config.PERMIT_REFERENCE_COLUMN]}"] + [
        _answer_line(column_name, value) for column_name, value in described.items()
        if column_name in config.FORM_ANSWER_LABELS]


def _none_or(items):
    return items if items else ["- none"]


def _requirement_lines(screening, request_profile=None, with_sentences=False):
    lines = []
    described = describe_record(request_profile) if request_profile else {}
    for requirement in screening["requirements"]:
        blank = ", ".join(config.FORM_ANSWER_LABELS.get(column, column)
                          for column in requirement["blank_answers"]) or "none"
        line = (f"- {requirement['permit_type']} standard ({requirement['standard_document']}), "
                f"{requirement['requirement_name']}: {requirement['result']}; answers left blank: "
                f"{blank}")
        if with_sentences:
            line += f"\n  the standard says: {requirement['requirement_sentence']}"
            columns = config.REQUIREMENT_ANSWER_COLUMNS.get(requirement["requirement_kind"], [])
            answers = "; ".join(f"{config.FORM_ANSWER_LABELS.get(column, column)}: "
                                f"{described.get(column, config.NOT_GIVEN_TEXT)}"
                                for column in columns)
            line += f"\n  the answers it reads: {answers}"
        lines.append(line)
    return _none_or(lines)


def _screening_summary(screening, register_count):
    return [
        f"- requirements Met: {screening[config.REQUIREMENTS_MET_COUNT_COLUMN]}, Not met: "
        f"{screening[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN]}, Not documented: "
        f"{screening[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN]}",
        "- permits live in the plant area: "
        + (", ".join(register_count["live_permits"]) or "none"),
        "- conflicting permits live in the plant area: "
        + (", ".join(register_count["conflicting_permits"]) or "none"),
        "- SIMOPS triggers that apply: " + ("; ".join(screening["simops_triggers"]) or "none"),
        f"- assessment: {screening['assessment']}",
        "- decisions open: " + ", ".join(screening["decisions_open"]),
    ]


def _band_lines(likelihood):
    return [f"- suspension likelihood: {likelihood['suspension_likelihood']:.{config.LIKELIHOOD_DECIMALS}f}"
            f" (band {likelihood['likelihood_band']})",
            f"- bands: {config.HIGH_LIKELIHOOD_BAND} from {config.HIGH_LIKELIHOOD_MINIMUM}, "
            f"{config.MEDIUM_LIKELIHOOD_BAND} from {config.MEDIUM_LIKELIHOOD_MINIMUM}, otherwise "
            f"{config.LOW_LIKELIHOOD_BAND}"]


def _section(title, lines):
    return title + ":\n" + "\n".join(lines)


# ---------------------------------------------------------------- the five reviewing agents


def plan_review(request_profile, request_documents, screening, register_count,
                coordinator_request):
    """Decide what the review has to establish about this request."""
    labels = ["PRIORITY", "CHECKS", "RATIONALE"]
    documents = [f"- {document.get(config.DOCUMENT_ID_COLUMN)}: "
                 f"{describe_record(document).get(config.DOCUMENT_TITLE_COLUMN)}"
                 for document in request_documents] or ["- no document is attached"]
    prompt = "\n\n".join([
        "You are the review planner. " + _PREAMBLE,
        "Do not alter any screening result, and name no figure beyond those given here.",
        _section("THE REQUEST", _request_lines(request_profile)),
        _section("ITS DOCUMENTS, IN ORDER", documents),
        _section("EVERY REQUIREMENT WITH ITS RESULT", _requirement_lines(screening)),
        _section("THE SCREENING", _screening_summary(screening, register_count)),
        _section("WHAT THE PERSON WHO OPENED THE REVIEW ASKED", [coordinator_request]),
        _layout(labels, {"PRIORITY": "one sentence naming what matters most for this review",
                         "CHECKS": "a numbered list of the checks the review must make",
                         "RATIONALE": "two or three sentences on why, from the facts given"}),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_PLANNER, labels)
    return {"review_plan": fields, "call_record": call_record}


def analyse_requirements(screening, request_profile, register_count, review_plan, simops_rules):
    """Account for each requirement's result, and for the SIMOPS triggers that apply."""
    labels = ["REQUIREMENTS MET", "REQUIREMENTS NOT MET", "NOT DOCUMENTED", "SIMOPS TRIGGERS",
              "REQUIREMENTS SUMMARY"]
    described = describe_record(request_profile)
    trigger_answers = [_answer_line(column, described.get(column, config.NOT_GIVEN_TEXT))
                       for column in config.TRIGGER_ANSWER_COLUMNS]
    prompt = "\n\n".join([
        "You are the requirements analyst. " + _PREAMBLE,
        "The screening's results are final as given: a blank answer stays blank, a result is "
        "never changed, and the request is never presented as decided.",
        _section("EVERY REQUIREMENT, IN THE SCREENING'S ORDER",
                 _requirement_lines(screening, request_profile, with_sentences=True)),
        _section("THE SCREENING", _screening_summary(screening, register_count)),
        _section("THE CONTROL OF WORK POLICY ON THE SIMOPS TRIGGERS", [simops_rules]),
        _section("THE ANSWERS THE TRIGGERS READ", trigger_answers),
        _section("THE PLAN'S CHECKS", [review_plan["CHECKS"]]),
        _layout(labels, {
            "REQUIREMENTS MET": "each requirement Met, with its standard, or none",
            "REQUIREMENTS NOT MET": "each requirement Not met, with its standard and why, or none",
            "NOT DOCUMENTED": "each requirement Not documented and the answers it needs, or none",
            "SIMOPS TRIGGERS": "each trigger that applies and why, or none",
            "REQUIREMENTS SUMMARY": "three or four sentences an Area Authority can check"}),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_REQUIREMENTS, labels)
    return {"requirements_analysis": fields, "call_record": call_record}


def review_evidence(evidence_findings, disagreements, reviewer_checks, request_profile, screening,
                    review_plan):
    """Set the evidence in the documents beside the request, with flags and the guides' checks."""
    labels = ["EVIDENCE FOUND", "DISAGREEMENTS", "CHECKS TO MAKE", "EVIDENCE SUMMARY"]
    described = describe_record(request_profile)
    columns = []
    for requirement in screening["requirements"]:
        columns += config.REQUIREMENT_ANSWER_COLUMNS.get(requirement["requirement_kind"], [])
    columns += config.STATEMENT_GUIDE_ANSWER_COLUMNS
    answers = [_answer_line(column, described.get(column, config.NOT_GIVEN_TEXT))
               for column in dict.fromkeys(columns)]
    sentences, current = [], None
    for finding in evidence_findings:
        if finding["document_id"] != current:
            current = finding["document_id"]
            sentences.append(f"{current} ({finding['document_type']}):")
        borne = ", ".join(f"{item['permit_type']} {item['requirement_name']}"
                          for item in finding.get("requirements_borne_on", [])) or "none"
        sentences.append(f"- [{finding['sentence_type']}] {finding['sentence']} (bears on: "
                         f"{borne})")
    flags = [f"- {flag['document_id']}: \"{flag['sentence']}\" flagged {flag['flag']}; it states "
             f"{flag['stated_readings']} where the request gives {flag['request_readings']}"
             for flag in disagreements]
    checks = [f"- {sentence_type} ({config.SENTENCE_TYPE_GUIDE_DOCUMENTS[sentence_type]}): {text}"
              for sentence_type, text in reviewer_checks.items()]
    prompt = "\n\n".join([
        "You are the evidence analyst. " + _PREAMBLE,
        "The documents are evidence only and never take the place of an answer on the request. "
        "The flags are left to the Area Authority; a difference that is not flagged may only be "
        "mentioned as advice.",
        _section("THE REQUEST'S ANSWERS", answers),
        _section("EVERY SENTENCE, DOCUMENT BY DOCUMENT", _none_or(sentences)),
        _section("FLAGGED SENTENCES", _none_or(flags)),
        _section("THE STATEMENT GUIDES' CHECKS", _none_or(checks)),
        _section("THE PLAN'S CHECKS", [review_plan["CHECKS"]]),
        _layout(labels, {
            "EVIDENCE FOUND": "what the documents show, document by document, or none",
            "DISAGREEMENTS": "each flagged sentence and its flag, or none",
            "CHECKS TO MAKE": "the guides' checks the Area Authority should make",
            "EVIDENCE SUMMARY": "two or three sentences"}),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_EVIDENCE, labels)
    return {"evidence_review": fields, "call_record": call_record}


def analyse_predictions(likelihood, work_hours, request_profile, screening, review_plan):
    """Interpret the suspension likelihood and the expected work hours, with their inputs."""
    labels = ["LIKELIHOOD", "EXPECTED WORK HOURS", "PREDICTION SUMMARY"]
    profile = dict(request_profile)
    for column, result in config.REQUIREMENT_COUNT_RESULTS.items():
        profile[column] = screening[column]
    for trigger, column in config.SIMOPS_TRIGGER_FLAG_COLUMNS.items():
        profile[column] = int(trigger in screening["simops_triggers"])
    described = describe_record(profile)
    suspension_inputs = [_answer_line(column, described.get(column, config.NOT_GIVEN_TEXT))
                         for column in (config.SUSPENSION_NUMERIC_FEATURES
                                        + config.SUSPENSION_CATEGORICAL_FEATURES)]
    hours_inputs = [_answer_line(column, described.get(column, config.NOT_GIVEN_TEXT))
                    for column in (config.WORK_HOURS_NUMERIC_FEATURES
                                   + config.WORK_HOURS_CATEGORICAL_FEATURES)]
    prompt = "\n\n".join([
        "You are the risk analyst. " + _PREAMBLE,
        "Neither figure has any bearing on a requirement, a SIMOPS trigger, a decision or a "
        "document. Do not recalculate either figure.",
        _section("THE SUSPENSION LIKELIHOOD", _band_lines(likelihood)),
        _section("THE SUSPENSION MODEL'S INPUTS", suspension_inputs),
        _section("THE EXPECTED WORK HOURS", [
            "- " + describe_expected_work_hours(work_hours),
            f"- an extension is expected when the expected work hours are more than "
            f"{config.EXTENSION_RATIO} times the planned work hours"]),
        _section("THE WORK HOURS MODEL'S INPUTS", hours_inputs),
        _section("BLANK INPUTS", [
            f"- a blank number is filled with the training data's {config.NUMBER_FILL_STRATEGY} "
            f"and a blank answer in words with its {config.CATEGORY_FILL_STRATEGY.replace('_', ' ')}"
            " value"]),
        _section("THE PLAN'S CHECKS", [review_plan["CHECKS"]]),
        _layout(labels, {"LIKELIHOOD": "what the likelihood and its band mean here",
                         "EXPECTED WORK HOURS": "what the expected work hours mean here",
                         "PREDICTION SUMMARY": "two sentences, saying neither figure decides"}),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_RISK, labels)
    return {"prediction_analysis": fields, "call_record": call_record}


def _recommended_decision_check(decisions_open, agent_name):
    def check(fields):
        written = fields["RECOMMENDED DECISION"]
        if "\n" in written:
            raise ValueError(f"{agent_name} wrote its RECOMMENDED DECISION over more than one line")
        decision = _undecorated(written).upper()
        if decision not in decisions_open:
            raise ValueError(f"{agent_name} recommended {written!r}, which is not among the "
                             "decisions open: " + ", ".join(decisions_open))
        fields["RECOMMENDED DECISION"] = decision
    return check


def write_review_recommendation(request_profile, coordinator_request, screening, register_count,
                                likelihood, work_hours, review_plan, requirements_analysis,
                                evidence_review, prediction_analysis, decision_rules):
    """Write the review an Area Authority decides from, recommending one decision open."""
    labels = ["RECOMMENDATION", "RECOMMENDED DECISION"]
    analysts = []
    for title, fields in [("THE REQUIREMENTS ANALYST", requirements_analysis),
                          ("THE EVIDENCE ANALYST", evidence_review),
                          ("THE RISK ANALYST", prediction_analysis)]:
        analysts.append(_section(title, [f"{label}: {text}" for label, text in fields.items()]))
    prompt = "\n\n".join([
        "You are the review writer. " + _PREAMBLE,
        "You only recommend; an Area Authority decides. Recommend the assessment unless the "
        "review argues for another decision open, and then give the reason. Neither the "
        "suspension likelihood nor the expected work hours may be a reason.",
        _section("THE REQUEST", _request_lines(request_profile)),
        _section("WHAT THE PERSON WHO OPENED THE REVIEW ASKED", [coordinator_request]),
        _section("THE PLAN", [f"PRIORITY: {review_plan['PRIORITY']}",
                              f"CHECKS: {review_plan['CHECKS']}"]),
        *analysts,
        _section("EVERY REQUIREMENT WITH ITS RESULT", _requirement_lines(screening)),
        _section("THE SCREENING", _screening_summary(screening, register_count)),
        _section("THE TWO FIGURES", _band_lines(likelihood)
                 + ["- " + describe_expected_work_hours(work_hours)]),
        _section("THE CONTROL OF WORK POLICY ON THE THREE DECISIONS", [decision_rules]),
        _layout(labels, {
            "RECOMMENDATION": "several sentences of prose; the last names the open decision "
                              "recommended and the reason",
            "RECOMMENDED DECISION": "exactly one of: " + ", ".join(screening["decisions_open"])}),
    ])
    fields, call_record = _ask(prompt, config.AGENT_NAME_WRITER, labels,
                               _recommended_decision_check(screening["decisions_open"],
                                                           config.AGENT_NAME_WRITER))
    return {"review_recommendation": fields, "call_record": call_record}


# ---------------------------------------------------------------- the two writing agents


def _flattened(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def _sign_off_form(text):
    return re.sub(config.SIGN_OFF_SPACING_PATTERN, " ", text).strip().lower()


def _check_written_document(agent_name, body_label, heading_label, must_name, forbidden_names=()):
    def check(fields):
        heading, body = fields[heading_label], fields[body_label]
        if "\n" in heading:
            raise ValueError(f"{agent_name} wrote its {heading_label} over more than one line")
        joined = _flattened(heading + " " + body)
        missing = [name for name in must_name if _flattened(str(name)) not in joined]
        if missing:
            raise ValueError(f"{agent_name} did not name " + ", ".join(map(str, missing)))
        spoken = [term for term in config.DOCUMENT_FORBIDDEN_TERMS if term.lower() in joined]
        if spoken:
            raise ValueError(f"{agent_name} spoke of " + ", ".join(spoken))
        named = [name for name in forbidden_names
                 if re.search(r"\b" + re.escape(name.lower()) + r"\b", joined)]
        if named:
            raise ValueError(f"{agent_name} named members of staff: " + ", ".join(named))
        if any(character in body for character in config.TEMPLATE_SLOT_CHARACTERS):
            raise ValueError(f"{agent_name} left a gap to fill in its {body_label}")
        if not _sign_off_form(body).endswith(_sign_off_form(config.DOCUMENT_SIGN_OFF)):
            raise ValueError(f"{agent_name}'s {body_label} does not end with "
                             f"{config.DOCUMENT_SIGN_OFF}")
    return check


def _detail_lines(details, leave_out_reference=False):
    described = describe_record({key: value for key, value in details.items()
                                 if key != "controls"})
    lines = [f"- {config.DOCUMENT_DETAIL_LABELS[key]}: {value}"
             for key, value in described.items() if key in config.DOCUMENT_DETAIL_LABELS
             and not (leave_out_reference and key == config.PERMIT_REFERENCE_COLUMN)]
    for permit_type, controls in details.get("controls", {}).items():
        lines.append(f"- controls under the {permit_type} standard:")
        lines += ["  " + _answer_line(column, value)
                  for column, value in describe_record(controls).items()] or ["  - none"]
    return lines


def _rework_lines(details):
    lines = [f"- Permit reference: {details['permit_reference']}",
             "- standards screened: " + ", ".join(details["permit_types"]),
             "- requirements to put right:"]
    for item in details["requirements_to_put_right"]:
        answers = "; ".join(_answer_line(column, value)[2:]
                            for column, value in describe_record(item["request_answers"]).items())
        needed = ", ".join(config.FORM_ANSWER_LABELS.get(column, column)
                           for column in item["blank_answers"]) or "none"
        lines.append(f"  - {item['requirement_name']} ({item['permit_type']} standard): "
                     f"{item['result']}. The standard says: {item['requirement_sentence']} The "
                     f"request gives: {answers or 'none'}. Answers it needs: {needed}.")
    if not details["requirements_to_put_right"]:
        lines.append("  - none")
    lines.append("- documentation to provide:")
    lines += [f"  - under the {permit_type} standard: {text}"
              for permit_type, text in details["documentation_to_provide"].items()] or ["  - none"]
    lines.append("- flagged statements:")
    lines += [f"  - {flag['document_id']}: \"{flag['sentence']}\" ({flag['flag']})"
              for flag in details["flagged_statements"]] or ["  - none"]
    return lines


def draft_permit_document(decision, request_profile, document_details, document_guidance):
    """Write the permit document or the rework request to the performing authority."""
    labels = ["SUBJECT", "DOCUMENT"]
    reference = request_profile[config.PERMIT_REFERENCE_COLUMN]
    if decision == config.DECISION_ISSUE:
        if not document_details.get(config.EQUIPMENT_TAG_COLUMN):
            raise ValueError(f"The permit details of {reference} give no equipment tag")
        details = _detail_lines(document_details)
        must_name = [document_details[config.PERMIT_TYPE_COLUMN],
                     document_details[config.PLANT_AREA_COLUMN],
                     document_details[config.EQUIPMENT_TAG_COLUMN]]
    else:
        details = _rework_lines(document_details)
        must_name = [item["requirement_name"]
                     for item in document_details["requirements_to_put_right"]]
    kind = config.DOCUMENT_KINDS[decision]
    prompt = "\n\n".join([
        f"You are the permit writer. Write the {kind} to the performing authority. " + _PREAMBLE,
        f"The decision: {decision} - {config.DECISION_DESCRIPTIONS[decision]}.",
        f"The request came through the {request_profile.get('request_channel')} channel.",
        _section("THE DETAILS", details),
        _section("THE GUIDANCE", [document_guidance]),
        "Give no figure from any model, name no member of staff, and invent no time, date or "
        "contact detail. Do not use the words " + ", ".join(
            f"'{term}'" for term in config.DOCUMENT_FORBIDDEN_TERMS)
        + ". Leave no gap to fill in and use no square brackets. The SUBJECT is one line naming "
        f"the permit reference {reference}. The DOCUMENT ends with the line: "
        f"{config.DOCUMENT_SIGN_OFF}",
        _layout(labels, {"SUBJECT": f"one line naming {reference} and what the {kind} is",
                         "DOCUMENT": f"the {kind}, in plain prose paragraphs"}),
    ])
    check = _check_written_document(config.AGENT_NAME_PERMIT_WRITER, "DOCUMENT", "SUBJECT",
                                    must_name)

    def accept(fields):
        if reference.lower() not in fields["SUBJECT"].lower():
            raise ValueError(f"{config.AGENT_NAME_PERMIT_WRITER}'s SUBJECT does not name "
                             f"{reference}")
        check(fields)

    fields, call_record = _ask(prompt, config.AGENT_NAME_PERMIT_WRITER, labels, accept)
    return {"permit_document": fields, "call_record": call_record}


def draft_work_party_briefing(permit_details, briefing_guidance):
    """Write the work party briefing for the people who will do the work."""
    labels = ["HEADLINE", "BRIEFING"]
    if not permit_details.get(config.EQUIPMENT_TAG_COLUMN):
        raise ValueError(f"The permit details of {permit_details.get('permit_reference')} give no "
                         "equipment tag")
    must_name = [permit_details[config.PERMIT_TYPE_COLUMN],
                 permit_details[config.EQUIPMENT_TAG_COLUMN]]
    linked = permit_details.get(config.LINKED_PERMIT_TYPE_COLUMN)
    if linked and linked != config.NO_LINKED_PERMIT:
        must_name.append(linked)
    staff_names = [account["username"] for account in config.SEED_USER_ACCOUNTS] + [
        account["display_name"] for account in config.SEED_USER_ACCOUNTS]
    prompt = "\n\n".join([
        "You are the briefing writer. Write the work party briefing for the people who will do "
        "the work, in plain words. " + _PREAMBLE,
        _section("THE DETAILS", _detail_lines(permit_details, leave_out_reference=True)),
        _section("THE GUIDANCE", [briefing_guidance]),
        "Give no figure from any model and none of the review's results, name no member of "
        "staff, and invent no time, date or contact detail. Do not use the words " + ", ".join(
            f"'{term}'" for term in config.DOCUMENT_FORBIDDEN_TERMS)
        + ". Name the permit type, the equipment tag and any linked permit type. Leave no gap "
        f"to fill in and use no square brackets. The BRIEFING ends with the line: "
        f"{config.DOCUMENT_SIGN_OFF}",
        _layout(labels, {"HEADLINE": "one line", "BRIEFING": "the briefing, in plain prose"}),
    ])
    check = _check_written_document(config.AGENT_NAME_BRIEFING_WRITER, "BRIEFING", "HEADLINE",
                                    must_name, staff_names)
    fields, call_record = _ask(prompt, config.AGENT_NAME_BRIEFING_WRITER, labels, check)
    return {"work_party_briefing": fields, "call_record": call_record}
