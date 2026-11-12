"""The ten steps of the review workflow, the router, and the context every node and the service
share. Each node does one thing and returns only the keys it wrote."""
import secrets

from langgraph.types import interrupt

from agentic.agents import (analyse_predictions, analyse_requirements, draft_permit_document,
                            draft_work_party_briefing, join_labelled_response, plan_review,
                            review_evidence, write_review_recommendation)
from config import config
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.standard_reading import read_permit_standard
from knowledge.vector_index import open_vector_index
from modeling.model_registry import load_feature_schema, load_model
from persistence.database import session_factory
from persistence.repository import (insert_agent_run, insert_llm_call, insert_permit_review,
                                    insert_review_decision, record_review_outcome,
                                    update_agent_run)
from tools.decision_tools import find_decision_refusal
from tools.evidence_tools import (find_disagreements, link_evidence_to_requirements,
                                  read_document_evidence)
from tools.permit_document_tools import (build_permit_details, build_rework_details,
                                         build_simops_brief)
from tools.prediction_tools import (describe_expected_work_hours, estimate_expected_work_hours,
                                    estimate_suspension_likelihood)
from tools.request_data_tools import (fetch_request, fetch_request_documents, load_live_register,
                                      load_open_requests, load_request_documents)
from tools.retrieval_tools import (read_briefing_guidance, read_documentation_to_provide,
                                   read_permit_document_guidance, read_policy_section,
                                   read_reviewer_checks)
from tools.screening_tools import count_live_permits, screen_request

suspension_model = load_model(config.SUSPENSION_MODEL_FILE)
work_hours_model = load_model(config.WORK_HOURS_MODEL_FILE)
sentence_model = load_model(config.SENTENCE_MODEL_FILE)
suspension_feature_names = load_feature_schema(config.SUSPENSION_SCHEMA_FILE)
work_hours_feature_names = load_feature_schema(config.WORK_HOURS_SCHEMA_FILE)
open_requests = load_open_requests()
open_request_documents = load_request_documents()
live_register = load_live_register()
knowledge_documents = extract_text_from_directory(config.DOCS_DIRECTORY)
knowledge_collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)


def _identifier(prefix):
    return prefix + secrets.token_hex(config.IDENTIFIER_HEX_LENGTH // 2).upper()


def _record_call(run_id, call_record):
    with session_factory() as session:
        insert_llm_call(session, run_id, call_record)
        session.commit()


def gather_permit_context(permit_reference):
    """Everything about one request that needs no language model."""
    request = fetch_request(open_requests, permit_reference)
    documents = fetch_request_documents(open_request_documents, request)
    permit_types = [request[config.PERMIT_TYPE_COLUMN]]
    if request.get(config.LINKED_PERMIT_TYPE_COLUMN) not in (None, config.NO_LINKED_PERMIT):
        permit_types.append(request[config.LINKED_PERMIT_TYPE_COLUMN])
    standards = {permit_type: read_permit_standard(permit_type, knowledge_documents)
                 for permit_type in permit_types}
    register_count = count_live_permits(live_register, request)
    profile = dict(request)
    for column_name in [config.CONCURRENT_LIVE_PERMITS_COLUMN,
                        config.CONFLICTING_LIVE_PERMITS_COLUMN]:
        profile[column_name] = register_count[column_name]
    screening = screen_request(profile, standards)
    findings = link_evidence_to_requirements(read_document_evidence(sentence_model, documents),
                                             screening)
    return {"request_profile": profile, "request_documents": documents,
            "register_count": register_count, "screening": screening,
            "evidence_findings": findings,
            "disagreements": find_disagreements(findings, profile),
            "likelihood": estimate_suspension_likelihood(suspension_model,
                                                         suspension_feature_names, profile,
                                                         screening),
            "work_hours": estimate_expected_work_hours(work_hours_model, work_hours_feature_names,
                                                       profile, screening)}


def load_permit_context_node(state):
    """Gather the request's context and open the run."""
    context = gather_permit_context(state["permit_reference"])
    run_id = _identifier(config.RUN_ID_PREFIX)
    review_id = _identifier(config.REVIEW_ID_PREFIX)
    with session_factory() as session:
        insert_agent_run(session, {"run_id": run_id, "thread_id": state["thread_id"],
                                   "permit_reference": state["permit_reference"],
                                   "coordinator_id": state["coordinator_id"],
                                   "coordinator_request": state["coordinator_request"],
                                   "status": config.AGENT_RUN_STATUS_RUNNING,
                                   "llm_call_count": 0})
        session.commit()
    return dict(context, run_id=run_id, review_id=review_id)


def plan_permit_review_node(state):
    """Ask the planner what the review has to establish."""
    planned = plan_review(state["request_profile"], state["request_documents"], state["screening"],
                          state["register_count"], state["coordinator_request"])
    _record_call(state["run_id"], planned["call_record"])
    return {"review_plan": planned["review_plan"], "llm_call_count": 1}


def assess_permit_requirements_node(state):
    """Ask the requirements analyst to explain the screening and the SIMOPS triggers."""
    simops_rules = read_policy_section(knowledge_collection, config.POLICY_SIMOPS_SECTION_NUMBER,
                                       config.POLICY_SIMOPS_SECTION_TITLE)
    analysed = analyse_requirements(state["screening"], state["request_profile"],
                                    state["register_count"], state["review_plan"], simops_rules)
    _record_call(state["run_id"], analysed["call_record"])
    return {"requirements_analysis": analysed["requirements_analysis"], "llm_call_count": 1}


def analyse_method_statements_node(state):
    """Ask the evidence analyst to review the documents with the guides' checks in hand."""
    found = {finding["sentence_type"] for finding in state["evidence_findings"]}
    reviewer_checks = {sentence_type: read_reviewer_checks(knowledge_collection, sentence_type)
                       for sentence_type in config.SENTENCE_TYPE_GUIDE_DOCUMENTS
                       if sentence_type in found}
    reviewed = review_evidence(state["evidence_findings"], state["disagreements"],
                               reviewer_checks, state["request_profile"], state["screening"],
                               state["review_plan"])
    _record_call(state["run_id"], reviewed["call_record"])
    return {"evidence_review": reviewed["evidence_review"], "llm_call_count": 1}


def interpret_model_predictions_node(state):
    """Ask the risk analyst to explain the two figures."""
    analysed = analyse_predictions(state["likelihood"], state["work_hours"],
                                   state["request_profile"], state["screening"],
                                   state["review_plan"])
    _record_call(state["run_id"], analysed["call_record"])
    return {"prediction_analysis": analysed["prediction_analysis"], "llm_call_count": 1}


def prepare_permit_recommendation_node(state):
    """Ask the writer for the recommendation, and store the review waiting at the gate."""
    decision_rules = read_policy_section(knowledge_collection,
                                         config.POLICY_DECISIONS_SECTION_NUMBER,
                                         config.POLICY_DECISIONS_SECTION_TITLE)
    written = write_review_recommendation(
        state["request_profile"], state["coordinator_request"], state["screening"],
        state["register_count"], state["likelihood"], state["work_hours"], state["review_plan"],
        state["requirements_analysis"], state["evidence_review"], state["prediction_analysis"],
        decision_rules)
    recommendation = written["review_recommendation"]
    write_ups = {"review_plan_text": join_labelled_response(state["review_plan"]),
                 "requirements_analysis_text": join_labelled_response(
                     state["requirements_analysis"]),
                 "evidence_review_text": join_labelled_response(state["evidence_review"]),
                 "prediction_analysis_text": join_labelled_response(state["prediction_analysis"]),
                 "recommendation_text": recommendation["RECOMMENDATION"]}
    short = [name for name, text in write_ups.items() if len(text.split()) < 8]
    if short or len(set(write_ups.values())) != len(write_ups):
        raise ValueError("The review's write-ups are too short or repeat one another: "
                         + ", ".join(short))
    profile, screening = state["request_profile"], state["screening"]
    keep = ["permit_type", "standard_document", "requirement_kind", "requirement_name",
            "requirement_sentence", "result", "blank_answers"]
    review = {
        "review_id": state["review_id"], "run_id": state["run_id"],
        "permit_reference": state["permit_reference"],
        "revision_number": int(profile[config.REVISION_NUMBER_COLUMN]),
        "permit_type": profile[config.PERMIT_TYPE_COLUMN],
        "linked_permit_type": profile[config.LINKED_PERMIT_TYPE_COLUMN],
        "plant_area": profile[config.PLANT_AREA_COLUMN],
        "coordinator_request": state["coordinator_request"],
        "requirements": [{key: requirement[key] for key in keep}
                         for requirement in screening["requirements"]],
        "live_permits": state["register_count"]["live_permits"],
        "conflicting_permits": state["register_count"]["conflicting_permits"],
        "simops_triggers": screening["simops_triggers"],
        "assessment": screening["assessment"],
        "decisions_open": screening["decisions_open"],
        "evidence": state["evidence_findings"], "disagreements": state["disagreements"],
        "suspension_likelihood": state["likelihood"]["suspension_likelihood"],
        "likelihood_band": state["likelihood"]["likelihood_band"],
        "expected_work_hours": state["work_hours"]["expected_work_hours"],
        "planned_work_hours": state["work_hours"]["planned_work_hours"],
        "extension_expected": state["work_hours"]["extension_expected"],
        "work_hours_text": describe_expected_work_hours(state["work_hours"]),
        "recommended_decision": recommendation["RECOMMENDED DECISION"],
        "status": config.REVIEW_STATUS_AWAITING,
    }
    for column_name in config.REQUIREMENT_COUNT_RESULTS:
        review[column_name] = screening[column_name]
    review.update(write_ups)
    with session_factory() as session:
        insert_llm_call(session, state["run_id"], written["call_record"])
        insert_permit_review(session, review)
        update_agent_run(session, state["run_id"], config.AGENT_RUN_STATUS_AWAITING_DECISION,
                         state["llm_call_count"] + 1)
        session.commit()
    return {"llm_call_count": 1}


def await_area_authority_decision_node(state):
    """Stop until an Area Authority decides; accept only a decision that may be recorded."""
    review_decision = interrupt({"review_id": state["review_id"]})
    refusal = find_decision_refusal(review_decision, state["request_profile"], state["screening"],
                                    state["coordinator_id"])
    if refusal:
        raise ValueError(refusal)
    return {"review_decision": review_decision}


def route_permit_decision(state):
    """The path out of the gate and out of the permit document is named by the decision."""
    return state["review_decision"]["decision"]


def draft_permit_document_node(state):
    """Write the permit document or the rework request."""
    decision = state["review_decision"]["decision"]
    profile, screening = state["request_profile"], state["screening"]
    if decision == config.DECISION_ISSUE:
        details = build_permit_details(profile, screening)
        guidance = "\n\n".join(config.STANDARD_PASSAGE_TEMPLATE.format(
            permit_type=permit_type,
            passage=read_permit_document_guidance(knowledge_collection, permit_type))
            for permit_type in screening["permit_types"])
    else:
        documentation = {permit_type: read_documentation_to_provide(knowledge_collection,
                                                                    permit_type)
                         for permit_type in screening["permit_types"]}
        details = build_rework_details(profile, screening, documentation, state["disagreements"])
        guidance = read_policy_section(knowledge_collection, config.POLICY_DOCUMENTS_SECTION_NUMBER,
                                       config.POLICY_DOCUMENTS_SECTION_TITLE)
    drafted = draft_permit_document(decision, profile, details, guidance)
    _record_call(state["run_id"], drafted["call_record"])
    return {"permit_document": drafted["permit_document"], "document_details": details,
            "llm_call_count": 1}


def draft_work_party_briefing_node(state):
    """Write the work party briefing, after a permit document."""
    guidance = "\n\n".join(config.STANDARD_PASSAGE_TEMPLATE.format(
        permit_type=permit_type, passage=read_briefing_guidance(knowledge_collection, permit_type))
        for permit_type in state["screening"]["permit_types"])
    drafted = draft_work_party_briefing(state["document_details"], guidance)
    _record_call(state["run_id"], drafted["call_record"])
    return {"work_party_briefing": drafted["work_party_briefing"], "llm_call_count": 1}


def record_permit_outcome_node(state):
    """Record the decision and everything it led to, and close the run."""
    decision = state["review_decision"]
    chosen = decision["decision"]
    outcome = {"status": config.REVIEW_STATUS_BY_DECISION[chosen], "permit_document_text": None,
               "document_details": None, "work_party_briefing_text": None, "simops_brief": None}
    if chosen in (config.DECISION_ISSUE, config.DECISION_RETURN):
        outcome["permit_document_text"] = join_labelled_response(state["permit_document"])
        outcome["document_details"] = state["document_details"]
    if chosen == config.DECISION_ISSUE:
        outcome["work_party_briefing_text"] = join_labelled_response(state["work_party_briefing"])
        texts = [outcome["permit_document_text"], outcome["work_party_briefing_text"]]
        if any(len(text.split()) < 8 for text in texts) or texts[0] == texts[1]:
            raise ValueError("The permit document and the work party briefing must each hold "
                             "eight words and differ")
    if chosen == config.DECISION_REFER:
        outcome["simops_brief"] = build_simops_brief(
            state["request_profile"], state["screening"], state["register_count"],
            state["likelihood"], state["work_hours"], state["disagreements"])
    with session_factory() as session:
        insert_review_decision(session, dict(
            {field: decision[field] for field in config.DECISION_RECORD_FIELDS},
            review_id=state["review_id"], run_id=state["run_id"]))
        record_review_outcome(session, state["review_id"], outcome)
        update_agent_run(session, state["run_id"], config.AGENT_RUN_STATUS_COMPLETE,
                         state["llm_call_count"])
        session.commit()
    return {}
