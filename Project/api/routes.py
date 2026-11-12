"""The REST service: the queue, the request detail, the review workflow, the decision, the model
calls, the chat and the audit trail, behind one sign-in and permission guard and one failure
handler. Every figure the dashboard shows comes from here."""
import hmac
import secrets
import threading
from decimal import Decimal
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors
from langgraph.types import Command
from sqlalchemy import and_, true

from access.password_security import hash_password
from access.permissions import SessionTokenHeader, authorize_plant_area, authorize_request
from agentic.graph import build_workflow
from agentic.nodes import (gather_permit_context, knowledge_collection, knowledge_documents,
                           open_requests)
from api.schemas import (ChatRequest, ChatResponse, ChatTurnResponse, DecisionRequest,
                         PermitTypeFilter, QueueRow, RequestDetailResponse, ReviewCallsResponse,
                         ReviewRecordResponse, ReviewRequest, ReviewStatusValue,
                         SignedInUserResponse, SignedOutResponse, SignInRequest)
from chat.conversation_memory import recall_conversation
from chat.permit_assistant import answer_from_documents
from chat.record_answer import answer_plant_area_figure, answer_request_record
from config import config
from persistence.database import session_factory
from persistence.models import AgentRun, ChatTurn, PermitReview, UserAccount
from persistence.repository import (fetch_chat_turns, fetch_llm_calls, fetch_review_records,
                                    fetch_user_account, insert_chat_turn, update_agent_run,
                                    update_session_token)
from tools.decision_tools import find_decision_refusal, find_review_conflict
from tools.request_data_tools import (describe_record, load_permit_history, plain_value,
                                      rank_request_queue)

SignedInUser = Annotated[dict, Depends(authorize_request)]


def _plain(value):
    """Every dictionary a reply carries holds only plain Python values."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    return plain_value(value)


review_workflow = build_workflow(config.CHECKPOINT_DATABASE_FILE)
permit_history = load_permit_history()
request_profiles = {}
request_details = {}
request_queue = []
for _position, _reference in enumerate(rank_request_queue(open_requests), start=1):
    _context = gather_permit_context(_reference)
    _profile = _context["request_profile"]
    request_profiles[_reference] = _profile
    _detail = {"queue_position": _position,
               "request": describe_record(_profile),
               "request_documents": [describe_record(document)
                                     for document in _context["request_documents"]],
               "screening": _plain(_context["screening"]),
               "register_count": _plain(_context["register_count"]),
               "evidence_findings": _plain(_context["evidence_findings"]),
               "disagreements": _plain(_context["disagreements"]),
               "likelihood": _plain(_context["likelihood"]),
               "work_hours": _plain(_context["work_hours"])}
    request_details[_reference] = _detail
    _screening, _count = _detail["screening"], _detail["register_count"]
    request_queue.append({
        "queue_position": _position, "permit_reference": _reference,
        "revision_number": int(_profile[config.REVISION_NUMBER_COLUMN]),
        "permit_type": _detail["request"][config.PERMIT_TYPE_COLUMN],
        "linked_permit_type": _detail["request"][config.LINKED_PERMIT_TYPE_COLUMN],
        "plant_area": _detail["request"][config.PLANT_AREA_COLUMN],
        "request_channel": _detail["request"]["request_channel"],
        "document_count": len(_detail["request_documents"]),
        "requirements_met_count": _screening[config.REQUIREMENTS_MET_COUNT_COLUMN],
        "requirements_not_met_count": _screening[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN],
        "requirements_not_documented_count": _screening[
            config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN],
        "live_permit_count": _count[config.CONCURRENT_LIVE_PERMITS_COLUMN],
        "conflicting_permit_count": _count[config.CONFLICTING_LIVE_PERMITS_COLUMN],
        "simops_trigger_count": len(_screening["simops_triggers"]),
        "assessment": _screening["assessment"],
        "suspension_likelihood": _detail["likelihood"]["suspension_likelihood"],
        "likelihood_band": _detail["likelihood"]["likelihood_band"],
        "expected_work_hours": _detail["work_hours"]["expected_work_hours"],
        "extension_expected": _detail["work_hours"]["extension_expected"],
        "disagreement_count": len(_detail["disagreements"])})
request_locks = {reference: threading.Lock() for reference in request_profiles}

public_router = APIRouter()
guarded_router = APIRouter(dependencies=[Depends(authorize_request)])


def _request_statuses(session):
    statuses = {}
    for record in fetch_review_records(session, true()):
        statuses[(record["permit_reference"], record["revision_number"])] = record["status"]
    return statuses


def _status_of(statuses, permit_reference):
    revision = int(request_profiles[permit_reference][config.REVISION_NUMBER_COLUMN])
    return statuses.get((permit_reference, revision), config.REQUEST_STATUS_AWAITING_REVIEW)


def _review_by_id(session, review_id):
    records = fetch_review_records(session, PermitReview.review_id == review_id)
    if not records:
        raise HTTPException(status_code=404, detail=config.REVIEW_NOT_FOUND_MESSAGE)
    return records[0]


def _thread(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _checkpoint_values(review):
    snapshot = review_workflow.get_state(_thread(review["thread_id"]))
    if not snapshot.values:
        raise ValueError(config.CHECKPOINT_MISSING_MESSAGE.format(
            review_id=review["review_id"], checkpoint_file=config.CHECKPOINT_DATABASE_FILE))
    return snapshot


# ---------------------------------------------------------------- signing in and out


@public_router.post("/auth/signin", response_model=SignedInUserResponse)
def sign_in_user(sign_in_request: SignInRequest):
    """Exchange a username and password for a session token."""
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.username == sign_in_request.username)
        if account is None or not hmac.compare_digest(
                hash_password(sign_in_request.password, account["password_salt"]),
                account["password_hash"]):
            raise HTTPException(status_code=401, detail=config.SIGN_IN_FAILED_MESSAGE)
        token = secrets.token_urlsafe(config.SESSION_TOKEN_BYTES)
        update_session_token(session, account["username"], token)
        session.commit()
    return {"username": account["username"], "display_name": account["display_name"],
            "role": account["role"], "area_group": account["area_group"],
            "session_token": token, "pages": config.ROLE_PAGE_ACCESS[account["role"]],
            "permissions": config.ROLE_PERMISSIONS[account["role"]]}


@public_router.post("/auth/signout", response_model=SignedOutResponse)
def sign_out_user(session_token: SessionTokenHeader = None):
    """End a session, so the same token opens nothing afterwards."""
    if not session_token:
        raise HTTPException(status_code=401, detail=config.SIGN_IN_REQUIRED_MESSAGE)
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.session_token == session_token)
        if account is None:
            raise HTTPException(status_code=401, detail=config.SIGN_IN_REQUIRED_MESSAGE)
        update_session_token(session, account["username"], None)
        session.commit()
    return {"username": account["username"], "signed_out": True}


# ---------------------------------------------------------------- the queue


@guarded_router.get("/queue", response_model=list[QueueRow])
def list_request_queue(permit_type: Annotated[PermitTypeFilter, Query()]):
    """The ranked list of open requests, filtered by permit type or linked permit type."""
    with session_factory() as session:
        statuses = _request_statuses(session)
    rows = []
    for row in request_queue:
        if permit_type != config.QUEUE_PERMIT_TYPE_ALL and permit_type not in (
                row["permit_type"], row["linked_permit_type"]):
            continue
        rows.append(dict(row, request_status=_status_of(statuses, row["permit_reference"])))
    return rows


@guarded_router.get("/queue/{permit_reference}", response_model=RequestDetailResponse)
def get_request_detail(permit_reference: str):
    """One request with everything the review would work from."""
    if permit_reference not in request_details:
        raise HTTPException(status_code=404, detail=config.UNKNOWN_REQUEST_MESSAGE.format(
            permit_reference=permit_reference))
    with session_factory() as session:
        statuses = _request_statuses(session)
    return dict(request_details[permit_reference],
                request_status=_status_of(statuses, permit_reference))


# ---------------------------------------------------------------- the review workflow


@guarded_router.post("/reviews", response_model=ReviewRecordResponse, status_code=201)
def start_permit_review(review_request: ReviewRequest, signed_in_user: SignedInUser,
                        http_request: Request):
    """Run the five reviewing agents on one request and leave it waiting at the gate."""
    reference = review_request.permit_reference
    if reference not in request_profiles:
        raise HTTPException(status_code=404, detail=config.UNKNOWN_REQUEST_MESSAGE.format(
            permit_reference=reference))
    profile = request_profiles[reference]
    authorize_plant_area(signed_in_user, profile[config.PLANT_AREA_COLUMN], reference)
    revision = int(profile[config.REVISION_NUMBER_COLUMN])
    with request_locks[reference]:
        with session_factory() as session:
            reviewed = fetch_review_records(session, and_(
                PermitReview.permit_reference == reference,
                PermitReview.revision_number == revision))
        if reviewed:
            raise HTTPException(status_code=409, detail=config.REQUEST_ALREADY_REVIEWED_MESSAGE
                                .format(revision_number=revision, permit_reference=reference))
        thread_id = (reference + "-"
                     + secrets.token_hex(config.IDENTIFIER_HEX_LENGTH // 2).upper())
        http_request.state.opened_thread_id = thread_id
        review_workflow.invoke({"permit_reference": reference,
                                "coordinator_request": review_request.coordinator_request,
                                "coordinator_id": signed_in_user["username"],
                                "thread_id": thread_id, "llm_call_count": 0},
                               _thread(thread_id))
        review_id = review_workflow.get_state(_thread(thread_id)).values["review_id"]
        with session_factory() as session:
            return _review_by_id(session, review_id)


@guarded_router.get("/reviews", response_model=list[ReviewRecordResponse])
def list_permit_reviews(status_value: Annotated[list[ReviewStatusValue], Query()]):
    """Every review whose status is one of those asked for, in permit-reference order."""
    with session_factory() as session:
        return fetch_review_records(session, PermitReview.status.in_(list(status_value)))


@guarded_router.post("/reviews/{review_id}/decision", response_model=ReviewRecordResponse)
def submit_review_decision(review_id: str, decision_request: DecisionRequest,
                           signed_in_user: SignedInUser):
    """Check the decision, then resume the paused run with it."""
    with session_factory() as session:
        review = _review_by_id(session, review_id)
    reference = review["permit_reference"]
    if reference not in request_profiles:
        raise ValueError(config.REVIEW_OUTSIDE_QUEUE_MESSAGE.format(review_id=review_id,
                                                                    permit_reference=reference))
    authorize_plant_area(signed_in_user, review["plant_area"], reference)
    decision = {"decision": decision_request.decision,
                "decided_by": signed_in_user["username"],
                "decided_by_role": signed_in_user["role"],
                "decided_by_area_group": signed_in_user["area_group"],
                "decision_reason": decision_request.decision_reason}
    with request_locks[reference]:
        with session_factory() as session:
            review = _review_by_id(session, review_id)
        snapshot = _checkpoint_values(review)
        values = snapshot.values
        conflict = find_review_conflict(review["status"], values.get("review_decision"), decision,
                                        reference)
        if conflict:
            raise HTTPException(status_code=409, detail=conflict)
        refusal = find_decision_refusal(decision, values["request_profile"], values["screening"],
                                        values["coordinator_id"])
        if refusal:
            raise HTTPException(status_code=409, detail=refusal)
        if values.get("review_decision") is None:
            review_workflow.invoke(Command(resume=decision), _thread(review["thread_id"]))
        else:
            review_workflow.invoke(None, _thread(review["thread_id"]))
        with session_factory() as session:
            return _review_by_id(session, review_id)


@guarded_router.get("/reviews/{review_id}/calls", response_model=ReviewCallsResponse)
def get_review_calls(review_id: str):
    """The model calls recorded for the review's run, against the number the workflow allows."""
    with session_factory() as session:
        review = _review_by_id(session, review_id)
        calls = fetch_llm_calls(session, review["run_id"])
    if review["decision"] is not None:
        expected = config.LLM_CALLS_BY_DECISION[review["decision"]]
    else:
        pending = _checkpoint_values(review).values.get("review_decision")
        expected = (config.LLM_CALLS_BY_DECISION[pending["decision"]] if pending
                    else config.LLM_CALLS_BEFORE_DECISION)
    return {"review_id": review_id, "run_id": review["run_id"], "call_count": len(calls),
            "expected_call_count": expected,
            "total_tokens": sum(call["total_tokens"] for call in calls),
            "total_latency_milliseconds": sum(call["latency_milliseconds"] for call in calls),
            "calls": calls}


# ---------------------------------------------------------------- the chat and the audit trail


@guarded_router.post("/chat/ask", response_model=ChatResponse)
def ask_permit_assistant(chat_request: ChatRequest, signed_in_user: SignedInUser):
    """Answer one question from the documents or the records, or refuse, and record it."""
    question = chat_request.question
    with session_factory() as session:
        conversation = recall_conversation(session, signed_in_user["username"], question)
        answered = answer_from_documents(knowledge_collection, knowledge_documents, question,
                                         conversation)
        if answered["question_type"] in config.CHAT_REQUEST_TYPES:
            answered["answer"] = answer_request_record(session, signed_in_user, open_requests,
                                                       answered["question_type"],
                                                       answered["permit_reference"])
        elif answered["question_type"] == config.CHAT_TYPE_PLANT_AREA_FIGURE:
            answered["answer"] = answer_plant_area_figure(permit_history, answered["plant_area"],
                                                          answered["permit_type"])
        insert_chat_turn(session, {
            "username": signed_in_user["username"], "user_role": signed_in_user["role"],
            "question_text": question, "question_type": answered["question_type"],
            "permit_reference": answered["permit_reference"],
            "plant_area": answered["plant_area"], "permit_type": answered["permit_type"],
            "answer_text": answered["answer"],
            "source_documents": ", ".join(answered["source_documents"]),
            "closest_distance": answered["closest_distance"],
            "was_refused": answered["was_refused"]})
        session.commit()
    return answered


@guarded_router.get("/chat/turns", response_model=list[ChatTurnResponse])
def list_own_chat_turns(signed_in_user: SignedInUser,
                        turn_limit: Annotated[int, Query(ge=1, le=config.CHAT_AUDIT_MAX_ROWS)]):
    """The signed-in person's own turns, newest first."""
    with session_factory() as session:
        return fetch_chat_turns(session, ChatTurn.username == signed_in_user["username"],
                                turn_limit)


@guarded_router.get("/audit/chat-turns", response_model=list[ChatTurnResponse])
def list_recent_chat_turns(turn_limit: Annotated[int, Query(ge=1,
                                                            le=config.CHAT_AUDIT_MAX_ROWS)]):
    """Every recent question, whoever asked it, newest first."""
    with session_factory() as session:
        return fetch_chat_turns(session, true(), turn_limit)


def report_failed_request(http_request: Request, failure: Exception):
    """Turn every failure into a 502 the dashboard can show, closing any run it opened."""
    thread_id = getattr(http_request.state, "opened_thread_id", None)
    if thread_id:
        with session_factory() as session:
            run = session.query(AgentRun).filter(AgentRun.thread_id == thread_id).first()
            if run is not None:
                update_agent_run(session, run.run_id, config.AGENT_RUN_STATUS_FAILED,
                                 len(fetch_llm_calls(session, run.run_id)))
                session.commit()
    return JSONResponse(status_code=502, content={"detail": str(failure)})


service_application = FastAPI(title=config.API_TITLE)
service_application.include_router(public_router)
service_application.include_router(guarded_router)
for _failure in (ValueError, genai_errors.APIError, httpx.HTTPError):
    service_application.add_exception_handler(_failure, report_failed_request)
