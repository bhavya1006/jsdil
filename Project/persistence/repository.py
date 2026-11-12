"""Every read and write against the six tables. A write joins the caller's transaction, and a
read returns plain records, never database objects."""
from sqlalchemy import select

from persistence.models import (AgentRun, ChatTurn, LlmCall, PermitReview, ReviewDecision,
                                UserAccount)

_DECISION_FIELDS = ["decision", "decided_by", "decided_by_role", "decided_by_area_group",
                    "decision_reason"]
_CALL_FIELDS = ["agent_name", "model_name", "prompt_tokens", "response_tokens", "total_tokens",
                "latency_milliseconds"]


def _as_record(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def insert_user_account(session, account_record):
    """Create one sign-in account."""
    session.add(UserAccount(**account_record))
    session.flush()


def fetch_user_account(session, condition):
    """One account with every column, or None."""
    account = session.execute(select(UserAccount).where(condition)).scalars().first()
    return None if account is None else _as_record(account)


def update_session_token(session, username, session_token):
    """Start or end one person's session."""
    account = session.get(UserAccount, username)
    if account is None:
        raise ValueError(f"No account has the username {username}")
    account.session_token = session_token
    session.flush()


def insert_agent_run(session, run_record):
    """Open the record of one workflow run."""
    session.add(AgentRun(**run_record))
    session.flush()


def update_agent_run(session, run_id, status_value, llm_call_count):
    """Move a run on, recording what it has cost."""
    run = session.get(AgentRun, run_id)
    if run is None:
        raise ValueError(f"No workflow run has the identifier {run_id}")
    run.status = status_value
    run.llm_call_count = int(llm_call_count)
    session.flush()


def insert_llm_call(session, run_id, call_record):
    """Record one language model call against its run."""
    session.add(LlmCall(run_id=run_id, **{field: call_record[field] for field in _CALL_FIELDS}))
    session.flush()


def fetch_llm_calls(session, run_id):
    """A run's calls in the order they were made."""
    calls = session.execute(select(LlmCall).where(LlmCall.run_id == run_id)
                            .order_by(LlmCall.call_id)).scalars().all()
    return [{field: getattr(call, field) for field in _CALL_FIELDS} for call in calls]


def insert_permit_review(session, review_record):
    """Store the review waiting at the gate."""
    session.add(PermitReview(**review_record))
    session.flush()


def fetch_review_records(session, condition):
    """Reviews with their run's thread, opener and calls, and their decision where taken."""
    rows = session.execute(
        select(PermitReview, AgentRun, ReviewDecision)
        .join(AgentRun, AgentRun.run_id == PermitReview.run_id, isouter=True)
        .join(ReviewDecision, ReviewDecision.review_id == PermitReview.review_id, isouter=True)
        .where(condition)
        .order_by(PermitReview.permit_reference, PermitReview.revision_number,
                  PermitReview.review_id)).all()
    records = []
    for review, run, decision in rows:
        record = _as_record(review)
        record["thread_id"] = run.thread_id if run else None
        record["coordinator_id"] = run.coordinator_id if run else None
        record["llm_call_count"] = run.llm_call_count if run else None
        for field in _DECISION_FIELDS:
            record[field] = getattr(decision, field) if decision else None
        records.append(record)
    return records


def insert_review_decision(session, decision_record):
    """Record an Area Authority's decision."""
    session.add(ReviewDecision(**decision_record))
    session.flush()


def record_review_outcome(session, review_id, outcome_record):
    """Write exactly the columns a decision changes onto its review."""
    review = session.get(PermitReview, review_id)
    if review is None:
        raise ValueError(f"No permit review has the identifier {review_id}")
    for column_name, value in outcome_record.items():
        setattr(review, column_name, value)
    session.flush()


def insert_chat_turn(session, turn_record):
    """Record one chat exchange, refusals included."""
    session.add(ChatTurn(**turn_record))
    session.flush()


def fetch_chat_turns(session, condition, turn_limit):
    """The most recent chat turns meeting the condition, newest first."""
    turns = session.execute(select(ChatTurn).where(condition).order_by(ChatTurn.turn_id.desc())
                            .limit(int(turn_limit))).scalars().all()
    return [_as_record(turn) for turn in turns]
