"""The six tables the system writes to. Only a column marked nullable accepts nothing."""
from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import config


class Base(DeclarativeBase):
    """The declarative base every table inherits from."""


class UserAccount(Base):
    """A person who signs in."""

    __tablename__ = config.USER_ACCOUNTS_TABLE

    username: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    area_group: Mapped[str | None] = mapped_column(String, nullable=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    password_salt: Mapped[str] = mapped_column(String, nullable=False)
    session_token: Mapped[str | None] = mapped_column(String, nullable=True, unique=True)


class AgentRun(Base):
    """One run of the review workflow."""

    __tablename__ = config.AGENT_RUNS_TABLE

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    permit_reference: Mapped[str] = mapped_column(String, nullable=False)
    coordinator_id: Mapped[str] = mapped_column(String, nullable=False)
    coordinator_request: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    llm_call_count: Mapped[int] = mapped_column(Integer, nullable=False)


class PermitReview(Base):
    """The review an Area Authority decides from."""

    __tablename__ = config.PERMIT_REVIEWS_TABLE

    review_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey(f"{config.AGENT_RUNS_TABLE}.run_id"),
                                        nullable=False, unique=True)
    permit_reference: Mapped[str] = mapped_column(String, nullable=False)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    permit_type: Mapped[str] = mapped_column(String, nullable=False)
    linked_permit_type: Mapped[str] = mapped_column(String, nullable=False)
    plant_area: Mapped[str] = mapped_column(String, nullable=False)
    coordinator_request: Mapped[str] = mapped_column(Text, nullable=False)
    requirements: Mapped[list] = mapped_column(JSON, nullable=False)
    requirements_met_count: Mapped[int] = mapped_column(Integer, nullable=False)
    requirements_not_met_count: Mapped[int] = mapped_column(Integer, nullable=False)
    requirements_not_documented_count: Mapped[int] = mapped_column(Integer, nullable=False)
    live_permits: Mapped[list] = mapped_column(JSON, nullable=False)
    conflicting_permits: Mapped[list] = mapped_column(JSON, nullable=False)
    simops_triggers: Mapped[list] = mapped_column(JSON, nullable=False)
    assessment: Mapped[str] = mapped_column(String, nullable=False)
    decisions_open: Mapped[list] = mapped_column(JSON, nullable=False)
    evidence: Mapped[list] = mapped_column(JSON, nullable=False)
    disagreements: Mapped[list] = mapped_column(JSON, nullable=False)
    suspension_likelihood: Mapped[float] = mapped_column(Float, nullable=False)
    likelihood_band: Mapped[str] = mapped_column(String, nullable=False)
    expected_work_hours: Mapped[float] = mapped_column(Float, nullable=False)
    planned_work_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    extension_expected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    work_hours_text: Mapped[str] = mapped_column(Text, nullable=False)
    review_plan_text: Mapped[str] = mapped_column(Text, nullable=False)
    requirements_analysis_text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_review_text: Mapped[str] = mapped_column(Text, nullable=False)
    prediction_analysis_text: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation_text: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_decision: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    permit_document_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    work_party_briefing_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    simops_brief: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ReviewDecision(Base):
    """An Area Authority's decision, one per review."""

    __tablename__ = config.REVIEW_DECISIONS_TABLE

    decision_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_id: Mapped[str] = mapped_column(
        String, ForeignKey(f"{config.PERMIT_REVIEWS_TABLE}.review_id"), nullable=False,
        unique=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey(f"{config.AGENT_RUNS_TABLE}.run_id"),
                                        nullable=False)
    decision: Mapped[str] = mapped_column(String, nullable=False)
    decided_by: Mapped[str] = mapped_column(String, nullable=False)
    decided_by_role: Mapped[str] = mapped_column(String, nullable=False)
    decided_by_area_group: Mapped[str] = mapped_column(String, nullable=False)
    decision_reason: Mapped[str] = mapped_column(Text, nullable=False)


class LlmCall(Base):
    """One language model call whose reply a review accepted."""

    __tablename__ = config.LLM_CALLS_TABLE

    call_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey(f"{config.AGENT_RUNS_TABLE}.run_id"),
                                        nullable=False)
    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    response_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_milliseconds: Mapped[int] = mapped_column(Integer, nullable=False)


class ChatTurn(Base):
    """One question the chat replied to, refusals included."""

    __tablename__ = config.CHAT_TURNS_TABLE

    turn_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, nullable=False)
    user_role: Mapped[str] = mapped_column(String, nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String, nullable=False)
    permit_reference: Mapped[str | None] = mapped_column(String, nullable=True)
    plant_area: Mapped[str | None] = mapped_column(String, nullable=True)
    permit_type: Mapped[str | None] = mapped_column(String, nullable=True)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_documents: Mapped[str] = mapped_column(Text, nullable=False)
    closest_distance: Mapped[float] = mapped_column(Float, nullable=False)
    was_refused: Mapped[bool] = mapped_column(Boolean, nullable=False)
