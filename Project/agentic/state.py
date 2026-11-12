"""What the workflow carries from one node to the next."""
import operator
from typing import Annotated, TypedDict


class PermitReviewState(TypedDict):
    """Every key the review workflow reads or writes. llm_call_count is added to by each node
    that returns it; every other key is replaced by whichever node writes it."""

    permit_reference: str
    coordinator_request: str
    coordinator_id: str
    thread_id: str
    run_id: str
    review_id: str
    request_profile: dict
    request_documents: list
    register_count: dict
    screening: dict
    evidence_findings: list
    disagreements: list
    likelihood: dict
    work_hours: dict
    review_plan: dict
    requirements_analysis: dict
    evidence_review: dict
    prediction_analysis: dict
    review_decision: dict
    permit_document: dict
    document_details: dict
    work_party_briefing: dict
    llm_call_count: Annotated[int, operator.add]
