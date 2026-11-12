"""The order the nodes run in, the gate, the branches and the checkpoint."""
import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agentic.nodes import (analyse_method_statements_node, assess_permit_requirements_node,
                           await_area_authority_decision_node, draft_permit_document_node,
                           draft_work_party_briefing_node, interpret_model_predictions_node,
                           load_permit_context_node, plan_permit_review_node,
                           prepare_permit_recommendation_node, record_permit_outcome_node,
                           route_permit_decision)
from agentic.state import PermitReviewState
from config import config


def build_workflow(checkpoint_path):
    """Assemble the ten nodes into a workflow that can be run, paused and resumed."""
    graph = StateGraph(PermitReviewState)
    in_order = [("load_permit_context", load_permit_context_node),
                ("plan_permit_review", plan_permit_review_node),
                ("assess_permit_requirements", assess_permit_requirements_node),
                ("analyse_method_statements", analyse_method_statements_node),
                ("interpret_model_predictions", interpret_model_predictions_node),
                ("prepare_permit_recommendation", prepare_permit_recommendation_node),
                ("await_area_authority_decision", await_area_authority_decision_node)]
    for name, node in in_order:
        graph.add_node(name, node)
    graph.add_node("draft_permit_document", draft_permit_document_node)
    graph.add_node("draft_work_party_briefing", draft_work_party_briefing_node)
    graph.add_node("record_permit_outcome", record_permit_outcome_node)
    graph.add_edge(START, in_order[0][0])
    for (source, _node), (target, _next) in zip(in_order, in_order[1:]):
        graph.add_edge(source, target)
    graph.add_conditional_edges("await_area_authority_decision", route_permit_decision, {
        config.DECISION_ISSUE: "draft_permit_document",
        config.DECISION_RETURN: "draft_permit_document",
        config.DECISION_REFER: "record_permit_outcome"})
    graph.add_conditional_edges("draft_permit_document", route_permit_decision, {
        config.DECISION_ISSUE: "draft_work_party_briefing",
        config.DECISION_RETURN: "record_permit_outcome"})
    graph.add_edge("draft_work_party_briefing", "record_permit_outcome")
    graph.add_edge("record_permit_outcome", END)
    folder = os.path.dirname(checkpoint_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    return graph.compile(checkpointer=SqliteSaver(connection))
