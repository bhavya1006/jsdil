"""Reads every sentence of a request's documents, links each to the requirements it bears on, and
flags gas testing sentences that disagree with the request."""
import re
from decimal import Decimal

from config import config
from data_pipeline.common_cleaning import to_decimal
from tools.request_data_tools import is_blank


def read_document_evidence(sentence_model, documents):
    """One finding per sentence, documents in the order given, sentences in the order written."""
    pieces = []
    for document in documents:
        text = document.get(config.DOCUMENT_TEXT_COLUMN)
        if is_blank(text) or not str(text).strip():
            raise ValueError(config.NO_STATEMENT_TEXT_MESSAGE.format(
                document_id=document.get(config.DOCUMENT_ID_COLUMN)))
        for sentence in re.split(config.STATEMENT_SENTENCE_PATTERN, str(text)):
            if sentence:
                pieces.append((document, sentence))
    if not pieces:
        return []
    sentence_types = sentence_model.predict([sentence for _document, sentence in pieces])
    return [{"document_id": document.get(config.DOCUMENT_ID_COLUMN),
             "document_type": document.get(config.DOCUMENT_TYPE_COLUMN),
             "sentence": sentence, "sentence_type": str(sentence_type)}
            for (document, sentence), sentence_type in zip(pieces, sentence_types)]


def link_evidence_to_requirements(findings, screening):
    """Each finding with the requirements of its request's own standards it bears on."""
    linked = []
    for finding in findings:
        sentence_type = finding.get("sentence_type")
        if sentence_type not in config.SENTENCE_TYPE_REQUIREMENT_KINDS:
            raise ValueError(config.NO_REQUIREMENT_KINDS_MESSAGE.format(sentence_type=sentence_type))
        kinds = config.SENTENCE_TYPE_REQUIREMENT_KINDS[sentence_type]
        borne = []
        for requirement in screening.get("requirements", []):
            entry = {"permit_type": requirement.get("permit_type"),
                     "requirement_name": requirement.get("requirement_name")}
            if requirement.get("requirement_kind") in kinds and entry not in borne:
                borne.append(entry)
        linked.append(dict(finding, requirements_borne_on=borne))
    return linked


def find_disagreements(findings, request):
    """Flag every gas testing sentence whose reading differs from the request's, or describes a
    reading the request left blank. Readings are compared as written, as exact decimals."""
    flags = []
    for finding in findings:
        if finding.get("sentence_type") != config.GAS_TESTING:
            continue
        sentence = str(finding.get("sentence", ""))
        stated = {}
        for column_name, pattern in config.STATEMENT_GAS_READING_PATTERNS.items():
            readings = [Decimal(value) for value in re.findall(pattern, sentence)]
            if readings:
                stated[column_name] = readings
        if not stated:
            continue
        answers = {column_name: request.get(column_name) for column_name in stated}
        if any(is_blank(answer) for answer in answers.values()):
            flag = config.DESCRIBES_BLANK_ANSWER
        elif any(reading != to_decimal(answers[column_name])
                 for column_name, readings in stated.items() for reading in readings):
            flag = config.DISAGREES_WITH_FORM
        else:
            continue
        flags.append({
            "document_id": finding.get("document_id"),
            "sentence": sentence,
            "stated_readings": {column_name: [float(reading) for reading in readings]
                                for column_name, readings in stated.items()},
            "request_readings": {column_name: (None if is_blank(answer) else float(answer))
                                 for column_name, answer in answers.items()},
            "flag": flag,
        })
    return flags
