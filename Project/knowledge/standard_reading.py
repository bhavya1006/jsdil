"""Reads a permit type's requirements out of that permit type's own permit standard.

This module keeps the requirements out of the code: every figure comes from the document.
"""
import re
from decimal import Decimal

from config import config
from knowledge.chunk_splitting import split_documents_into_sections

_FIGURE_RULES = [config.MINIMUM_RULE, config.MAXIMUM_RULE, config.THRESHOLD_RULE]


def _read_requirement(text, document_name):
    fitting = [(kind, re.fullmatch(pattern, text))
               for kind, pattern in config.REQUIREMENT_SENTENCE_PATTERNS.items()]
    fitting = [(kind, match) for kind, match in fitting if match]
    if len(fitting) != 1:
        raise ValueError(f"{document_name} states a requirement that "
                         f"{'no' if not fitting else 'more than one'} form fits: {text!r}")
    kind, match = fitting[0]
    parts = match.groupdict()
    rules = {rule: Decimal(parts[rule]) for rule in _FIGURE_RULES if parts.get(rule) is not None}
    if config.MINIMUM_RULE in rules and config.MAXIMUM_RULE in rules:
        low, high = sorted([rules[config.MINIMUM_RULE], rules[config.MAXIMUM_RULE]])
        rules[config.MINIMUM_RULE], rules[config.MAXIMUM_RULE] = low, high
    return {"requirement_kind": kind,
            "requirement_name": parts[config.REQUIREMENT_NAME_PART],
            "requirement_sentence": parts[config.REQUIREMENT_SENTENCE_PART],
            "rules": rules}


def read_permit_standard(permit_type, document_entries):
    """Establish every requirement of one permit type, with its own figures, from its own standard."""
    if permit_type not in config.PERMIT_STANDARD_DOCUMENTS:
        raise ValueError(config.NO_STANDARD_CONFIGURED_MESSAGE.format(permit_type=permit_type))
    document_name = config.PERMIT_STANDARD_DOCUMENTS[permit_type]
    held = [entry for entry in document_entries if entry.get("document_name") == document_name]
    if len(held) != 1:
        raise ValueError(f"The documents hold {document_name} {len(held)} times where they hold "
                         "it once")
    sections = {section["section_number"]: section["section_text"]
                for section in split_documents_into_sections(held)}
    for number in [config.PERMIT_TYPE_SECTION_NUMBER, config.REQUIREMENTS_SECTION_NUMBER]:
        if number not in sections:
            raise ValueError(f"{document_name} has no section {number}")
    stated = set(re.findall(config.PERMIT_TYPE_PATTERN, sections[config.PERMIT_TYPE_SECTION_NUMBER]))
    if stated != {permit_type}:
        raise ValueError(f"{document_name} states the permit types {sorted(stated)} where it is "
                         f"the standard of {permit_type}")
    match = re.fullmatch(config.REQUIREMENTS_SECTION_PATTERN,
                         sections[config.REQUIREMENTS_SECTION_NUMBER])
    if not match:
        raise ValueError(f"Section {config.REQUIREMENTS_SECTION_NUMBER} of {document_name} is not "
                         "laid out as the Permit Rules set out")
    pieces = re.split(config.REQUIREMENT_NUMBER_PATTERN, match.group(1))
    numbers = [int(number) for number in pieces[1::2]]
    if pieces[0].strip() or numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"{document_name} numbers its requirements {numbers} where they run "
                         "from 1 in order with nothing before the first")
    requirements = [_read_requirement(text.strip(), document_name) for text in pieces[2::2]]
    kinds = [requirement["requirement_kind"] for requirement in requirements]
    repeated = sorted({kind for kind in kinds if kinds.count(kind) > 1})
    if repeated:
        raise ValueError(f"{document_name} states the requirement kinds {repeated} twice")
    return {"permit_type": permit_type, "document_name": document_name,
            "requirements": requirements}
