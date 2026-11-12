"""One question, one language model call, answered from Hexley's documents and nothing else."""
import re

from agentic.agents import _ask, _undecorated
from config import config
from knowledge.standard_reading import read_permit_standard
from knowledge.vector_index import search_documents
from tools.retrieval_tools import read_documentation_to_provide, read_reviewer_checks

_LABELS = ["QUESTION TYPE", "REQUEST", "ANSWER"]


def _is_no_value(text):
    return _undecorated(str(text)).strip().lower() == config.CHAT_NO_VALUE_REPLY.lower()


def _canonical_reference(text):
    match = re.fullmatch(config.PERMIT_REFERENCE_PATTERN, text.strip().upper())
    if match:
        return config.PERMIT_REFERENCE_PREFIX + match.group(1).zfill(config.PERMIT_REFERENCE_DIGITS)
    return text.strip()


def _named_in(reference, texts):
    lowered = [text.lower() for text in texts]
    if any(reference.lower() in text for text in lowered):
        return True
    if re.fullmatch(config.CANONICAL_PERMIT_REFERENCE_PATTERN, reference):
        number = str(int(reference[len(config.PERMIT_REFERENCE_PREFIX):]))
        pattern = config.NUMBER_IN_TEXT_PATTERN.format(number=number)
        return any(re.search(pattern, text) for text in lowered)
    return False


def _one_named(text, names_and_patterns):
    found = [name for name, pattern in names_and_patterns if re.search(pattern, text)]
    return found


def _plant_area_and_type(question_text, previous_question):
    areas = [(area, config.NAME_IN_TEXT_PATTERN.format(name=re.escape(area.lower())))
             for group in config.AREA_GROUPS.values() for area in group]
    types = list(config.CHAT_PERMIT_TYPE_PATTERNS.items())
    found_areas = _one_named(question_text.lower(), areas) or _one_named(
        previous_question.lower(), areas)
    found_types = _one_named(question_text.lower(), types) or _one_named(
        previous_question.lower(), types)
    if len(found_areas) == 1 and len(found_types) == 1:
        return found_areas[0], found_types[0]
    return None, None


def _standing_knowledge(collection, document_entries):
    lines = ["(Reference material. Cite only the file name in brackets.)"]
    for permit_type, document_name in config.PERMIT_STANDARD_DOCUMENTS.items():
        standard = read_permit_standard(permit_type, document_entries)
        lines.append(f"{permit_type} ({document_name}):")
        lines += [f"- {requirement['requirement_name']}: {requirement['requirement_sentence']}"
                  for requirement in standard["requirements"]]
        lines.append(f"- documentation to provide when returned: "
                     f"{read_documentation_to_provide(collection, permit_type)}")
    lines.append("")
    lines.append("(Statement guides: what the reviewer checks.)")
    for sentence_type, document_name in config.SENTENCE_TYPE_GUIDE_DOCUMENTS.items():
        lines.append(f"- {sentence_type} ({document_name}): "
                     f"{read_reviewer_checks(collection, sentence_type)}")
    return "\n".join(lines)


def answer_from_documents(collection, document_entries, question_text, conversation):
    """Classify the question and answer it from the documents, refusing when they do not."""
    found = search_documents(collection, [question_text, conversation["search_text"]])
    passages, seen = [], set()
    for passage in sorted([passage for results in found for passage in results],
                          key=lambda passage: passage["distance"]):
        if passage["chunk_id"] not in seen:
            seen.add(passage["chunk_id"])
            passages.append(passage)
    closest = passages[0]["distance"] if passages else 2.0
    passage_text = "\n".join(f"[{passage['document_name']}] {passage['chunk_text']}"
                             for passage in passages) or "none"
    kinds = "\n".join(f"- {kind}" for kind in config.CHAT_TYPE_OPTIONS)
    prompt = "\n\n".join([
        f"You are the permit assistant of {config.ORGANISATION_NAME} at {config.SITE_NAME}. "
        "Answer only from the passages and the standing knowledge below, and from nothing else. "
        "Each requirement belongs to its own permit type's standard; never lend one standard's "
        "figure to another permit type.",
        "Classify the question as exactly one of these kinds:\n" + kinds + "\n"
        f"{config.CHAT_TYPE_PERMIT_STANDARD}: what a permit standard asks for. "
        f"{config.CHAT_TYPE_CONTROL_OF_WORK}: the Control of Work Policy or the Permit Rules. "
        f"{config.CHAT_TYPE_STATEMENT_GUIDANCE}: what a statement guide says the reviewer checks. "
        f"{config.CHAT_TYPE_REQUEST_STATUS}: the status or review of one named permit request. "
        f"{config.CHAT_TYPE_OUTCOME_PREDICTION}: whether one request will be suspended or how "
        "long its work will take. "
        f"{config.CHAT_TYPE_PLANT_AREA_FIGURE}: how many permits of a type were worked in a "
        f"plant area. {config.CHAT_TYPE_OUT_OF_SCOPE}: anything else.",
        "You know nothing of any single permit request or plant area: questions about them are "
        "answered from the records, so for those kinds write "
        f"{config.CHAT_NO_VALUE_REPLY} as the ANSWER.",
        "CONVERSATION SO FAR:\n" + conversation["history_text"],
        "PASSAGES FOUND, NEAREST FIRST:\n" + passage_text,
        _standing_knowledge(collection, document_entries),
        "QUESTION: " + question_text,
        "Reply with exactly these labelled fields, each label in capitals at the start of its "
        "own line followed by a colon:\n"
        "QUESTION TYPE: one kind from the list, on one line\n"
        "REQUEST: the permit reference the question is about, on one line, or "
        f"{config.CHAT_NO_VALUE_REPLY}\n"
        "ANSWER: the answer in plain prose, citing each document by its file name (for example "
        f"{config.PERMIT_STANDARD_DOCUMENTS[config.HOT_WORK]}) and never by a heading of these "
        f"instructions, or {config.CHAT_NO_VALUE_REPLY} when the documents do not answer it",
    ])
    references_in_question = re.findall(config.PERMIT_REFERENCE_IN_TEXT_PATTERN, question_text)
    agent = config.AGENT_NAME_PERMIT_ASSISTANT
    read = {}

    def accept(fields):
        for label in ["QUESTION TYPE", "REQUEST"]:
            if "\n" in fields[label]:
                raise ValueError(f"{agent} wrote its {label} over more than one line")
        question_type = _undecorated(fields["QUESTION TYPE"]).upper()
        if question_type not in config.CHAT_TYPE_OPTIONS:
            raise ValueError(f"{agent} classified the question as {fields['QUESTION TYPE']!r}")
        if not _undecorated(fields["REQUEST"]):
            raise ValueError(f"{agent} left its REQUEST empty")
        if question_type in config.CHAT_ANSWERED_TYPES and references_in_question:
            question_type = config.CHAT_TYPE_REQUEST_STATUS
        if question_type in config.CHAT_ANSWERED_TYPES and _is_no_value(fields["ANSWER"]):
            raise ValueError(f"{agent} gave no answer to a question of the type {question_type}")
        reference = None
        if question_type in config.CHAT_REQUEST_TYPES:
            written = (references_in_question[0] if references_in_question
                       else _undecorated(fields["REQUEST"]))
            if written and not _is_no_value(written):
                reference = _canonical_reference(written)
                if not _named_in(reference, [question_text, conversation["history_text"]]):
                    raise ValueError(f"{agent} named the request {reference}, which the "
                                     "conversation never names")
        read.update(question_type=question_type, permit_reference=reference,
                    answer=fields["ANSWER"])

    _ask(prompt, agent, _LABELS, accept)
    question_type = read["question_type"]
    plant_area = permit_type = None
    if question_type == config.CHAT_TYPE_PLANT_AREA_FIGURE:
        plant_area, permit_type = _plant_area_and_type(question_text,
                                                       conversation["previous_question"])
    answered = (question_type in config.CHAT_ANSWERED_TYPES
                and closest <= config.CHAT_MAX_RELEVANCE_DISTANCE)
    refused = not answered and question_type not in config.CHAT_RECORD_TYPES
    sources = []
    if answered:
        if question_type == config.CHAT_TYPE_PERMIT_STANDARD:
            sources += list(config.PERMIT_STANDARD_DOCUMENTS.values())
        if question_type == config.CHAT_TYPE_STATEMENT_GUIDANCE:
            sources += list(config.SENTENCE_TYPE_GUIDE_DOCUMENTS.values())
        sources += [passage["document_name"] for passage in passages]
    return {"question_type": config.CHAT_TYPE_OUT_OF_SCOPE if refused else question_type,
            "permit_reference": read["permit_reference"],
            "plant_area": plant_area, "permit_type": permit_type,
            "answer": config.CHAT_REFUSAL_MESSAGE if refused else read["answer"],
            "source_documents": list(dict.fromkeys(sources)),
            "closest_distance": float(closest),
            "was_refused": refused}
