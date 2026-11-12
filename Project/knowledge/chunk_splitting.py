"""Cuts the documents into sections, and the sections into overlapping passages."""
import re

from config import config


def _single_spaced(text):
    return re.sub(config.WHITESPACE_RUN_PATTERN, " ", text).strip()


def split_documents_into_sections(document_entries):
    """Divide each document at its numbered headings, leaving out its title and header."""
    sections = []
    for entry in document_entries:
        name = entry["document_name"]
        text = entry["document_text"]
        headers = re.findall(config.DOCUMENT_HEADER_PATTERN, text)
        if len(headers) != 1:
            raise ValueError(f"{name} carries {len(headers)} header lines where it carries one")
        header_end = re.search(config.DOCUMENT_HEADER_PATTERN, text).end()
        body = text[header_end:]
        headings = list(re.finditer(config.SECTION_HEADING_PATTERN, body))
        numbers = [int(match.group(1)) for match in headings]
        if not numbers or numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"{name} is not divided into sections numbered from 1 in order: "
                             f"{numbers[:12]}")
        for position, match in enumerate(headings):
            end = headings[position + 1].start() if position + 1 < len(headings) else len(body)
            sections.append({"document_name": name, "section_number": numbers[position],
                             "section_text": _single_spaced(body[match.start():end])})
    return sections


def _split_sentences(section_text):
    return [piece for piece in re.split(config.SENTENCE_BOUNDARY_PATTERN, section_text) if piece]


def _passages_of_section(sentences):
    """Pack whole sentences into passages, each opening with the end of the one before."""
    size, overlap_size = config.CHUNK_SIZE_CHARACTERS, config.CHUNK_OVERLAP_CHARACTERS
    passages = []
    position = 0
    previous = []
    while position < len(sentences):
        carried = []
        for sentence in reversed(previous):
            candidate = [sentence] + carried
            if len(" ".join(candidate)) > overlap_size:
                break
            carried = candidate
        while carried and len(" ".join(carried + [sentences[position]])) > size:
            carried = carried[1:]
        current = list(carried)
        new_count = 0
        while position < len(sentences):
            candidate = current + [sentences[position]]
            if new_count > 0 and len(" ".join(candidate)) > size:
                break
            current = candidate
            position += 1
            new_count += 1
        carried_characters = len(" ".join(carried)) + 1 if carried else 0
        passages.append((" ".join(current), carried_characters))
        previous = current
    return passages


def build_chunk_records(section_entries):
    """Cut each section into passages a search can return, with a stable identifier each."""
    records = []
    passage_numbers = {}
    for section in section_entries:
        name = section["document_name"]
        for chunk_text, carried_characters in _passages_of_section(
                _split_sentences(section["section_text"])):
            passage_number = passage_numbers.get(name, 0)
            passage_numbers[name] = passage_number + 1
            records.append({
                "chunk_id": f"{name}::passage-{passage_number:04d}",
                "chunk_text": chunk_text,
                "document_name": name,
                "section_number": int(section["section_number"]),
                "passage_number": passage_number,
                "carried_characters": carried_characters,
            })
    return records
