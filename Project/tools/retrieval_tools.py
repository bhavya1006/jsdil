"""The five section reads the workflow and the chat need, each returned without its heading."""
from config import config
from knowledge.vector_index import read_document_section


def _read_titled_section(collection, document_name, section_number, section_title):
    text = read_document_section(collection, document_name, section_number)
    heading = f"{section_number}. {section_title} "
    if not text.startswith(heading):
        raise ValueError(config.SECTION_TITLE_MISMATCH_MESSAGE.format(
            section_number=section_number, document_name=document_name,
            section_title=section_title))
    return text[len(heading):]


def _standard_document(permit_type):
    if permit_type not in config.PERMIT_STANDARD_DOCUMENTS:
        raise ValueError(config.NO_STANDARD_CONFIGURED_MESSAGE.format(permit_type=permit_type))
    return config.PERMIT_STANDARD_DOCUMENTS[permit_type]


def read_reviewer_checks(collection, sentence_type):
    """What the statement guide for a type of sentence says the reviewer checks."""
    if sentence_type not in config.SENTENCE_TYPE_GUIDE_DOCUMENTS:
        raise ValueError(config.NO_GUIDE_CONFIGURED_MESSAGE.format(sentence_type=sentence_type))
    return _read_titled_section(collection, config.SENTENCE_TYPE_GUIDE_DOCUMENTS[sentence_type],
                                config.REVIEWER_CHECKS_SECTION_NUMBER,
                                config.REVIEWER_CHECKS_SECTION_TITLE)


def read_documentation_to_provide(collection, permit_type):
    """What a permit type's own standard asks for when a request is returned."""
    return _read_titled_section(collection, _standard_document(permit_type),
                                config.DOCUMENTATION_SECTION_NUMBER,
                                config.DOCUMENTATION_SECTION_TITLE)


def read_permit_document_guidance(collection, permit_type):
    """What a permit type's own standard says the permit document holds."""
    return _read_titled_section(collection, _standard_document(permit_type),
                                config.PERMIT_DOCUMENT_SECTION_NUMBER,
                                config.PERMIT_DOCUMENT_SECTION_TITLE)


def read_briefing_guidance(collection, permit_type):
    """What a permit type's own standard says the work party briefing tells the work party."""
    return _read_titled_section(collection, _standard_document(permit_type),
                                config.WORK_PARTY_BRIEFING_SECTION_NUMBER,
                                config.WORK_PARTY_BRIEFING_SECTION_TITLE)


def read_policy_section(collection, section_number, section_title):
    """One numbered section of the Control of Work Policy, in the policy's own words."""
    return _read_titled_section(collection, config.CONTROL_OF_WORK_POLICY_DOCUMENT,
                                section_number, section_title)
