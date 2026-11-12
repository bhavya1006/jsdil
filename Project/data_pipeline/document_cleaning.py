"""Cleaning of document rows, whether past statement sentences or live request documents."""
from config import config
from data_pipeline.common_cleaning import (clean_statement_text, convert_permit_references,
                                           convert_whole_numbers, normalize_category_values,
                                           refuse_missing_values, refuse_repeated_references,
                                           refuse_unreadable_identifiers,
                                           remove_duplicate_records)


def clean_document_records(frame, value_mappings, whole_number_patterns,
                           identifier_patterns, id_column, text_column):
    """Clean the rows of documents: keys checked, answers canonical, text plain."""
    result = remove_duplicate_records(frame)
    result = refuse_missing_values(result, [id_column, config.PERMIT_REFERENCE_COLUMN])
    result = refuse_unreadable_identifiers(result, identifier_patterns)
    result = refuse_repeated_references(result, id_column)
    result = normalize_category_values(result, value_mappings)
    result[config.PERMIT_REFERENCE_COLUMN] = convert_permit_references(
        result[config.PERMIT_REFERENCE_COLUMN])
    result = convert_whole_numbers(result, whole_number_patterns)
    result[text_column] = clean_statement_text(result[text_column])
    return result.reset_index(drop=True)
