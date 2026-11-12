"""The repairs a permit request's answers need, and the cleaning of the live permit register."""
from config import config
from data_pipeline.common_cleaning import (convert_permit_references, convert_readings,
                                           convert_whole_numbers,
                                           keep_latest_revision_of_each_request,
                                           normalize_category_values,
                                           refuse_missing_values, refuse_repeated_references,
                                           refuse_unreadable_identifiers,
                                           remove_duplicate_records,
                                           treat_values_above_maximum_as_missing,
                                           treat_values_below_minimum_as_missing)


def read_request_answers(frame, value_mappings, whole_number_patterns):
    """Read every answer on a request as the Permit Rules read it; no row is removed."""
    result = normalize_category_values(frame, value_mappings)
    result[config.PERMIT_REFERENCE_COLUMN] = convert_permit_references(
        result[config.PERMIT_REFERENCE_COLUMN])
    result = refuse_unreadable_identifiers(result, config.FORM_IDENTIFIER_PATTERNS)
    result = convert_whole_numbers(result, whole_number_patterns)
    return convert_readings(result, config.FORM_READINGS)


def clean_request_records(frame, value_mappings, whole_number_patterns):
    """Turn raw permit requests into one clean row per request, at its highest revision."""
    result = remove_duplicate_records(frame)
    result = read_request_answers(result, value_mappings, whole_number_patterns)
    result = treat_values_below_minimum_as_missing(result, config.FORM_MINIMUM_VALUES)
    result = treat_values_above_maximum_as_missing(result, config.FORM_MAXIMUM_VALUES)
    return keep_latest_revision_of_each_request(result)


def clean_register_records(frame):
    """Turn the raw live permit register into one clean row per permit."""
    result = remove_duplicate_records(frame)
    result = refuse_missing_values(result, config.REGISTER_REQUIRED_COLUMNS)
    result = normalize_category_values(result, config.REGISTER_VALUE_MAPPINGS)
    result[config.PERMIT_REFERENCE_COLUMN] = convert_permit_references(
        result[config.PERMIT_REFERENCE_COLUMN])
    result = refuse_unreadable_identifiers(result, config.FORM_IDENTIFIER_PATTERNS)
    result = refuse_repeated_references(result, config.PERMIT_REFERENCE_COLUMN)
    return result.reset_index(drop=True)
