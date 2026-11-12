"""The table the work hours model learns from."""
from config import config
from data_pipeline.common_cleaning import (convert_permit_references, convert_readings,
                                           convert_whole_numbers, normalize_category_values,
                                           refuse_missing_values, refuse_repeated_references,
                                           refuse_unreadable_identifiers,
                                           remove_duplicate_records, remove_rows_missing_target,
                                           treat_values_below_minimum_as_missing)


def read_log_answers(frame):
    """Read every answer of a job log as the Permit Rules read it; no row is removed."""
    result = normalize_category_values(frame, config.LOG_VALUE_MAPPINGS)
    result[config.PERMIT_REFERENCE_COLUMN] = convert_permit_references(
        result[config.PERMIT_REFERENCE_COLUMN])
    result = refuse_unreadable_identifiers(result, config.FORM_IDENTIFIER_PATTERNS)
    result = convert_whole_numbers(result, config.LOG_WHOLE_NUMBER_PATTERNS)
    return convert_readings(result, config.LOG_READINGS)


def clean_job_log_dataset(frame):
    """One row per permit, with its actual work hours."""
    result = remove_duplicate_records(frame)
    result = refuse_missing_values(result, [config.PERMIT_REFERENCE_COLUMN])
    result = read_log_answers(result)
    result = refuse_repeated_references(result, config.PERMIT_REFERENCE_COLUMN)
    result = treat_values_below_minimum_as_missing(result, config.LOG_MINIMUM_VALUES)
    return remove_rows_missing_target(result, config.LOG_TARGET_COLUMN)
