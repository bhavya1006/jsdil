"""The table the suspension model learns from."""
from config import config
from data_pipeline.common_cleaning import remove_rows_missing_target
from data_pipeline.request_cleaning import clean_request_records


def clean_history_dataset(frame):
    """One row per request at its highest revision, kept only when that revision is answered."""
    cleaned = clean_request_records(frame, config.HISTORY_VALUE_MAPPINGS,
                                    config.HISTORY_WHOLE_NUMBER_PATTERNS)
    return remove_rows_missing_target(cleaned, config.HISTORY_TARGET_COLUMN)
