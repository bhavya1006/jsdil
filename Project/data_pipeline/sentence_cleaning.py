"""The table the sentence model learns from."""
import pandas as pd

from config import config
from data_pipeline.document_cleaning import clean_document_records


def clean_sentence_dataset(frame):
    """Clean sentences, measure them again, and keep the labelled ones long enough to read."""
    result = clean_document_records(frame, config.SENTENCE_VALUE_MAPPINGS,
                                    config.SENTENCE_WHOLE_NUMBER_PATTERNS,
                                    config.SENTENCE_IDENTIFIER_PATTERNS,
                                    config.SENTENCE_ID_COLUMN, config.SENTENCE_TEXT_COLUMN)
    texts = result[config.SENTENCE_TEXT_COLUMN]
    result[config.SENTENCE_LENGTH_COLUMN] = pd.array(
        [pd.NA if pd.isna(text) else len(text) for text in texts],
        dtype=config.WHOLE_NUMBER_DTYPE)
    words = pd.Series([0 if pd.isna(text) else len(str(text).split()) for text in texts],
                      index=result.index)
    keep = result[config.SENTENCE_TARGET_COLUMN].notna() & (words >= config.MINIMUM_SENTENCE_WORDS)
    return result[keep].reset_index(drop=True)
