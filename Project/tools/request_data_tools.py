"""Reads the live queue, its documents, the live register and the permit history, through the same
cleaning as the extracts, and serves one request at a time."""
import math
from decimal import Decimal

import numpy as np
import pandas as pd

from config import config
from data_pipeline.dataset_loader import load_processed_dataset, load_raw_dataset
from data_pipeline.document_cleaning import clean_document_records
from data_pipeline.request_cleaning import clean_register_records, clean_request_records


def is_blank(value):
    """A blank answer arrives as None from fetch_request and as NaN from a table row."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def plain_value(value):
    """A cell as a plain Python value, a missing one as None."""
    if is_blank(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    return value


def load_open_requests():
    """The live queue, each open request once at its highest revision."""
    return clean_request_records(load_raw_dataset(config.OPEN_PERMIT_REQUESTS_FILE),
                                 config.FORM_VALUE_MAPPINGS, config.LIVE_WHOLE_NUMBER_PATTERNS)


def load_request_documents():
    """The documents attached to the open requests."""
    return clean_document_records(load_raw_dataset(config.REQUEST_DOCUMENTS_FILE),
                                  config.DOCUMENT_VALUE_MAPPINGS,
                                  config.DOCUMENT_WHOLE_NUMBER_PATTERNS,
                                  config.DOCUMENT_IDENTIFIER_PATTERNS,
                                  config.DOCUMENT_ID_COLUMN, config.DOCUMENT_TEXT_COLUMN)


def load_live_register():
    """The permits issued on site and not yet handed back."""
    return clean_register_records(load_raw_dataset(config.LIVE_PERMIT_REGISTER_FILE))


def load_permit_history():
    """The permits already issued, training history then evaluation history; never the queue."""
    return pd.concat([load_processed_dataset(config.CLEAN_PERMIT_HISTORY_TRAIN_FILE),
                      load_processed_dataset(config.CLEAN_PERMIT_HISTORY_EVAL_FILE)],
                     ignore_index=True)


def fetch_request(requests, permit_reference):
    """One request's own answers, column by column, a missing value as None."""
    matching = requests[requests[config.PERMIT_REFERENCE_COLUMN] == permit_reference]
    if len(matching) == 0:
        raise ValueError(config.UNKNOWN_REQUEST_MESSAGE.format(permit_reference=permit_reference))
    row = matching.iloc[0]
    return {str(column_name): plain_value(row[column_name]) for column_name in requests.columns}


def fetch_request_documents(documents, request):
    """Every document attached to one request, in document identifier order."""
    reference = request[config.PERMIT_REFERENCE_COLUMN]
    own = documents[documents[config.PERMIT_REFERENCE_COLUMN] == reference]
    own = own.sort_values(config.DOCUMENT_ID_COLUMN, kind="stable")
    return [{str(column_name): plain_value(row[column_name]) for column_name in documents.columns}
            for _, row in own.iterrows()]


def rank_request_queue(requests):
    """Confined space entries (as permit or linked type) first, each group by reference."""
    first_type = config.QUEUE_FIRST_PERMIT_TYPE
    references = list(dict.fromkeys(requests[config.PERMIT_REFERENCE_COLUMN]))
    firsts = set(requests.loc[(requests[config.PERMIT_TYPE_COLUMN] == first_type)
                              | (requests[config.LINKED_PERMIT_TYPE_COLUMN] == first_type),
                              config.PERMIT_REFERENCE_COLUMN])
    return ([reference for reference in sorted(references) if reference in firsts]
            + [reference for reference in sorted(references) if reference not in firsts])


def describe_record(record):
    """Every value as the text a page and a prompt show."""
    described = {}
    for key, value in record.items():
        value = plain_value(value)
        if value is None:
            described[key] = config.NOT_GIVEN_TEXT
        elif isinstance(value, bool):
            described[key] = config.BOOLEAN_YES if value else config.BOOLEAN_NO
        elif isinstance(value, int):
            described[key] = f"{value:,}"
        elif isinstance(value, float) and math.isfinite(value) and value == int(value):
            described[key] = f"{int(value):,}"
        else:
            described[key] = str(value)
    return described
