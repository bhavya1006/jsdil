"""Measures how dirty each raw extract was, and writes the evidence out.

Every figure is counted on the extract's first copies of each row as they arrived, unless said
otherwise.
"""
import json
import os
import re

import pandas as pd

from config import config
from data_pipeline.common_cleaning import (convert_markup_to_plain_text,
                                           convert_permit_references, convert_readings,
                                           convert_whole_numbers, to_decimal)


def _missing_cells(frame):
    counts = frame.isna().sum()
    return {str(column_name): int(count) for column_name, count in counts.items() if count > 0}


def profile_dataset(raw_frame, clean_frame, target_column):
    """Account for one dataset: what arrived, what survived, and what was missing."""
    first_copies = raw_frame.drop_duplicates(keep="first")
    missing_target = (int(first_copies[target_column].isna().sum())
                      if target_column in first_copies.columns else 0)
    return {
        "rows_received": int(len(raw_frame)),
        "rows_kept": int(len(clean_frame)),
        "rows_removed": int(len(raw_frame) - len(clean_frame)),
        "duplicate_rows": int(raw_frame.duplicated(keep="first").sum()),
        "rows_missing_target": missing_target,
        "columns": int(clean_frame.shape[1]),
        "missing_cells_as_received": _missing_cells(first_copies),
        "missing_cells_after_cleaning": _missing_cells(clean_frame),
    }


def _references_not_canonical(frame):
    references = frame[config.PERMIT_REFERENCE_COLUMN].dropna().astype(str)
    return int((~references.str.fullmatch(config.CANONICAL_PERMIT_REFERENCE_PATTERN)).sum())


def _readings_by_unit(frame, readings):
    counted = {}
    for column_name, reading in readings.items():
        if column_name not in frame.columns:
            continue
        by_unit = {unit_name: 0 for unit_name in reading["notations"]}
        for value in frame[column_name].dropna().astype(str):
            for unit_name, pattern in reading["notations"].items():
                if re.fullmatch(pattern, value):
                    by_unit[unit_name] += 1
                    break
        counted[column_name] = by_unit
    return counted


def _count_outside(read_frame, limits, outside):
    counted = {}
    for column_name, limit in limits.items():
        if column_name not in read_frame.columns:
            continue
        bound = to_decimal(limit)
        counted[column_name] = int(sum(1 for value in read_frame[column_name].dropna()
                                       if outside(to_decimal(value), bound)))
    return counted


def _categories_not_canonical(frame, value_mappings):
    counted = {}
    for column_name, mapping in value_mappings.items():
        if column_name not in frame.columns:
            continue
        canonical = set(mapping.values())
        counted[column_name] = int(sum(1 for value in frame[column_name].dropna()
                                       if value not in canonical))
    return counted


def _target_shares(clean_frame, target_column):
    shares = clean_frame[target_column].value_counts(normalize=True)
    return {str(answer): round(float(share), config.TARGET_SHARE_DECIMALS)
            for answer, share in sorted(shares.items(), key=lambda item: str(item[0]))}


def _read_numbers(frame, whole_number_patterns, readings):
    read = frame.copy()
    read[config.PERMIT_REFERENCE_COLUMN] = convert_permit_references(
        read[config.PERMIT_REFERENCE_COLUMN])
    read = convert_whole_numbers(read, whole_number_patterns)
    return convert_readings(read, readings)


def profile_history_details(raw_frame, clean_frame):
    """Account for the defects peculiar to the permit history extracts."""
    first_copies = raw_frame.drop_duplicates(keep="first").reset_index(drop=True)
    read = _read_numbers(first_copies, config.HISTORY_WHOLE_NUMBER_PATTERNS, config.FORM_READINGS)
    reference, revision = config.PERMIT_REFERENCE_COLUMN, config.REVISION_NUMBER_COLUMN
    highest = read.groupby(reference)[revision].transform("max")
    set_aside = (read[revision] < highest).fillna(False).astype(bool)
    answered = read[config.HISTORY_TARGET_COLUMN].notna()
    return {
        "answered_earlier_revisions_removed": int((set_aside & answered).sum()),
        "permit_references_not_written_canonically": _references_not_canonical(first_copies),
        "readings_by_unit_as_written": _readings_by_unit(first_copies, config.FORM_READINGS),
        "values_below_minimum_as_received": _count_outside(
            read, config.FORM_MINIMUM_VALUES, lambda value, bound: value < bound),
        "values_above_maximum_as_received": _count_outside(
            read, config.FORM_MAXIMUM_VALUES, lambda value, bound: value > bound),
        "category_values_not_written_canonically": _categories_not_canonical(
            first_copies, config.HISTORY_VALUE_MAPPINGS),
        "target_shares": _target_shares(clean_frame, config.HISTORY_TARGET_COLUMN),
    }


def profile_log_details(raw_frame, clean_frame):
    """Account for the defects peculiar to the job log extracts."""
    first_copies = raw_frame.drop_duplicates(keep="first").reset_index(drop=True)
    read = _read_numbers(first_copies, config.LOG_WHOLE_NUMBER_PATTERNS, config.LOG_READINGS)
    target_minimum = to_decimal(config.LOG_MINIMUM_VALUES[config.LOG_TARGET_COLUMN])
    impossible = sum(1 for value in read[config.LOG_TARGET_COLUMN].dropna()
                     if to_decimal(value) < target_minimum)
    return {
        "rows_with_impossible_target": int(impossible),
        "values_below_minimum_as_received": _count_outside(
            read, config.LOG_MINIMUM_VALUES, lambda value, bound: value < bound),
        "permit_references_not_written_canonically": _references_not_canonical(first_copies),
        "readings_by_unit_as_written": _readings_by_unit(first_copies, config.LOG_READINGS),
        "category_values_not_written_canonically": _categories_not_canonical(
            first_copies, config.LOG_VALUE_MAPPINGS),
        "median_actual_work_hours": float(clean_frame[config.LOG_TARGET_COLUMN].median()),
    }


def profile_sentence_details(raw_frame, clean_frame):
    """Account for the defects peculiar to free text."""
    first_copies = raw_frame.drop_duplicates(keep="first").reset_index(drop=True)
    texts = first_copies[config.SENTENCE_TEXT_COLUMN].fillna("").astype(str)
    plain = convert_markup_to_plain_text(first_copies[config.SENTENCE_TEXT_COLUMN]).fillna("")
    lengths = pd.to_numeric(first_copies[config.SENTENCE_LENGTH_COLUMN], errors="coerce")
    labelled = first_copies[config.SENTENCE_TARGET_COLUMN].notna()
    word_counts = plain.map(lambda text: len(str(text).split()))
    clean_words = clean_frame[config.SENTENCE_TEXT_COLUMN].astype(str).map(lambda t: len(t.split()))
    return {
        "sentences_with_markup": int(texts.str.contains(config.MARKUP_PATTERN, regex=True).sum()),
        "sentences_with_encoding_artifacts": int(
            texts.str.contains(config.ENCODING_ARTIFACT_PATTERN, regex=True).sum()),
        "sentences_in_capitals": int(plain.map(lambda text: str(text).isupper()).sum()),
        "sentence_lengths_disagreeing_with_plain_text_as_received": int(
            (lengths != plain.str.len()).sum()),
        "rows_with_too_few_words": int(
            (labelled & (word_counts < config.MINIMUM_SENTENCE_WORDS)).sum()),
        "shortest_sentence_words": int(clean_words.min()) if len(clean_words) else 0,
        "permit_references_not_written_canonically": _references_not_canonical(first_copies),
        "category_values_not_written_canonically": _categories_not_canonical(
            first_copies, config.SENTENCE_VALUE_MAPPINGS),
        "target_shares": _target_shares(clean_frame, config.SENTENCE_TARGET_COLUMN),
    }


def write_quality_report(profiles):
    """Write every dataset's profile as sorted, indented JSON."""
    folder = os.path.dirname(config.DATA_QUALITY_REPORT_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(config.DATA_QUALITY_REPORT_FILE, "w", encoding="utf-8") as report_file:
        json.dump(profiles, report_file, indent=config.JSON_INDENT, sort_keys=True)
