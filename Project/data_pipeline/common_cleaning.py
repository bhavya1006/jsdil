"""The repairs shared by the cleaning runs, the live files and the detail profiles.

Every function returns a repaired copy and leaves what it was given untouched. A value in a
notation a function does not recognise is refused with a ValueError naming the column and values.
"""
import html
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from config import config


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _refuse(column_name, values, reason):
    shown = ", ".join(repr(str(value)) for value in list(dict.fromkeys(values))[:10])
    raise ValueError(f"{column_name}: {reason}: {shown}")


def to_decimal(value):
    """An answer held as a number taken as the decimal it is written as."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (bool, np.bool_)):
        return Decimal(int(value))
    if isinstance(value, (int, np.integer)):
        return Decimal(int(value))
    if isinstance(value, (float, np.floating)):
        return Decimal(repr(float(value)))
    return Decimal(str(value))


def remove_duplicate_records(dataframe):
    """Rows identical in every column appear once, the first kept, numbered from nought."""
    return dataframe.drop_duplicates(keep="first").reset_index(drop=True)


def refuse_missing_values(dataframe, required_columns):
    """Refuse a table in which a required key is blank."""
    for column_name in required_columns:
        if column_name not in dataframe.columns:
            raise ValueError(f"{column_name}: the column is missing from the table")
        blank = int(dataframe[column_name].isna().sum())
        if blank:
            raise ValueError(f"{column_name}: {blank} rows leave it blank")
    return dataframe


def refuse_repeated_references(dataframe, reference_column):
    """Refuse a table in which two rows claim the same identity."""
    repeated = dataframe.loc[dataframe[reference_column].duplicated(keep=False), reference_column]
    if len(repeated):
        values = sorted({str(value) for value in repeated})
        raise ValueError(f"{reference_column}: repeated on more than one row: "
                         + ", ".join(values))
    return dataframe


def remove_rows_missing_target(dataframe, target_column):
    """Keep every row holding an answer, numbered from nought."""
    return dataframe[dataframe[target_column].notna()].reset_index(drop=True)


def normalize_category_values(dataframe, value_mappings):
    """Fold every spelling of a value onto its canonical value."""
    result = dataframe.copy()
    for column_name, mapping in value_mappings.items():
        if column_name not in result.columns:
            continue
        written = result[column_name]
        unknown = [value for value in written
                   if not _is_missing(value) and str(value).strip().lower() not in mapping]
        if unknown:
            _refuse(column_name, unknown, "values with no canonical spelling")
        result[column_name] = pd.Series(
            [np.nan if _is_missing(value) else mapping[str(value).strip().lower()]
             for value in written], index=result.index, dtype=object)
    return result


def convert_permit_references(references):
    """Write every permit reference as PTW and seven digits."""
    pattern = re.compile(config.PERMIT_REFERENCE_PATTERN)
    converted, unreadable = [], []
    for value in references:
        if _is_missing(value):
            converted.append(np.nan)
            continue
        match = pattern.fullmatch(str(value).strip())
        if not match:
            unreadable.append(value)
            continue
        converted.append(config.PERMIT_REFERENCE_PREFIX
                         + match.group(1).zfill(config.PERMIT_REFERENCE_DIGITS))
    if unreadable:
        _refuse(references.name, unreadable, "permit references in no allowed notation")
    return pd.Series(converted, index=references.index, name=references.name, dtype=object)


def refuse_unreadable_identifiers(dataframe, identifier_patterns):
    """Refuse an identifier written in a form no channel uses; a missing value passes."""
    for column_name, pattern in identifier_patterns.items():
        if column_name not in dataframe.columns:
            continue
        compiled = re.compile(pattern)
        unreadable = [value for value in dataframe[column_name]
                      if not _is_missing(value) and not compiled.fullmatch(str(value))]
        if unreadable:
            _refuse(column_name, unreadable, "identifiers in no form a channel uses")
    return dataframe


def convert_whole_numbers(dataframe, number_patterns):
    """Read revisions, work parties and counts as whole numbers."""
    result = dataframe.copy()
    for column_name, pattern in number_patterns.items():
        if column_name not in result.columns:
            continue
        compiled = re.compile(pattern)
        numbers, unreadable = [], []
        for value in result[column_name]:
            if _is_missing(value):
                numbers.append(pd.NA)
                continue
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                value = str(int(value))
            match = compiled.fullmatch(str(value).strip())
            if not match:
                unreadable.append(value)
                continue
            numbers.append(int(match.group(1)))
        if unreadable:
            _refuse(column_name, unreadable, "values that are not whole numbers")
        result[column_name] = pd.array(numbers, dtype=config.WHOLE_NUMBER_DTYPE)
    return result


def _read_reading(text, reading):
    for unit_name, pattern in reading["notations"].items():
        match = re.fullmatch(pattern, text)
        if match:
            number = (Decimal(match.group(1)) * reading["multipliers"][unit_name]
                      / reading["divisors"][unit_name])
            return number.quantize(reading["step"], rounding=ROUND_HALF_UP)
    return None


def convert_readings(dataframe, readings):
    """Read every gas reading, length, wind, voltage and time in its standard's unit."""
    result = dataframe.copy()
    for column_name, reading in readings.items():
        if column_name not in result.columns:
            continue
        values, unreadable = [], []
        for value in result[column_name]:
            if _is_missing(value):
                values.append(None)
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                value = format(to_decimal(value), "f")
            number = _read_reading(str(value), reading)
            if number is None:
                unreadable.append(value)
                continue
            values.append(number)
        if unreadable:
            _refuse(column_name, unreadable, "readings in no notation the channels write")
        if reading["dtype"] == config.WHOLE_NUMBER_DTYPE:
            result[column_name] = pd.array([pd.NA if value is None else int(value)
                                            for value in values], dtype=config.WHOLE_NUMBER_DTYPE)
        else:
            result[column_name] = pd.Series([np.nan if value is None else float(value)
                                             for value in values], index=result.index,
                                            dtype=reading["dtype"])
    return result


def _blank_where(dataframe, limits, outside):
    result = dataframe.copy()
    for column_name, limit in limits.items():
        if column_name not in result.columns:
            continue
        bound = to_decimal(limit)
        column = result[column_name]
        mask = [not _is_missing(value) and outside(to_decimal(value), bound) for value in column]
        if any(mask):
            column = column.copy()
            column[pd.Series(mask, index=result.index)] = (
                pd.NA if str(column.dtype) == config.WHOLE_NUMBER_DTYPE else np.nan)
            result[column_name] = column
    return result


def treat_values_below_minimum_as_missing(dataframe, minimum_values):
    """Blank the readings below the least value that can be true."""
    return _blank_where(dataframe, minimum_values, lambda value, bound: value < bound)


def treat_values_above_maximum_as_missing(dataframe, maximum_values):
    """Blank the readings above the greatest value that can be true."""
    return _blank_where(dataframe, maximum_values, lambda value, bound: value > bound)


def keep_latest_revision_of_each_request(dataframe):
    """Keep each request once, as its highest revision, in table order."""
    reference, revision = config.PERMIT_REFERENCE_COLUMN, config.REVISION_NUMBER_COLUMN
    blank = dataframe[dataframe[reference].isna() | dataframe[revision].isna()]
    if len(blank):
        named = sorted({str(value) for value in blank[reference]})
        raise ValueError(f"Requests {', '.join(named)} leave {reference} or {revision} blank")
    repeated = dataframe[dataframe.duplicated([reference, revision], keep=False)]
    if len(repeated):
        named = sorted({str(value) for value in repeated[reference]})
        raise ValueError(f"Requests {', '.join(named)} hold two rows at the same {revision}")
    highest = dataframe.groupby(reference)[revision].transform("max")
    return dataframe[dataframe[revision] == highest].reset_index(drop=True)


def convert_markup_to_plain_text(texts):
    """Recover the words a writer meant from text with markup and escaped characters."""
    def plain(value):
        if _is_missing(value):
            return np.nan
        text = re.sub(config.MARKUP_PATTERN, " ", str(value))
        text = html.unescape(text)
        text = re.sub(config.WHITESPACE_RUN_PATTERN, " ", text)
        return text.strip()
    return pd.Series([plain(value) for value in texts], index=texts.index, name=texts.name,
                     dtype=object)


def clean_statement_text(texts):
    """Plain words, and a text written wholly in capitals written in small letters."""
    plain = convert_markup_to_plain_text(texts)
    return pd.Series([value if _is_missing(value) or not str(value).isupper()
                      else str(value).lower() for value in plain],
                     index=plain.index, name=plain.name, dtype=object)
