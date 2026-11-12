"""The suspension likelihood and the expected work hours of one request, from the saved models."""
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from config import config
from data_pipeline.common_cleaning import to_decimal
from data_pipeline.feature_builder import build_feature_frame
from tools.request_data_tools import is_blank


def _refuse_other_screening(request, screening):
    reference = request.get(config.PERMIT_REFERENCE_COLUMN)
    if screening.get("permit_reference") != reference:
        raise ValueError(config.SCREENING_OF_ANOTHER_REQUEST_MESSAGE.format(
            screening_reference=screening.get("permit_reference"), permit_reference=reference))


def _model_inputs(request, screening, feature_names, numeric_columns, categorical_columns):
    row = dict(request)
    for column_name in config.REQUIREMENT_COUNT_RESULTS:
        row[column_name] = screening[column_name]
    for trigger_name, column_name in config.SIMOPS_TRIGGER_FLAG_COLUMNS.items():
        row[column_name] = int(trigger_name in screening.get("simops_triggers", []))
    inputs = build_feature_frame(pd.DataFrame([row]), numeric_columns, categorical_columns)
    if list(inputs.columns) != list(feature_names):
        raise ValueError(f"The model inputs {list(inputs.columns)} are not the saved schema "
                         f"{list(feature_names)}")
    return inputs


def _band(likelihood):
    if likelihood >= config.HIGH_LIKELIHOOD_MINIMUM:
        return config.HIGH_LIKELIHOOD_BAND
    if likelihood >= config.MEDIUM_LIKELIHOOD_MINIMUM:
        return config.MEDIUM_LIKELIHOOD_BAND
    return config.LOW_LIKELIHOOD_BAND


def estimate_suspension_likelihood(suspension_model, feature_names, request, screening):
    """The model's probability of suspension, rounded half up, and its band."""
    _refuse_other_screening(request, screening)
    inputs = _model_inputs(request, screening, feature_names, config.SUSPENSION_NUMERIC_FEATURES,
                           config.SUSPENSION_CATEGORICAL_FEATURES)
    classes = list(suspension_model.classes_)
    positive = classes.index(config.SUSPENSION_TARGET_ENCODING[config.BOOLEAN_YES])
    probability = float(suspension_model.predict_proba(inputs)[0, positive])
    rounded = Decimal(repr(probability)).quantize(config.LIKELIHOOD_STEP, rounding=ROUND_HALF_UP)
    return {"suspension_likelihood": float(rounded), "likelihood_band": _band(rounded)}


def estimate_expected_work_hours(work_hours_model, feature_names, request, screening):
    """The forecast work hours, rounded half up, and whether an extension is expected."""
    _refuse_other_screening(request, screening)
    inputs = _model_inputs(request, screening, feature_names, config.WORK_HOURS_NUMERIC_FEATURES,
                           config.WORK_HOURS_CATEGORICAL_FEATURES)
    forecast = float(work_hours_model.predict(inputs)[0])
    expected = Decimal(repr(forecast)).quantize(config.TENTH_STEP, rounding=ROUND_HALF_UP)
    planned = request.get(config.PLANNED_WORK_HOURS_COLUMN)
    if is_blank(planned):
        return {"expected_work_hours": float(expected), "planned_work_hours": None,
                "extension_expected": None}
    extension = bool(expected > to_decimal(planned) * config.EXTENSION_RATIO)
    return {"expected_work_hours": float(expected), "planned_work_hours": float(planned),
            "extension_expected": extension}


def describe_expected_work_hours(work_hours):
    """The expected work hours in the configuration's words."""
    places = "." + str(config.WORK_HOURS_DECIMALS) + "f"
    expected = format(work_hours["expected_work_hours"], places)
    if work_hours.get("planned_work_hours") is None or work_hours.get("extension_expected") is None:
        return config.EXPECTED_HOURS_WITHOUT_PLAN_TEMPLATE.format(expected_work_hours=expected)
    planned = format(work_hours["planned_work_hours"], places)
    template = (config.EXTENSION_TEMPLATE if work_hours["extension_expected"]
                else config.NO_EXTENSION_TEMPLATE)
    return template.format(expected_work_hours=expected, planned_work_hours=planned)
