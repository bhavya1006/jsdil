"""The requirement counts, the SIMOPS trigger flags, the model inputs in their saved order, and the
preparation every tabular model is fitted with."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import config


def add_requirement_counts(dataframe, standards):
    """Each request's Met, Not met and Not documented counts under its own standards."""
    from tools.screening_tools import screen_request

    result = dataframe.copy()
    screenings = [screen_request(record, standards) for record in result.to_dict("records")]
    for column_name in config.REQUIREMENT_COUNT_RESULTS:
        result[column_name] = [int(screening[column_name]) for screening in screenings]
    return result


def add_simops_trigger_flags(dataframe):
    """A 0 or 1 flag for each SIMOPS trigger each row meets, as the screening finds it."""
    from tools.screening_tools import find_simops_triggers

    result = dataframe.copy()
    triggers = [find_simops_triggers(record) for record in result.to_dict("records")]
    for trigger_name, column_name in config.SIMOPS_TRIGGER_FLAG_COLUMNS.items():
        result[column_name] = [int(trigger_name in found) for found in triggers]
    return result


def build_feature_frame(dataframe, numeric_columns, categorical_columns):
    """The model inputs, numbers then words, in the configured order, gaps as NaN."""
    features = pd.DataFrame(index=dataframe.index)
    for column_name in numeric_columns:
        features[column_name] = pd.to_numeric(
            dataframe[column_name].astype(object).where(dataframe[column_name].notna(), np.nan),
            errors="coerce").astype(float)
    for column_name in categorical_columns:
        column = dataframe[column_name].astype(object)
        features[column_name] = pd.Series(
            [np.nan if pd.isna(value) else value for value in column], index=dataframe.index,
            dtype=object)
    return features


def build_model_pipeline(estimator, numeric_columns, categorical_columns):
    """The input preparation and the estimator joined into one fittable pipeline."""
    numeric = Pipeline([("fill", SimpleImputer(strategy=config.NUMBER_FILL_STRATEGY, keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("fill", SimpleImputer(strategy=config.CATEGORY_FILL_STRATEGY,
                                                   missing_values=np.nan, keep_empty_features=True)),
                            ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preparation = ColumnTransformer([("numbers", numeric, list(numeric_columns)),
                                     ("words", categorical, list(categorical_columns))])
    return Pipeline([(config.PREPROCESSING_STEP_NAME, preparation),
                     (config.MODEL_STEP_NAME, estimator)])
