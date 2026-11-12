"""The expected work hours model."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from config import config
from data_pipeline.feature_builder import build_model_pipeline
from modeling.model_evaluation import evaluate_regressor


def train_work_hours_model(train_features, train_target, held_out_features, held_out_target):
    """Gradient boosting on the logarithm of the hours, so a forecast is always positive and
    multiplicative effects such as a live unit scale with the size of the job."""
    booster = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                            random_state=config.RANDOM_SEED)
    estimator = TransformedTargetRegressor(regressor=booster, func=np.log, inverse_func=np.exp)
    model = build_model_pipeline(estimator, config.WORK_HOURS_NUMERIC_FEATURES,
                                 config.WORK_HOURS_CATEGORICAL_FEATURES)
    model.fit(train_features, train_target)
    scores = evaluate_regressor(model, train_features, train_target, held_out_features,
                                held_out_target)
    return model, scores
