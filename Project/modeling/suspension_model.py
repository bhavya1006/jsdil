"""The suspension likelihood model."""
from sklearn.linear_model import LogisticRegression

from config import config
from data_pipeline.feature_builder import build_model_pipeline
from modeling.model_evaluation import evaluate_classifier


def train_suspension_model(train_features, train_target, held_out_features, held_out_target):
    """A calibrated logistic model, monotone in its inputs, fitted on the training rows only."""
    estimator = LogisticRegression(C=1.0, max_iter=5000, random_state=config.RANDOM_SEED)
    model = build_model_pipeline(estimator, config.SUSPENSION_NUMERIC_FEATURES,
                                 config.SUSPENSION_CATEGORICAL_FEATURES)
    model.fit(train_features, train_target)
    scores = evaluate_classifier(model, train_features, train_target, held_out_features,
                                 held_out_target)
    return model, scores
