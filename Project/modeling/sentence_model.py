"""The sentence type model."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import config
from modeling.model_evaluation import evaluate_text_classifier


def train_sentence_model(train_text, train_target, held_out_text, held_out_target):
    """Word and word-pair weights into a linear classifier, fitted on the training sentences."""
    model = Pipeline([
        (config.PREPROCESSING_STEP_NAME, TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                                         min_df=2, lowercase=True)),
        (config.MODEL_STEP_NAME, LinearSVC(C=1.0, class_weight="balanced",
                                           random_state=config.RANDOM_SEED)),
    ])
    model.fit(list(train_text), list(train_target))
    scores = evaluate_text_classifier(model, train_text, train_target, held_out_text,
                                      held_out_target)
    return model, scores
