"""Trains the three models in one run, and saves them only when every one reaches its minimum."""
import sklearn
from sklearn.model_selection import train_test_split

from config import config
from data_pipeline.dataset_loader import load_processed_dataset
from data_pipeline.feature_builder import (add_requirement_counts, add_simops_trigger_flags,
                                           build_feature_frame)
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.standard_reading import read_permit_standard
from modeling.model_registry import save_feature_schema, save_metrics, save_model
from modeling.sentence_model import train_sentence_model
from modeling.suspension_model import train_suspension_model
from modeling.work_hours_model import train_work_hours_model


def _encode_suspension(frame):
    answers = frame[config.HISTORY_TARGET_COLUMN]
    unknown = sorted({str(value) for value in answers
                      if value not in config.SUSPENSION_TARGET_ENCODING})
    if unknown:
        raise ValueError(f"{config.HISTORY_TARGET_COLUMN}: answers the encoding does not know: "
                         + ", ".join(unknown))
    return answers.map(config.SUSPENSION_TARGET_ENCODING).astype(int)


def _algorithm(model, whole_pipeline=False):
    if whole_pipeline:
        return " + ".join(type(step).__name__ for _name, step in model.steps)
    return type(model.named_steps[config.MODEL_STEP_NAME]).__name__


def main():
    """Produce the three saved models, their schemas and their scores."""
    documents = extract_text_from_directory(config.DOCS_DIRECTORY)
    standards = {permit_type: read_permit_standard(permit_type, documents)
                 for permit_type in config.PERMIT_STANDARD_DOCUMENTS}
    print("training three models")

    history_train, history_eval = [
        add_simops_trigger_flags(add_requirement_counts(load_processed_dataset(path), standards))
        for path in [config.CLEAN_PERMIT_HISTORY_TRAIN_FILE, config.CLEAN_PERMIT_HISTORY_EVAL_FILE]]
    suspension_inputs = [build_feature_frame(frame, config.SUSPENSION_NUMERIC_FEATURES,
                                             config.SUSPENSION_CATEGORICAL_FEATURES)
                         for frame in [history_train, history_eval]]
    suspension_model, suspension_scores = train_suspension_model(
        suspension_inputs[0], _encode_suspension(history_train),
        suspension_inputs[1], _encode_suspension(history_eval))

    logs_train, logs_eval = [add_simops_trigger_flags(load_processed_dataset(path))
                             for path in [config.CLEAN_JOB_LOGS_TRAIN_FILE,
                                          config.CLEAN_JOB_LOGS_EVAL_FILE]]
    hours_inputs = [build_feature_frame(frame, config.WORK_HOURS_NUMERIC_FEATURES,
                                        config.WORK_HOURS_CATEGORICAL_FEATURES)
                    for frame in [logs_train, logs_eval]]
    work_hours_model, work_hours_scores = train_work_hours_model(
        hours_inputs[0], logs_train[config.LOG_TARGET_COLUMN],
        hours_inputs[1], logs_eval[config.LOG_TARGET_COLUMN])

    sentences = load_processed_dataset(config.CLEAN_STATEMENT_SENTENCES_FILE)
    train_text, held_out_text, train_types, held_out_types = train_test_split(
        sentences[config.SENTENCE_TEXT_COLUMN], sentences[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE,
        stratify=sentences[config.SENTENCE_TARGET_COLUMN], random_state=config.RANDOM_SEED)
    sentence_model, sentence_scores = train_sentence_model(train_text, train_types,
                                                           held_out_text, held_out_types)

    shortfalls = []
    for label, score, minimum in [
            ("suspension held-out ROC AUC", suspension_scores["held_out_roc_auc"],
             config.MINIMUM_SUSPENSION_ROC_AUC),
            ("work hours held-out R2", work_hours_scores["held_out_r2"],
             config.MINIMUM_WORK_HOURS_R2),
            ("sentence held-out macro F1", sentence_scores["held_out_macro_f1"],
             config.MINIMUM_SENTENCE_MACRO_F1)]:
        if score < minimum:
            shortfalls.append(f"{label} {score} is below the minimum {minimum}")
    if shortfalls:
        raise SystemExit("No model was saved: " + "; ".join(shortfalls))

    save_model(suspension_model, config.SUSPENSION_MODEL_FILE)
    save_model(work_hours_model, config.WORK_HOURS_MODEL_FILE)
    save_model(sentence_model, config.SENTENCE_MODEL_FILE)
    save_feature_schema(list(suspension_inputs[0].columns), config.SUSPENSION_SCHEMA_FILE)
    save_feature_schema(list(hours_inputs[0].columns), config.WORK_HOURS_SCHEMA_FILE)
    version = sklearn.__version__
    save_metrics({
        "suspension": {"algorithm": _algorithm(suspension_model),
                       "training_rows": len(history_train), "held_out_rows": len(history_eval),
                       "scores": suspension_scores,
                       "minimum_required": config.MINIMUM_SUSPENSION_ROC_AUC,
                       "scikit_learn_version": version},
        "work_hours": {"algorithm": _algorithm(work_hours_model),
                       "training_rows": len(logs_train), "held_out_rows": len(logs_eval),
                       "scores": work_hours_scores,
                       "minimum_required": config.MINIMUM_WORK_HOURS_R2,
                       "scikit_learn_version": version},
        "sentence": {"algorithm": _algorithm(sentence_model, whole_pipeline=True),
                     "training_rows": len(train_text), "held_out_rows": len(held_out_text),
                     "scores": sentence_scores,
                     "minimum_required": config.MINIMUM_SENTENCE_MACRO_F1,
                     "scikit_learn_version": version},
    })
    print(f"  suspension      held-out ROC AUC {suspension_scores['held_out_roc_auc']:.4f}  "
          f"(train {suspension_scores['train_roc_auc']:.4f}, "
          f"gap {suspension_scores['generalisation_gap']:+.4f})")
    print(f"  work hours      held-out R2 {work_hours_scores['held_out_r2']:.4f}  "
          f"MAE {work_hours_scores['mean_absolute_error']:.4f} hours  "
          f"(train {work_hours_scores['train_r2']:.4f}, "
          f"gap {work_hours_scores['generalisation_gap']:+.4f})")
    print(f"  sentence        held-out macro F1 {sentence_scores['held_out_macro_f1']:.4f}  "
          f"(train {sentence_scores['train_macro_f1']:.4f}, "
          f"gap {sentence_scores['generalisation_gap']:+.4f})")
    print(f"metrics written to {config.MODEL_METRICS_FILE}")
    print("every model reaches the minimum the configuration sets for it")


if __name__ == "__main__":
    main()
