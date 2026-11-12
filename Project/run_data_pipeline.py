"""Cleans the five raw extracts and writes the evidence that each one needed cleaning."""
import os

from config import config
from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset
from data_pipeline.history_cleaning import clean_history_dataset
from data_pipeline.job_log_cleaning import clean_job_log_dataset
from data_pipeline.quality_profiler import (profile_dataset, profile_history_details,
                                            profile_log_details, profile_sentence_details,
                                            write_quality_report)
from data_pipeline.sentence_cleaning import clean_sentence_dataset

DATASET_JOBS = [
    (config.RAW_PERMIT_HISTORY_TRAIN_FILE, config.CLEAN_PERMIT_HISTORY_TRAIN_FILE,
     clean_history_dataset, config.HISTORY_TARGET_COLUMN, profile_history_details),
    (config.RAW_PERMIT_HISTORY_EVAL_FILE, config.CLEAN_PERMIT_HISTORY_EVAL_FILE,
     clean_history_dataset, config.HISTORY_TARGET_COLUMN, profile_history_details),
    (config.RAW_JOB_LOGS_TRAIN_FILE, config.CLEAN_JOB_LOGS_TRAIN_FILE,
     clean_job_log_dataset, config.LOG_TARGET_COLUMN, profile_log_details),
    (config.RAW_JOB_LOGS_EVAL_FILE, config.CLEAN_JOB_LOGS_EVAL_FILE,
     clean_job_log_dataset, config.LOG_TARGET_COLUMN, profile_log_details),
    (config.RAW_STATEMENT_SENTENCES_FILE, config.CLEAN_STATEMENT_SENTENCES_FILE,
     clean_sentence_dataset, config.SENTENCE_TARGET_COLUMN, profile_sentence_details),
]


def main():
    """Clean every extract first; write nothing unless all five are cleaned."""
    cleaned, profiles = [], {}
    for raw_path, clean_path, cleaner, target_column, detail_profiler in DATASET_JOBS:
        raw_frame = load_raw_dataset(raw_path)
        clean_frame = cleaner(raw_frame)
        name = os.path.splitext(os.path.basename(raw_path))[0]
        profile = profile_dataset(raw_frame, clean_frame, target_column)
        profile.update(detail_profiler(raw_frame, clean_frame))
        profiles[name] = profile
        cleaned.append((name, clean_path, clean_frame, profile))
    for name, clean_path, clean_frame, profile in cleaned:
        save_processed_dataset(clean_frame, clean_path)
        print(f"{name:<{config.DATASET_NAME_WIDTH}}"
              f"{profile['rows_received']:>{config.ROW_COUNT_WIDTH}} received -> "
              f"{profile['rows_kept']:>{config.ROW_COUNT_WIDTH}} kept  "
              f"({profile['rows_removed']} removed)")
    write_quality_report(profiles)
    print(f"quality report written to {config.DATA_QUALITY_REPORT_FILE}")


if __name__ == "__main__":
    main()
