# PermitWise AI: the data, its preparation and the models

This document explains every data file, what each column means, which column is the **target**,
what is wrong with the raw data, every step taken **before** a model is built, which models we use
(and which others could be used), and how well they score. All numbers come from this project's
own files: `output/data_quality_report.json`, `output/model_metrics.json`, and analysis of the
cleaned data.

For the system as a whole see [README_ARCHITECTURE.md](README_ARCHITECTURE.md).

---

## 1. All the data at a glance

| File (under `Project/`) | What one row is | Raw rows → clean rows | Cols | Used for | Target |
|---|---|---|---|---|---|
| `data/permit_history_train.csv` | one past permit request (one revision) | 29,796 → **27,797** | 40 | training the **suspension model** | `permit_suspended` (Yes/No) |
| `data/permit_history_eval.csv` | same, later requests | 7,455 → **6,935** | 40 | scoring the suspension model (never trained on) | `permit_suspended` |
| `data/job_logs_train.csv` | one permit, from issue to hand-back | 28,339 → **27,705** | 24 | training the **work hours model** | `actual_work_hours` (number) |
| `data/job_logs_eval.csv` | same, later permits | 7,078 → **6,930** | 24 | scoring the work hours model | `actual_work_hours` |
| `data/statement_sentences.csv` | one sentence of a document attached to an old request | 36,666 → **35,709** | 8 | training the **sentence model** | `sentence_type` (9 classes) |
| `data/input/open_permit_requests.csv` | a live request waiting for review | 45 → **40** | 36 | the live queue (scored, never trained on) | none (unknown) |
| `data/input/request_documents.csv` | a document attached to a live request | 92 → **90** | 6 | statement evidence | none |
| `data/input/live_permit_register.csv` | a permit already live on site | 29 → **28** | 8 | counting permits beside a request; SIMOPS | none |
| `data/docs/*.pdf` (16) | permit standards, guides, policy, rules, casebook | 136 sections, 220 passages | — | requirements, rules, chat answers | none |

**Why "train" and "eval" are separate files:** the eval files hold the requests and permits that
came *after* the training ones. Scoring on them is a **time-based hold-out**. It shows how the
model will do on future requests, which is exactly how it is used. The models never see these
rows while fitting.

---

## 2. The permit history (suspension model data)

### 2.1 Columns

| Group | Columns | Meaning |
|---|---|---|
| Identity (never a feature) | `permit_reference`, `revision_number`, `contractor_reference`, `equipment_tag` | who and what; identifiers only |
| Request context | `request_channel` (Site portal / Contractor app / Scanned paper), `plant_area` (29), `area_classification` (Zone 1 / Zone 2 / Non-hazardous), `process_unit_status` (Operating / Shut down / No process plant), `work_location` (Indoors / Outdoors), `work_activity` (25 kinds) | where and how the job is done |
| Permit types | `permit_type` (5 types), `linked_permit_type` (5 types or *No linked permit*) | which standards apply |
| Size of job | `work_party_size`, `planned_work_hours` | people and planned time |
| Gas readings | `flammable_gas_lel_percent`, `oxygen_percent`, `hydrogen_sulphide_ppm`, `carbon_monoxide_ppm` | the gas test at the work point |
| Hot work answers | `combustibles_cleared_radius_m`, `fire_watch_minutes` | fire precautions |
| Confined space answers | `rescue_plan_in_place` | rescue cover |
| Electrical answers | `system_voltage_volts`, `isolation_proved_dead`, `personal_locks_applied`, `circuit_earths_applied` | isolation |
| Height answers | `working_height_m`, `access_method`, `fall_protection_fitted`, `scaffold_inspected`, `wind_speed_mps` | access and falls |
| Excavation answers | `excavation_depth_m`, `services_located`, `trench_support_fitted`, `work_area_barriered` | digging |
| Every request | `risk_assessment_attached` | task risk assessment attached |
| Site at the time | `concurrent_live_permits`, `conflicting_live_permits` | permits already live in the same area |
| **Outcome (never a feature)** | `review_outcome`, **`permit_suspended`**, `suspension_reason` | known only *after* the decision or the work |

Most answers only apply to one permit type, so they are naturally blank for the others. For
example, `excavation_depth_m` is blank on about 85 % of rows. This is expected, not dirt.

### 2.2 Target: `permit_suspended`

| Value | Train share | Eval share | Encoded as |
|---|---|---|---|
| No | 84.3 % | 84.7 % | 0 |
| **Yes** | **15.7 %** | **15.3 %** | 1 |

The classes are **imbalanced** (about 1 in 6 suspended), so accuracy would be misleading: always
saying "No" scores 84 %. We therefore use **ROC AUC** (how well the model ranks risky requests above
safe ones) and check **calibration** (the average predicted probability must match the real rate).

What the target looks like in the cleaned training data:

| Slice | Suspension rate |
|---|---|
| Confined space entry / Excavation / Hot work / Work at height / Electrical isolation | 28.7 % / 18.9 % / 16.4 % / 15.6 % / 5.1 % |
| With a conflicting live permit / without | **39.7 %** / 11.6 % |
| Any requirement Not met / none | **24.6 %** / 14.4 % |
| Suspension reasons (Yes rows) | Gas alarm 1,282 · Conflicting work 917 · Unsafe condition 852 · Process upset 765 · Site alarm 361 · High wind 191 |

Counts by permit type (train): Work at height 6,934 · Hot work 6,520 · Electrical isolation 6,252 ·
Excavation 4,304 · Confined space entry 3,787. 18.4 % of requests have a linked permit type.

### 2.3 Defects found in the raw history (train file)

| Defect | How many | Example |
|---|---|---|
| Exact duplicate rows | 365 | the same row sent twice |
| Earlier revisions set aside (they had an answer, but a later revision exists) | 1,430 | rev 1 and rev 2 of one request |
| Latest revision has no outcome yet (row dropped) | 204 | target `N/A` |
| References not written `PTW` + 7 digits | 15,326 | `PTW-33897`, `34166` |
| Permit type not written canonically | 15,749 | `EI`, `Electrical Isolation Permit`, `CSE` |
| Yes/No not written canonically | e.g. `risk_assessment_attached` 15,643 | `Y`, `true`, `FALSE`, `1` |
| Readings in another unit | planned hours in minutes 9,825; voltage in kV 1,743; depth in ft 2,108; H₂S in mg/m³ 628; wind in mph 821 … | `690 min`, `6.6kV`, `4.5 ft` |
| Impossible readings (blanked, row kept) | below minimum: planned hours 41, work party 32, flammable gas 23, height 20, depth 10 …; oxygen above 100 %: 24 | `-2`, `0`, `104%LEL` |
| Blank written many ways | — | `N/A`, `-`, `null`, `illegible`, `not recorded`, empty |

The eval file has the same defects in proportion (83 duplicates, 370 earlier revisions, 67 with no
outcome).

---

## 3. The job logs (work hours model data)

### 3.1 Columns

The job context and permit types are the same as in the history, plus the job-specific answers
(`fire_watch_minutes`, `system_voltage_volts`, `working_height_m`, `access_method`,
`excavation_depth_m`, `trench_support_fitted`, `wind_speed_mps`, `work_party_size`), the live and
conflicting permit counts, and `planned_work_hours`.

The outcome columns `permit_suspended` and `completion_code` (Completed / Completed after extension
/ Completed after suspension) are only known **after** the work, so they are **never features**.

### 3.2 Target: `actual_work_hours`

| Statistic | Actual hours | Planned hours |
|---|---|---|
| Mean | 14.2 | 11.2 |
| Median | 11.4 | 9.5 |
| Min / Max | 1.3 / 197.0 | 1.5 / 72.0 |
| Quartiles | 6.8 – 18.1 | 6.0 – 14.0 |

The target is **right-skewed**: most jobs take a few hours, a few take days. That is why we model
`log(hours)` (section 7.2). Jobs usually overrun the plan. The median actual ÷ planned is 1.28 on
an **Operating** unit, 1.14 when **Shut down**, and 0.96 with **No process plant**.

Completion codes: Completed 17,348 · Completed after extension 5,999 · Completed after
suspension 4,358.

### 3.3 Defects (train file)

Duplicates 339 · no actual hours 239 · actual hours impossible (≤ 0) 56 · references not
canonical 14,584 · hours written in minutes 9,449 · voltage in kV 1,664 · depth in feet 2,013 ·
wind in mph 783 · height in feet 1,239.

---

## 4. The statement sentences (sentence model data)

### 4.1 Columns

| Column | Meaning |
|---|---|
| `sentence_id` | `SEN-` + 7 digits (unique) |
| `permit_reference`, `request_channel` | the request it came from |
| `document_type` | Method statement 19,020 · Task risk assessment 8,233 · Gas test record 4,968 · Rescue plan 3,488 |
| `sentence_number` | its place in the document |
| **`sentence_text`** | the only model input |
| `sentence_length_characters` | re-measured after cleaning (the raw value was wrong on 6,222 rows) |
| **`sentence_type`** | **target**, 9 classes |

### 4.2 Target: `sentence_type`

| Class | Share | Bears on requirements of the kind |
|---|---|---|
| Background | 16.4 % | none |
| Simultaneous operations | 15.8 % | none |
| Access | 11.9 % | fall protection, scaffold, barriers, wind, trench support |
| Gas testing | 11.8 % | flammable gas, oxygen, H₂S, CO (also checked for readings that disagree with the form) |
| Rescue | 11.7 % | rescue plan |
| Isolation | 9.6 % | proved dead, personal locks, circuit earths, services located |
| Personal protective equipment | 9.0 % | none |
| Housekeeping and waste | 8.6 % | none |
| Fire precautions | 5.3 % | combustibles cleared, fire watch |

The classes are moderately imbalanced (5 % to 16 %), so we score with **macro F1**, which weighs
every class equally.

### 4.3 Defects

Markup such as `<b>`, `<br/>` in 3,656 sentences · escaped characters such as `&amp;`, `&nbsp;` in
2,839 · written in CAPITALS in 544 · labels spelled 30+ ways (`GAS TEST`, `gas-testing`, `PPE`,
`SIMOPS`, `context` …) · 450 duplicates · 291 unlabelled · 216 shorter than 3 words.

---

## 5. Live files and documents (used at run time, not for training)

- **open_permit_requests.csv**: the same 35 answer columns as the history, plus
  `supporting_document_count`. It has **no outcome and no live-permit counts**: those are
  counted from the register at run time. After cleaning it holds 40 requests (repeats and earlier
  revisions removed).
- **request_documents.csv**: `document_id`, `permit_reference`, `request_channel`,
  `document_type`, `document_title`, `document_text`. Each document is split into sentences, and
  each sentence is classified by the sentence model.
- **live_permit_register.csv**: `permit_reference`, `plant_area`, `permit_type`,
  `linked_permit_type`, `work_activity`, `equipment_tag`, `contractor_reference`, `permit_status`
  (Live / Suspended). Only **Live** permits are counted.
- **16 PDFs**: the 5 permit standards hold each permit type's requirements and figures, read at
  run time. They are never typed into the code.

---

## 6. Every step before a model is built

The steps run in this order (`run_data_pipeline.py`, then `run_model_training.py`). Each step is
one function in `data_pipeline/`.

### Stage A: cleaning (`run_data_pipeline.py`)

| # | Step | Function | Example |
|---|---|---|---|
| 1 | Read every column **as text**, guess nothing; strip spaces | `load_raw_dataset` | `" 5 hrs "` → `"5 hrs"` |
| 2 | Only the 7 null tokens become blank | `load_raw_dataset` | `N/A`, `-`, `illegible` → blank; `None` is **not** blank (it gets refused) |
| 3 | Remove exact duplicate rows (keep the first) | `remove_duplicate_records` | 365 rows (history train) |
| 4 | Refuse rows missing a key | `refuse_missing_values` | a job log with no reference → error |
| 5 | Fold every spelling onto one canonical value | `normalize_category_values` | `HW`, `Hot Work Permit` → `Hot work`; `Y`, `TRUE`, `1` → `Yes` |
| 6 | Canonical permit references | `convert_permit_references` | `PTW-52118`, `52118` → `PTW0052118`; `PTW52118` → refused |
| 7 | Check identifiers | `refuse_unreadable_identifiers` | `CTR-12345` → refused |
| 8 | Whole numbers | `convert_whole_numbers` | revision `2` → 2 (`Int64`) |
| 9 | **Unit conversion, exact decimals, rounded half up once** | `convert_readings` | `30 ft` → 9.1 m · `7.1 mg/m3` H₂S → 5.0 ppm · `27 mph` → 12.1 m/s · `11kV` → 11000 V · `1.5 hrs` fire watch → 90 min · `390 min` → 6.5 h |
| 10 | Blank impossible readings (never "fix" them; keep the row) | `treat_values_below_minimum_as_missing` / `…above_maximum…` | `-2` % LEL, `104%LEL`, a work party of 0 → blank |
| 11 | Keep the **latest revision** of each request | `keep_latest_revision_of_each_request` | rev 2 kept, rev 1 dropped; two rows at the same revision → refused |
| 12 | Drop rows with no target | `remove_rows_missing_target` | an outcome is never borrowed from an earlier revision |
| 13 | Text only: strip markup, unescape, collapse spaces, lower whole-CAPITAL sentences | `clean_statement_text` | `<b>THE GAS&nbsp;TEST</b>` → `the gas test` |
| 14 | Text only: re-measure the length; drop sentences under 3 words | `clean_sentence_dataset` | — |
| 15 | Profile and write the evidence report | `quality_profiler` | every removed row has a reason |

A value in a notation we do not recognise is **refused with an error naming the column and the
value**. It is never guessed or silently blanked, because a wrong unit would pass or fail the
wrong safety limit.

### Stage B: feature engineering (`run_model_training.py`, `feature_builder.py`)

| # | Step | Why |
|---|---|---|
| 16 | Read the 5 permit standards from their PDFs | the source of every requirement and figure |
| 17 | **Requirement counts**: screen every past request under its own (and linked) standard, exactly as a live request is screened, and add `requirements_met_count`, `requirements_not_met_count` and `requirements_not_documented_count` | turns ~25 sparse answers into 3 dense, meaningful numbers; training and live use the same logic |
| 18 | **SIMOPS trigger flags** (0/1): `conflicting_permit_trigger`, `live_process_trigger`, `high_voltage_trigger` | the three policy triggers as features for both tabular models |
| 19 | **Leave out leakage and identifiers**: `permit_suspended`, `review_outcome`, `suspension_reason`, `completion_code`, `actual_work_hours`, references, contractor, equipment tag | these are known only after the fact, or identify the row rather than describe the job |
| 20 | Lay out the inputs in a fixed order: numbers, then categories (`build_feature_frame`); blanks become `NaN` | the saved schema must match at scoring time |
| 21 | Encode the target: Yes → 1, No → 0 (suspension); an unknown answer is refused | — |
| 22 | Sentence data: **stratified 80/20 split**, seed 42 (28,567 train / 7,142 held out) | there is no separate eval file for sentences |

### Stage C: inside the model pipeline (`build_model_pipeline`)

All of these are **learned on the training rows only** and saved inside the model, so live
requests are prepared identically.

| Column type | Steps |
|---|---|
| Numeric | fill blanks with the training **median**, then **standard-scale** |
| Categorical | fill blanks with the training **most frequent** value, then **one-hot encode** (unknown categories ignored) |
| Text (sentence model) | **TF-IDF** on words and word pairs, sublinear term frequency, terms seen at least twice |

---

## 7. The models

### 7.1 Suspension likelihood: classification

**Chosen:** `LogisticRegression` (C = 1.0) inside the pipeline above.
**Inputs (27):** 20 numeric (`work_party_size`, `planned_work_hours`, the 4 gas readings,
clearance, fire watch, voltage, height, wind, depth, the 2 permit counts, the 3 requirement counts,
the 3 trigger flags) and 7 categorical (`permit_type`, `linked_permit_type`, `work_activity`,
`area_classification`, `process_unit_status`, `work_location`, `access_method`).

Why logistic regression wins here:

1. **Calibrated probabilities.** The output is read as a likelihood (LOW < 0.20 ≤ MEDIUM < 0.40 ≤
   HIGH), so the average must match reality. It does: mean prediction 0.1528 vs. real rate 0.1528.
2. **Monotone behaviour required by the spec.** One more requirement Not met must raise the score
   for ≥ 95 % of requests (+0.008 on average). One more conflicting permit must raise it for ≥ 95 %
   (+0.05). A linear model does this for 100 % of requests (+0.016 and +0.16). Tree models can
   stay flat or even go down in places.
3. Explainable to an Area Authority, fast, and stable.

**Result:** held-out ROC AUC **0.829** (minimum 0.79), train 0.833, gap +0.004 (no overfitting),
Brier score 0.095.

**Most important features** (drop in held-out ROC AUC when the column is shuffled):

| Feature | Importance |
|---|---|
| conflicting_live_permits | 0.040 |
| planned_work_hours | 0.035 |
| process_unit_status | 0.027 |
| flammable_gas_lel_percent | 0.021 |
| area_classification | 0.021 |
| wind_speed_mps | 0.020 |
| permit_type | 0.012 |
| work_activity | 0.010 |
| concurrent_live_permits | 0.006 |

**Other models that can be used:**

| Model | Pros | Cons here |
|---|---|---|
| HistGradientBoostingClassifier with `monotonic_cst` | captures interactions, may reach a higher AUC | needs calibration (`CalibratedClassifierCV`); monotone constraints give "never lower", not "always higher" |
| Random Forest | robust, little tuning | poorly calibrated, not monotone, larger file |
| XGBoost / LightGBM | strong on tabular data | extra dependency; the same calibration and monotonicity issues |
| Logistic regression with interaction terms | keeps calibration | more features to explain |

### 7.2 Expected work hours: regression

**Chosen:** `HistGradientBoostingRegressor` (400 iterations, learning rate 0.08, seed 42), wrapped in
`TransformedTargetRegressor(func=log, inverse_func=exp)`.
**Inputs (20):** 12 numeric (`work_party_size`, `fire_watch_minutes`, `system_voltage_volts`,
`working_height_m`, `excavation_depth_m`, `wind_speed_mps`, `planned_work_hours`, the 2 permit
counts, the 3 trigger flags) and 8 categorical (as above, plus `trench_support_fitted`).

Why:

1. **The log target** suits right-skewed hours. Effects are multiplicative: a live unit adds a
   *percentage* to a job, not fixed hours. Forecasts are always positive.
2. **Boosted trees** capture the interactions between planned hours, unit status and conflicts.
3. Required behaviours hold. An operating unit instead of a shut-down one lengthens 100 % of
   forecasts (+3.5 h on average; at least +1.0 h is required). One more conflicting permit
   lengthens 99.9 % of them (+2.35 h). No forecast falls below the shortest real job (lowest
   1.75 h vs. 1.3 h).

**Result:** held-out R² **0.890** (minimum 0.84), train 0.903, gap +0.014. MAE **2.24 h**, against
7.53 h for always guessing the average (the spec requires < 0.75 × that; we reach 0.30 ×).

**Most important features** (drop in held-out R² when shuffled):

| Feature | Importance |
|---|---|
| planned_work_hours | 1.452 (dominant: a job's plan is the best base estimate) |
| live_process_trigger | 0.035 |
| area_classification | 0.025 |
| conflicting_live_permits | 0.024 |
| linked_permit_type | 0.014 |
| process_unit_status | 0.012 |
| wind_speed_mps | 0.008 |

**Other models that can be used:**

| Model | Pros | Cons here |
|---|---|---|
| Ridge / linear regression on log hours | simple, explainable | tested: **R² −2.3**, because one extreme forecast ruins it; misses interactions |
| Random Forest regressor | robust | cannot extrapolate; larger file |
| XGBoost / LightGBM | similar accuracy | extra dependency |
| Predicting the ratio actual ÷ planned | very interpretable | blank planned hours need a separate path |

### 7.3 Sentence type: text classification

**Chosen:** `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2)` →
`LinearSVC(class_weight="balanced")`.

Why: sentences are short (median 14 words) and their type is signalled by key words and word pairs
("tested at the top", "standby person", "isolation certificate"). A linear model on TF-IDF is fast,
accurate and works offline. `class_weight="balanced"` protects the rarest class (Fire precautions,
5 %). Any text, even nonsense, maps to one of the 9 classes.

**Result:** held-out macro F1 **0.981** (minimum 0.95), accuracy 0.982, gap +0.0002. F1 per class:

| Class | F1 | Class | F1 |
|---|---|---|---|
| Background | 0.987 | Personal protective equipment | 0.983 |
| Housekeeping and waste | 0.986 | Isolation | 0.982 |
| Rescue | 0.986 | Access | 0.969 |
| Simultaneous operations | 0.985 | Fire precautions | 0.968 |
| Gas testing | 0.985 | | |

**Other models that can be used:** multinomial Naive Bayes (very fast, slightly weaker), logistic
regression on TF-IDF (gives probabilities, similar F1), sentence-embedding models plus a classifier
(the project already ships `all-MiniLM-L6-v2`; heavier, and unnecessary at 98 %), or a fine-tuned
transformer such as DistilBERT (best on harder text, but needs a GPU and is overkill here).

---

## 8. How the models are scored and saved

| | Suspension | Work hours | Sentence |
|---|---|---|---|
| Train rows / held-out rows | 27,797 / 6,935 | 27,705 / 6,930 | 28,567 / 7,142 |
| Held-out set | later requests (eval file) | later permits (eval file) | stratified 20 % |
| Main metric (minimum) | ROC AUC ≥ 0.79 | R² ≥ 0.84 | macro F1 ≥ 0.95 |
| Our score | **0.829** | **0.890** | **0.981** |
| Also reported | Brier, mean prediction vs. positive share | MAE vs. average-guess MAE, lowest/highest forecast | accuracy, weighted F1, F1 per class |
| Saved to | `output/models/suspension_model.pkl` + `suspension_features.json` | `work_hours_model.pkl` + `work_hours_features.json` | `sentence_model.pkl` |

- `generalisation_gap` = train score − held-out score. All three gaps are tiny (≤ 0.014), so the
  models **learned patterns, not rows**.
- Training **stops and saves nothing** if any model misses its minimum.
- Every score is in `output/model_metrics.json`, together with the algorithm and the
  scikit-learn version.

## 9. How a live request uses the models

1. The request is cleaned with the same steps 1–11 (no target).
2. Its live and conflicting permits are counted from the register.
3. It is screened under its standards, which gives the 3 counts; the SIMOPS triggers give the 3
   flags.
4. `build_feature_frame` lays out the inputs in the saved schema order.
5. The **suspension likelihood** is rounded half up to 0.01, then banded. The **expected hours**
   are rounded to 0.1, and an extension is expected when they exceed 1.3 × the planned hours.
6. Each document sentence gets a **sentence type**, the requirements it bears on, and a check of
   its gas readings against the form.

**The figures never decide anything:** they are advice next to the screening, which an Area
Authority decides from.

## 10. Re-running and experimenting

```bash
python3 run_data_pipeline.py      # stages A
python3 run_model_training.py     # stages B-C, training, scoring, saving
python3 -m pytest tests.py -v -k "suspension or work_hours or sentence or inputs_follow or models_fit or schemas"
```

To try another estimator, change only the estimator inside `modeling/suspension_model.py`,
`work_hours_model.py` or `sentence_model.py`. Keep `build_model_pipeline` so the preparation and
the saved schema stay the same. Then re-run training and the model tests.
