# PermitWise AI

PermitWise AI is a permit-to-work review platform for Hexley Chemicals' Thornmere Works. It
cleans five raw extracts, reads every permit type's requirements out of its own permit standard,
screens each open request under its own and its linked type's standard, counts the permits live
beside it, estimates the suspension likelihood and the expected work hours, reads every sentence
of the attached documents, and runs a five-agent review that stops at an Area Authority's gate.
After the decision it writes a permit document and work party briefing, a rework request, or a
SIMOPS brief. Everything is served by a FastAPI service and a Streamlit dashboard.

No requirement figure is written in the code: each is read at run time from its standard's PDF.

More documentation:

- [README_ARCHITECTURE.md](README_ARCHITECTURE.md): how the system is built, stage by stage, with flow diagrams
- [README_DATA_AND_MODELS.md](README_DATA_AND_MODELS.md): every data file, its columns and target, each preparation step, and the models with alternatives and scores
- [README_DASHBOARD_GUIDE.md](README_DASHBOARD_GUIDE.md): how to run and use the dashboard, the logins, sample scenarios, and the checks performed
- [README_TEAM_PLAN.md](README_TEAM_PLAN.md): who owns what, and the 5-day upload plan to `main`

## Run order

Every command is run from this `Project` folder.

```bash
pip install -r requirements.txt
python3 run_database_setup.py      # database/permitwise.db, six tables, sixteen accounts
python3 run_data_pipeline.py       # processed/*.csv and output/data_quality_report.json
python3 run_model_training.py      # output/models/*.pkl, schemas, output/model_metrics.json
python3 run_knowledge_indexing.py  # database/chroma_store (Chroma, all-MiniLM-L6-v2)
python3 run_api_service.py         # REST service on port 8081, docs at /docs
python3 -m streamlit run run_dashboard.py   # dashboard on port 8080 (second terminal)
```

The Gemini key and model are read from `.env` (`GEMINI_API_KEY`, `GEMINI_MODEL`).

Run the tests with `python3 -m pytest tests.py -v`. The suite opens one real review run and
asks the chat one question, so it needs the steps above and a working Gemini key. Each run of
the suite reviews one not-yet-reviewed request assessed ISSUE PERMIT.

## Layout

| Folder | What it does |
|---|---|
| `data_pipeline/` | Loading, cleaning (references, units, codes, revisions), quality profiling, model features |
| `knowledge/` | PDF text, sections and overlapping passages, the Chroma index, standard reading |
| `modeling/` | Suspension (logistic regression), work hours (gradient boosting on log hours), sentence type (TF-IDF + linear SVM) |
| `tools/` | Request data, screening, evidence, predictions, retrieval, documents, decision checks |
| `agentic/` | Gemini client and seven agents, LangGraph nodes, the graph with a SQLite checkpoint |
| `persistence/` | SQLAlchemy engine, six tables, every query |
| `access/` | Password hashing, the sign-in and permission guard, area-group appointment |
| `chat/` | Conversation memory, the document-grounded assistant, answers from the records |
| `api/` | Request/response schemas and the routes |
| `dashboard/` | Sign-in and five role-gated pages, all data fetched from the service |

## Roles

| Role | Pages |
|---|---|
| Area Authority | Request queue, Request detail, Decision queue, Audit trail, Permit chat |
| Permit Coordinator | Request queue, Request detail, Audit trail, Permit chat |
| Control Room Operator | Permit chat (status only; never a prediction) |

Seed accounts and passwords are in `config/config.py` (`SEED_USER_ACCOUNTS`).
