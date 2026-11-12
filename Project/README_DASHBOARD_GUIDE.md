# PermitWise AI: dashboard guide

How to start the dashboard, sign in, and walk through each page with the sample data. The last
section lists the checks that were run on the dashboard and their results. For how the system works
inside, see [README_ARCHITECTURE.md](README_ARCHITECTURE.md).

---

## 1. Start it

Run from the `Project/` folder. Steps 1–4 are needed once (they are already done in this
workspace).

```bash
pip install -r requirements.txt
python3 run_database_setup.py
python3 run_data_pipeline.py
python3 run_model_training.py
python3 run_knowledge_indexing.py
```

Then use two terminals:

```bash
# terminal 1: the REST service (wait for "Application startup complete", about 30-60 s)
python3 run_api_service.py

# terminal 2: the dashboard
python3 -m streamlit run run_dashboard.py
```

Open **http://localhost:8080**. The API docs are at http://localhost:8081/docs.

`.env` must hold `GEMINI_API_KEY` and `GEMINI_MODEL` (for example `gemini-3.5-flash-lite`). Without
them the queue and detail pages still work, but starting a review, writing documents and the chat
will fail with a clear message.

## 2. Logins

These are the demo accounts seeded from `config/config.py`. Every account in a role shares that
role's password.

| Username | Name | Role | Area group | Password |
|---|---|---|---|---|
| `dara` | Dara Whitfield | Area Authority | Process units | `11111111` |
| `imogen` | Imogen Hale | Area Authority | Process units | `11111111` |
| `kofi` | Kofi Mensah | Area Authority | Storage areas | `11111111` |
| `ruth` | Ruth Calder | Area Authority | Storage areas | `11111111` |
| `tomasz` | Tomasz Nowak | Area Authority | Utilities | `11111111` |
| `elena` | Elena Varga | Area Authority | Utilities | `11111111` |
| `gareth` | Gareth Lloyd | Area Authority | Non-process areas | `11111111` |
| `sunita` | Sunita Rao | Area Authority | Non-process areas | `11111111` |
| `marcus` | Marcus Reed | Permit Coordinator | — | `22222222` |
| `fiona` | Fiona Blake | Permit Coordinator | — | `22222222` |
| `hassan` | Hassan Idris | Permit Coordinator | — | `22222222` |
| `chloe` | Chloe Mercer | Permit Coordinator | — | `22222222` |
| `liam` | Liam Doherty | Control Room Operator | — | `33333333` |
| `aisha` | Aisha Khan | Control Room Operator | — | `33333333` |
| `nathan` | Nathan Cole | Control Room Operator | — | `33333333` |
| `rosa` | Rosa Delgado | Control Room Operator | — | `33333333` |

Which plant areas belong to which group:

| Area group | Plant areas |
|---|---|
| Process units | Reactor House A, Reactor House B, Distillation Unit, Solvent Recovery, Chlorine Unit, Caustic Evaporation, Drum Filling, Flare Area, East Pipe Bridge, West Pipe Bridge |
| Storage areas | Tank Farm North, Tank Farm South, Road Loading Bay, Rail Sidings |
| Utilities | Effluent Treatment, Boiler House, Cooling Towers, Compressor House, Main Substation, Water Treatment |
| Non-process areas | Site Roads and Drains, Engineering Workshops, Central Warehouse, Laboratory, Control Building, Administration Block, Gatehouse, Fire Station, Weighbridge |

**Pages by role:** an Area Authority sees all five pages; a Permit Coordinator sees everything
except the Decision queue; a Control Room Operator sees only Permit chat. The service enforces
this too: a page you cannot see is also refused by the API with 403.

One account holds one session. Signing in on another browser ends the first session, and that
browser returns to sign-in with the message "Your session ended…".

## 3. The pages

### Request queue
- Counts across the top: open requests, how many are confined space entry (always at the top of the
  queue), awaiting review, awaiting a decision, and how many are assessed each decision.
- The **Permit type** filter matches the permit type *or* the linked permit type.
- The table shows: position, reference, revision, permit types, plant area, channel, status,
  assessment, Met / Not met / Not documented, live and conflicting permits, SIMOPS triggers, the
  likelihood bar, band, expected hours, extension, flagged sentences and documents.
- **Select a row**, then click **Open request detail**.

### Request detail
- Headline badges: the assessment (green ISSUE, orange RETURN, violet REFER), the status and the
  queue position.
- Eight metric cards: requirements met of total, not met, not documented, SIMOPS triggers, permits
  live, conflicting, suspension likelihood with its band, and expected vs. planned hours (a red
  *extension expected* badge when the expected hours exceed 1.3 × planned).
- Tabs:
  - **Screening:** every requirement of each standard with its result and the standard's own
    sentence, then the decisions open, the triggers and the live permits.
  - **Statement evidence:** every sentence of every attached document, with its type and the
    requirements it bears on. A gas reading that disagrees with the form is shown in a yellow
    flag box.
  - **Request:** every answer the review uses, and what it is used for.
  - **Review:** the stored review and any decision. If the request has no review yet, you can
    start one: type what the review should look at (5 to 1000 characters) and click
    **Start the review**. Five AI agents run; this takes about 15–40 s.
- An Area Authority outside the request's area group is told which group must open it.

### Decision queue (Area Authorities only)
- Lists reviews waiting for a decision **in your own area group only**.
- For the chosen review it shows the recommendation, both figures, the requirements, the triggers,
  the live permits, any flagged sentences, and a section showing each agent's write-up.
- **Your decision** offers only the decisions open, with a reason of at least 10 characters. If
  *you* opened the review, the form is replaced by a note: the other Area Authority of your group
  must decide it.
- After recording, a green message reports the outcome. For example: "…is issued: the work may go
  ahead…"

### Audit trail
- **Permit reviews:** filter by status (the decided statuses are selected at first) and select a
  row to see the whole record: the recommendation, the decision and reason, each agent's
  write-up, the permit document or rework request, the work party briefing, the SIMOPS brief,
  and the **call budget** (calls recorded vs. expected, total tokens, total latency, and every
  call).
- **Chat log:** the most recent 200 chat turns from everyone.

### Permit chat
- Your own conversation (the last 20 turns). Each answer carries a type badge and its sources.
- It answers from Hexley's documents only, and refuses anything else.

## 4. Try it: sample scenarios

The data below is the **current state of this workspace's database**. States change as people use
the system, and each review can be started only once per revision.

**Already done (to look at in Audit trail / chat):**

| Request | What | Status |
|---|---|---|
| PTW0052106 | Excavation, Central Warehouse | Issued (permit document + briefing) |
| PTW0052144 | Electrical isolation, Central Warehouse | Issued |
| PTW0052135 | Confined space entry, Boiler House | Returned to requester (rework request) |
| PTW0052112 | Excavation + Electrical isolation, Main Substation | Referred for SIMOPS review (SIMOPS brief) |

**Scenario A: issue a permit (Process units).**
1. Sign in as `marcus` and open Request detail for **PTW0052111** (Electrical isolation, Drum
   Filling, assessed ISSUE PERMIT).
2. On the Review tab, start the review.
3. Sign out and sign in as `dara`. In the Decision queue, choose PTW0052111, choose **ISSUE
   PERMIT**, give a reason, and record it.
4. Audit trail → Issued → select it: you will see the permit document, the briefing and
   *Model calls 7 of 7*.

**Scenario B: the opener cannot decide.** Sign in as `dara` and start the review of
**PTW0052149** (Work at height, Distillation Unit). On the Decision queue, `dara` sees *"You
opened this review…"*. Sign in as `imogen` to decide it.

**Scenario C: area group guard.** As `kofi` (Storage areas), open **PTW0052117** (Reactor House A).
The Review tab says it belongs to Process units and offers no button.

**Scenario D: a return.** Review **PTW0052127** (Work at height, Chlorine Unit), assessed RETURN
TO REQUESTER. Its decision options are only RETURN and REFER. Choosing RETURN writes a rework
request listing every requirement to put right (*6 of 6* calls).

**Scenario E: a SIMOPS referral.** **PTW0052129** (Confined space entry, Tank Farm South) has a
flagged gas sentence and a HIGH likelihood. Review it as `marcus`, then refer it as `kofi` or
`ruth`. A SIMOPS brief appears in the Audit trail (*5 of 5* calls).

**Chat questions to try** (any role):

| Question | Expected |
|---|---|
| What does the hot work permit standard ask for before a permit is issued? | PERMIT STANDARD answer citing Permit-Standard-Hot-Work.pdf |
| What does the gas testing guide say the reviewer checks? | STATEMENT GUIDANCE answer |
| Who decides a permit request, and can the person who opened the review decide it? | CONTROL OF WORK answer |
| What is the status of PTW0052106? | Operator: *"PTW0052106 (Excavation) in Central Warehouse is issued. Equipment tag: … Work party size: …"*. Coordinator/Area Authority: the full review summary |
| Will PTW0052106 be suspended? | Operator: the fixed line *"…the control room is not told…"* |
| How many hot work permits have been worked in Reactor House A? | A count from the permit history (only given if ≥ 5) |
| And in Gatehouse? | The follow-up uses the previous question's permit type |
| What is the capital of France? | OUT OF SCOPE refusal |

## 5. Badge colours

| Badge | Colours |
|---|---|
| Decision / assessment | ISSUE PERMIT green · RETURN TO REQUESTER orange · REFER FOR SIMOPS REVIEW violet |
| Status | Awaiting review gray · Awaiting a decision blue · Issued green · Returned orange · Referred violet |
| Likelihood band | HIGH red · MEDIUM yellow · LOW green |
| Requirement result | Met green · Not met red · Not documented orange |
| Sentence type | blue · extension expected red · everything else gray |

## 6. Troubleshooting

| You see | Do this |
|---|---|
| "The PermitWise AI API is not answering at http://localhost:8081" | Start `python3 run_api_service.py` and wait for start-up to finish. |
| "The service cannot start. … is missing; it comes from …" | Run the named script (setup / pipeline / training / indexing). |
| "Your session ended…" | The account signed in somewhere else, or signed out. Sign in again. |
| "The review could not be completed: … Ask it again" (502) | Usually a Gemini error or a bad key. Check `.env`. Nothing was stored, so start the review again. |
| "The decision could not be finished…" | Send the same decision with the same reason again to finish it. |
| 409 "Revision N of request … has already been reviewed" | Each revision is reviewed once; read it on the Review tab. |
| `run_database_setup.py` says a table "was made by an earlier version" | Move `database/permitwise.db` and `database/checkpoint.db` aside, then run the setup again. |
| Tests say no ISSUE PERMIT request is left unreviewed | Reset the database as in the line above. This also clears all demo reviews. |

## 7. Checks performed on the dashboard (10 Nov 2026)

| Check | How | Result |
|---|---|---|
| Every page renders for every role with no exception or error box | Streamlit AppTest against the live API: `dara` (5 pages), `marcus` (4), `liam` (1) | ✅ 10/10 pages clean |
| Real browser rendering of sign-in, queue, detail (all 4 tabs), decision queue, audit trail, chat | Headless Chromium screenshots | ✅ Layout, badges and metric cards render as designed |
| Start a review from Request detail (Coordinator) | AppTest: typed the request and clicked **Start the review** on PTW0052135 and PTW0052112 | ✅ Status became *Awaiting a decision* |
| Decide from the Decision queue (Utilities Area Authority `tomasz`) | AppTest: picked the review, the decision and a reason, then **Record the decision** | ✅ RETURN → *returned to requester*, rework request 238 words, **6 of 6** calls; REFER → *referred for SIMOPS review*, SIMOPS brief stored, **5 of 5** calls |
| Audit trail after decisions | AppTest render | ✅ no errors |
| Operator chat via the chat box | AppTest: "What is the status of PTW0052135?" and "Will PTW0052135 be suspended?" | ✅ status line only; the fixed "not told" line for the prediction |
| Chat refusal and grounding | Direct calls | ✅ standard/guide questions answered and cited by file name; "capital of France" refused |
| Role and sign-in guards | API | ✅ operator → `/queue` 403; wrong password 401 |
| Full marking suite | `pytest tests.py` | ✅ **50 passed** |
