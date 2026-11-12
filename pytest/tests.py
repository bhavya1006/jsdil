import pytest


@pytest.fixture(scope="session")
def processed_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path in [config.CLEAN_PERMIT_HISTORY_TRAIN_FILE,
                          config.CLEAN_PERMIT_HISTORY_EVAL_FILE,
                          config.CLEAN_JOB_LOGS_TRAIN_FILE,
                          config.CLEAN_JOB_LOGS_EVAL_FILE,
                          config.CLEAN_STATEMENT_SENTENCES_FILE]:
            loaded[file_path] = pd.read_csv(file_path, keep_default_na=False, na_values=[""],
                                            low_memory=False)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The cleaned extracts could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on cleaning, features and models reads the files the data "
            "pipeline writes, so the pipeline has to have run and written all five."
        )
    return loaded


@pytest.fixture(scope="session")
def raw_frames():
    import pandas as pd

    from config import config

    loaded = {"problem": None}
    try:
        for file_path in [config.RAW_PERMIT_HISTORY_TRAIN_FILE,
                          config.RAW_PERMIT_HISTORY_EVAL_FILE,
                          config.RAW_JOB_LOGS_TRAIN_FILE,
                          config.RAW_JOB_LOGS_EVAL_FILE,
                          config.RAW_STATEMENT_SENTENCES_FILE]:
            loaded[file_path] = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The raw extracts supplied with the project could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The data folder has to stay as it was supplied."
        )
    return loaded


@pytest.fixture(scope="session")
def saved_models():
    from config import config

    loaded = {"problem": None}
    try:
        from modeling.model_registry import load_feature_schema, load_model

        loaded.update({
            "suspension": load_model(config.SUSPENSION_MODEL_FILE),
            "work_hours": load_model(config.WORK_HOURS_MODEL_FILE),
            "sentence": load_model(config.SENTENCE_MODEL_FILE),
            "suspension_schema": list(load_feature_schema(config.SUSPENSION_SCHEMA_FILE)),
            "work_hours_schema": list(load_feature_schema(config.WORK_HOURS_SCHEMA_FILE)),
        })
        for name, label, method_name in [("suspension", "suspension", "predict_proba"),
                                         ("work_hours", "work hours", "predict"),
                                         ("sentence", "sentence", "predict")]:
            assert callable(getattr(loaded[name], method_name, None)), (
                "load_model returned "
                + type(loaded[name]).__name__
                + " for the "
                + label
                + " model, where it returns the fitted model training saved, which has "
                + method_name
                + "."
            )
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The saved models and their feature schemas could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Model training saves them, and every check on the models scores what "
            "training saved."
        )
    return loaded


@pytest.fixture(scope="session")
def live_files():
    from config import config

    loaded = {"problem": None}
    try:
        from data_pipeline.dataset_loader import load_raw_dataset
        from tools.request_data_tools import (fetch_request, load_live_register,
                                              load_open_requests, load_request_documents)

        loaded.update({
            "raw_requests": load_raw_dataset(config.OPEN_PERMIT_REQUESTS_FILE),
            "requests": load_open_requests(),
            "documents": load_request_documents(),
            "register": load_live_register(),
        })
        for name, column_name in [("raw_requests", config.PERMIT_REFERENCE_COLUMN),
                                  ("requests", config.PERMIT_TYPE_COLUMN),
                                  ("documents", config.DOCUMENT_TEXT_COLUMN),
                                  ("register", config.PERMIT_STATUS_COLUMN)]:
            assert column_name in getattr(loaded[name], "columns", []), (
                "The live "
                + name.replace("_", " ")
                + " came back as "
                + type(loaded[name]).__name__
                + " where they are a table holding the column "
                + column_name
                + "."
            )
        first_reference = loaded["requests"][config.PERMIT_REFERENCE_COLUMN].iloc[0]
        first_request = fetch_request(loaded["requests"], first_reference)
        assert isinstance(first_request, dict) and first_request.get(
            config.PERMIT_REFERENCE_COLUMN) == first_reference, (
            "fetch_request returned "
            + str(first_request)[:200]
            + " for "
            + str(first_reference)
            + ", where it returns that request's answers keyed by their columns."
        )
    except (Exception, SystemExit) as error:
        loaded["problem"] = (
            "The live files could not be loaded ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). The queue, the screening of a live request and the workflow all start from "
            "the open permit requests, the documents attached to them and the live permit "
            "register, read through the same cleaning as the extracts."
        )
    return loaded


@pytest.fixture(scope="session")
def knowledge_collection():
    from config import config

    opened = {"problem": None}
    try:
        from knowledge.vector_index import open_vector_index

        opened["collection"] = open_vector_index(config.VECTOR_STORE_DIRECTORY)
        assert callable(getattr(opened["collection"], "query", None)), (
            "open_vector_index returned "
            + type(opened["collection"]).__name__
            + " where it returns the collection the passages are stored in."
        )
    except (Exception, SystemExit) as error:
        opened["problem"] = (
            "The knowledge index could not be opened ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Knowledge indexing builds it, and every check on retrieval reads it."
        )
    return opened


@pytest.fixture(scope="session")
def document_texts():
    from config import config

    read = {"problem": None}
    try:
        from knowledge.pdf_extraction import extract_text_from_directory

        read["documents"] = extract_text_from_directory(config.DOCS_DIRECTORY)
        assert isinstance(read["documents"], list) and len(read["documents"]) > 0, (
            "extract_text_from_directory returned "
            + type(read["documents"]).__name__
            + " where it returns one entry for each PDF in the folder."
        )
        unkeyed = [entry for entry in read["documents"]
                   if not isinstance(entry, dict)
                   or not {"document_name", "document_text"} <= set(entry)]
        assert len(unkeyed) == 0, (
            str(len(unkeyed))
            + " entries read from the document folder do not name their document and carry its "
            "text, for example "
            + str(unkeyed[:1])[:200]
            + "."
        )
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "The document folder could not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on the knowledge base starts from the text of the sixteen "
            "documents."
        )
    return read


@pytest.fixture(scope="session")
def document_passages(document_texts):
    cut = {"problem": document_texts["problem"]}
    if cut["problem"] is not None:
        return cut
    try:
        from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections

        cut["documents"] = document_texts["documents"]
        cut["sections"] = split_documents_into_sections(cut["documents"])
        cut["chunks"] = build_chunk_records(cut["sections"])
        assert isinstance(cut["sections"], list) and isinstance(cut["chunks"], list), (
            "Splitting the documents returned "
            + type(cut["sections"]).__name__
            + " and cutting the sections "
            + type(cut["chunks"]).__name__
            + ", where each returns a list of entries."
        )
        for entry_name, entries, keys in [
                ("section", cut["sections"], ["document_name", "section_number", "section_text"]),
                ("passage", cut["chunks"], ["chunk_id", "chunk_text", "document_name"])]:
            unkeyed = [entry for entry in entries
                       if not isinstance(entry, dict) or not set(keys) <= set(entry)]
            assert len(entries) > 0 and len(unkeyed) == 0, (
                "Every "
                + entry_name
                + " carries "
                + str(keys)
                + ", but "
                + str(len(unkeyed))
                + " of the "
                + str(len(entries))
                + " do not, for example "
                + str(unkeyed[:1])[:200]
                + "."
            )
    except (Exception, SystemExit) as error:
        cut["problem"] = (
            "The documents could not be split into sections and cut into passages ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on the knowledge base starts from the passages the sixteen "
            "documents are cut into."
        )
    return cut


@pytest.fixture(scope="session")
def permit_standards(document_texts):
    from config import config

    read = {"problem": document_texts["problem"], "standards": {}}
    if read["problem"] is not None:
        return read
    try:
        from knowledge.standard_reading import read_permit_standard

        for permit_type in config.PERMIT_STANDARD_DOCUMENTS:
            read["standards"][permit_type] = read_permit_standard(
                permit_type, document_texts["documents"])
            assert isinstance(read["standards"][permit_type], dict) and isinstance(
                read["standards"][permit_type].get("requirements"), list), (
                "read_permit_standard returned "
                + str(read["standards"][permit_type])[:200]
                + " for "
                + permit_type
                + ", where it returns the standard with its requirements."
            )
            unkeyed = [requirement for requirement
                       in read["standards"][permit_type]["requirements"]
                       if not isinstance(requirement, dict)
                       or not {"requirement_kind", "requirement_name", "requirement_sentence",
                               "rules"} <= set(requirement)]
            assert len(unkeyed) == 0, (
                "The standard read for "
                + permit_type
                + " holds requirements without their kind, name, sentence and rules, for "
                "example "
                + str(unkeyed[:1])[:200]
                + "."
            )
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "The permit standards could not be read from their documents ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Each request is screened against the requirements of its own permit type's "
            "standard, and of its linked type's, read from the sections of that standard's own "
            "document."
        )
    return read


@pytest.fixture(scope="session")
def invented_standards():
    from config import config

    invented = {
        config.HOT_WORK: [
            (config.RISK_ASSESSMENT_REQUIREMENT, "Risk assessment",
             "A task risk assessment for the job is attached to the request."),
            (config.FLAMMABLE_GAS_REQUIREMENT, "Flammable gas",
             "The flammable gas reading at the work point is at most 4 percent of the lower "
             "explosive limit."),
            (config.OXYGEN_REQUIREMENT, "Oxygen",
             "The oxygen reading is at least 19.8 percent and at most 22.6 percent."),
            (config.COMBUSTIBLES_CLEARED_REQUIREMENT, "Combustibles cleared",
             "Combustible materials are cleared to at least 11.5 metres around the work."),
            (config.FIRE_WATCH_REQUIREMENT, "Fire watch",
             "A fire watch stays for at least 45 minutes after the work ends.")],
        config.CONFINED_SPACE_ENTRY: [
            (config.RISK_ASSESSMENT_REQUIREMENT, "Risk assessment",
             "A task risk assessment for the job is attached to the request."),
            (config.FLAMMABLE_GAS_REQUIREMENT, "Flammable gas",
             "The flammable gas reading at the work point is at most 7 percent of the lower "
             "explosive limit."),
            (config.OXYGEN_REQUIREMENT, "Oxygen",
             "The oxygen reading is at least 19.7 percent and at most 22.9 percent."),
            (config.HYDROGEN_SULPHIDE_REQUIREMENT, "Hydrogen sulphide",
             "The hydrogen sulphide reading is at most 4.5 ppm."),
            (config.CARBON_MONOXIDE_REQUIREMENT, "Carbon monoxide",
             "The carbon monoxide reading is at most 25 ppm."),
            (config.RESCUE_PLAN_REQUIREMENT, "Rescue cover",
             "A rescue plan is in place with a standby person at the entry.")],
        config.ELECTRICAL_ISOLATION: [
            (config.PROVED_DEAD_REQUIREMENT, "Proved dead",
             "The isolation is proved dead at the point of work."),
            (config.CIRCUIT_EARTHS_REQUIREMENT, "Circuit earths",
             "On a system above 650 volts, circuit earths are applied at the point of work."),
            (config.PERSONAL_LOCKS_REQUIREMENT, "Personal locks",
             "Every member of the work party has applied a personal lock to the isolation."),
            (config.RISK_ASSESSMENT_REQUIREMENT, "Risk assessment",
             "A task risk assessment for the job is attached to the request.")],
        config.WORK_AT_HEIGHT: [
            (config.RISK_ASSESSMENT_REQUIREMENT, "Risk assessment",
             "A task risk assessment for the job is attached to the request."),
            (config.SCAFFOLD_INSPECTION_REQUIREMENT, "Scaffold tag",
             "A scaffold used for access is inspected and tagged."),
            (config.FALL_PROTECTION_REQUIREMENT, "Fall protection",
             "Fall protection is fitted, as edge protection or a harness clipped to an anchor."),
            (config.WIND_SPEED_REQUIREMENT, "Wind speed",
             "Outdoors, the wind at the work position is at most 9.5 metres per second."),
            (config.BARRIERS_REQUIREMENT, "Barriers",
             "The area below or around the work is barriered.")],
        config.EXCAVATION: [
            (config.SERVICES_LOCATED_REQUIREMENT, "Services located",
             "Underground services are located before digging starts."),
            (config.TRENCH_SUPPORT_REQUIREMENT, "Trench support",
             "A trench or pit deeper than 1.5 metres has its sides supported."),
            (config.BARRIERS_REQUIREMENT, "Barriers",
             "The area below or around the work is barriered."),
            (config.RISK_ASSESSMENT_REQUIREMENT, "Risk assessment",
             "A task risk assessment for the job is attached to the request.")],
    }
    documents = []
    for position, (permit_type, requirements) in enumerate(invented.items(), start=1):
        requirement_lines = []
        for number, requirement in enumerate(requirements, start=1):
            requirement_lines.append(str(number) + "\n" + requirement[1] + ". " + requirement[2])
        documents.append({
            "document_name": config.PERMIT_STANDARD_DOCUMENTS[permit_type],
            "document_text": (
                "HEXLEY CHEMICALS\nPermit Standard: "
                + permit_type.title()
                + "\nDocument Standard 9"
                + str(position)
                + " | Issue 1 | Owner: Process Safety | Review cycle: every twelve months\n1. "
                "Purpose and scope\nThis standard is written for the checks alone. Its figures "
                "belong to no standard of Hexley Chemicals.\n2. What this permit covers\nThis "
                "standard covers the work its permit type names, anywhere at Thornmere Works.\nThe "
                "permit type is "
                + permit_type
                + ".\n3. Who requests\nA request is made by the performing authority who will "
                "lead the job.\n4. Requirements\nEvery requirement below must be Met for an Area "
                "Authority to issue the permit. Each requirement is stated once, in one sentence, "
                "and a request is screened against all of them, in this order:\n"
                + "\n".join(requirement_lines)
                + "\nA requirement is never met by a sentence in a statement, and is never "
                "waived.\n5. Reading the requirements\nEach requirement is Met, Not met or Not "
                "documented on the request as the Permit Rules read it.\n"),
        })
    read = {"problem": None, "invented": invented, "documents": documents, "standards": {}}
    try:
        from knowledge.standard_reading import read_permit_standard

        for permit_type in invented:
            read["standards"][permit_type] = read_permit_standard(permit_type, documents)
            standard = read["standards"][permit_type]
            assert (isinstance(standard, dict) and isinstance(standard.get("requirements"), list)
                    and all(isinstance(requirement, dict)
                            for requirement in standard["requirements"])), (
                "read_permit_standard returned "
                + str(standard)[:200]
                + " for "
                + permit_type
                + ", where it returns the standard with its requirements, each a record of its "
                "own."
            )
    except (Exception, SystemExit) as error:
        read["problem"] = (
            "Five standards written for the checks, every figure different from Hexley's, could "
            "not be read ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). They are laid out as Hexley's standards are, and each permit type is read from "
            "its own standard among them with the project's own reader."
        )
    return read


@pytest.fixture(scope="session")
def service_client():
    service = {"problem": None}
    try:
        from fastapi.testclient import TestClient

        from api.routes import service_application

        service["client"] = TestClient(service_application, raise_server_exceptions=False)
    except (Exception, SystemExit) as error:
        service["problem"] = (
            "The REST service could not be started ("
            + type(error).__name__
            + ": "
            + str(error)[:300]
            + "). Every check on sign-in, access, the queue, the workflow and the chat goes "
            "through it."
        )
    return service


@pytest.fixture(scope="session")
def staff(service_client):
    from config import config

    signed_in = {"problem": service_client["problem"], "authorities": {}}
    if signed_in["problem"] is not None:
        return signed_in
    try:
        coordinators = [account for account in config.SEED_USER_ACCOUNTS
                        if account["role"] == config.ROLE_PERMIT_COORDINATOR]
        operators = [account for account in config.SEED_USER_ACCOUNTS
                     if account["role"] == config.ROLE_CONTROL_ROOM_OPERATOR]
        authorities = [account for account in config.SEED_USER_ACCOUNTS
                       if account["role"] == config.ROLE_AREA_AUTHORITY]
        for staff_key, account in ([("coordinator", coordinators[0]), ("operator", operators[0])]
                                   + [(account["username"], account) for account in authorities]):
            reply = service_client["client"].post(
                "/auth/signin",
                json={"username": account["username"], "password": account["password"]})
            assert reply.status_code == 200, (
                "Signing in as "
                + account["username"]
                + " with the password the configuration seeds returned "
                + str(reply.status_code)
                + ", so no check that needs a signed-in "
                + account["role"]
                + " can run."
            )
            signed = dict(reply.json(), headers={
                config.SESSION_TOKEN_HEADER: reply.json()["session_token"]})
            if account["role"] == config.ROLE_AREA_AUTHORITY:
                signed_in["authorities"].setdefault(account["area_group"], []).append(signed)
            else:
                signed_in[staff_key] = signed
    except (Exception, SystemExit) as error:
        signed_in["problem"] = (
            "The seeded staff could not be signed in ("
            + type(error).__name__
            + ": "
            + str(error)[:400]
            + "). The accounts are the ones the configuration seeds through the database "
            "setup, and a sign-in answers with the user's display name, role, area group and a "
            "session token."
        )
    return signed_in


@pytest.fixture(scope="session", autouse=True)
def language_model_requests(request):
    from concurrent.futures import ThreadPoolExecutor

    from config import config

    question = "What does the hot work permit standard ask for before a permit is issued?"
    started = {"review": {"problem": "The review run was never started."},
               "chat": {"problem": "The chat question was never asked.", "question": question,
                        "latest_turn_before": 0}}
    wanted = set()
    for item in request.session.items:
        wanted.update({"saved_run", "chat_replies"} & set(item.fixturenames))
    with ThreadPoolExecutor(max_workers=2) as executor:
        service_client = {"problem": "The REST service was never started."}
        staff = {"problem": None}
        if wanted:
            service_client = request.getfixturevalue("service_client")
            staff = request.getfixturevalue("staff")
        service_ready = service_client["problem"] is None and staff["problem"] is None
        if "saved_run" in wanted and service_ready:
            client = service_client["client"]
            coordinator = staff["coordinator"]
            review = {"problem": None,
                      "coordinator_request": ("Are the requirements of both standards met on "
                                              "the request, and does any document disagree "
                                              "with it?")}
            try:
                queue = client.get("/queue", params={"permit_type": config.QUEUE_PERMIT_TYPE_ALL},
                                   headers=coordinator["headers"])
                assert queue.status_code == 200 and isinstance(queue.json(), list), (
                    "The request queue returned "
                    + str(queue.status_code)
                    + " with "
                    + queue.text[:300]
                    + ", so no request could be chosen to review."
                )
                reviewed = client.get("/reviews",
                                      params={"status_value": config.REVIEW_STATUS_OPTIONS},
                                      headers=coordinator["headers"])
                assert reviewed.status_code == 200, (
                    "Listing the permit reviews returned "
                    + str(reviewed.status_code)
                    + " with "
                    + reviewed.text[:300]
                    + ", so a request nobody has reviewed yet could not be chosen."
                )
                already_reviewed = {(record["permit_reference"], record["revision_number"])
                                    for record in reviewed.json()}
                choices = []
                for queue_row in queue.json():
                    if (queue_row["assessment"] == config.DECISION_ISSUE
                            and (queue_row["permit_reference"], queue_row["revision_number"])
                            not in already_reviewed):
                        choices.append((queue_row["document_count"],
                                        queue_row["permit_reference"], queue_row["plant_area"],
                                        queue_row["revision_number"]))
                assert len(choices) > 0, (
                    "No request in the queue whose assessment is ISSUE PERMIT is left "
                    "unreviewed, so the one review run cannot be opened. Start again from an "
                    "empty database."
                )
                chosen = sorted(choices)[0]
                review["permit_reference"] = chosen[1]
                review["revision_number"] = chosen[3]
                appointed = [group for group, plant_areas in config.AREA_GROUPS.items()
                             if chosen[2] in plant_areas]
                assert len(appointed) == 1 and len(staff["authorities"].get(appointed[0],
                                                                            [])) >= 2, (
                    "The plant area of "
                    + chosen[1]
                    + ", "
                    + str(chosen[2])
                    + ", belongs to the area groups "
                    + str(appointed)
                    + ", where every plant area belongs to one group with two Area Authorities "
                    "appointed for it."
                )
                review["opener"] = staff["authorities"][appointed[0]][0]
                review["decider"] = staff["authorities"][appointed[0]][1]
                review["outsider"] = [signed for group, signed_list
                                      in staff["authorities"].items()
                                      if group != appointed[0] for signed in signed_list][0]
                reply = client.get("/queue/" + review["permit_reference"],
                                   headers=coordinator["headers"])
                assert reply.status_code == 200, (
                    "The detail of "
                    + review["permit_reference"]
                    + " returned "
                    + str(reply.status_code)
                    + " with "
                    + reply.text[:300]
                    + ", so what the review starts from cannot be read."
                )
                review["detail"] = reply.json()
                from persistence.database import session_factory
                from persistence.models import AgentRun

                with session_factory() as session:
                    review["runs_before"] = session.query(AgentRun).filter(
                        AgentRun.permit_reference == review["permit_reference"]).count()
                review["opened"] = executor.submit(
                    client.post, "/reviews",
                    json={"permit_reference": review["permit_reference"],
                          "coordinator_request": review["coordinator_request"]},
                    headers=review["opener"]["headers"])
            except (Exception, SystemExit) as error:
                review["problem"] = (
                    "The one review run that the agent, workflow and record checks read could "
                    "not be opened ("
                    + type(error).__name__
                    + ": "
                    + str(error)[:600]
                    + ")."
                )
            started["review"] = review
        if "chat_replies" in wanted and service_ready:
            client = service_client["client"]
            asker = staff["coordinator"]
            chat = {"problem": None, "question": question, "latest_turn_before": 0}
            try:
                earlier = client.get("/chat/turns", params={"turn_limit": 1},
                                     headers=asker["headers"])
                if earlier.status_code == 200:
                    chat["latest_turn_before"] = max([turn["turn_id"] for turn in earlier.json()]
                                                     + [0])
                chat["asked"] = executor.submit(client.post, "/chat/ask",
                                                json={"question": question},
                                                headers=asker["headers"])
            except (Exception, SystemExit) as error:
                chat["problem"] = (
                    "The chat question could not be asked ("
                    + type(error).__name__
                    + ": "
                    + str(error)[:300]
                    + ")."
                )
            started["chat"] = chat
        yield started


@pytest.fixture(scope="session")
def saved_run(language_model_requests, service_client, staff):
    run = {"problem": service_client["problem"] or staff["problem"], "decision_problem": None}
    if run["problem"] is not None:
        return run
    from config import config

    client = service_client["client"]
    coordinator = staff["coordinator"]
    review = language_model_requests["review"]
    run["problem"] = review["problem"]
    if run["problem"] is not None:
        return run
    for key in ["coordinator_request", "permit_reference", "revision_number", "detail",
                "runs_before", "opener", "decider", "outsider"]:
        run[key] = review[key]
    opener, decider, outsider = run["opener"], run["decider"], run["outsider"]
    run["decision_reason"] = ("Every requirement of the standard is met on the request and no "
                              "SIMOPS trigger applies, so the work may go ahead as requested.")
    for name in ["calls_at_gate", "short_reason", "unknown_decision", "decided", "final_calls",
                 "final_record", "second_review", "short_request"]:
        run[name] = {"status": None, "body": "never requested"}
    for name in ["own_decision", "outside_decision"]:
        run[name] = {"status": None, "body": "never requested", "calls_before": None,
                     "calls_after": None}
    run["run_record"] = {"problem": "never read"}
    run["heard_at_gate"] = {"problem": "never asked"}
    run["heard_after"] = {"problem": "never asked"}
    try:
        opened = review["opened"].result()
        assert opened.status_code == 201, (
            "Opening a review of "
            + run["permit_reference"]
            + " returned "
            + str(opened.status_code)
            + " with "
            + opened.text[:400]
            + ", so the workflow never reached an Area Authority."
        )
        run["at_gate"] = opened.json()
        review_id = run["at_gate"]["review_id"]
    except (Exception, SystemExit) as error:
        run["problem"] = (
            "The one review run that the agent, workflow and record checks read could not be "
            "opened ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )
        return run

    calls_path = "/reviews/" + review_id + "/calls"
    decision_path = "/reviews/" + review_id + "/decision"
    try:
        reply = client.get(calls_path, headers=coordinator["headers"])
        run["calls_at_gate"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["calls_at_gate"]["body"] = type(error).__name__ + ": " + str(error)[:300]

    try:
        from chat.record_answer import answer_request_record
        from persistence.database import session_factory
        from tools.request_data_tools import load_open_requests

        open_requests = load_open_requests()
        heard = {"problem": None}
        with session_factory() as session:
            for heard_key, asker, question_type in [
                    ("operator_status", staff["operator"], config.CHAT_TYPE_REQUEST_STATUS),
                    ("operator_prediction", staff["operator"],
                     config.CHAT_TYPE_OUTCOME_PREDICTION),
                    ("team", coordinator, config.CHAT_TYPE_REQUEST_STATUS)]:
                heard[heard_key] = answer_request_record(
                    session, asker, open_requests, question_type, run["permit_reference"])
        run["heard_at_gate"] = heard
    except (Exception, SystemExit) as error:
        run["heard_at_gate"] = {"problem": (
            "What is heard about the request while the review waits could not be worked out ("
            + type(error).__name__ + ": " + str(error)[:300] + ")."
        )}

    try:
        for refused_key, refused_by in [("own_decision", opener), ("outside_decision", outsider)]:
            before = client.get(calls_path, headers=coordinator["headers"]).json()
            reply = client.post(decision_path, json={"decision": config.DECISION_ISSUE,
                                                     "decision_reason": run["decision_reason"]},
                                headers=refused_by["headers"])
            after = client.get(calls_path, headers=coordinator["headers"]).json()
            run[refused_key] = {"status": reply.status_code, "body": reply.json(),
                                "calls_before": before.get("call_count"),
                                "calls_after": after.get("call_count")}
        for refusal_key, payload in [
                ("short_reason", {"decision": config.DECISION_ISSUE,
                                  "decision_reason": "Fine."}),
                ("unknown_decision", {"decision": "PROCEED",
                                      "decision_reason": run["decision_reason"]})]:
            reply = client.post(decision_path, json=payload, headers=decider["headers"])
            run[refusal_key] = {"status": reply.status_code, "body": reply.json()}
        reply = client.post(decision_path, json={"decision": config.DECISION_ISSUE,
                                                 "decision_reason": run["decision_reason"]},
                            headers=decider["headers"])
        run["decided"] = {"status": reply.status_code, "body": reply.json()}
        assert reply.status_code == 200, (
            "Issuing the permit as "
            + decider["username"]
            + ", an Area Authority appointed for the plant area who did not open the review, "
            "returned "
            + str(reply.status_code)
            + " with "
            + reply.text[:400]
            + "."
        )
    except (Exception, SystemExit) as error:
        run["decision_problem"] = (
            "The decision on the one review run could not be recorded ("
            + type(error).__name__
            + ": "
            + str(error)[:600]
            + ")."
        )

    try:
        reply = client.get(calls_path, headers=coordinator["headers"])
        run["final_calls"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/reviews", params={"status_value": config.REVIEW_STATUS_OPTIONS},
                           headers=coordinator["headers"])
        matching = [record for record in reply.json() if record["review_id"] == review_id]
        run["final_record"] = {"status": reply.status_code, "body": matching}
        reply = client.post("/reviews",
                            json={"permit_reference": run["permit_reference"],
                                  "coordinator_request": run["coordinator_request"]},
                            headers=coordinator["headers"])
        run["second_review"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.post("/reviews",
                            json={"permit_reference": run["permit_reference"],
                                  "coordinator_request": "ok"},
                            headers=coordinator["headers"])
        run["short_request"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        run["final_record"]["body"] = type(error).__name__ + ": " + str(error)[:300]

    try:
        from persistence.database import session_factory
        from persistence.models import AgentRun

        with session_factory() as session:
            stored_run = session.get(AgentRun, run["at_gate"]["run_id"])
            run["run_record"] = {"problem": None, "status": stored_run.status,
                                 "llm_call_count": stored_run.llm_call_count,
                                 "thread_id": stored_run.thread_id,
                                 "runs_opened": session.query(AgentRun).filter(
                                     AgentRun.permit_reference
                                     == run["permit_reference"]).count()
                                 - run["runs_before"]}
    except (Exception, SystemExit) as error:
        run["run_record"] = {"problem": (
            "The run's own record could not be read (" + type(error).__name__ + ": "
            + str(error)[:300] + ")."
        )}

    try:
        from chat.record_answer import answer_request_record
        from persistence.database import session_factory
        from tools.request_data_tools import load_open_requests

        open_requests = load_open_requests()
        heard = {"problem": None}
        with session_factory() as session:
            for heard_key, asker, question_type in [
                    ("operator_status", staff["operator"], config.CHAT_TYPE_REQUEST_STATUS),
                    ("operator_prediction", staff["operator"],
                     config.CHAT_TYPE_OUTCOME_PREDICTION),
                    ("team", opener, config.CHAT_TYPE_OUTCOME_PREDICTION)]:
                heard[heard_key] = answer_request_record(
                    session, asker, open_requests, question_type, run["permit_reference"])
        run["heard_after"] = heard
    except (Exception, SystemExit) as error:
        run["heard_after"] = {"problem": (
            "What is heard about the request after the decision could not be worked out ("
            + type(error).__name__ + ": " + str(error)[:300] + ")."
        )}
    return run


@pytest.fixture(scope="session")
def chat_replies(language_model_requests, service_client, staff):
    chat = language_model_requests["chat"]
    replies = {"problem": service_client["problem"] or staff["problem"] or chat["problem"],
               "standard_question": chat["question"],
               "standard": {"status": None, "body": "never asked"},
               "turns": {"status": None, "body": "never read"},
               "latest_turn_before": chat["latest_turn_before"]}
    if replies["problem"] is not None:
        return replies
    client = service_client["client"]
    asker = staff["coordinator"]
    replies["asker"] = asker["username"]
    replies["someone_else"] = staff["operator"]["username"]
    try:
        reply = chat["asked"].result()
        replies["standard"] = {"status": reply.status_code, "body": reply.json()}
        reply = client.get("/chat/turns", params={"turn_limit": 5}, headers=asker["headers"])
        replies["turns"] = {"status": reply.status_code, "body": reply.json()}
    except (Exception, SystemExit) as error:
        replies["standard"]["body"] = type(error).__name__ + ": " + str(error)[:300]
    return replies


def test_cleaning_keeps_latest_revision_once(raw_frames, processed_frames, tmp_path):
    import os

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    try:
        cleaned_again = clean_history_dataset(
            load_raw_dataset(config.RAW_PERMIT_HISTORY_EVAL_FILE))
    except Exception as error:
        cleaned_again = error
    assert isinstance(cleaned_again, pd.DataFrame), (
        "clean_history_dataset gave "
        + repr(cleaned_again)[:200]
        + " for "
        + config.RAW_PERMIT_HISTORY_EVAL_FILE
        + ", where it returns the cleaned table."
    )
    rebuilt_path = str(tmp_path / "permit_history_eval_cleaned_again.csv")
    save_processed_dataset(cleaned_again, rebuilt_path)
    assert os.path.isfile(rebuilt_path), (
        "save_processed_dataset wrote nothing to "
        + rebuilt_path
        + ", where it writes the table it is given to the path it is given."
    )
    rebuilt = pd.read_csv(rebuilt_path, keep_default_na=False, na_values=[""], low_memory=False)
    saved = processed_frames[config.CLEAN_PERMIT_HISTORY_EVAL_FILE]
    assert rebuilt.shape == saved.shape and rebuilt.equals(saved), (
        "Cleaning "
        + config.RAW_PERMIT_HISTORY_EVAL_FILE
        + " again gives "
        + str(rebuilt.shape)
        + " rows and columns where "
        + config.CLEAN_PERMIT_HISTORY_EVAL_FILE
        + " holds "
        + str(saved.shape)
        + ", or the same shape with different values. Every check reads the cleaned files, "
        "so they have to be what the cleaning code produces now: run the data pipeline "
        "again after changing it."
    )
    for clean_path in [config.CLEAN_PERMIT_HISTORY_TRAIN_FILE,
                       config.CLEAN_PERMIT_HISTORY_EVAL_FILE]:
        clean_frame = processed_frames[clean_path]
        repeated = int(clean_frame[config.PERMIT_REFERENCE_COLUMN].duplicated().sum())
        assert repeated == 0, (
            clean_path
            + " still holds "
            + str(repeated)
            + " rows whose permit reference appears on another row. A request sent again is "
            "one request, and a model fitted on it twice weighs it twice."
        )
        outcomes = sorted(str(value) for value in
                          clean_frame[config.HISTORY_TARGET_COLUMN].dropna().unique())
        unanswered = int(clean_frame[config.HISTORY_TARGET_COLUMN].isna().sum())
        assert unanswered == 0 and outcomes == sorted(config.SUSPENSION_TARGET_ENCODING), (
            "The outcome column of "
            + clean_path
            + " holds "
            + str(unanswered)
            + " blanks and the outcomes "
            + str(outcomes[:6])
            + " where every spelling is folded onto "
            + str(sorted(config.SUSPENSION_TARGET_ENCODING))
            + " and no row is kept without one."
        )
    raw_frame = raw_frames[config.RAW_PERMIT_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    templates = raw_frame[answered].drop_duplicates(config.PERMIT_REFERENCE_COLUMN).head(2)
    rows = pd.concat([templates.iloc[[0]]] * 3 + [templates.iloc[[1]]] * 2
                     + [templates.iloc[[0]]] * 2, ignore_index=True)
    rows[config.PERMIT_REFERENCE_COLUMN] = ["PTW-950001", "PTW0950001", "PTW0950001", "950002",
                                            "950002", "PTW0950003", "PTW-950003"]
    rows[config.REVISION_NUMBER_COLUMN] = ["1", "2", "2", "1", "1", "1", "2"]
    rows[config.PLANNED_WORK_HOURS_COLUMN] = ["11", "12", "12", "13", "13", "14", "15"]
    rows.loc[6, config.HISTORY_TARGET_COLUMN] = "N/A"
    planted_path = str(tmp_path / "planted_revisions.csv")
    rows.iloc[[2, 6, 3, 0, 5, 4, 1]].to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    needed = [config.PERMIT_REFERENCE_COLUMN, config.REVISION_NUMBER_COLUMN,
              config.PLANNED_WORK_HOURS_COLUMN]
    assert isinstance(cleaned, pd.DataFrame) and set(needed) <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for seven planted rows, where it returns the cleaned table with the columns "
        + str(needed)
        + "."
    )
    got = sorted(zip(cleaned[config.PERMIT_REFERENCE_COLUMN].map(str),
                     cleaned[config.REVISION_NUMBER_COLUMN].map(str),
                     cleaned[config.PLANNED_WORK_HOURS_COLUMN].map(str), strict=True))
    expected = [("PTW0950001", "2", "12.0"), ("PTW0950002", "1", "13.0")]
    assert got == expected, (
        "Seven planted rows - a request written PTW-950001 at revision 1 and PTW0950001 at "
        "revision 2, its second revision repeated exactly and standing first in the file; a "
        "request written 950002 repeated exactly; and a request answered at revision 1, written "
        "PTW0950003, whose revision 2, written PTW-950003, records no outcome yet - each telling "
        "its revisions apart by its planned work hours, cleaned to "
        + str(got)
        + " where they are "
        + str(expected)
        + ". A row repeated exactly is one row, the highest revision is the request, and a "
        "request whose highest revision records no outcome is not kept: an outcome is never "
        "taken from an earlier revision."
    )


def test_readings_made_standard_units(raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_frame = raw_frames[config.RAW_PERMIT_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_frame[answered].drop_duplicates(config.PERMIT_REFERENCE_COLUMN).head(1)
    flammable, oxygen = "flammable_gas_lel_percent", config.OXYGEN_COLUMN
    sulphide, monoxide = "hydrogen_sulphide_ppm", "carbon_monoxide_ppm"
    cleared, height = "combustibles_cleared_radius_m", "working_height_m"
    depth, wind = config.EXCAVATION_DEPTH_COLUMN, config.WIND_SPEED_COLUMN
    voltage, watch = config.SYSTEM_VOLTAGE_COLUMN, "fire_watch_minutes"
    planned = config.PLANNED_WORK_HOURS_COLUMN
    planted = [(flammable, "3", 3), (flammable, "3%LEL", 3), (flammable, "3 % LEL", 3),
               (flammable, "2.5", 3), (flammable, "2.4%LEL", 2), (flammable, "4.5 % LEL", 5),
               (oxygen, "20.9", 20.9), (oxygen, "20.9%", 20.9), (oxygen, "20.9 % O2", 20.9),
               (oxygen, "19.45%", 19.5), (oxygen, "19.44", 19.4), (oxygen, "23.45 % O2", 23.5),
               (sulphide, "2.5", 2.5), (sulphide, "2.5 ppm", 2.5), (sulphide, "3.55 mg/m3", 2.5),
               (sulphide, "7.2 mg/m3", 5.1), (sulphide, "7.1 mg/m3", 5.0),
               (monoxide, "4.68 mg/m3", 4.0), (monoxide, "23.5 mg/m3", 20.1),
               (monoxide, "7.1 mg/m3", 6.1), (monoxide, "20 ppm", 20.0),
               (cleared, "10", 10.0), (cleared, "10 m", 10.0), (cleared, "33 ft", 10.1),
               (cleared, "32 ft", 9.8), (cleared, "30 ft", 9.1), (height, "3.9 ft", 1.2),
               (height, "9.5 m", 9.5), (height, "2.25", 2.3), (depth, "4.2 ft", 1.3),
               (depth, "1.25", 1.3), (depth, "1.2 m", 1.2), (wind, "6.2", 6.2),
               (wind, "6.2 m/s", 6.2), (wind, "14 mph", 6.3), (wind, "26.8 mph", 12.0),
               (wind, "27 mph", 12.1), (wind, "28 mph", 12.5), (voltage, "415", 415),
               (voltage, "415V", 415), (voltage, "0.415 kV", 415), (voltage, "3.3kV", 3300),
               (voltage, "1kV", 1000), (voltage, "0.4165 kV", 417), (watch, "60", 60),
               (watch, "60 min", 60), (watch, "1 hr", 60), (watch, "1.5 hrs", 90),
               (watch, "0.75 hrs", 45), (watch, "0.175 hrs", 11), (planned, "6.5", 6.5),
               (planned, "390 min", 6.5), (planned, "6.5 hrs", 6.5), (planned, "1 hr", 1.0),
               (planned, "4.35 hrs", 4.4), (planned, "261 min", 4.4)]
    rows = pd.concat([template] * len(planted), ignore_index=True)
    for position, planted_reading in enumerate(planted):
        rows.loc[position, config.PERMIT_REFERENCE_COLUMN] = "PTW09" + str(60001 + position)
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
        rows.loc[position, planted_reading[0]] = planted_reading[1]
    planted_path = str(tmp_path / "planted_readings.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    read_columns = {config.PERMIT_REFERENCE_COLUMN}.union(
        planted_reading[0] for planted_reading in planted)
    assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for "
        + str(len(planted))
        + " planted readings, where it returns the cleaned table with the columns "
        + str(sorted(read_columns))
        + "."
    )
    cleaned = cleaned.set_index(config.PERMIT_REFERENCE_COLUMN)
    wrong = []
    for position, (column_name, written, wanted) in enumerate(planted):
        reference = "PTW09" + str(60001 + position)
        if reference not in cleaned.index:
            wrong.append(column_name + " " + repr(written) + " lost its row")
            continue
        got = cleaned.loc[reference, column_name]
        if pd.isna(got) or got != wanted:
            wrong.append(column_name + " " + repr(written) + " read as " + str(got) + ", not "
                         + str(wanted))
    assert len(wrong) == 0, (
        "Readings are not read as the Permit Rules set out: "
        + "; ".join(wrong[:6])
        + ". Each reading is made its standard's unit - the flammable gas a whole percent of the "
        "lower explosive limit, oxygen a percent, hydrogen sulphide and carbon monoxide ppm, "
        "milligrams per cubic metre divided by 1.42 and by 1.17, lengths metres, feet times "
        "0.3048, wind metres per second, miles per hour times 0.44704, voltages whole volts, "
        "kilovolts times 1,000, the fire watch whole minutes, hours times 60, and work hours "
        "hours, minutes divided by 60 - converted exactly and rounded half up once, so 19.45% "
        "of oxygen is 19.5 and 4.35 hrs is 4.4."
    )
    for column_name, written in [(flammable, "3% LEL"), (flammable, "3 %LEL"),
                                 (oxygen, "20.9 %"), (oxygen, "20.9% O2"),
                                 (sulphide, "2.5ppm"), (sulphide, "3.55 mg/m3 H2S"),
                                 (monoxide, "4.68mg/m3"), (cleared, "10m"), (cleared, "33 feet"),
                                 (height, "3.9ft"), (wind, "14mph"), (wind, "6.2 km/h"),
                                 (voltage, "415 V"), (voltage, "1 kV AC"), (watch, "60mins"),
                                 (watch, "1 hour"), (planned, "390min"), (planned, "6.5 h")]:
        unreadable = template.copy()
        unreadable[config.REVISION_NUMBER_COLUMN] = "1"
        unreadable[column_name] = written
        unreadable_path = str(tmp_path / "unreadable_reading.csv")
        unreadable.to_csv(unreadable_path, index=False)
        refusal = None
        try:
            clean_history_dataset(load_raw_dataset(unreadable_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and column_name in refusal and written in refusal, (
            "Cleaning a "
            + column_name
            + " written "
            + repr(written)
            + " gave "
            + str(refusal)
            + ". A reading in no notation the three channels write - another unit or another "
            "spacing - is refused with a ValueError naming the column and the value as written, "
            "never guessed at or left blank."
        )


def test_references_types_and_places_canonical(raw_frames, processed_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    raw_frame = raw_frames[config.RAW_PERMIT_HISTORY_EVAL_FILE]
    answered = ~raw_frame[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_frame[answered].drop_duplicates(config.PERMIT_REFERENCE_COLUMN).head(1)
    hot, entry = config.HOT_WORK, config.CONFINED_SPACE_ENTRY
    isolation, height = config.ELECTRICAL_ISOLATION, config.WORK_AT_HEIGHT
    dig, unlinked = config.EXCAVATION, config.NO_LINKED_PERMIT
    zone_1, zone_2, safe = config.ZONE_1, config.ZONE_2, config.NON_HAZARDOUS
    operating, shut, no_plant = config.OPERATING, config.SHUT_DOWN, config.NO_PROCESS_PLANT
    inside, outside = config.INDOORS, config.OUTDOORS
    scaffold, platform_lift = config.SCAFFOLD, config.MOBILE_ELEVATING_WORK_PLATFORM
    ladder, fixed, rope = config.LADDER, config.FIXED_PLATFORM, config.ROPE_ACCESS
    columns = [config.PERMIT_TYPE_COLUMN, config.LINKED_PERMIT_TYPE_COLUMN, "area_classification",
               config.PROCESS_UNIT_STATUS_COLUMN, config.WORK_LOCATION_COLUMN,
               config.ACCESS_METHOD_COLUMN]
    planted = [
        ("PTW0970001", "PTW0970001", ["Hot work", "No linked permit", "Zone 1", "Operating",
                                      "Indoors", "Scaffold"],
         [hot, unlinked, zone_1, operating, inside, scaffold]),
        ("PTW-970002", "PTW0970002", ["Hot Work Permit", "No link", "Z1", "Running", "Inside",
                                      "SCAF"],
         [hot, unlinked, zone_1, operating, inside, scaffold]),
        ("970003", "PTW0970003", ["HW", "NIL", "ZONE-1", "OPER", "IN", "scaffold"],
         [hot, unlinked, zone_1, operating, inside, scaffold]),
        ("PTW0970004", "PTW0970004", ["Confined space entry", "Hot work", "Zone 2", "Shut down",
                                      "Outdoors", "Mobile elevating work platform"],
         [entry, hot, zone_2, shut, outside, platform_lift]),
        ("PTW-970005", "PTW0970005", ["Confined Space Permit", "Hot Work Permit", "z2",
                                      "Off-line", "Outside", "MEWP"],
         [entry, hot, zone_2, shut, outside, platform_lift]),
        ("970006", "PTW0970006", ["CSE", "hw", "Zone-2", "S/D", "OUT", "mewp"],
         [entry, hot, zone_2, shut, outside, platform_lift]),
        ("PTW0970007", "PTW0970007", ["Electrical isolation", "Confined space entry",
                                      "Non-hazardous", "No process plant", "indoors", "Ladder"],
         [isolation, entry, safe, no_plant, inside, ladder]),
        ("PTW-970008", "PTW0970008", ["Electrical Isolation Permit", "CSE", "Safe area",
                                      "No plant", "inside", "LADDER"],
         [isolation, entry, safe, no_plant, inside, ladder]),
        ("970009", "PTW0970009", ["EI", "Confined Space Permit", "NON-HAZ", "no process plant",
                                  "in", "Fixed platform"],
         [isolation, entry, safe, no_plant, inside, fixed]),
        ("PTW0970010", "PTW0970010", ["Work at height", "Excavation", "zone 1", "operating",
                                      "outdoors", "Platform"],
         [height, dig, zone_1, operating, outside, fixed]),
        ("PTW-970011", "PTW0970011", ["Work at Height Permit", "Excavation Permit", "Z2",
                                      "RUNNING", "OUTSIDE", "Rope access"],
         [height, dig, zone_2, operating, outside, rope]),
        ("970012", "PTW0970012", ["WAH", "EXC", "safe area", "s/d", "out", "ROPE"],
         [height, dig, safe, shut, outside, rope]),
        ("PTW0970013", "PTW0970013", ["Excavation", "Work at height", "Non-Hazardous",
                                      "Shut Down", "Indoors", "Rope Access"],
         [dig, height, safe, shut, inside, rope]),
        ("PTW-970014", "PTW0970014", ["Excavation Permit", "WAH", "Zone 1", "OFF-LINE",
                                      "Outdoors", "Mobile Elevating Work Platform"],
         [dig, height, zone_1, shut, outside, platform_lift]),
        ("97", "PTW0000097", ["EXC", "Electrical isolation", "Z1", "No Plant", "Inside",
                              "fixed platform"],
         [dig, isolation, zone_1, no_plant, inside, fixed])]
    rows = pd.concat([template] * len(planted), ignore_index=True)
    for position, (written_reference, _wanted_reference, written, _wanted) in enumerate(planted):
        rows.loc[position, config.PERMIT_REFERENCE_COLUMN] = written_reference
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
        for column_name, value in zip(columns, written, strict=True):
            rows.loc[position, column_name] = value
    planted_path = str(tmp_path / "planted_notations.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    assert isinstance(cleaned, pd.DataFrame) and set(
            [config.PERMIT_REFERENCE_COLUMN] + columns) <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for fifteen planted notations, where it returns the cleaned table."
    )
    wrong = []
    for written_reference, wanted_reference, written, wanted in planted:
        matching = cleaned[cleaned[config.PERMIT_REFERENCE_COLUMN] == wanted_reference]
        if len(matching) != 1:
            wrong.append(written_reference + " is kept " + str(len(matching)) + " times as "
                         + wanted_reference)
            continue
        got = [matching[column_name].iloc[0] for column_name in columns]
        if any(pd.isna(value) for value in got) or got != wanted:
            wrong.append(str(written) + " read as " + str(got))
    assert len(wrong) == 0, (
        "References, permit types, classifications, unit statuses, locations and access methods "
        "are not read as the Permit Rules set out: "
        + "; ".join(wrong[:5])
        + ". A reference written PTW and seven digits, PTW- and the number or the number alone "
        "is PTW and the number in seven digits; and each named answer, written in full, as the "
        "contractor app writes it or as scanned paper's code, in any capitals, is its one "
        "spelling: HW, Hot Work Permit and Hot work are Hot work, Z1 and Zone-1 are Zone 1, OPER "
        "and Running are Operating, OUT and Outside are Outdoors, MEWP is Mobile elevating work "
        "platform."
    )
    for column_name, written in [(config.PERMIT_REFERENCE_COLUMN, "PTW52118"),
                                 (config.PERMIT_REFERENCE_COLUMN, "PT-52118"),
                                 (config.PERMIT_REFERENCE_COLUMN, "PTW-12345678"),
                                 (config.PERMIT_REFERENCE_COLUMN, "12345678"),
                                 (config.PERMIT_TYPE_COLUMN, "Hot works"),
                                 (config.LINKED_PERMIT_TYPE_COLUMN, "None"),
                                 ("area_classification", "Zone 0"),
                                 (config.PROCESS_UNIT_STATUS_COLUMN, "Idle"),
                                 (config.WORK_LOCATION_COLUMN, "Roof"),
                                 (config.ACCESS_METHOD_COLUMN, "Cherry picker"),
                                 (config.PLANT_AREA_COLUMN, "Reactor House C"),
                                 ("contractor_reference", "CTR-12345")]:
        unreadable = template.copy()
        unreadable[config.REVISION_NUMBER_COLUMN] = "1"
        unreadable[column_name] = written
        unreadable_path = str(tmp_path / "unreadable_notation.csv")
        unreadable.to_csv(unreadable_path, index=False)
        refusal = None
        try:
            clean_history_dataset(load_raw_dataset(unreadable_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and column_name in refusal and written in refusal, (
            "Cleaning a "
            + column_name
            + " written "
            + repr(written)
            + " gave "
            + str(refusal)
            + ". A value in no notation the three channels write is refused with a ValueError "
            "naming the column and the value as written."
        )
    permit_types = sorted(config.PERMIT_STANDARD_DOCUMENTS)
    for clean_path in [config.CLEAN_PERMIT_HISTORY_TRAIN_FILE,
                       config.CLEAN_PERMIT_HISTORY_EVAL_FILE,
                       config.CLEAN_JOB_LOGS_TRAIN_FILE,
                       config.CLEAN_JOB_LOGS_EVAL_FILE,
                       config.CLEAN_STATEMENT_SENTENCES_FILE]:
        clean_frame = processed_frames[clean_path]
        references = clean_frame[config.PERMIT_REFERENCE_COLUMN].dropna().astype(str)
        stray = sorted(references[~references.str.fullmatch(
            config.CANONICAL_PERMIT_REFERENCE_PATTERN)].unique())
        assert len(references) == len(clean_frame) and len(stray) == 0, (
            clean_path
            + " holds "
            + str(len(clean_frame) - len(references))
            + " blank permit references and "
            + str(len(stray))
            + " that are not canonical, for example "
            + str(stray[:3])
            + "."
        )
        if config.PERMIT_TYPE_COLUMN in clean_frame.columns:
            found_types = sorted(str(value) for value in
                                 clean_frame[config.PERMIT_TYPE_COLUMN].dropna().unique())
            found_links = sorted(str(value) for value in
                                 clean_frame[config.LINKED_PERMIT_TYPE_COLUMN].dropna().unique())
            untyped = int(clean_frame[[config.PERMIT_TYPE_COLUMN,
                                       config.LINKED_PERMIT_TYPE_COLUMN]].isna().sum().sum())
            assert (found_types == permit_types and untyped == 0
                    and found_links == sorted(permit_types + [unlinked])), (
                clean_path
                + " holds the permit types "
                + str(found_types)
                + ", the linked types "
                + str(found_links)
                + " and "
                + str(untyped)
                + " blanks among them, where a permit type is one of the five, a linked type one "
                "of the five or No linked permit, and neither is ever blank."
            )


def test_answers_codes_and_register_read_canonically(raw_frames, processed_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset
    from data_pipeline.job_log_cleaning import clean_job_log_dataset
    from data_pipeline.request_cleaning import clean_register_records
    from data_pipeline.sentence_cleaning import clean_sentence_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    for clean_path, value_mappings in [
            (config.CLEAN_PERMIT_HISTORY_TRAIN_FILE, config.HISTORY_VALUE_MAPPINGS),
            (config.CLEAN_PERMIT_HISTORY_EVAL_FILE, config.HISTORY_VALUE_MAPPINGS),
            (config.CLEAN_JOB_LOGS_TRAIN_FILE, config.LOG_VALUE_MAPPINGS),
            (config.CLEAN_JOB_LOGS_EVAL_FILE, config.LOG_VALUE_MAPPINGS),
            (config.CLEAN_STATEMENT_SENTENCES_FILE, config.SENTENCE_VALUE_MAPPINGS)]:
        clean_frame = processed_frames[clean_path]
        for column_name, value_mapping in value_mappings.items():
            found = set(clean_frame[column_name].dropna().astype(str))
            stray = sorted(found - set(value_mapping.values()))
            assert len(stray) == 0, (
                column_name
                + " in "
                + clean_path
                + " holds "
                + str(stray[:5])
                + ", which are not among its canonical values "
                + str(sorted(set(value_mapping.values()))[:6])
                + ". Every spelling a channel writes is folded onto one."
            )
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    risk, rescue = "risk_assessment_attached", "rescue_plan_in_place"
    dead, locks = "isolation_proved_dead", "personal_locks_applied"
    earths, fall = config.CIRCUIT_EARTHS_COLUMN, "fall_protection_fitted"
    inspected, services = config.SCAFFOLD_INSPECTED_COLUMN, "services_located"
    support, barriered = config.TRENCH_SUPPORT_COLUMN, "work_area_barriered"
    suspended, outcome = config.HISTORY_TARGET_COLUMN, "review_outcome"
    reason = "suspension_reason"
    as_requested, reworked = config.OUTCOME_ISSUED_AS_REQUESTED, config.OUTCOME_ISSUED_AFTER_REWORK
    after_simops = config.OUTCOME_ISSUED_AFTER_SIMOPS_REVIEW
    history_planted = [
        ({risk: "Yes", rescue: "No", outcome: "Issued as requested", suspended: "Yes",
          reason: "Gas alarm"},
         {risk: answered_yes, rescue: answered_no, outcome: as_requested,
          suspended: answered_yes, reason: config.GAS_ALARM}),
        ({risk: "Y", rescue: "N", outcome: "Issued", suspended: "Y", reason: "gas"},
         {risk: answered_yes, rescue: answered_no, outcome: as_requested,
          suspended: answered_yes, reason: config.GAS_ALARM}),
        ({risk: "TRUE", rescue: "FALSE", outcome: "ISS", suspended: "TRUE",
          reason: "Process upset"},
         {risk: answered_yes, rescue: answered_no, outcome: as_requested,
          suspended: answered_yes, reason: config.PROCESS_UPSET}),
        ({risk: "true", rescue: "false", outcome: "Issued after rework", suspended: "true",
          reason: "PROC"},
         {risk: answered_yes, rescue: answered_no, outcome: reworked, suspended: answered_yes,
          reason: config.PROCESS_UPSET}),
        ({risk: "1", rescue: "0", outcome: "Issued - rework", suspended: "1",
          reason: "Site alarm"},
         {risk: answered_yes, rescue: answered_no, outcome: reworked, suspended: answered_yes,
          reason: config.SITE_ALARM}),
        ({risk: "y", rescue: "n", outcome: "ISS-RW", suspended: "y", reason: "alarm"},
         {risk: answered_yes, rescue: answered_no, outcome: reworked, suspended: answered_yes,
          reason: config.SITE_ALARM}),
        ({risk: "yes", rescue: "no", outcome: "Issued after SIMOPS review", suspended: "yes",
          reason: "Conflicting work"},
         {risk: answered_yes, rescue: answered_no, outcome: after_simops,
          suspended: answered_yes, reason: config.CONFLICTING_WORK}),
        ({dead: "YES", locks: "NO", outcome: "Issued - SIMOPS", suspended: "YES",
          reason: "Conflict"},
         {dead: answered_yes, locks: answered_no, outcome: after_simops,
          suspended: answered_yes, reason: config.CONFLICTING_WORK}),
        ({earths: "True", fall: "False", outcome: "iss-sim", suspended: "True",
          reason: "SIMOPS"},
         {earths: answered_yes, fall: answered_no, outcome: after_simops,
          suspended: answered_yes, reason: config.CONFLICTING_WORK}),
        ({inspected: "Yes", services: "No", outcome: "issued as requested", suspended: "Yes",
          reason: "Unsafe condition"},
         {inspected: answered_yes, services: answered_no, outcome: as_requested,
          suspended: answered_yes, reason: config.UNSAFE_CONDITION}),
        ({support: "Y", barriered: "N", outcome: "ISSUED", suspended: "Y", reason: "unsafe"},
         {support: answered_yes, barriered: answered_no, outcome: as_requested,
          suspended: answered_yes, reason: config.UNSAFE_CONDITION}),
        ({risk: "Yes", outcome: "Issued - Rework", suspended: "Yes", reason: "High wind"},
         {risk: answered_yes, outcome: reworked, suspended: answered_yes,
          reason: config.HIGH_WIND}),
        ({rescue: "No", outcome: "Iss", suspended: "Yes", reason: "WIND"},
         {rescue: answered_no, outcome: as_requested, suspended: answered_yes,
          reason: config.HIGH_WIND}),
        ({risk: "No", outcome: "Issued", suspended: "No", reason: "N/A"},
         {risk: answered_no, outcome: as_requested, suspended: answered_no, reason: None}),
        ({risk: "N", outcome: "ISS", suspended: "false", reason: ""},
         {risk: answered_no, outcome: as_requested, suspended: answered_no, reason: None}),
        ({risk: "0", outcome: "Issued", suspended: "0", reason: "-"},
         {risk: answered_no, outcome: as_requested, suspended: answered_no, reason: None})]
    raw_history = raw_frames[config.RAW_PERMIT_HISTORY_EVAL_FILE]
    answered = ~raw_history[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    history_template = raw_history[answered].drop_duplicates(
        config.PERMIT_REFERENCE_COLUMN).head(1)
    log_template = raw_frames[config.RAW_JOB_LOGS_EVAL_FILE].drop_duplicates(
        config.PERMIT_REFERENCE_COLUMN).head(1)
    sentence_template = raw_frames[config.RAW_STATEMENT_SENTENCES_FILE].head(1)
    completed, extended = config.COMPLETED, config.COMPLETED_AFTER_EXTENSION
    after_suspension = config.COMPLETED_AFTER_SUSPENSION
    code = "completion_code"
    log_planted = [
        ({code: "Completed", suspended: "No", support: "Yes"},
         {code: completed, suspended: answered_no, support: answered_yes}),
        ({code: "Done", suspended: "false", support: "false"},
         {code: completed, suspended: answered_no, support: answered_no}),
        ({code: "COMP", suspended: "N", support: "N"},
         {code: completed, suspended: answered_no, support: answered_no}),
        ({code: "Completed after extension", suspended: "No"},
         {code: extended, suspended: answered_no}),
        ({code: "Done - extended", suspended: "0"}, {code: extended, suspended: answered_no}),
        ({code: "COMP-EXT", suspended: "n"}, {code: extended, suspended: answered_no}),
        ({code: "Completed after suspension", suspended: "Yes"},
         {code: after_suspension, suspended: answered_yes}),
        ({code: "Done - suspended", suspended: "true"},
         {code: after_suspension, suspended: answered_yes}),
        ({code: "comp-sus", suspended: "Y"}, {code: after_suspension, suspended: answered_yes}),
        ({code: "done", suspended: "FALSE"}, {code: completed, suspended: answered_no})]
    label, kind = config.SENTENCE_TARGET_COLUMN, config.DOCUMENT_TYPE_COLUMN
    sentence_planted = [
        ({label: "Isolation", kind: "Method statement"},
         {label: config.ISOLATION, kind: config.METHOD_STATEMENT}),
        ({label: "isolations", kind: "TASK RISK ASSESSMENT"},
         {label: config.ISOLATION, kind: config.TASK_RISK_ASSESSMENT}),
        ({label: "Gas testing", kind: "gas test record"},
         {label: config.GAS_TESTING, kind: config.GAS_TEST_RECORD}),
        ({label: "GAS TEST", kind: "Rescue Plan"},
         {label: config.GAS_TESTING, kind: config.RESCUE_PLAN}),
        ({label: "gas-testing"}, {label: config.GAS_TESTING}),
        ({label: "Fire precautions"}, {label: config.FIRE_PRECAUTIONS}),
        ({label: "fire"}, {label: config.FIRE_PRECAUTIONS}),
        ({label: "Fire-Precautions"}, {label: config.FIRE_PRECAUTIONS}),
        ({label: "Rescue"}, {label: config.RESCUE}),
        ({label: "emergency rescue"}, {label: config.RESCUE}),
        ({label: "Access"}, {label: config.ACCESS}),
        ({label: "Access and egress"}, {label: config.ACCESS}),
        ({label: "Personal protective equipment"}, {label: config.PERSONAL_PROTECTIVE_EQUIPMENT}),
        ({label: "PPE"}, {label: config.PERSONAL_PROTECTIVE_EQUIPMENT}),
        ({label: "personal-protective-equipment"}, {label: config.PERSONAL_PROTECTIVE_EQUIPMENT}),
        ({label: "Simultaneous operations"}, {label: config.SIMULTANEOUS_OPERATIONS}),
        ({label: "SIMOPS"}, {label: config.SIMULTANEOUS_OPERATIONS}),
        ({label: "simultaneous-operations"}, {label: config.SIMULTANEOUS_OPERATIONS}),
        ({label: "Housekeeping and waste"}, {label: config.HOUSEKEEPING_AND_WASTE}),
        ({label: "housekeeping & waste"}, {label: config.HOUSEKEEPING_AND_WASTE}),
        ({label: "Housekeeping"}, {label: config.HOUSEKEEPING_AND_WASTE}),
        ({label: "housekeeping-and-waste"}, {label: config.HOUSEKEEPING_AND_WASTE}),
        ({label: "Background"}, {label: config.BACKGROUND}),
        ({label: "context"}, {label: config.BACKGROUND}),
        ({label: "General"}, {label: config.BACKGROUND})]
    raw_register = load_raw_dataset(config.LIVE_PERMIT_REGISTER_FILE)
    area, status = config.PLANT_AREA_COLUMN, config.PERMIT_STATUS_COLUMN
    permit_type, linked = config.PERMIT_TYPE_COLUMN, config.LINKED_PERMIT_TYPE_COLUMN
    register_planted = [
        ({area: "reactor house a", permit_type: "HW", linked: "No link",
          "work_activity": "welding", status: "LIVE"},
         {area: config.REACTOR_HOUSE_A, permit_type: config.HOT_WORK,
          linked: config.NO_LINKED_PERMIT, "work_activity": config.WELDING,
          status: config.PERMIT_STATUS_LIVE}),
        ({area: "TANK FARM SOUTH", permit_type: "Confined Space Permit", linked: "nil",
          "work_activity": "TANK CLEANING", status: "suspended"},
         {area: config.TANK_FARM_SOUTH, permit_type: config.CONFINED_SPACE_ENTRY,
          linked: config.NO_LINKED_PERMIT, "work_activity": config.TANK_CLEANING,
          status: config.PERMIT_STATUS_SUSPENDED}),
        ({area: "Boiler house", permit_type: "excavation", linked: "Hot Work Permit",
          "work_activity": "Trial Hole", status: "Live"},
         {area: config.BOILER_HOUSE, permit_type: config.EXCAVATION, linked: config.HOT_WORK,
          "work_activity": config.TRIAL_HOLE, status: config.PERMIT_STATUS_LIVE})]
    register_references = [("PTW-51207", "PTW0051207"), ("51208", "PTW0051208"),
                           ("PTW0051209", "PTW0051209")]
    wrong = []
    for name, template, planted, key_column, keys, cleaner in [
            ("permit history", history_template, history_planted,
             config.PERMIT_REFERENCE_COLUMN,
             [("PTW09" + str(81001 + position), "PTW09" + str(81001 + position))
              for position in range(len(history_planted))], clean_history_dataset),
            ("job log", log_template, log_planted, config.PERMIT_REFERENCE_COLUMN,
             [("PTW09" + str(82001 + position), "PTW09" + str(82001 + position))
              for position in range(len(log_planted))], clean_job_log_dataset),
            ("statement sentence", sentence_template, sentence_planted,
             config.SENTENCE_ID_COLUMN,
             [("SEN-98" + str(30001 + position), "SEN-98" + str(30001 + position))
              for position in range(len(sentence_planted))], clean_sentence_dataset),
            ("live permit register", raw_register.head(1), register_planted,
             config.PERMIT_REFERENCE_COLUMN, register_references, clean_register_records)]:
        rows = pd.concat([template] * len(planted), ignore_index=True)
        for position, (written, _wanted) in enumerate(planted):
            rows.loc[position, key_column] = keys[position][0]
            if config.REVISION_NUMBER_COLUMN in rows.columns:
                rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
            for column_name, value in written.items():
                rows.loc[position, column_name] = value
        planted_path = str(tmp_path / "planted_answers.csv")
        rows.to_csv(planted_path, index=False)
        try:
            cleaned = cleaner(load_raw_dataset(planted_path))
        except Exception as error:
            cleaned = error
        read_columns = {key_column}.union(*[set(written) for written, _wanted in planted])
        assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
            "Cleaning "
            + str(len(planted))
            + " planted rows of the "
            + name
            + " gave "
            + repr(cleaned)[:200]
            + ", where it returns the cleaned table with the columns "
            + str(sorted(read_columns))
            + "."
        )
        cleaned = cleaned.set_index(key_column)
        for position, (written, wanted) in enumerate(planted):
            key = keys[position][1]
            if key not in cleaned.index:
                wrong.append("the planted " + name + " row " + key + " was lost")
                continue
            for column_name, expected in wanted.items():
                got = cleaned.loc[key, column_name]
                if (expected is None and pd.isna(got)) or (not pd.isna(got) and got == expected):
                    continue
                wrong.append(name + " " + column_name + " " + repr(written[column_name])
                             + " read as " + repr(got) + ", not " + repr(expected))
    assert len(wrong) == 0, (
        "Answers are not read as the Permit Rules set out: "
        + "; ".join(wrong[:6])
        + ". Yes is Yes, Y, TRUE or 1 and no is No, N, FALSE or 0, in any capitals; a review "
        "outcome, a suspension reason, a completion code, a sentence label and a document type "
        "take their one spelling whichever way a channel writes them; a blank is a blank; and "
        "the live permit register's plant areas, permit types, work activities and statuses "
        "are read whatever their capitals."
    )
    for name, template, cleaner, column_name, written in [
            ("permit history", history_template, clean_history_dataset, outcome, "Withdrawn"),
            ("permit history", history_template, clean_history_dataset, reason, "Lightning"),
            ("permit history", history_template, clean_history_dataset, risk, "Maybe"),
            ("permit history", history_template, clean_history_dataset, suspended, "Perhaps"),
            ("job log", log_template, clean_job_log_dataset, code, "Abandoned"),
            ("job log", log_template, clean_job_log_dataset, suspended, "Unknown"),
            ("statement sentence", sentence_template, clean_sentence_dataset, label, "Rumour"),
            ("statement sentence", sentence_template, clean_sentence_dataset, kind, "Drawing"),
            ("live permit register", raw_register.head(1), clean_register_records, status,
             "Closed"),
            ("live permit register", raw_register.head(1), clean_register_records, area,
             "Reactor House C")]:
        unreadable = template.copy()
        if config.REVISION_NUMBER_COLUMN in unreadable.columns:
            unreadable[config.REVISION_NUMBER_COLUMN] = "1"
        unreadable[column_name] = written
        unreadable_path = str(tmp_path / "unreadable_answer.csv")
        unreadable.to_csv(unreadable_path, index=False)
        refusal = None
        try:
            cleaner(load_raw_dataset(unreadable_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and column_name in refusal and written in refusal, (
            "Cleaning a "
            + name
            + " row whose "
            + column_name
            + " is written "
            + repr(written)
            + " gave "
            + str(refusal)
            + ". A value with no reading in the Permit Rules is refused with a ValueError naming "
            "the column and the value as written."
        )


def test_impossible_readings_blanked_bad_keys_refused(raw_frames, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.history_cleaning import clean_history_dataset
    from data_pipeline.job_log_cleaning import clean_job_log_dataset
    from data_pipeline.request_cleaning import clean_register_records
    from data_pipeline.sentence_cleaning import clean_sentence_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    raw_history = raw_frames[config.RAW_PERMIT_HISTORY_EVAL_FILE]
    answered = ~raw_history[config.HISTORY_TARGET_COLUMN].str.strip().str.lower().isin(
        config.NULL_TOKENS)
    template = raw_history[answered].drop_duplicates(config.PERMIT_REFERENCE_COLUMN).head(1)
    rows = pd.concat([template] * 5, ignore_index=True)
    for position in range(5):
        rows.loc[position, config.PERMIT_REFERENCE_COLUMN] = "PTW09" + str(90001 + position)
        rows.loc[position, config.REVISION_NUMBER_COLUMN] = "1"
    flammable, oxygen = "flammable_gas_lel_percent", config.OXYGEN_COLUMN
    sulphide, monoxide = "hydrogen_sulphide_ppm", "carbon_monoxide_ppm"
    cleared, height = "combustibles_cleared_radius_m", "working_height_m"
    depth, wind = config.EXCAVATION_DEPTH_COLUMN, config.WIND_SPEED_COLUMN
    voltage, watch = config.SYSTEM_VOLTAGE_COLUMN, "fire_watch_minutes"
    planned, party = config.PLANNED_WORK_HOURS_COLUMN, "work_party_size"
    impossible = [(0, flammable, "-2"), (0, oxygen, "100.5"), (0, sulphide, "-0.5"),
                  (0, monoxide, "-1 ppm"), (0, cleared, "0"), (0, watch, "0"), (0, voltage, "0"),
                  (0, height, "-3 m"), (0, wind, "-2 m/s"), (0, depth, "0"), (0, planned, "0"),
                  (0, party, "0"), (1, flammable, "104%LEL"), (1, oxygen, "-0.1"),
                  (1, cleared, "-1 ft"), (1, planned, "-30 min"), (1, watch, "-1 hr")]
    for position, column_name, written in impossible:
        rows.loc[position, column_name] = written
    possible = [(flammable, "100", 100), (oxygen, "0", 0.0), (wind, "0", 0.0),
                (depth, "0.1", 0.1), (watch, "1", 1), (voltage, "1", 1), (planned, "0.1", 0.1),
                (party, "1", 1), (cleared, "0.1 m", 0.1)]
    for column_name, written, _wanted in possible:
        rows.loc[2, column_name] = written
    blanked = ["risk_assessment_attached", flammable, config.PROCESS_UNIT_STATUS_COLUMN,
               config.ACCESS_METHOD_COLUMN]
    for position, blank_tokens in [(3, [" N/A ", "not recorded", "Illegible", "NULL"]),
                                   (4, ["-", "NA", " n/a", ""])]:
        for column_name, blank_token in zip(blanked, blank_tokens, strict=True):
            rows.loc[position, column_name] = blank_token
    planted_path = str(tmp_path / "planted_impossible.csv")
    rows.to_csv(planted_path, index=False)
    try:
        cleaned = clean_history_dataset(load_raw_dataset(planted_path))
    except Exception as error:
        cleaned = error
    read_columns = {config.PERMIT_REFERENCE_COLUMN}.union(
        [column_name for _position, column_name, _written in impossible], blanked)
    assert isinstance(cleaned, pd.DataFrame) and read_columns <= set(cleaned.columns), (
        "clean_history_dataset gave "
        + repr(cleaned)[:200]
        + " for five planted requests, where it returns the cleaned table with the columns "
        + str(sorted(read_columns))
        + "."
    )
    cleaned = cleaned.set_index(config.PERMIT_REFERENCE_COLUMN)
    planted_references = ["PTW09" + str(90001 + position) for position in range(5)]
    assert sorted(cleaned.index) == planted_references, (
        "Five planted requests cleaned to the rows "
        + str(sorted(cleaned.index))
        + ". A reading that cannot be true, or an answer left blank, never takes its row with "
        "it."
    )
    still_read = [column_name + " " + written for position, column_name, written in impossible
                  if not pd.isna(cleaned.loc[planted_references[position], column_name])]
    assert len(still_read) == 0, (
        "Readings that cannot be true were kept: "
        + str(still_read)
        + ". A negative reading, a flammable gas or oxygen reading above 100 percent, a "
        "clearance, height or depth of nought or less, a negative wind, a fire watch, voltage or "
        "planned work hours of nought or less and a work party of nobody cannot be true, so the "
        "reading is made blank, never corrected, and the row stays."
    )
    lost = [column_name + " " + written + " read as "
            + str(cleaned.loc[planted_references[2], column_name])
            for column_name, written, wanted in possible
            if pd.isna(cleaned.loc[planted_references[2], column_name])
            or cleaned.loc[planted_references[2], column_name] != wanted]
    assert len(lost) == 0, (
        "Readings at the edge of what can be true were not kept as written: "
        + str(lost)
        + ". A flammable gas reading of 100 percent, oxygen or wind of nought, and a depth, "
        "clearance or planned work hours of 0.1, a fire watch of one minute, one volt or a work "
        "party of one are readings like any other."
    )
    unblanked = [str(position) + " " + column_name for position in [3, 4]
                 for column_name in blanked
                 if not pd.isna(cleaned.loc[planted_references[position], column_name])]
    assert len(unblanked) == 0, (
        "An answer written N/A, not recorded, illegible, null, - or NA, with spaces around it or "
        "in any capitals, or left empty, was read as a value on planted rows "
        + str(unblanked)
        + ". Each channel's way of leaving an answer blank is a blank answer."
    )
    raw_logs = raw_frames[config.RAW_JOB_LOGS_EVAL_FILE]
    log_template = raw_logs.drop_duplicates(
        config.PERMIT_REFERENCE_COLUMN).head(4).reset_index(drop=True).copy()
    log_rows = log_template.copy()
    log_rows[config.PERMIT_REFERENCE_COLUMN] = ["PTW0991001", "PTW0991002", "PTW0991003",
                                                "PTW0991004"]
    log_rows[config.LOG_TARGET_COLUMN] = ["-2", "0", "12 hrs", "6 min"]
    log_rows.loc[2, planned] = "-3 hrs"
    log_path = str(tmp_path / "planted_logs.csv")
    log_rows.to_csv(log_path, index=False)
    try:
        cleaned_logs = clean_job_log_dataset(load_raw_dataset(log_path))
    except Exception as error:
        cleaned_logs = error
    assert isinstance(cleaned_logs, pd.DataFrame) and {
            config.PERMIT_REFERENCE_COLUMN, planned,
            config.LOG_TARGET_COLUMN} <= set(cleaned_logs.columns), (
        "clean_job_log_dataset gave "
        + repr(cleaned_logs)[:200]
        + " for four planted job logs, where it returns the cleaned table."
    )
    kept = cleaned_logs[config.PERMIT_REFERENCE_COLUMN].astype(str).tolist()
    actual_hours = cleaned_logs[config.LOG_TARGET_COLUMN].tolist()
    assert (kept == ["PTW0991003", "PTW0991004"] and actual_hours == [12.0, 0.1]
            and pd.isna(cleaned_logs[planned].iloc[0])), (
        "Four planted job logs cleaned to "
        + str(cleaned_logs[[config.PERMIT_REFERENCE_COLUMN, planned,
                            config.LOG_TARGET_COLUMN]].to_dict("records"))
        + ". Actual work hours of -2 or of nought cannot be true and are made blank, and being "
        "the answer the log cannot be learned from, so the row goes; six minutes are 0.1 hours "
        "and stay; negative planned work hours are made blank and the log stays."
    )
    raw_sentences = raw_frames[config.RAW_STATEMENT_SENTENCES_FILE]
    raw_register = load_raw_dataset(config.LIVE_PERMIT_REGISTER_FILE)
    history_keys = pd.concat([template] * 2, ignore_index=True)
    history_keys[config.PERMIT_REFERENCE_COLUMN] = "PTW0992001"
    history_keys[config.REVISION_NUMBER_COLUMN] = ["1", "1"]
    history_keys[planned] = ["10", "15"]
    blank_revision = template.copy()
    blank_revision[config.PERMIT_REFERENCE_COLUMN] = "PTW0992002"
    blank_revision[config.REVISION_NUMBER_COLUMN] = "N/A"
    shared_logs = log_template.head(2).copy()
    shared_logs[config.PERMIT_REFERENCE_COLUMN] = ["PTW-992003", "992003"]
    unnamed_logs = log_template.head(2).copy()
    unnamed_logs.loc[unnamed_logs.index[0], config.PERMIT_REFERENCE_COLUMN] = "-"
    unnamed_sentences = raw_sentences.head(2).copy()
    unnamed_sentences[config.SENTENCE_ID_COLUMN] = ["", "SEN-9920041"]
    shared_sentences = raw_sentences.head(2).copy()
    shared_sentences[config.SENTENCE_ID_COLUMN] = "SEN-9920042"
    shared_register = raw_register.head(2).copy()
    shared_register[config.PERMIT_REFERENCE_COLUMN] = ["PTW-51207", "51207"]
    unstated_register = raw_register.head(2).copy()
    unstated_register.loc[unstated_register.index[1], config.PERMIT_STATUS_COLUMN] = None
    for label, cleaner, frame, named in [
            ("two rows of the same request at the same revision", clean_history_dataset,
             history_keys, ["PTW0992001", config.REVISION_NUMBER_COLUMN]),
            ("a request whose revision is left blank", clean_history_dataset,
             blank_revision, ["PTW0992002", config.REVISION_NUMBER_COLUMN]),
            ("two job logs of one request", clean_job_log_dataset,
             shared_logs, ["PTW0992003", config.PERMIT_REFERENCE_COLUMN]),
            ("a job log with no permit reference", clean_job_log_dataset,
             unnamed_logs, [config.PERMIT_REFERENCE_COLUMN]),
            ("a sentence with no identifier", clean_sentence_dataset, unnamed_sentences,
             [config.SENTENCE_ID_COLUMN]),
            ("two sentences sharing an identifier", clean_sentence_dataset, shared_sentences,
             ["SEN-9920042", config.SENTENCE_ID_COLUMN]),
            ("two permits of the register written PTW-51207 and 51207", clean_register_records,
             shared_register, ["PTW0051207", config.PERMIT_REFERENCE_COLUMN]),
            ("a permit of the register with no status", clean_register_records,
             unstated_register, [config.PERMIT_STATUS_COLUMN])]:
        keyed_path = str(tmp_path / "keyed.csv")
        frame.to_csv(keyed_path, index=False)
        refusal = None
        try:
            cleaner(load_raw_dataset(keyed_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and all(word in refusal for word in named), (
            "Cleaning "
            + label
            + " gave "
            + str(refusal)
            + ". Every request revision, job log, sentence and permit of the register is named "
            "and appears once: a blank or repeated key is refused with a ValueError naming "
            + " and ".join(named)
            + ", never dropped or numbered quietly."
        )


def test_schemas_leave_out_outcome_and_identifiers(saved_models):
    from config import config

    assert saved_models["problem"] is None, saved_models["problem"]
    suspension_schema = saved_models["suspension_schema"]
    work_hours_schema = saved_models["work_hours_schema"]
    for schema_name, schema, configured in [
            ("suspension", suspension_schema,
             config.SUSPENSION_NUMERIC_FEATURES + config.SUSPENSION_CATEGORICAL_FEATURES),
            ("work hours", work_hours_schema,
             config.WORK_HOURS_NUMERIC_FEATURES + config.WORK_HOURS_CATEGORICAL_FEATURES)]:
        assert len(schema) == len(set(schema)) and sorted(schema) == sorted(configured), (
            "The saved "
            + schema_name
            + " schema names "
            + str(schema)
            + " where the configuration's features are "
            + str(configured)
            + ", each once."
        )
    suspension_never = [config.HISTORY_TARGET_COLUMN, "review_outcome", "suspension_reason",
                        config.PERMIT_REFERENCE_COLUMN, config.REVISION_NUMBER_COLUMN,
                        "contractor_reference", config.EQUIPMENT_TAG_COLUMN]
    work_hours_never = [config.LOG_TARGET_COLUMN, "completion_code", config.HISTORY_TARGET_COLUMN,
                        config.PERMIT_REFERENCE_COLUMN, "contractor_reference"]
    for schema_name, schema, never in [("suspension", suspension_schema, suspension_never),
                                       ("work hours", work_hours_schema, work_hours_never)]:
        leaked = sorted(set(schema) & set(never))
        assert len(leaked) == 0, (
            "The saved "
            + schema_name
            + " schema feeds the model "
            + str(leaked)
            + ". The outcome, an identifier, and anything recorded only once the decision was "
            "taken or the work was done are never features: a model fed them learns what it "
            "cannot know when the request is reviewed, or learns which request it is rather "
            "than what was asked."
        )
    counts_used = [column_name for column_name in config.REQUIREMENT_COUNT_RESULTS
                   if column_name in suspension_schema]
    flag_columns = list(config.SIMOPS_TRIGGER_FLAG_COLUMNS.values())
    flags_used = {schema_name: [column_name for column_name in flag_columns
                                if column_name in schema]
                  for schema_name, schema in [("suspension", suspension_schema),
                                              ("work hours", work_hours_schema)]}
    assert (counts_used == list(config.REQUIREMENT_COUNT_RESULTS)
            and flags_used == {"suspension": flag_columns, "work hours": flag_columns}), (
        "The suspension schema uses the requirement counts "
        + str(counts_used)
        + ", and the two schemas use the SIMOPS trigger flags "
        + str(flags_used)
        + ", where the suspension model is fed all three counts - requirements Met, Not met and "
        "Not documented - and both models every SIMOPS trigger's flag, so they learn from the "
        "screening a live request is shown with."
    )


def test_counts_and_triggers_use_own_standards(processed_frames, permit_standards):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import add_requirement_counts, add_simops_trigger_flags
    from tools.screening_tools import find_simops_triggers, screen_request

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    standards = permit_standards["standards"]
    history = processed_frames[config.CLEAN_PERMIT_HISTORY_EVAL_FILE]
    sample = history.groupby(config.PERMIT_TYPE_COLUMN, group_keys=False).head(40)
    sample = sample.reset_index(drop=True)
    linked_rows = int((sample[config.LINKED_PERMIT_TYPE_COLUMN] != config.NO_LINKED_PERMIT).sum())
    assert linked_rows > 0, (
        "The first past requests of each permit type hold no request with a linked permit "
        "type, where the cleaned permit history holds many."
    )
    try:
        counted = add_requirement_counts(sample, standards)
    except Exception as error:
        counted = error
    assert isinstance(counted, pd.DataFrame), (
        "add_requirement_counts gave "
        + repr(counted)[:200]
        + ", where it returns the requests it was given with their requirement counts beside "
        "them."
    )
    missing = [column_name for column_name in config.REQUIREMENT_COUNT_RESULTS
               if column_name not in counted.columns]
    assert len(missing) == 0 and len(counted) == len(sample), (
        "add_requirement_counts returned "
        + str(len(counted))
        + " rows of the "
        + str(len(sample))
        + " it was given, without the columns "
        + str(missing)
        + "."
    )
    assert config.PERMIT_REFERENCE_COLUMN in counted and list(
            counted[config.PERMIT_REFERENCE_COLUMN]) == list(
            sample[config.PERMIT_REFERENCE_COLUMN]), (
        "add_requirement_counts returned the requests in another order, so each row's counts "
        "no longer sit beside its own answers."
    )
    try:
        flagged = add_simops_trigger_flags(counted)
    except Exception as error:
        flagged = error
    flag_columns = list(config.SIMOPS_TRIGGER_FLAG_COLUMNS.values())
    assert isinstance(flagged, pd.DataFrame) and set(flag_columns) <= set(flagged.columns) and list(
            flagged[config.PERMIT_REFERENCE_COLUMN]) == list(
            sample[config.PERMIT_REFERENCE_COLUMN]), (
        "add_simops_trigger_flags gave "
        + repr(flagged)[:200]
        + ", where it returns the requests it was given, in their order, with a flag for each "
        "SIMOPS trigger "
        + str(flag_columns)
        + " beside them."
    )
    wrong = []
    for request, flagged_row in zip(sample.to_dict("records"), flagged.to_dict("records"),
                                    strict=True):
        screening = screen_request(request, standards)
        assert isinstance(screening, dict) and set(config.REQUIREMENT_COUNT_RESULTS) <= set(
                screening), (
            "screen_request returned "
            + str(screening)[:200]
            + " for "
            + request[config.PERMIT_REFERENCE_COLUMN]
            + ", where it returns the screening with its counts "
            + str(list(config.REQUIREMENT_COUNT_RESULTS))
            + "."
        )
        own_count = len(standards[request[config.PERMIT_TYPE_COLUMN]]["requirements"])
        if request[config.LINKED_PERMIT_TYPE_COLUMN] != config.NO_LINKED_PERMIT:
            own_count += len(standards[request[config.LINKED_PERMIT_TYPE_COLUMN]]["requirements"])
        expected = [screening[column_name] for column_name in config.REQUIREMENT_COUNT_RESULTS]
        got = [flagged_row[column_name] for column_name in config.REQUIREMENT_COUNT_RESULTS]
        if got != expected or sum(got) != own_count:
            wrong.append(request[config.PERMIT_REFERENCE_COLUMN] + " counted " + str(got)
                         + " where its standards' screening gives " + str(expected)
                         + " of " + str(own_count) + " requirements")
        triggers = find_simops_triggers(request)
        expected_flags = [int(trigger_name in triggers)
                          for trigger_name in config.SIMOPS_TRIGGER_FLAG_COLUMNS]
        got_flags = [flagged_row[column_name] for column_name in flag_columns]
        if got_flags != expected_flags:
            wrong.append(request[config.PERMIT_REFERENCE_COLUMN] + " flagged " + str(got_flags)
                         + " where its SIMOPS triggers " + str(triggers) + " give "
                         + str(expected_flags))
    assert len(wrong) == 0, (
        "The requirement counts and the trigger flags disagree with the screening under each "
        "request's own standards: "
        + "; ".join(wrong[:4])
        + ". Each past request is screened against the standard of its own permit type and of "
        "its linked type, as a live request is; its requirements Met, Not met and Not documented "
        "add up to both standards' requirements, and each SIMOPS trigger it meets is flagged 1, "
        "the others 0."
    )
    logs = processed_frames[config.CLEAN_JOB_LOGS_EVAL_FILE]
    log_sample = logs.groupby(config.PERMIT_TYPE_COLUMN, group_keys=False).head(30).reset_index(
        drop=True)
    try:
        flagged_logs = add_simops_trigger_flags(log_sample)
    except Exception as error:
        flagged_logs = error
    assert isinstance(flagged_logs, pd.DataFrame) and set(flag_columns) <= set(
            flagged_logs.columns), (
        "add_simops_trigger_flags gave "
        + repr(flagged_logs)[:200]
        + " for job logs, where the work hours features flag the same SIMOPS triggers."
    )
    log_wrong = [log[config.PERMIT_REFERENCE_COLUMN]
                 for log, flagged_log in zip(log_sample.to_dict("records"),
                                             flagged_logs.to_dict("records"), strict=True)
                 if [flagged_log[column_name] for column_name in flag_columns]
                 != [int(trigger_name in find_simops_triggers(log))
                     for trigger_name in config.SIMOPS_TRIGGER_FLAG_COLUMNS]]
    assert len(log_wrong) == 0, (
        "The job logs "
        + str(log_wrong[:4])
        + " are flagged otherwise than their own SIMOPS triggers. A job log is flagged the way "
        "a past request is."
    )
    crossed = dict(standards)
    crossed[config.HOT_WORK] = standards[config.CONFINED_SPACE_ENTRY]
    hot_work = sample[(sample[config.PERMIT_TYPE_COLUMN] == config.HOT_WORK)].head(1)
    refusal = None
    try:
        add_requirement_counts(hot_work, crossed)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    entry_document = config.PERMIT_STANDARD_DOCUMENTS[config.CONFINED_SPACE_ENTRY]
    assert refusal is not None and config.HOT_WORK in refusal and entry_document in refusal, (
        "A hot work request screened with the confined space entry standard handed over under "
        "its permit type gave "
        + str(refusal)
        + ". The two standards are twins written from one template, and a standard written for "
        "another permit type is refused, naming the permit type and the document, never used "
        "to screen it."
    )
    linked_to_dig = hot_work.copy()
    linked_to_dig[config.LINKED_PERMIT_TYPE_COLUMN] = config.EXCAVATION
    without_standard = dict(standards)
    del without_standard[config.EXCAVATION]
    for meaning, requests in [
            ("an excavation request",
             sample[sample[config.PERMIT_TYPE_COLUMN] == config.EXCAVATION].head(1)),
            ("a hot work request linked to an excavation", linked_to_dig)]:
        refusal = None
        try:
            add_requirement_counts(requests, without_standard)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert (refusal is not None and config.EXCAVATION in refusal
                and requests[config.PERMIT_REFERENCE_COLUMN].iloc[0] in refusal), (
            meaning[0].upper()
            + meaning[1:]
            + " with no excavation standard handed over gave "
            + str(refusal)
            + ". A request whose permit type or linked type has no standard cannot be screened "
            "and is refused, naming the request and the permit type."
        )


def test_inputs_follow_schema_order_with_gaps(processed_frames, saved_models, permit_standards):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_requirement_counts, add_simops_trigger_flags,
                                               build_feature_frame)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    try:
        history = add_simops_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_PERMIT_HISTORY_EVAL_FILE].head(60),
            permit_standards["standards"]))
        logs = add_simops_trigger_flags(processed_frames[config.CLEAN_JOB_LOGS_EVAL_FILE].head(60))
    except Exception as error:
        history = logs = error
    suspension_columns = (config.SUSPENSION_NUMERIC_FEATURES
                          + config.SUSPENSION_CATEGORICAL_FEATURES)
    work_hours_columns = (config.WORK_HOURS_NUMERIC_FEATURES
                          + config.WORK_HOURS_CATEGORICAL_FEATURES)
    assert (isinstance(history, pd.DataFrame) and set(suspension_columns) <= set(history.columns)
            and isinstance(logs, pd.DataFrame) and set(work_hours_columns) <= set(logs.columns)), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(history)[:200]
        + ", where it returns the requests with every suspension feature "
        + str(suspension_columns)
        + " beside them, and the job logs with every work hours feature."
    )
    for name, frame, numeric_columns, categorical_columns, schema in [
            ("suspension", history, config.SUSPENSION_NUMERIC_FEATURES,
             config.SUSPENSION_CATEGORICAL_FEATURES, saved_models["suspension_schema"]),
            ("work hours", logs, config.WORK_HOURS_NUMERIC_FEATURES,
             config.WORK_HOURS_CATEGORICAL_FEATURES, saved_models["work_hours_schema"])]:
        frame = frame.copy()
        for column_name in numeric_columns:
            frame[column_name] = frame[column_name].astype(float)
            frame.loc[frame.index[0], column_name] = float("nan")
        for column_name in categorical_columns:
            frame[column_name] = frame[column_name].astype(object)
            frame.loc[frame.index[1], column_name] = None
        shuffled = frame[list(reversed(list(frame.columns)))]
        features = build_feature_frame(shuffled, numeric_columns, categorical_columns)
        assert isinstance(features, pd.DataFrame), (
            "build_feature_frame returned "
            + type(features).__name__
            + " for the "
            + name
            + " inputs, where it returns a table of the model's inputs."
        )
        assert list(features.columns) == schema, (
            "The "
            + name
            + " inputs are built in the order "
            + str(list(features.columns))
            + " where the saved schema reads "
            + str(schema)
            + ". The model takes its inputs in the order it was trained on, whatever order the "
            "table it is given holds them in."
        )
        assert len(features) == len(frame) and list(features.index) == list(frame.index), (
            "build_feature_frame returned "
            + str(len(features))
            + " rows for "
            + str(len(frame))
            + ", or in another order."
        )
        for column_name in numeric_columns:
            same = np.allclose(features[column_name].to_numpy(dtype=float),
                               frame[column_name].to_numpy(dtype=float), equal_nan=True)
            assert same, (
                column_name
                + " is changed on its way into the "
                + name
                + " model. A number goes in as it was cleaned, a gap as a gap."
            )
        for column_name in categorical_columns:
            gaps = frame[column_name].isna()
            built = features[column_name]
            gaps_as_nan = all(isinstance(value, float) and value != value
                              for value in built[gaps])
            assert gaps_as_nan and list(built[~gaps]) == list(frame.loc[~gaps, column_name]), (
                column_name
                + " is changed on its way into the "
                + name
                + " model: a gap became "
                + str(sorted({repr(value) for value in built[gaps]})[:3])
                + ". An answer in words goes in as cleaned and a gap as NaN, which the model's "
                "own imputer fills, never None, the text 'nan' or an empty string."
            )


def test_models_fit_schema_and_fill_gaps(processed_frames, saved_models, permit_standards):
    import numpy as np
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_requirement_counts, add_simops_trigger_flags,
                                               build_feature_frame)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    for name, label, schema in [("suspension", "suspension", saved_models["suspension_schema"]),
                                ("work_hours", "work hours", saved_models["work_hours_schema"])]:
        fitted_on = list(getattr(saved_models[name], "feature_names_in_", []))
        assert fitted_on == schema, (
            "The saved "
            + label
            + " model was fitted on "
            + str(fitted_on)
            + " while its schema names "
            + str(schema)
            + ". A model saved beside a schema it was not fitted on scores live requests on the "
            "wrong columns."
        )
    planned, classification = config.PLANNED_WORK_HOURS_COLUMN, "area_classification"
    location = config.WORK_LOCATION_COLUMN
    history_train = processed_frames[config.CLEAN_PERMIT_HISTORY_TRAIN_FILE]
    try:
        history = add_simops_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_PERMIT_HISTORY_EVAL_FILE].dropna(
                subset=[planned, classification]).head(25),
            permit_standards["standards"]))
    except Exception as error:
        history = error
    assert isinstance(history, pd.DataFrame), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(history)[:200]
        + ", where it returns the requests with them beside their answers."
    )
    features = build_feature_frame(history, config.SUSPENSION_NUMERIC_FEATURES,
                                   config.SUSPENSION_CATEGORICAL_FEATURES)
    log_features = build_feature_frame(
        add_simops_trigger_flags(processed_frames[config.CLEAN_JOB_LOGS_EVAL_FILE].dropna(
            subset=[planned, location]).head(25)),
        config.WORK_HOURS_NUMERIC_FEATURES, config.WORK_HOURS_CATEGORICAL_FEATURES)
    assert isinstance(features, pd.DataFrame) and isinstance(log_features, pd.DataFrame), (
        "build_feature_frame returned "
        + type(features).__name__
        + " and "
        + type(log_features).__name__
        + " for the suspension and work hours inputs, where it returns a table of the model's "
        "inputs."
    )
    with_gaps = features.copy()
    with_gaps[planned] = np.nan
    with_gaps[classification] = np.nan
    filled = features.copy()
    filled[planned] = float(history_train[planned].median())
    filled[classification] = history_train[classification].mode().iloc[0]
    gap_scores = saved_models["suspension"].predict_proba(with_gaps)[:, 1]
    filled_scores = saved_models["suspension"].predict_proba(filled)[:, 1]
    assert np.isfinite(gap_scores).all() and np.allclose(gap_scores, filled_scores), (
        "A request with its planned work hours and its hazardous area classification left blank "
        "scores "
        + str(np.round(gap_scores[:3], 4))
        + " where the same request with the training median and the commonest classification "
        "scores "
        + str(np.round(filled_scores[:3], 4))
        + ". The pipeline fills a blank number with the training data's "
        + config.NUMBER_FILL_STRATEGY
        + " and a blank answer in words with its "
        + config.CATEGORY_FILL_STRATEGY
        + " value, so a live request with a gap is still scored."
    )
    logs_train = processed_frames[config.CLEAN_JOB_LOGS_TRAIN_FILE]
    log_gaps = log_features.copy()
    log_gaps[planned] = np.nan
    log_gaps[location] = np.nan
    log_filled = log_features.copy()
    log_filled[planned] = float(logs_train[planned].median())
    log_filled[location] = logs_train[location].mode().iloc[0]
    gap_hours = saved_models["work_hours"].predict(log_gaps)
    filled_hours = saved_models["work_hours"].predict(log_filled)
    assert np.isfinite(gap_hours).all() and np.allclose(gap_hours, filled_hours), (
        "A job with its planned work hours and its location left blank is expected to take "
        + str(np.round(gap_hours[:3], 1))
        + " work hours where the same job filled as training filled it gives "
        + str(np.round(filled_hours[:3], 1))
        + "."
    )


def test_quality_report_accounts_for_rows(raw_frames, processed_frames):
    import json
    import os

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.quality_profiler import profile_dataset

    assert raw_frames["problem"] is None, raw_frames["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert os.path.isfile(config.DATA_QUALITY_REPORT_FILE), (
        "There is no quality report at "
        + config.DATA_QUALITY_REPORT_FILE
        + ". The data pipeline writes it when it cleans the five extracts."
    )
    with open(config.DATA_QUALITY_REPORT_FILE, encoding="utf-8") as report_file:
        report = json.load(report_file)
    datasets = [
        ("permit_history_train", config.RAW_PERMIT_HISTORY_TRAIN_FILE,
         config.CLEAN_PERMIT_HISTORY_TRAIN_FILE, config.HISTORY_TARGET_COLUMN),
        ("permit_history_eval", config.RAW_PERMIT_HISTORY_EVAL_FILE,
         config.CLEAN_PERMIT_HISTORY_EVAL_FILE, config.HISTORY_TARGET_COLUMN),
        ("job_logs_train", config.RAW_JOB_LOGS_TRAIN_FILE, config.CLEAN_JOB_LOGS_TRAIN_FILE,
         config.LOG_TARGET_COLUMN),
        ("job_logs_eval", config.RAW_JOB_LOGS_EVAL_FILE, config.CLEAN_JOB_LOGS_EVAL_FILE,
         config.LOG_TARGET_COLUMN),
        ("statement_sentences", config.RAW_STATEMENT_SENTENCES_FILE,
         config.CLEAN_STATEMENT_SENTENCES_FILE, config.SENTENCE_TARGET_COLUMN)]
    assert isinstance(report, dict) and sorted(report) == sorted(
            dataset[0] for dataset in datasets) and all(
            isinstance(profile, dict) for profile in report.values()), (
        config.DATA_QUALITY_REPORT_FILE
        + " profiles "
        + str(sorted(report))[:300]
        + " where it profiles each of the five extracts under its own name."
    )
    reasons = ["duplicate_rows", "rows_missing_target", "answered_earlier_revisions_removed",
               "rows_with_impossible_target", "rows_with_too_few_words"]
    counted_keys = ["rows_received", "rows_kept", "rows_removed", "duplicate_rows",
                    "rows_missing_target", "columns"]
    for name, raw_path, clean_path, target_column in datasets:
        profile = report[name]
        try:
            profiled = profile_dataset(load_raw_dataset(raw_path), processed_frames[clean_path],
                                       target_column)
        except Exception as error:
            profiled = error
        assert isinstance(profiled, dict), (
            "profile_dataset gave "
            + repr(profiled)[:200]
            + " for "
            + name
            + ", where it returns the counts that profile the extract."
        )
        expected = {key: profiled.get(key) for key in counted_keys}
        expected.update({
            "rows_received": len(raw_frames[raw_path]),
            "rows_kept": len(processed_frames[clean_path]),
            "rows_removed": len(raw_frames[raw_path]) - len(processed_frames[clean_path]),
            "columns": int(processed_frames[clean_path].shape[1]),
        })
        got = {key: profile.get(key) for key in counted_keys}
        assert got == expected, (
            "The quality report gives "
            + name
            + " as "
            + str(got)
            + " where the files and profile_dataset give "
            + str(expected)
            + ". Rows received are the raw rows and rows kept the cleaned rows, and the report "
            "holds what profile_dataset counts in the extract as the pipeline read it."
        )
        explained = sum(int(profile.get(reason, 0)) for reason in reasons)
        assert explained == expected["rows_removed"], (
            "The quality report's reasons for "
            + name
            + " add up to "
            + str(explained)
            + " rows where "
            + str(expected["rows_removed"])
            + " were removed: "
            + str({reason: profile.get(reason) for reason in reasons if reason in profile})
            + ". Every removed row is a repeat, a row with no answer, an earlier revision, an "
            "answer that cannot be true or a sentence too short to read."
        )
    planted_raw = pd.DataFrame({
        config.PERMIT_REFERENCE_COLUMN: ["PTW0960001", "PTW0960001", "PTW0960002", "PTW0960002",
                                         "PTW0960003", "PTW0960004"],
        config.PLANT_AREA_COLUMN: [config.BOILER_HOUSE, config.BOILER_HOUSE, config.GATEHOUSE,
                                   config.GATEHOUSE, config.LABORATORY, config.FLARE_AREA],
        config.HISTORY_TARGET_COLUMN: [config.BOOLEAN_NO, config.BOOLEAN_NO, None, None, None,
                                       config.BOOLEAN_YES]})
    planted_clean = planted_raw.iloc[[0, 5]].reset_index(drop=True)
    try:
        profiled = profile_dataset(planted_raw, planted_clean, config.HISTORY_TARGET_COLUMN)
    except Exception as error:
        profiled = error
    expected = {"rows_received": 6, "rows_kept": 2, "rows_removed": 4, "duplicate_rows": 2,
                "rows_missing_target": 2, "columns": 3}
    got = ({key: profiled.get(key) for key in expected} if isinstance(profiled, dict)
           else profiled)
    assert got == expected, (
        "Six planted rows - a request sent twice exactly, a request with no outcome sent twice "
        "exactly, another request with no outcome and one answered - profiled against the two "
        "rows kept as "
        + str(got)
        + " where they are "
        + str(expected)
        + ". A repeat is counted once, beside its first copy, and a row missing its outcome is "
        "counted among the first copies only."
    )


def test_suspension_model_auc_and_calibration(processed_frames, saved_models, permit_standards):
    import copy

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from config import config
    from data_pipeline.feature_builder import (add_requirement_counts, add_simops_trigger_flags,
                                               build_feature_frame)
    from modeling.model_evaluation import evaluate_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    standards = permit_standards["standards"]
    history_train = processed_frames[config.CLEAN_PERMIT_HISTORY_TRAIN_FILE]
    try:
        held_out = add_simops_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_PERMIT_HISTORY_EVAL_FILE], standards))
        trained_on = add_simops_trigger_flags(add_requirement_counts(
            history_train[history_train[config.PERMIT_TYPE_COLUMN]
                          == config.ELECTRICAL_ISOLATION].head(3000), standards))
    except Exception as error:
        held_out = trained_on = error
    assert all(isinstance(frame, pd.DataFrame) and config.HISTORY_TARGET_COLUMN in frame
               for frame in [held_out, trained_on]), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(held_out)[:200]
        + ", where it returns the requests with their outcomes, requirement counts and trigger "
        "flags."
    )
    held_out_features = build_feature_frame(held_out, config.SUSPENSION_NUMERIC_FEATURES,
                                            config.SUSPENSION_CATEGORICAL_FEATURES)
    train_features = build_feature_frame(trained_on, config.SUSPENSION_NUMERIC_FEATURES,
                                         config.SUSPENSION_CATEGORICAL_FEATURES)
    assert all(isinstance(frame, pd.DataFrame) for frame in [held_out_features, train_features]), (
        "build_feature_frame returned "
        + type(held_out_features).__name__
        + ", where it returns a table of the model's inputs."
    )
    held_out_answers = held_out[config.HISTORY_TARGET_COLUMN].map(
        config.SUSPENSION_TARGET_ENCODING)
    train_answers = trained_on[config.HISTORY_TARGET_COLUMN].map(
        config.SUSPENSION_TARGET_ENCODING)
    model = saved_models["suspension"]
    scores = model.predict_proba(held_out_features)[:, 1]
    measured = roc_auc_score(held_out_answers, scores)
    assert measured >= config.MINIMUM_SUSPENSION_ROC_AUC, (
        "The suspension model ranks the held-out requests with a ROC AUC of "
        + str(round(measured, 4))
        + ", below the "
        + str(config.MINIMUM_SUSPENSION_ROC_AUC)
        + " the configuration sets. It is scored on the later requests, which it never saw, "
        "with the requirement counts worked out under each request's own standards."
    )
    assert abs(float(scores.mean()) - float(held_out_answers.mean())) <= 0.03, (
        "The suspension model's scores average "
        + str(round(float(scores.mean()), 4))
        + " on the held-out requests, where "
        + str(round(float(held_out_answers.mean()), 4))
        + " of their permits were suspended. A suspension likelihood is read as the share of "
        "permits like this one that were suspended, so the scores average the share that were."
    )
    evaluation = evaluate_classifier(copy.deepcopy(model), train_features, train_answers,
                                     held_out_features, held_out_answers)
    missing = sorted({"train_roc_auc", "held_out_roc_auc", "generalisation_gap"}
                     - set(evaluation if isinstance(evaluation, dict) else {}))
    assert len(missing) == 0, (
        "evaluate_classifier returned "
        + type(evaluation).__name__
        + " without "
        + str(missing)
        + ", so the suspension model's scores cannot be read."
    )
    train_measured = roc_auc_score(train_answers, model.predict_proba(train_features)[:, 1])
    assert (abs(evaluation["held_out_roc_auc"] - measured) < 0.006
            and abs(evaluation["train_roc_auc"] - train_measured) < 0.006
            and abs(evaluation["generalisation_gap"]
                    - (evaluation["train_roc_auc"] - evaluation["held_out_roc_auc"])) < 0.011), (
        "Given the held-out requests and the first training requests of one permit type, the "
        "evaluation reports a held-out ROC AUC of "
        + str(evaluation["held_out_roc_auc"])
        + ", a training ROC AUC of "
        + str(evaluation["train_roc_auc"])
        + " and a gap of "
        + str(evaluation["generalisation_gap"])
        + ", where the model's own scores on those rows give "
        + str(round(measured, 4))
        + " and "
        + str(round(train_measured, 4))
        + ". evaluate_classifier scores the fitted model it is given on the rows it is given, "
        "without fitting it again, and the gap is the training score less the held-out one."
    )


def test_work_hours_model_r2_and_floor(processed_frames, saved_models):
    import copy

    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score

    from config import config
    from data_pipeline.feature_builder import add_simops_trigger_flags, build_feature_frame
    from modeling.model_evaluation import evaluate_regressor

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    training_logs = processed_frames[config.CLEAN_JOB_LOGS_TRAIN_FILE]
    try:
        held_out = add_simops_trigger_flags(processed_frames[config.CLEAN_JOB_LOGS_EVAL_FILE])
        trained_on = add_simops_trigger_flags(
            training_logs[training_logs[config.PERMIT_TYPE_COLUMN]
                          == config.WORK_AT_HEIGHT].head(3000))
    except Exception as error:
        held_out = trained_on = error
    assert all(isinstance(frame, pd.DataFrame) and config.LOG_TARGET_COLUMN in frame
               for frame in [held_out, trained_on]), (
        "add_simops_trigger_flags gave "
        + repr(held_out)[:200]
        + ", where it returns the job logs with their trigger flags beside them."
    )
    held_out_features = build_feature_frame(held_out, config.WORK_HOURS_NUMERIC_FEATURES,
                                            config.WORK_HOURS_CATEGORICAL_FEATURES)
    train_features = build_feature_frame(trained_on, config.WORK_HOURS_NUMERIC_FEATURES,
                                         config.WORK_HOURS_CATEGORICAL_FEATURES)
    assert all(isinstance(frame, pd.DataFrame) for frame in [held_out_features, train_features]), (
        "build_feature_frame returned "
        + type(held_out_features).__name__
        + ", where it returns a table of the model's inputs."
    )
    model = saved_models["work_hours"]
    forecast = model.predict(held_out_features)
    measured = r2_score(held_out[config.LOG_TARGET_COLUMN], forecast)
    assert measured >= config.MINIMUM_WORK_HOURS_R2, (
        "The work hours model explains the held-out actual work hours with an R2 of "
        + str(round(measured, 4))
        + ", below the "
        + str(config.MINIMUM_WORK_HOURS_R2)
        + " the configuration sets."
    )
    floor = float(training_logs[config.LOG_TARGET_COLUMN].min())
    assert float(forecast.min()) >= floor, (
        "The work hours model expects a held-out job to take "
        + str(round(float(forecast.min()), 1))
        + " work hours, fewer than the "
        + str(floor)
        + " of the quickest job it learned from. No job was ever done in fewer, so a shorter "
        "forecast is the model inventing one."
    )
    training_average = float(training_logs[config.LOG_TARGET_COLUMN].mean())
    average_error = mean_absolute_error(held_out[config.LOG_TARGET_COLUMN],
                                        [training_average] * len(held_out))
    model_error = mean_absolute_error(held_out[config.LOG_TARGET_COLUMN], forecast)
    assert model_error < 0.75 * average_error, (
        "The work hours model is off by "
        + str(round(model_error, 1))
        + " work hours on average, against "
        + str(round(average_error, 1))
        + " for the average job's hours given to every job. A forecast that does little better "
        "than the average tells the Area Authority nothing about this request."
    )
    evaluation = evaluate_regressor(copy.deepcopy(model), train_features,
                                    trained_on[config.LOG_TARGET_COLUMN], held_out_features,
                                    held_out[config.LOG_TARGET_COLUMN])
    missing = sorted({"train_r2", "held_out_r2", "generalisation_gap"}
                     - set(evaluation if isinstance(evaluation, dict) else {}))
    assert len(missing) == 0, (
        "evaluate_regressor returned "
        + type(evaluation).__name__
        + " without "
        + str(missing)
        + ", so the work hours model's scores cannot be read."
    )
    train_measured = r2_score(trained_on[config.LOG_TARGET_COLUMN], model.predict(train_features))
    assert (abs(evaluation["held_out_r2"] - measured) < 0.006
            and abs(evaluation["train_r2"] - train_measured) < 0.006
            and abs(evaluation["generalisation_gap"]
                    - (evaluation["train_r2"] - evaluation["held_out_r2"])) < 0.011), (
        "Given the held-out logs and the first training logs of one permit type, "
        "evaluate_regressor reports "
        + str({key: evaluation.get(key) for key in ["train_r2", "held_out_r2",
                                                    "generalisation_gap"]})
        + " where the model's own forecasts on those rows give a held-out R2 of "
        + str(round(measured, 4))
        + " and a training R2 of "
        + str(round(train_measured, 4))
        + ". It scores the fitted model it is given on the rows it is given, without fitting it "
        "again."
    )


def test_suspension_rises_with_failures_and_conflicts(processed_frames, saved_models,
                                                      permit_standards):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_requirement_counts, add_simops_trigger_flags,
                                               build_feature_frame)

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    try:
        held_out = add_simops_trigger_flags(add_requirement_counts(
            processed_frames[config.CLEAN_PERMIT_HISTORY_EVAL_FILE],
            permit_standards["standards"]))
    except Exception as error:
        held_out = error
    assert isinstance(held_out, pd.DataFrame), (
        "Adding the requirement counts and the trigger flags gave "
        + repr(held_out)[:200]
        + ", where it returns the requests with them beside their answers."
    )
    features = build_feature_frame(held_out, config.SUSPENSION_NUMERIC_FEATURES,
                                   config.SUSPENSION_CATEGORICAL_FEATURES)
    assert isinstance(features, pd.DataFrame) and set(config.REQUIREMENT_COUNT_RESULTS) <= set(
            features.columns), (
        "build_feature_frame returned "
        + type(features).__name__
        + ", where it returns the model's inputs with the requirement counts among them."
    )
    model = saved_models["suspension"]
    met_column = config.REQUIREMENTS_MET_COUNT_COLUMN
    before = features[features[met_column] >= 1]
    assert len(before) > 50, (
        "Only "
        + str(len(before))
        + " held-out requests are counted with a requirement Met, where the screening finds far "
        "more."
    )
    after = before.copy()
    after[met_column] = after[met_column] - 1
    after[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN] = (
        after[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN] + 1)
    before_scores = model.predict_proba(before)[:, 1]
    after_scores = model.predict_proba(after)[:, 1]
    rising = float((after_scores > before_scores).mean())
    assert rising >= 0.95 and (after_scores.mean() - before_scores.mean()) >= 0.008, (
        "Turning one requirement from Met to Not met raises the suspension score for "
        + str(round(100 * rising, 1))
        + "% of the "
        + str(len(before))
        + " held-out requests it applies to, and moves the average from "
        + str(round(float(before_scores.mean()), 4))
        + " to "
        + str(round(float(after_scores.mean()), 4))
        + ". A permit issued on a request that fails more of its standards' requirements was the "
        "more often suspended, so the model must have learned from the screening counts."
    )
    clear = features[features[config.CONFLICTING_LIVE_PERMITS_COLUMN] == 0]
    beside = clear.copy()
    beside[config.CONFLICTING_LIVE_PERMITS_COLUMN] = 1
    beside[config.CONCURRENT_LIVE_PERMITS_COLUMN] = (
        beside[config.CONCURRENT_LIVE_PERMITS_COLUMN] + 1)
    beside[config.CONFLICTING_PERMIT_TRIGGER_COLUMN] = 1
    clear_scores = model.predict_proba(clear)[:, 1]
    beside_scores = model.predict_proba(beside)[:, 1]
    rising = float((beside_scores > clear_scores).mean())
    assert len(clear) > 50 and rising >= 0.95 and (
            beside_scores.mean() - clear_scores.mean()) >= 0.05, (
        "Putting one conflicting permit live beside the work, with its SIMOPS trigger flagged, "
        "raises the suspension score for "
        + str(round(100 * rising, 1))
        + "% of the "
        + str(len(clear))
        + " held-out requests with none, and moves the average from "
        + str(round(float(clear_scores.mean()), 4))
        + " to "
        + str(round(float(beside_scores.mean()), 4))
        + ". A permit worked beside a conflicting one is the more often suspended for "
        "conflicting work, so the model must have learned it from the register counts."
    )


def test_expected_hours_rise_on_live_unit_and_conflict(processed_frames, saved_models):
    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import add_simops_trigger_flags, build_feature_frame

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    logs = processed_frames[config.CLEAN_JOB_LOGS_EVAL_FILE]
    shut_down = logs[logs[config.PROCESS_UNIT_STATUS_COLUMN] == config.SHUT_DOWN]
    operating = shut_down.assign(**{config.PROCESS_UNIT_STATUS_COLUMN: config.OPERATING})
    clear = logs[logs[config.CONFLICTING_LIVE_PERMITS_COLUMN] == 0]
    beside = clear.assign(**{config.CONFLICTING_LIVE_PERMITS_COLUMN: 1,
                             config.CONCURRENT_LIVE_PERMITS_COLUMN:
                                 clear[config.CONCURRENT_LIVE_PERMITS_COLUMN] + 1})
    try:
        flagged = [add_simops_trigger_flags(frame) for frame in [shut_down, operating, clear,
                                                                 beside]]
    except Exception as error:
        flagged = error
    assert isinstance(flagged, list) and all(isinstance(frame, pd.DataFrame)
                                             for frame in flagged), (
        "add_simops_trigger_flags gave "
        + repr(flagged)[:200]
        + ", where it returns the job logs with their trigger flags beside them."
    )
    features = [build_feature_frame(frame, config.WORK_HOURS_NUMERIC_FEATURES,
                                    config.WORK_HOURS_CATEGORICAL_FEATURES) for frame in flagged]
    assert all(isinstance(frame, pd.DataFrame) for frame in features), (
        "build_feature_frame returned "
        + type(features[0]).__name__
        + ", where it returns the work hours model's inputs."
    )
    model = saved_models["work_hours"]
    for label, shorter, longer, count in [
            ("on an operating unit rather than one shut down", features[0], features[1],
             len(shut_down)),
            ("beside a conflicting permit rather than with none live", features[2], features[3],
             len(clear))]:
        shorter_hours = model.predict(shorter)
        longer_hours = model.predict(longer)
        lengthened = float((longer_hours > shorter_hours).mean())
        assert count > 50 and lengthened >= 0.9 and (
                longer_hours.mean() - shorter_hours.mean()) >= 1.0, (
            "With the work "
            + label
            + ", the expected work hours rise for "
            + str(round(100 * lengthened, 1))
            + "% of the "
            + str(count)
            + " held-out jobs and on average from "
            + str(round(float(shorter_hours.mean()), 1))
            + " to "
            + str(round(float(longer_hours.mean()), 1))
            + " work hours. Work on a live unit waits for the process and work beside a "
            "conflicting permit for the other job, so each takes longer, and the model is fed "
            "the unit's status, the register counts and the SIMOPS triggers."
        )


def test_bands_and_extension_follow_thresholds(live_files, saved_models, permit_standards):
    from decimal import Decimal

    import pandas as pd

    from config import config
    from data_pipeline.feature_builder import (add_requirement_counts, add_simops_trigger_flags,
                                               build_feature_frame)
    from tools.prediction_tools import (describe_expected_work_hours,
                                        estimate_expected_work_hours,
                                        estimate_suspension_likelihood)
    from tools.request_data_tools import fetch_request
    from tools.screening_tools import count_live_permits, screen_request

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    standards = permit_standards["standards"]
    band_ranges = {
        config.HIGH_LIKELIHOOD_BAND: (float(config.HIGH_LIKELIHOOD_MINIMUM), float("inf")),
        config.MEDIUM_LIKELIHOOD_BAND: (float(config.MEDIUM_LIKELIHOOD_MINIMUM),
                                        float(config.HIGH_LIKELIHOOD_MINIMUM)),
        config.LOW_LIKELIHOOD_BAND: (float("-inf"), float(config.MEDIUM_LIKELIHOOD_MINIMUM))}
    bands = {band: 0 for band in band_ranges}
    given = []
    for permit_reference in live_files["requests"][config.PERMIT_REFERENCE_COLUMN]:
        request = fetch_request(live_files["requests"], permit_reference)
        try:
            register_count = count_live_permits(live_files["register"], request)
            for count_column in [config.CONCURRENT_LIVE_PERMITS_COLUMN,
                                 config.CONFLICTING_LIVE_PERMITS_COLUMN]:
                request[count_column] = register_count[count_column]
            screening = screen_request(request, standards)
            likelihood = estimate_suspension_likelihood(
                saved_models["suspension"], saved_models["suspension_schema"], request, screening)
            work_hours = estimate_expected_work_hours(
                saved_models["work_hours"], saved_models["work_hours_schema"], request, screening)
        except Exception as error:
            screening = likelihood = work_hours = error
        assert (isinstance(screening, dict)
                and set(config.REQUIREMENT_COUNT_RESULTS) <= set(screening)
                and isinstance(screening.get("simops_triggers"), list)
                and isinstance(likelihood, dict)
                and {"suspension_likelihood", "likelihood_band"} <= set(likelihood)
                and isinstance(work_hours, dict)
                and {"expected_work_hours", "planned_work_hours", "extension_expected"}
                <= set(work_hours)
                and isinstance(work_hours["expected_work_hours"], (int, float))), (
            "For "
            + permit_reference
            + " the screening, the likelihood and the expected work hours came back as "
            + repr(screening)[:120]
            + ", "
            + repr(likelihood)[:120]
            + " and "
            + repr(work_hours)[:120]
            + ", where the screening carries its counts and SIMOPS triggers, the likelihood its "
            "figure and band, and the expected work hours a number with the planned work hours "
            "and whether an extension is expected."
        )
        given.append((request, screening, likelihood, work_hours))
    try:
        frame = add_simops_trigger_flags(add_requirement_counts(
            pd.DataFrame([request for request, _screening, _likelihood, _work_hours in given]),
            standards))
    except Exception as error:
        frame = error
    assert isinstance(frame, pd.DataFrame) and len(frame) == len(given), (
        "add_requirement_counts and add_simops_trigger_flags gave "
        + repr(frame)[:200]
        + " for the live queue, where they give its requests back in order, with their "
        "requirement counts and trigger flags, prepared as the models were trained."
    )
    suspension_inputs = build_feature_frame(frame, config.SUSPENSION_NUMERIC_FEATURES,
                                            config.SUSPENSION_CATEGORICAL_FEATURES)
    work_hours_inputs = build_feature_frame(frame, config.WORK_HOURS_NUMERIC_FEATURES,
                                            config.WORK_HOURS_CATEGORICAL_FEATURES)
    assert isinstance(suspension_inputs, pd.DataFrame) and isinstance(work_hours_inputs,
                                                                      pd.DataFrame), (
        "build_feature_frame returned "
        + type(suspension_inputs).__name__
        + ", where it returns a table of the model's inputs."
    )
    scores = saved_models["suspension"].predict_proba(suspension_inputs)[:, 1]
    forecasts = saved_models["work_hours"].predict(work_hours_inputs)
    extensions = {True: 0, False: 0}
    for (request, _screening, likelihood, work_hours), score, forecast in zip(
            given, scores, forecasts, strict=True):
        figure = likelihood["suspension_likelihood"]
        band = likelihood["likelihood_band"]
        lowest, highest = band_ranges.get(band, (None, None))
        assert (isinstance(figure, float) and abs(figure - float(score)) <= 0.005 + 1e-9
                and abs(round(figure * 100) - figure * 100) < 1e-6
                and lowest is not None and lowest <= figure < highest), (
            request[config.PERMIT_REFERENCE_COLUMN]
            + " is given a suspension likelihood of "
            + str(figure)
            + ", band "
            + str(band)
            + ", where its saved model scores it "
            + str(round(float(score), 4))
            + ". The figure is the model's score to "
            + str(config.LIKELIHOOD_DECIMALS)
            + " decimals, and its band follows config.HIGH_LIKELIHOOD_MINIMUM ("
            + str(config.HIGH_LIKELIHOOD_MINIMUM)
            + ") and config.MEDIUM_LIKELIHOOD_MINIMUM ("
            + str(config.MEDIUM_LIKELIHOOD_MINIMUM)
            + "), applied to the rounded figure."
        )
        bands[band] += 1
        planned = request[config.PLANNED_WORK_HOURS_COLUMN]
        expected_hours = work_hours["expected_work_hours"]
        flag = work_hours["extension_expected"]
        flagged_right = (
            (planned is None and flag is None)
            or (planned is not None and flag is (Decimal(str(expected_hours))
                                                 > Decimal(str(planned))
                                                 * config.EXTENSION_RATIO)))
        if planned is not None and isinstance(flag, bool):
            extensions[flag] += 1
        assert (abs(round(expected_hours * 10) - expected_hours * 10) < 1e-6
                and abs(expected_hours - float(forecast)) <= 0.05 + 1e-9
                and work_hours["planned_work_hours"] == planned and flagged_right), (
            request[config.PERMIT_REFERENCE_COLUMN]
            + " is expected to take "
            + str(expected_hours)
            + " work hours against "
            + str(planned)
            + " planned, with an extension expected "
            + str(flag)
            + ", where its saved model gives "
            + str(round(float(forecast), 2))
            + ". The expected work hours are the model's forecast to "
            + str(config.WORK_HOURS_DECIMALS)
            + " decimal, and extension_expected is True only when that rounded figure exceeds "
            + "config.EXTENSION_RATIO ("
            + str(config.EXTENSION_RATIO)
            + ") times the planned work hours."
        )
        expected_text = format(expected_hours, "." + str(config.WORK_HOURS_DECIMALS) + "f")
        planned_text = (None if planned is None
                        else format(planned, "." + str(config.WORK_HOURS_DECIMALS) + "f"))
        wording = {
            None: config.EXPECTED_HOURS_WITHOUT_PLAN_TEMPLATE.format(
                expected_work_hours=expected_text),
            True: config.EXTENSION_TEMPLATE.format(expected_work_hours=expected_text,
                                                   planned_work_hours=planned_text),
            False: config.NO_EXTENSION_TEMPLATE.format(expected_work_hours=expected_text,
                                                       planned_work_hours=planned_text)}
        assert describe_expected_work_hours(work_hours) == wording[flag], (
            "The expected work hours of "
            + request[config.PERMIT_REFERENCE_COLUMN]
            + " are described as "
            + repr(describe_expected_work_hours(work_hours))
            + " where the configuration's words for an extension expected "
            + str(flag)
            + " give "
            + repr(wording[flag])
            + "."
        )
    assert min(bands.values()) > 0 and min(extensions.values()) > 0, (
        "The live queue's likelihoods fall only in the bands "
        + str(bands)
        + " and its extensions only in "
        + str(extensions)
        + ", so the thresholds cannot all be told apart."
    )
    unplanned = dict(given[0][0])
    unplanned[config.PLANNED_WORK_HOURS_COLUMN] = None
    unplanned_hours = estimate_expected_work_hours(
        saved_models["work_hours"], saved_models["work_hours_schema"], unplanned, given[0][1])
    assert (unplanned_hours["extension_expected"] is None
            and unplanned_hours["planned_work_hours"] is None
            and describe_expected_work_hours(unplanned_hours)
            == config.EXPECTED_HOURS_WITHOUT_PLAN_TEMPLATE.format(
                expected_work_hours=format(unplanned_hours["expected_work_hours"],
                                           "." + str(config.WORK_HOURS_DECIMALS) + "f"))), (
        "A request with no planned work hours is described as "
        + repr(describe_expected_work_hours(unplanned_hours))
        + " with an extension expected "
        + str(unplanned_hours["extension_expected"])
        + ". With nothing planned there is nothing to compare: no extension is worked out."
    )
    first_request, first_screening = given[0][0], given[0][1]
    other_request = given[1][0]
    for label, estimate_name in [("suspension likelihood", "likelihood"),
                                 ("expected work hours", "work_hours")]:
        refusal = None
        try:
            if estimate_name == "likelihood":
                estimate_suspension_likelihood(saved_models["suspension"],
                                               saved_models["suspension_schema"],
                                               other_request, first_screening)
            else:
                estimate_expected_work_hours(saved_models["work_hours"],
                                             saved_models["work_hours_schema"],
                                             other_request, first_screening)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert (refusal is not None
                and first_request[config.PERMIT_REFERENCE_COLUMN] in refusal
                and other_request[config.PERMIT_REFERENCE_COLUMN] in refusal), (
            "A "
            + label
            + " worked out for "
            + other_request[config.PERMIT_REFERENCE_COLUMN]
            + " with the screening of "
            + first_request[config.PERMIT_REFERENCE_COLUMN]
            + " gave "
            + str(refusal)
            + ". A screening belongs to one request and is refused for any other, naming both."
        )


def test_sentence_model_macro_f1(processed_frames, saved_models):
    import copy

    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    from config import config
    from modeling.model_evaluation import evaluate_text_classifier

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    sentences = processed_frames[config.CLEAN_STATEMENT_SENTENCES_FILE]
    train_text, held_out_text, train_answers, held_out_answers = train_test_split(
        sentences[config.SENTENCE_TEXT_COLUMN], sentences[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE,
        stratify=sentences[config.SENTENCE_TARGET_COLUMN], random_state=config.RANDOM_SEED)
    model = saved_models["sentence"]
    measured = f1_score(held_out_answers, model.predict(held_out_text), average="macro")
    assert measured >= config.MINIMUM_SENTENCE_MACRO_F1, (
        "The sentence model scored "
        + str(round(measured, 4))
        + " macro F1 on the held-out statement sentences, below the "
        + str(config.MINIMUM_SENTENCE_MACRO_F1)
        + " the configuration sets. Macro F1 weighs the rarest type of sentence as much as the "
        "commonest: a gas test read as background is a reading never held to the request."
    )
    from_gas_records = (sentences.loc[train_text.index, config.DOCUMENT_TYPE_COLUMN]
                        == config.GAS_TEST_RECORD)
    probe_text = train_text[from_gas_records].head(3000)
    probe_answers = train_answers[from_gas_records].head(3000)
    evaluation = evaluate_text_classifier(copy.deepcopy(model), probe_text, probe_answers,
                                          held_out_text, held_out_answers)
    missing = sorted({"held_out_macro_f1", "train_macro_f1", "generalisation_gap",
                      "per_class_f1"} - set(evaluation if isinstance(evaluation, dict) else {}))
    assert len(missing) == 0, (
        "evaluate_text_classifier returned "
        + type(evaluation).__name__
        + " without "
        + str(missing)
        + ", so the sentence model's scores cannot be read."
    )
    trained_on = f1_score(probe_answers, model.predict(probe_text), average="macro")
    assert (abs(evaluation["held_out_macro_f1"] - measured) < 0.006
            and abs(evaluation["train_macro_f1"] - trained_on) < 0.006
            and abs(evaluation["generalisation_gap"]
                    - (evaluation["train_macro_f1"] - evaluation["held_out_macro_f1"])) < 0.011), (
        "Given the held-out sentences and the first training sentences of gas test records, the "
        "evaluation reports macro F1 "
        + str(evaluation["held_out_macro_f1"])
        + " held out and "
        + str(evaluation["train_macro_f1"])
        + " on training, with a gap of "
        + str(evaluation["generalisation_gap"])
        + ", but the model's own predictions give "
        + str(round(measured, 4))
        + " and "
        + str(round(trained_on, 4))
        + ". evaluate_text_classifier scores the fitted model it is given on the sentences it "
        "is given, without fitting it again."
    )


def test_sentence_predictions_reach_every_type(processed_frames, saved_models):
    from sklearn.model_selection import train_test_split

    from config import config

    assert processed_frames["problem"] is None, processed_frames["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    vocabulary = sorted(set(config.SENTENCE_TYPE_MAPPING.values()))
    sentences = processed_frames[config.CLEAN_STATEMENT_SENTENCES_FILE]
    found_types = sorted(str(value) for value in
                         sentences[config.SENTENCE_TARGET_COLUMN].dropna().unique())
    unlabelled = int(sentences[config.SENTENCE_TARGET_COLUMN].isna().sum())
    assert unlabelled == 0 and found_types == vocabulary, (
        "The cleaned sentences hold "
        + str(unlabelled)
        + " with no label, and are labelled "
        + str(found_types)
        + " where the sentence types are "
        + str(vocabulary)
        + "."
    )
    model = saved_models["sentence"]
    learned = sorted(str(label) for label in getattr(model, "classes_", []))
    assert learned == vocabulary, (
        "The sentence model knows the classes "
        + str(learned)
        + " where the sentence types are "
        + str(vocabulary)
        + ". A type of sentence the model cannot predict is a type no Area Authority ever sees."
    )
    split = train_test_split(
        sentences[config.SENTENCE_TEXT_COLUMN], sentences[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE,
        stratify=sentences[config.SENTENCE_TARGET_COLUMN], random_state=config.RANDOM_SEED)
    held_out_text, held_out_answers = split[1], split[3]
    predicted = [str(label) for label in model.predict(held_out_text)]
    stray = sorted(set(predicted) - set(vocabulary))
    unreached = sorted(set(vocabulary) - set(predicted))
    assert len(stray) == 0 and len(unreached) == 0, (
        "On the held-out sentences the sentence model predicts "
        + str(stray)
        + " outside the vocabulary and never predicts "
        + str(unreached)
        + "."
    )
    for sentence_type in vocabulary:
        shown = held_out_text[held_out_answers == sentence_type]
        right = sum(1 for label in model.predict(shown) if str(label) == sentence_type)
        assert right >= 0.75 * len(shown), (
            "Of the "
            + str(len(shown))
            + " held-out sentences labelled "
            + sentence_type
            + ", the model reads "
            + str(right)
            + " as that type. Every type of statement sentence is told apart, not only the "
            "common ones."
        )
    unseen = [str(label) for label in model.predict(["Zzq wvx plorth.",
                                                     "No words of this sort were ever written."])]
    assert all(label in vocabulary for label in unseen), (
        "Sentences of words never seen in training were read as "
        + str(unseen)
        + ", outside the sentence types."
    )


def test_statement_text_cleaning_leaves_plain_words(processed_frames):
    import html

    import pandas as pd

    from config import config
    from data_pipeline.common_cleaning import clean_statement_text

    assert processed_frames["problem"] is None, processed_frames["problem"]
    sentences = processed_frames[config.CLEAN_STATEMENT_SENTENCES_FILE]
    texts = sentences[config.SENTENCE_TEXT_COLUMN].map(str)
    left_markup = int(texts.str.contains(r"</?[A-Za-z][A-Za-z0-9]*[ />]", regex=True).sum())
    left_entities = int(texts.str.contains(r"&[#a-z0-9]+;|\xa0", regex=True).sum())
    doubled_spaces = int(texts.str.contains(r"\s{2,}|^\s|\s$", regex=True).sum())
    assert left_markup == 0 and left_entities == 0 and doubled_spaces == 0, (
        config.CLEAN_STATEMENT_SENTENCES_FILE
        + " still holds "
        + str(left_markup)
        + " sentences with markup, "
        + str(left_entities)
        + " with escaped characters and "
        + str(doubled_spaces)
        + " with stray spaces. A statement sentence is read as plain words."
    )
    lengths = sentences[config.SENTENCE_LENGTH_COLUMN]
    miscounted = int((lengths != texts.str.len()).sum())
    assert miscounted == 0, (
        str(miscounted)
        + " cleaned sentences carry a length that is not the length of their cleaned text. The "
        "length is counted again once the markup is gone."
    )
    plain = texts[texts.str.contains("[A-Z]", regex=True)
                  & texts.str.contains("[a-z]", regex=True)
                  & ~texts.str.contains("[&<>]", regex=True)].head(40).reset_index(drop=True)
    expected = clean_statement_text(plain)
    assert isinstance(expected, pd.Series) and len(expected) == len(plain), (
        "clean_statement_text returned "
        + type(expected).__name__
        + " for "
        + str(len(plain))
        + " plain sentences, where it returns a series of the same length, one cleaned text "
        "for each."
    )
    variants = {
        "wrapped in markup": "<div>" + plain.str.replace(" ", " <br/> ", n=1, regex=False)
                             + "</span>",
        "with its characters escaped": plain.map(html.escape).str.replace(" ", "&nbsp;", n=1,
                                                                          regex=False),
        "with no-break and extra spaces": "  " + plain.str.replace(" ", "\xa0 ", n=1,
                                                                   regex=False).str.replace(
            " ", "   ", regex=False) + " ",
    }
    for label, written in variants.items():
        cleaned = clean_statement_text(pd.Series(written))
        assert isinstance(cleaned, pd.Series) and len(cleaned) == len(plain), (
            "clean_statement_text returned "
            + type(cleaned).__name__
            + " for sentences "
            + label
            + ", where it returns a series of the same length."
        )
        differing = [(written.iloc[position], cleaned.iloc[position])
                     for position in range(len(plain))
                     if cleaned.iloc[position] != expected.iloc[position]]
        assert len(differing) == 0, (
            "A sentence "
            + label
            + " cleans to something else than the plain sentence does, for example "
            + str(differing[:2])
            + ". Markup, character entities, no-break spaces and runs of spaces are taken out, "
            "leaving the words the writer meant."
        )
    shouted = clean_statement_text(plain.str.upper())
    assert isinstance(shouted, pd.Series) and len(shouted) == len(plain), (
        "clean_statement_text returned "
        + type(shouted).__name__
        + " for sentences written in capitals, where it returns a series of the same length."
    )
    differing = [(plain.iloc[position], shouted.iloc[position]) for position in range(len(plain))
                 if shouted.iloc[position] != str(expected.iloc[position]).lower()]
    assert len(differing) == 0, (
        "A sentence written in capitals cleans to something other than the plain sentence in "
        "small letters, for example "
        + str(differing[:2])
        + ". Capitals carry no meaning the model was trained on, and the case cannot be "
        "restored further than small letters."
    )
    kept_case = clean_statement_text(plain)
    assert list(kept_case) == list(plain), (
        "A plain sentence written in mixed case is changed by cleaning, for example "
        + str([(plain.iloc[position], kept_case.iloc[position]) for position in range(len(plain))
               if kept_case.iloc[position] != plain.iloc[position]][:2])
        + ". Only a sentence written wholly in capitals is lowered."
    )


def test_statements_split_into_ordered_sentences(live_files, saved_models):
    import re

    from config import config
    from tools.evidence_tools import read_document_evidence
    from tools.request_data_tools import fetch_request, fetch_request_documents

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    vocabulary = set(config.SENTENCE_TYPE_MAPPING.values())
    requests = live_files["requests"]
    documents_file = live_files["documents"]
    read_requests = 0
    for permit_reference in list(requests[config.PERMIT_REFERENCE_COLUMN])[:8]:
        request = fetch_request(requests, permit_reference)
        documents = fetch_request_documents(documents_file, request)
        assert isinstance(documents, list) and all(
                isinstance(document, dict) and {config.DOCUMENT_ID_COLUMN,
                                                config.DOCUMENT_TYPE_COLUMN,
                                                config.DOCUMENT_TEXT_COLUMN} <= set(document)
                for document in documents), (
            "fetch_request_documents returned "
            + str(documents)[:200]
            + " for "
            + permit_reference
            + ", where it returns the documents attached to the request, each with its "
            "identifier, type and text."
        )
        attached = set(documents_file.loc[documents_file[config.PERMIT_REFERENCE_COLUMN]
                                          == permit_reference, config.DOCUMENT_ID_COLUMN])
        fetched = [document[config.DOCUMENT_ID_COLUMN] for document in documents]
        assert set(fetched) == attached and len(fetched) == len(attached) and fetched == sorted(
                fetched), (
            "The documents of "
            + permit_reference
            + " were fetched as "
            + str(fetched)
            + " where the documents attached to it are "
            + str(sorted(attached))
            + ". Every document attached to a request is fetched once, in the order of the "
            "document identifiers, whatever order they stand in the file."
        )
        try:
            findings = read_document_evidence(saved_models["sentence"], documents)
        except Exception as error:
            findings = error
        assert isinstance(findings, list) and all(
                isinstance(finding, dict) and {"document_id", "document_type", "sentence",
                                               "sentence_type"} <= set(finding)
                and isinstance(finding["sentence"], str) for finding in findings), (
            "read_document_evidence gave "
            + repr(findings)[:200]
            + " for the documents of "
            + permit_reference
            + ", where it returns one finding for each sentence, with its document's identifier "
            "and type, the sentence and the type the model reads in it."
        )
        read_requests += 1
        runs = [finding["document_id"] for position, finding in enumerate(findings)
                if position == 0 or findings[position - 1]["document_id"] != finding["document_id"]]
        assert runs == fetched, (
            "The sentences of "
            + permit_reference
            + " come from the documents in the runs "
            + str(runs)
            + " where the documents run "
            + str(fetched)
            + ". Each document's sentences are read together, in the order written, and the "
            "documents in their own order."
        )
        for document in documents:
            own = [finding["sentence"] for finding in findings
                   if finding["document_id"] == document[config.DOCUMENT_ID_COLUMN]]
            assert " ".join(own) == document[config.DOCUMENT_TEXT_COLUMN], (
                "The sentences read from "
                + document[config.DOCUMENT_ID_COLUMN]
                + " join back to "
                + repr(" ".join(own)[:200])
                + " instead of the document's own text. A sentence is kept whole, as written."
            )
            unsplit = [sentence for sentence in own
                       if re.search(config.STATEMENT_SENTENCE_PATTERN, sentence)]
            cut_short = ([sentence for sentence in own[:-1]
                          if not sentence.endswith((".", "!", "?"))]
                         + [sentence for sentence in own[1:] if not re.match("[A-Za-z0-9]",
                                                                             sentence)])
            assert len(unsplit) == 0 and len(cut_short) == 0, (
                "The document "
                + document[config.DOCUMENT_ID_COLUMN]
                + " was split into pieces that still hold the end of a sentence, "
                + str(unsplit[:2])
                + ", or that were cut where no sentence ends, "
                + str(cut_short[:2])
                + ". A sentence ends at a full stop, question mark or exclamation mark followed "
                "by a space and the next word, and only there."
            )
        for finding in findings:
            assert (finding.get("sentence_type") in vocabulary
                    and finding.get("document_type") == [
                        document[config.DOCUMENT_TYPE_COLUMN] for document in documents
                        if document[config.DOCUMENT_ID_COLUMN] == finding["document_id"]][0]), (
                "A sentence of "
                + str(finding.get("document_id"))
                + " is read as "
                + repr(finding.get("sentence_type"))
                + " in a document of the type "
                + repr(finding.get("document_type"))
                + ". Each sentence carries the type the model reads in it and the type of the "
                "document it comes from."
            )
    assert read_requests == 8, "Fewer than eight live requests could be read."
    assert read_document_evidence(saved_models["sentence"], []) == [], (
        "A request with no document attached gives statement evidence. With nothing attached, "
        "nothing is read."
    )
    lowered_text = ("the gas test read 0% lel at the work point. the fire watch stays for an "
                    "hour after the work. the area below is barriered.")
    lowered = read_document_evidence(saved_models["sentence"], [
        {config.DOCUMENT_ID_COLUMN: "DOC-990001",
         config.DOCUMENT_TYPE_COLUMN: config.METHOD_STATEMENT,
         config.DOCUMENT_TEXT_COLUMN: lowered_text}])
    pieces = ([finding.get("sentence") for finding in lowered if isinstance(finding, dict)]
              if isinstance(lowered, list) else [])
    assert len(pieces) == 3 and " ".join(str(piece) for piece in pieces) == lowered_text, (
        "A document of three sentences written in small letters, as cleaning leaves a document "
        "written in capitals, was split into "
        + str(pieces)
        + ". A sentence ends at its full stop whatever the case of the next word."
    )
    refusal = None
    try:
        read_document_evidence(saved_models["sentence"], [
            {config.DOCUMENT_ID_COLUMN: "DOC-990002",
             config.DOCUMENT_TYPE_COLUMN: config.TASK_RISK_ASSESSMENT,
             config.DOCUMENT_TEXT_COLUMN: None}])
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "DOC-990002" in refusal, (
        "A document whose text is blank gave "
        + str(refusal)
        + ". A document with nothing in it cannot be read and is refused with a ValueError "
        "naming the document."
    )


def test_sentences_link_requirements_and_gas_readings(live_files, saved_models,
                                                      permit_standards):
    from config import config
    from tools.evidence_tools import (find_disagreements, link_evidence_to_requirements,
                                      read_document_evidence)
    from tools.request_data_tools import fetch_request, fetch_request_documents
    from tools.screening_tools import count_live_permits, screen_request

    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    standards = permit_standards["standards"]
    requests = live_files["requests"]
    seen_types = set()
    for permit_reference in requests[config.PERMIT_REFERENCE_COLUMN]:
        request = fetch_request(requests, permit_reference)
        documents = fetch_request_documents(live_files["documents"], request)
        if request[config.PERMIT_TYPE_COLUMN] in seen_types or not documents:
            continue
        seen_types.add(request[config.PERMIT_TYPE_COLUMN])
        try:
            register_count = count_live_permits(live_files["register"], request)
            for count_column in [config.CONCURRENT_LIVE_PERMITS_COLUMN,
                                 config.CONFLICTING_LIVE_PERMITS_COLUMN]:
                request[count_column] = register_count[count_column]
            screening = screen_request(request, standards)
            findings = read_document_evidence(saved_models["sentence"], documents)
        except Exception as error:
            screening = findings = error
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and isinstance(findings, list)
                and all(isinstance(finding, dict) for finding in findings)), (
            "For "
            + permit_reference
            + " the screening came back as "
            + repr(screening)[:150]
            + " and the statement evidence as "
            + repr(findings)[:150]
            + ", where the screening holds its standards' requirements and the evidence one "
            "finding for each sentence."
        )
        linked = link_evidence_to_requirements(findings, screening)
        assert isinstance(linked, list) and len(linked) == len(findings), (
            "link_evidence_to_requirements returned "
            + str(linked)[:150]
            + " for the "
            + str(len(findings))
            + " sentences of "
            + permit_reference
            + ", where it returns one finding for each."
        )
        own_requirements = [{"permit_type": requirement.get("permit_type"),
                             "requirement_name": requirement.get("requirement_name")}
                            for requirement in screening["requirements"]]
        for finding, linked_finding in zip(findings, linked, strict=True):
            borne = linked_finding.get("requirements_borne_on")
            assert (isinstance(borne, list)
                    and borne == [requirement for requirement in own_requirements
                                  if requirement in borne]
                    and len(borne) == len({str(requirement) for requirement in borne})
                    and (finding.get("sentence_type") != config.BACKGROUND or borne == [])
                    and linked_finding.get("sentence") == finding.get("sentence")
                    and linked_finding.get("sentence_type") == finding.get("sentence_type")
                    and linked_finding.get("document_id") == finding.get("document_id")), (
                "A sentence of "
                + permit_reference
                + " read as "
                + str(finding.get("sentence_type"))
                + " bears on "
                + str(borne)
                + " where the request's own standards have the requirements "
                + str(own_requirements)
                + ". A sentence bears only on requirements its request's own standards have, "
                "each named with its permit type, in the screening's order, background on none, "
                "and keeps its sentence, type and document."
            )
    assert seen_types == set(config.PERMIT_STANDARD_DOCUMENTS), (
        "The live requests read covered the permit types "
        + str(sorted(seen_types))
        + ", where every permit type of the configuration has a live request with a document."
    )
    hot, entry = config.HOT_WORK, config.CONFINED_SPACE_ENTRY
    isolation, height, dig = config.ELECTRICAL_ISOLATION, config.WORK_AT_HEIGHT, config.EXCAVATION
    named = {"permit_reference": "PTW0950110", "permit_types": [hot, entry, isolation, height, dig],
             "requirements": []}
    for permit_type, kind, name in [
            (hot, config.RISK_ASSESSMENT_REQUIREMENT, "Job study"),
            (hot, config.FLAMMABLE_GAS_REQUIREMENT, "Gas at the torch"),
            (hot, config.OXYGEN_REQUIREMENT, "Air quality"),
            (hot, config.COMBUSTIBLES_CLEARED_REQUIREMENT, "Clear zone"),
            (hot, config.FIRE_WATCH_REQUIREMENT, "Watch after work"),
            (entry, config.HYDROGEN_SULPHIDE_REQUIREMENT, "Sour gas"),
            (entry, config.CARBON_MONOXIDE_REQUIREMENT, "Exhaust gas"),
            (entry, config.RESCUE_PLAN_REQUIREMENT, "Standby cover"),
            (isolation, config.PROVED_DEAD_REQUIREMENT, "Dead test"),
            (isolation, config.PERSONAL_LOCKS_REQUIREMENT, "Own padlocks"),
            (isolation, config.CIRCUIT_EARTHS_REQUIREMENT, "Earthing leads"),
            (height, config.FALL_PROTECTION_REQUIREMENT, "Harness"),
            (height, config.SCAFFOLD_INSPECTION_REQUIREMENT, "Tagged scaffold"),
            (height, config.BARRIERS_REQUIREMENT, "Drop zone"),
            (height, config.WIND_SPEED_REQUIREMENT, "Gust limit"),
            (dig, config.SERVICES_LOCATED_REQUIREMENT, "Cable scan"),
            (dig, config.TRENCH_SUPPORT_REQUIREMENT, "Shoring")]:
        named["requirements"].append({"permit_type": permit_type, "requirement_kind": kind,
                                      "requirement_name": name,
                                      "requirement_sentence": "An invented requirement: " + name
                                                              + ".",
                                      "result": config.REQUIREMENT_MET, "blank_answers": []})
    wanted = {
        config.ISOLATION: [(isolation, "Dead test"), (isolation, "Own padlocks"),
                           (isolation, "Earthing leads"), (dig, "Cable scan")],
        config.GAS_TESTING: [(hot, "Gas at the torch"), (hot, "Air quality"), (entry, "Sour gas"),
                             (entry, "Exhaust gas")],
        config.FIRE_PRECAUTIONS: [(hot, "Clear zone"), (hot, "Watch after work")],
        config.RESCUE: [(entry, "Standby cover")],
        config.ACCESS: [(height, "Harness"), (height, "Tagged scaffold"), (height, "Drop zone"),
                        (height, "Gust limit"), (dig, "Shoring")],
        config.PERSONAL_PROTECTIVE_EQUIPMENT: [], config.SIMULTANEOUS_OPERATIONS: [],
        config.HOUSEKEEPING_AND_WASTE: [], config.BACKGROUND: []}
    wanted = {sentence_type: [{"permit_type": permit_type, "requirement_name": name}
                              for permit_type, name in pairs]
              for sentence_type, pairs in wanted.items()}
    linked = link_evidence_to_requirements(
        [{"document_id": "DOC-990003", "document_type": config.METHOD_STATEMENT,
          "sentence": "A sentence read as " + sentence_type + ".", "sentence_type": sentence_type}
         for sentence_type in wanted], named)
    got = ({finding.get("sentence_type"): finding.get("requirements_borne_on")
            for finding in linked if isinstance(finding, dict)}
           if isinstance(linked, list) else linked)
    assert got == wanted, (
        "Under standards whose requirements carry names no Hexley standard uses, one sentence of "
        "each type bears on "
        + str(got)
        + " where they bear on "
        + str(wanted)
        + ". Each type of sentence bears on the requirements of the kinds its statement guide "
        "names, found by their kind, in the screening's own order and named with their permit "
        "type, and personal protective equipment, simultaneous operations, housekeeping and "
        "waste and background on none."
    )
    refusal = None
    try:
        link_evidence_to_requirements(
            [{"document_id": "DOC-990004", "document_type": config.METHOD_STATEMENT,
              "sentence": "A sentence of a type nobody configured.", "sentence_type": "Rumour"}],
            named)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "Rumour" in refusal, (
        "A sentence of a type the configuration gives no requirements for gave "
        + str(refusal)
        + ". It is refused with a ValueError naming the type."
    )
    gas, flammable, oxygen = config.GAS_TESTING, "flammable_gas_lel_percent", config.OXYGEN_COLUMN
    sulphide, monoxide = "hydrogen_sulphide_ppm", "carbon_monoxide_ppm"
    disagrees, blank = config.DISAGREES_WITH_FORM, config.DESCRIBES_BLANK_ANSWER
    hot_sentences = [
        (gas, "The pre-job gas test read 0% LEL and 20.9% oxygen."),
        (gas, "A second test read 2% LEL at the flange."),
        (gas, "The tester recorded 20.90% oxygen before work."),
        (gas, "The monitor showed 3 ppm hydrogen sulphide by the drain."),
        (gas, "Readings were taken at the top, middle and bottom."),
        (config.BACKGROUND, "The last job here read 5% LEL before the tank was cleaned."),
        (gas, "Gas monitor: 2% lel at the flange, 19.45% oxygen."),
        (config.FIRE_PRECAUTIONS, "A fire watch of 60 minutes follows at 0% LEL.")]
    entry_sentences = [
        (gas, "The entry test read 4% LEL, 20.9% oxygen, 1.5 ppm hydrogen sulphide and 12 ppm "
              "carbon monoxide."),
        (gas, "Two readings of 4% LEL and 5% LEL were logged."),
        (gas, "Carbon monoxide stayed at 10.0 ppm carbon monoxide throughout.")]
    hot_request = {config.PERMIT_REFERENCE_COLUMN: "PTW0950111", flammable: 0, oxygen: 20.9,
                   sulphide: None, monoxide: None}
    entry_request = {config.PERMIT_REFERENCE_COLUMN: "PTW0950112", flammable: 4, oxygen: 20.9,
                     sulphide: 1.5, monoxide: 10.0}
    cases = [
        ("a hot work request reading 0 and 20.9, with no toxic gas reading", hot_sentences,
         hot_request,
         [(hot_sentences[1][1], {flammable: [2.0]}, {flammable: 0.0}, disagrees),
          (hot_sentences[3][1], {sulphide: [3.0]}, {sulphide: None}, blank),
          (hot_sentences[6][1], {flammable: [2.0], oxygen: [19.45]},
           {flammable: 0.0, oxygen: 20.9}, disagrees)]),
        ("a confined space entry reading 4, 20.9, 1.5 and 10.0", entry_sentences, entry_request,
         [(entry_sentences[0][1], {flammable: [4.0], oxygen: [20.9], sulphide: [1.5],
                                   monoxide: [12.0]},
           {flammable: 4.0, oxygen: 20.9, sulphide: 1.5, monoxide: 10.0}, disagrees),
          (entry_sentences[1][1], {flammable: [4.0, 5.0]}, {flammable: 4.0}, disagrees)])]
    for meaning, sentences, request, expected in cases:
        findings = [{"document_id": "DOC-990005", "document_type": config.GAS_TEST_RECORD,
                     "sentence": sentence, "sentence_type": sentence_type}
                    for sentence_type, sentence in sentences]
        flagged = find_disagreements(findings, request)
        assert isinstance(flagged, list) and all(isinstance(flag, dict) for flag in flagged), (
            "find_disagreements returned "
            + str(flagged)[:200]
            + ", where it returns one flag for each gas testing sentence it flags."
        )
        got = [(flag.get("sentence"), flag.get("stated_readings"), flag.get("request_readings"),
                flag.get("flag")) for flag in flagged]
        right_document = all(flag.get("document_id") == "DOC-990005" for flag in flagged)
        assert got == expected and right_document, (
            "For "
            + meaning
            + ", the gas testing sentences were flagged as "
            + str(got)
            + " where the Permit Rules flag "
            + str(expected)
            + ". A reading written as a number and % LEL, % oxygen, ppm hydrogen sulphide or "
            "ppm carbon monoxide, in any capitals, is held as written, without rounding, to the "
            "request's reading of the same gas: a sentence with no reading is not flagged, one "
            "reading a gas the request leaves blank describes a blank answer, any other "
            "difference disagrees with the form, and only gas testing sentences are read."
        )


def test_every_document_gives_up_text(document_texts, tmp_path):
    import os
    import shutil

    import pypdf

    from config import config
    from knowledge.pdf_extraction import extract_text_from_directory

    assert document_texts["problem"] is None, document_texts["problem"]
    in_folder = sorted(name for name in os.listdir(config.DOCS_DIRECTORY)
                       if name.lower().endswith(".pdf"))
    documents = document_texts["documents"]
    assert len(documents) > 0 and {"document_name", "document_text"} <= set(documents[0]), (
        "Reading the document folder returned "
        + str(len(documents))
        + " entries"
        + ("" if len(documents) == 0 else ", the first keyed " + str(sorted(documents[0])))
        + ", where every entry names its document and carries its text."
    )
    names = sorted(document["document_name"] for document in documents)
    assert names == in_folder, (
        "Reading the document folder returned "
        + str(names)
        + " where the folder holds "
        + str(in_folder)
        + ". A document never read is a question answered from general knowledge instead."
    )
    for document in documents:
        assert len(document["document_text"].split()) > 250, (
            document["document_name"]
            + " gave up only "
            + str(len(document["document_text"].split()))
            + " words. A PDF that will not give up its text is indexed as an empty passage."
        )
    for required in (list(config.PERMIT_STANDARD_DOCUMENTS.values())
                     + list(config.SENTENCE_TYPE_GUIDE_DOCUMENTS.values())
                     + config.CHAT_EXCLUDED_DOCUMENTS):
        assert required in names, (
            required
            + " was not read, and the configuration reads it by name."
        )
    folder = tmp_path / "mixed_folder"
    folder.mkdir()
    standard_document = config.PERMIT_STANDARD_DOCUMENTS[config.EXCAVATION]
    guide_document = config.SENTENCE_TYPE_GUIDE_DOCUMENTS[config.GAS_TESTING]
    for name in [standard_document, guide_document]:
        shutil.copy(os.path.join(config.DOCS_DIRECTORY, name), str(folder / name))
    (folder / "reading-list.txt").write_text("Not a document of the permit office.")
    (folder / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    mixed_names = None
    try:
        mixed_names = sorted(entry["document_name"]
                             for entry in extract_text_from_directory(str(folder)))
    except Exception as error:
        mixed_names = type(error).__name__ + ": " + str(error)[:200]
    assert mixed_names == sorted([standard_document, guide_document]), (
        "A folder holding two PDFs, a text file and an image was read as "
        + str(mixed_names)
        + ". Only the PDFs are the documents; anything else in the folder is passed over."
    )
    blank_folder = tmp_path / "blank_page_folder"
    blank_folder.mkdir()
    writer = pypdf.PdfWriter()
    writer.add_page(pypdf.PdfReader(os.path.join(config.DOCS_DIRECTORY,
                                                 standard_document)).pages[0])
    writer.add_blank_page()
    with open(str(blank_folder / "Standard-With-A-Blank-Page.pdf"), "wb") as blank_file:
        writer.write(blank_file)
    empty_folder = tmp_path / "empty_folder"
    empty_folder.mkdir()
    (empty_folder / "notes.txt").write_text("No PDF here.")
    for label, folder_path, named in [
            ("A PDF whose second page gives up no text", blank_folder,
             "Standard-With-A-Blank-Page.pdf"),
            ("A folder with no PDF in it", empty_folder, str(empty_folder))]:
        refusal = None
        try:
            extract_text_from_directory(str(folder_path))
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and named in refusal, (
            label
            + " gave "
            + str(refusal)
            + ". A page that cannot be read may be the one that states a requirement, and a "
            "folder with no document has nothing to answer from, so each is refused with a "
            "ValueError naming the file or the folder."
        )


def test_documents_cut_into_sections_and_passages(document_passages, permit_standards):
    import re

    from config import config
    from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections

    assert document_passages["problem"] is None, document_passages["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    documents = document_passages["documents"]
    sections = document_passages["sections"]
    chunks = document_passages["chunks"]
    assert len(chunks) > 0 and {"chunk_id", "chunk_text", "document_name"} <= set(chunks[0]), (
        "Cutting the documents returned "
        + str(len(chunks))
        + " passages"
        + ("" if len(chunks) == 0 else ", the first keyed " + str(sorted(chunks[0])))
        + ", where every passage carries its identifier, its text and its document."
    )
    assert len(sections) > 5 * len(documents), (
        str(len(documents))
        + " documents split into "
        + str(len(sections))
        + " sections. Each document is written in numbered sections, and a section is the unit "
        "the tools read and a question is usually about."
    )
    with_header = [chunk["chunk_id"] for chunk in chunks
                   if re.search(config.DOCUMENT_HEADER_PATTERN, chunk["chunk_text"])]
    assert len(with_header) == 0, (
        "These passages still carry the document header: "
        + str(with_header[:5])
        + ". The header repeats the same words in every document, so it matches every question "
        "equally and pushes the passages that answer it out of the results."
    )
    lengths = [len(chunk["chunk_text"]) for chunk in chunks]
    average = sum(lengths) / len(lengths)
    assert (config.CHUNK_SIZE_CHARACTERS / 5 <= average <= config.CHUNK_SIZE_CHARACTERS
            and max(lengths) <= config.CHUNK_SIZE_CHARACTERS), (
        "Passages average "
        + str(int(average))
        + " characters, the longest "
        + str(max(lengths))
        + ", where the configuration asks for passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + ". Passages far longer return a whole document for every question; far shorter lose "
        "the sentence that gave them their meaning."
    )
    identifiers = [chunk["chunk_id"] for chunk in chunks]
    assert len(identifiers) == len(set(identifiers)), (
        "Passage identifiers repeat, so storing them would overwrite one passage with another."
    )
    words_by_document = {document["document_name"]: set(document["document_text"].split())
                         for document in documents}
    for chunk in chunks:
        words = chunk["chunk_text"].split()
        foreign = [word for word in words
                   if word not in words_by_document.get(chunk["document_name"], set())]
        assert len(words) > 0 and len(foreign) <= len(words) // 10, (
            "Passage "
            + str(chunk["chunk_id"])
            + " says it comes from "
            + str(chunk["document_name"])
            + " but "
            + str(len(foreign))
            + " of its "
            + str(len(words))
            + " words are not in that document, for example "
            + str(foreign[:5])
            + ". A passage has to cite the document it was cut from."
        )
    for permit_type, standard in permit_standards["standards"].items():
        own_chunks = [re.sub(r"\s+", " ", chunk["chunk_text"]) for chunk in chunks
                      if chunk["document_name"] == config.PERMIT_STANDARD_DOCUMENTS[permit_type]]
        broken = [requirement["requirement_sentence"] for requirement in standard["requirements"]
                  if not any(requirement["requirement_sentence"] in chunk_text
                             for chunk_text in own_chunks)]
        assert len(broken) == 0, (
            "These requirements of "
            + config.PERMIT_STANDARD_DOCUMENTS[permit_type]
            + " are not whole inside any one of its passages: "
            + str(broken[:3])
            + ". A requirement cut across two passages can no longer be read from either."
        )
    header = ("Document Standard 99 | Issue 1 | Owner: Process Safety | Review cycle: every six "
              "months")
    lines = []
    for first in "abcdefgh":
        for second in "abcdefg":
            lines.append("Reading " + first + second + " of the overlap check names break "
                         + first + second + ".")
    synthetic = [{"document_name": "Overlap-Check.pdf",
                  "document_text": "HEXLEY CHEMICALS\nOverlap Check\n" + header
                                   + "\n1. Overlap check\n" + " ".join(lines)}]
    synthetic_chunks = build_chunk_records(split_documents_into_sections(synthetic))
    assert isinstance(synthetic_chunks, list) and all(
            isinstance(chunk, dict) and "chunk_text" in chunk for chunk in synthetic_chunks), (
        "Cutting a document of one long section returned "
        + str(synthetic_chunks)[:200]
        + ", where it returns a list of passages, each with its text."
    )
    pieces = [str(chunk["chunk_text"]) for chunk in synthetic_chunks]
    assert len(pieces) >= 3, (
        "One numbered section of "
        + str(len(" ".join(lines)))
        + " characters was cut into "
        + str(len(pieces))
        + " passages, where passages of up to "
        + str(config.CHUNK_SIZE_CHARACTERS)
        + " characters need at least three."
    )
    unshared = [position for position in range(1, len(pieces))
                if pieces[position].strip()[:20] not in pieces[position - 1]]
    assert len(unshared) == 0, (
        "Passages "
        + str(unshared)
        + " of one long section open with text the passage before them does not carry. "
        "Neighbouring passages of a section overlap by up to "
        + str(config.CHUNK_OVERLAP_CHARACTERS)
        + " characters, so a sentence cut at a boundary is still whole in one of them."
    )
    for description, document_text in [
            ("no header line", "HEXLEY CHEMICALS\n1. Overlap check\n" + " ".join(lines)),
            ("two header lines", header + "\n" + header + "\n1. Overlap check\n"
             + " ".join(lines)),
            ("sections numbered from 2", header + "\n2. Overlap check\n" + " ".join(lines))]:
        refusal = None
        try:
            split_documents_into_sections([{"document_name": "Broken-Layout.pdf",
                                            "document_text": document_text}])
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and "Broken-Layout.pdf" in refusal, (
            "A document with "
            + description
            + " was split, giving "
            + str(refusal)
            + ". Every document carries one header line and sections numbered from 1, so a "
            "document laid out otherwise has been read wrongly and is refused, naming it."
        )


def test_index_holds_every_passage_once(knowledge_collection, document_passages):
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    assert document_passages["problem"] is None, document_passages["problem"]
    collection = knowledge_collection["collection"]
    documents = document_passages["documents"]
    chunks = document_passages["chunks"]
    stored = collection.get(include=["metadatas", "documents"])
    assert collection.count() == len(chunks) == len(stored["ids"]), (
        "The knowledge index holds "
        + str(collection.count())
        + " passages where cutting the documents gives "
        + str(len(chunks))
        + ". More means the index was added to twice and repeats itself in every result; fewer "
        "means passages no search can reach."
    )
    assert len(set(stored["ids"])) == len(stored["ids"]), (
        "Stored passage identifiers repeat, so one passage has overwritten another and its text "
        "can no longer be found."
    )
    names = {document["document_name"] for document in documents}
    cited = [(metadata or {}).get("document_name") for metadata in stored["metadatas"]]
    unnamed = [value for value in cited if value not in names]
    assert len(unnamed) == 0, (
        str(len(unnamed))
        + " stored passages name no document from the folder, for example "
        + str(unnamed[:3])
        + ". The name is what lets an answer cite its source and a section be read from its "
        "own document alone."
    )
    by_document = {}
    for chunk in chunks:
        by_document[chunk["document_name"]] = by_document.get(chunk["document_name"], 0) + 1
    stored_by_document = {name: cited.count(name) for name in names}
    assert stored_by_document == by_document, (
        "Passages stored per document "
        + str(stored_by_document)
        + " differ from passages cut per document "
        + str(by_document)
        + ". A document with passages missing answers less than it says."
    )
    assert sorted(stored["documents"]) == sorted(chunk["chunk_text"] for chunk in chunks), (
        "The texts in the knowledge index are not the passages the documents are cut into, so "
        "a section read from the index is not the section the document states."
    )
    for chunk_id, text, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"],
                                        strict=True):
        twins = [chunk for chunk in chunks if chunk["chunk_id"] == chunk_id]
        assert (len(twins) == 1 and twins[0]["chunk_text"] == text
                and twins[0]["document_name"] == (metadata or {}).get("document_name")), (
            "The stored passage "
            + str(chunk_id)
            + " does not carry the text and document of the passage cut under that identifier."
        )


def test_tools_read_whole_own_sections(knowledge_collection, document_passages):
    from config import config
    from knowledge.vector_index import read_document_section
    from tools.retrieval_tools import (read_briefing_guidance, read_documentation_to_provide,
                                       read_permit_document_guidance, read_reviewer_checks)

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    assert document_passages["problem"] is None, document_passages["problem"]
    collection = knowledge_collection["collection"]
    unread = []
    for section in document_passages["sections"]:
        read_back = read_document_section(collection, section["document_name"],
                                          section["section_number"])
        if read_back != section["section_text"]:
            unread.append(section["document_name"] + " section "
                          + str(section["section_number"]) + " read as "
                          + repr(str(read_back)[:120]))
    assert len(unread) == 0, (
        str(len(unread))
        + " sections of the "
        + str(len(document_passages["sections"]))
        + " read back from the index differ from the documents, for example "
        + str(unread[:2])
        + ". A section is read back from its own document's passages alone, whole and only "
        "once, exactly as the documents were divided."
    )
    wanted = []
    for permit_type, document_name in config.PERMIT_STANDARD_DOCUMENTS.items():
        wanted.append(("the documentation to provide for " + permit_type, document_name,
                       config.DOCUMENTATION_SECTION_NUMBER, config.DOCUMENTATION_SECTION_TITLE,
                       read_documentation_to_provide(collection, permit_type)))
        wanted.append(("what the permit document of " + permit_type + " says", document_name,
                       config.PERMIT_DOCUMENT_SECTION_NUMBER,
                       config.PERMIT_DOCUMENT_SECTION_TITLE,
                       read_permit_document_guidance(collection, permit_type)))
        wanted.append(("what the work party briefing of " + permit_type + " says", document_name,
                       config.WORK_PARTY_BRIEFING_SECTION_NUMBER,
                       config.WORK_PARTY_BRIEFING_SECTION_TITLE,
                       read_briefing_guidance(collection, permit_type)))
    for sentence_type, document_name in config.SENTENCE_TYPE_GUIDE_DOCUMENTS.items():
        wanted.append(("what the reviewer checks in a sentence of the type " + sentence_type,
                       document_name, config.REVIEWER_CHECKS_SECTION_NUMBER,
                       config.REVIEWER_CHECKS_SECTION_TITLE,
                       read_reviewer_checks(collection, sentence_type)))
    for label, document_name, section_number, section_title, passage in wanted:
        heading = str(section_number) + ". " + section_title + " "
        sections = [section["section_text"] for section in document_passages["sections"]
                    if section["document_name"] == document_name
                    and section["section_text"].startswith(heading)]
        assert len(sections) == 1, (
            document_name
            + " holds "
            + str(len(sections))
            + " sections headed "
            + repr(heading.strip())
            + " where it holds one."
        )
        expected = sections[0][len(heading):]
        assert passage == expected, (
            "Reading "
            + label
            + " gave "
            + repr(str(passage)[:160])
            + " where "
            + document_name
            + "'s section "
            + repr(heading.strip())
            + " reads "
            + repr(expected[:160])
            + ". The tools read that section whole, from that document alone, without its "
            "heading."
        )
    for label, call_name, unknown in [
            ("the documentation of a permit type with no standard", "documentation",
             "Diving operations"),
            ("the permit document of a permit type with no standard", "permit", "Lifting"),
            ("the work party briefing of a permit type with no standard", "briefing",
             "Radiography"),
            ("the reviewer's checks for a sentence type with no guide", "checks", "Rumour"),
            ("a section the index does not hold", "section",
             config.PERMIT_STANDARD_DOCUMENTS[config.HOT_WORK])]:
        refusal = None
        try:
            if call_name == "documentation":
                read_documentation_to_provide(collection, unknown)
            elif call_name == "permit":
                read_permit_document_guidance(collection, unknown)
            elif call_name == "briefing":
                read_briefing_guidance(collection, unknown)
            elif call_name == "checks":
                read_reviewer_checks(collection, unknown)
            else:
                read_document_section(collection, unknown, 40)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and unknown in refusal, (
            "Asking for "
            + label
            + ", "
            + unknown
            + ", gave "
            + str(refusal)
            + ". It is refused with a ValueError naming it, never answered from another "
            "document or section."
        )


def test_chat_search_skips_casebook_nearest_first(knowledge_collection):
    from config import config
    from knowledge.vector_index import search_documents

    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    collection = knowledge_collection["collection"]
    stored = collection.get(include=["documents", "metadatas"])
    casebook_passages = [passage_id for passage_id, metadata
                         in zip(stored["ids"], stored["metadatas"], strict=True)
                         if (metadata or {}).get("document_name") in config.CHAT_EXCLUDED_DOCUMENTS]
    assert len(casebook_passages) > 0, (
        "The index holds no passage of "
        + str(config.CHAT_EXCLUDED_DOCUMENTS)
        + ", which is indexed with the other documents and left out of the chat's search only."
    )
    longest = {}
    for text, metadata in zip(stored["documents"], stored["metadatas"], strict=True):
        document_name = (metadata or {}).get("document_name")
        if (document_name not in config.CHAT_EXCLUDED_DOCUMENTS
                and len(text) > len(longest.get(document_name, ""))):
            longest[document_name] = text
    questions = [(text, True) for text in longest.values()]
    questions.append(("What did the SIMOPS panel decide when a past request was referred to it?",
                      False))
    questions.append(("PTW0041207 was sent on scanned paper for hot work on a flange in Reactor "
                      "House A.", False))
    results = search_documents(collection, [pair[0] for pair in questions])
    assert isinstance(results, list) and len(results) == len(questions), (
        "search_documents answered "
        + str(results)[:200]
        + " for "
        + str(len(questions))
        + " questions, where it answers one list of passages a question, in order."
    )
    for (question, stored_passage), passages in zip(questions, results, strict=True):
        assert (isinstance(passages, list) and len(passages) == config.CHAT_RETRIEVAL_RESULT_COUNT
                and all(isinstance(passage, dict)
                        and {"chunk_id", "chunk_text", "document_name", "distance"} <= set(passage)
                        for passage in passages)), (
            "The search for "
            + repr(question[:80])
            + " returned "
            + str(passages)[:200]
            + " where it returns "
            + str(config.CHAT_RETRIEVAL_RESULT_COUNT)
            + " passages, each with its identifier, text, document and distance."
        )
        distances = [float(passage["distance"]) for passage in passages]
        found = [passage["document_name"] for passage in passages]
        assert distances == sorted(distances) and all(-0.0001 <= value <= 2.0001
                                                      for value in distances), (
            "The search for "
            + repr(question[:80])
            + " returned distances "
            + str([round(value, 3) for value in distances])
            + ". Passages come nearest first, at cosine distances between 0 and 2."
        )
        excluded = [name for name in found if name in config.CHAT_EXCLUDED_DOCUMENTS]
        assert len(excluded) == 0, (
            "The search for "
            + repr(question[:80])
            + " returned passages of "
            + str(excluded)
            + ". The casebook records past reviews of named requests and is never quoted by the "
            "chat, however close its passages are."
        )
        if stored_passage:
            assert passages[0]["chunk_text"] == question, (
                "Searching with the words of the stored passage "
                + repr(question[:80])
                + " returned "
                + repr(str(passages[0]["chunk_text"])[:80])
                + " first. No passage is nearer to a passage's own words than the passage "
                "itself, so it comes back first."
            )


def test_each_permit_type_read_from_own_standard(invented_standards):
    from config import config

    assert invented_standards["problem"] is None, invented_standards["problem"]
    standards = invented_standards["standards"]
    assert sorted(standards) == sorted(config.PERMIT_STANDARD_DOCUMENTS), (
        "Standards were read for "
        + str(sorted(standards))
        + " where every one of the five permit types has its own."
    )
    for permit_type, requirements in invented_standards["invented"].items():
        standard = standards[permit_type]
        document_name = config.PERMIT_STANDARD_DOCUMENTS[permit_type]
        assert (standard.get("permit_type") == permit_type
                and standard.get("document_name") == document_name), (
            "The standard read for "
            + permit_type
            + " names the permit type "
            + repr(standard.get("permit_type"))
            + " and the document "
            + repr(standard.get("document_name"))
            + ". A standard is read from the document configured for its own permit type and "
            "names both."
        )
        read = [(requirement.get("requirement_name"), requirement.get("requirement_kind"))
                for requirement in standard.get("requirements", [])]
        stated = [(requirement[1], requirement[0]) for requirement in requirements]
        assert read == stated, (
            "A standard written for the checks under the name "
            + document_name
            + " was read as the requirements "
            + str(read)
            + " where its section 4 states "
            + str(stated)
            + ", in that order. Each numbered requirement is read whole, named as the standard "
            "names it, from the standard of its own permit type alone: the hot work and confined "
            "space entry standards read alike, and neither lends the other a requirement."
        )


def test_requirements_keep_sentence_and_figures(invented_standards):
    from decimal import Decimal

    from config import config

    assert invented_standards["problem"] is None, invented_standards["problem"]
    lowest, highest, threshold = config.MINIMUM_RULE, config.MAXIMUM_RULE, config.THRESHOLD_RULE
    stated = {
        config.HOT_WORK: [{}, {highest: Decimal("4")},
                          {lowest: Decimal("19.8"), highest: Decimal("22.6")},
                          {lowest: Decimal("11.5")}, {lowest: Decimal("45")}],
        config.CONFINED_SPACE_ENTRY: [{}, {highest: Decimal("7")},
                                      {lowest: Decimal("19.7"), highest: Decimal("22.9")},
                                      {highest: Decimal("4.5")}, {highest: Decimal("25")}, {}],
        config.ELECTRICAL_ISOLATION: [{}, {threshold: Decimal("650")}, {}, {}],
        config.WORK_AT_HEIGHT: [{}, {}, {}, {highest: Decimal("9.5")}, {}],
        config.EXCAVATION: [{}, {threshold: Decimal("1.5")}, {}, {}],
    }
    for permit_type, figures in stated.items():
        document_name = config.PERMIT_STANDARD_DOCUMENTS[permit_type]
        requirements = invented_standards["standards"][permit_type].get("requirements", [])
        written = invented_standards["invented"][permit_type]
        assert len(requirements) == len(figures), (
            document_name
            + " was read as "
            + str(len(requirements))
            + " requirements where it states "
            + str(len(figures))
            + "."
        )
        for requirement, wanted, stated_requirement in zip(requirements, figures, written,
                                                           strict=True):
            sentence = requirement.get("requirement_sentence")
            assert sentence == stated_requirement[2], (
                "The requirement "
                + str(requirement.get("requirement_name"))
                + " of "
                + document_name
                + " keeps the sentence "
                + repr(str(sentence)[:160])
                + " where the standard states "
                + repr(stated_requirement[2])
                + ". A requirement keeps the sentence it was read from, word for word, so the "
                "review and the rework request quote the standard itself."
            )
            rules = requirement.get("rules", {})
            exact = isinstance(rules, dict) and all(isinstance(figure, Decimal)
                                                    for figure in rules.values())
            assert rules == wanted and exact, (
                "The requirement "
                + str(requirement.get("requirement_name"))
                + " of "
                + document_name
                + " was read with the figures "
                + repr(rules)
                + " where its sentence states "
                + repr(wanted)
                + ". Each requirement keeps its own figures, read from its own standard as exact "
                "decimals, and no others; none of these figures is Hexley's, so none can come "
                "from anywhere but the standard read."
            )


def test_twin_gas_figures_read_from_own_documents(invented_standards):
    from decimal import Decimal

    from config import config
    from knowledge.standard_reading import read_permit_standard

    assert invented_standards["problem"] is None, invented_standards["problem"]
    documents = invented_standards["documents"]
    reordered = list(reversed(documents))
    for permit_type in config.PERMIT_STANDARD_DOCUMENTS:
        again = read_permit_standard(permit_type, reordered)
        assert again == invented_standards["standards"][permit_type], (
            "The standard of "
            + permit_type
            + " reads differently when the documents are handed over in another order. A "
            "standard is found by its document's name, wherever it stands among the documents."
        )
    entry_document = config.PERMIT_STANDARD_DOCUMENTS[config.CONFINED_SPACE_ENTRY]
    amended = []
    for document in documents:
        document_text = document["document_text"]
        if document["document_name"] == entry_document:
            document_text = document_text.replace("is at most 7 percent of the lower",
                                                  "is at most 8 percent of the lower")
        amended.append({"document_name": document["document_name"],
                        "document_text": document_text})
    amended_figures = {}
    for permit_type in [config.HOT_WORK, config.CONFINED_SPACE_ENTRY]:
        amended_figures[permit_type] = [
            requirement.get("rules", {}).get(config.MAXIMUM_RULE)
            for requirement in read_permit_standard(permit_type, amended).get("requirements", [])
            if requirement.get("requirement_kind") == config.FLAMMABLE_GAS_REQUIREMENT]
    assert amended_figures == {config.HOT_WORK: [Decimal("4")],
                               config.CONFINED_SPACE_ENTRY: [Decimal("8")]}, (
        "With only the confined space entry standard's flammable gas figure changed to 8 percent "
        "of the lower explosive limit, the two twins read "
        + str(amended_figures)
        + ". Changing one standard changes that permit type's limit and no other's, because "
        "each is read from its own document."
    )
    copied = documents + [dict(documents[0])]
    refusal = None
    try:
        read_permit_standard(config.HOT_WORK, copied)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    hot_document = config.PERMIT_STANDARD_DOCUMENTS[config.HOT_WORK]
    assert refusal is not None and hot_document in refusal, (
        "Reading the hot work standard from documents holding two copies of it gave "
        + str(refusal)
        + ". Which copy states the requirements cannot be told, so it is refused with a "
        "ValueError naming the document."
    )


def test_band_and_thresholds_read_however_laid_out(invented_standards):
    from decimal import Decimal

    from config import config
    from knowledge.standard_reading import read_permit_standard

    assert invented_standards["problem"] is None, invented_standards["problem"]
    standard_documents = set(config.PERMIT_STANDARD_DOCUMENTS.values())
    relaid = []
    for document in invented_standards["documents"]:
        document_text = document["document_text"]
        if document["document_name"] in standard_documents:
            document_text = (document_text.replace(" is at least ", " is at\nleast  ")
                             .replace(" percent and at most ", " percent\nand at   most ")
                             .replace(" volts, circuit ", "\nvolts,  circuit ")
                             .replace(" metres has its ", " metres\nhas   its ")
                             .replace(" is at most ", " is at\nmost  "))
        relaid.append({"document_name": document["document_name"],
                       "document_text": document_text})
    differing = []
    for permit_type in config.PERMIT_STANDARD_DOCUMENTS:
        try:
            again = read_permit_standard(permit_type, relaid)
        except Exception as error:
            again = type(error).__name__ + ": " + str(error)
        if again != invented_standards["standards"][permit_type]:
            differing.append(permit_type + " read as " + repr(again)[:200])
    assert len(differing) == 0, (
        "With each standard's lines broken in other places and runs of spaces between its "
        "words, as text taken from a PDF can come, the standards read differently: "
        + str(differing[:2])
        + ". A standard is read from its sections, in which every run of spaces and line breaks "
        "is one space, so the same standard reads the same however its lines fall."
    )
    replacements = {
        config.HOT_WORK: ("at least 19.8 percent and at most 22.6 percent",
                          "at least 18.6 percent and at most 21.4 percent"),
        config.ELECTRICAL_ISOLATION: ("above 650 volts", "above 400 volts"),
        config.WORK_AT_HEIGHT: ("at most 9.5 metres per second", "at most 7.5 metres per second"),
        config.EXCAVATION: ("deeper than 1.5 metres", "deeper than 0.9 metres")}
    changed = []
    for document in invented_standards["documents"]:
        document_text = document["document_text"]
        for permit_type, (before, after) in replacements.items():
            if document["document_name"] == config.PERMIT_STANDARD_DOCUMENTS[permit_type]:
                document_text = document_text.replace(before, after)
        changed.append({"document_name": document["document_name"],
                        "document_text": document_text})
    lowest, highest, threshold = config.MINIMUM_RULE, config.MAXIMUM_RULE, config.THRESHOLD_RULE
    wanted = {
        config.HOT_WORK: (config.OXYGEN_REQUIREMENT,
                          {lowest: Decimal("18.6"), highest: Decimal("21.4")}),
        config.CONFINED_SPACE_ENTRY: (config.OXYGEN_REQUIREMENT,
                                      {lowest: Decimal("19.7"), highest: Decimal("22.9")}),
        config.ELECTRICAL_ISOLATION: (config.CIRCUIT_EARTHS_REQUIREMENT,
                                      {threshold: Decimal("400")}),
        config.WORK_AT_HEIGHT: (config.WIND_SPEED_REQUIREMENT, {highest: Decimal("7.5")}),
        config.EXCAVATION: (config.TRENCH_SUPPORT_REQUIREMENT, {threshold: Decimal("0.9")})}
    got = {}
    for permit_type, (kind, _figures) in wanted.items():
        try:
            requirements = read_permit_standard(permit_type, changed).get("requirements", [])
            got[permit_type] = (kind, [requirement.get("rules") for requirement in requirements
                                       if requirement.get("requirement_kind") == kind])
        except Exception as error:
            got[permit_type] = (kind, type(error).__name__ + ": " + str(error)[:120])
    expected = {permit_type: (kind, [figures]) for permit_type, (kind, figures) in wanted.items()}
    assert got == expected, (
        "With the hot work standard's oxygen band changed to 18.6 to 21.4 percent, the circuit "
        "earths asked above 400 volts, the wind capped at 7.5 metres per second and trench "
        "support asked deeper than 0.9 metres, the standards read "
        + str(got)
        + " where they state "
        + str(expected)
        + ". The oxygen band keeps both its figures, the lower as its minimum and the higher as "
        "its maximum; each follow-on question keeps the threshold its own standard states; and "
        "the confined space entry standard's band, left alone, is unchanged."
    )


def test_unknown_type_refused_reading_repeatable(invented_standards):
    from config import config
    from knowledge.standard_reading import read_permit_standard

    assert invented_standards["problem"] is None, invented_standards["problem"]
    documents = invented_standards["documents"]
    refusal = None
    try:
        read_permit_standard("Diving operations", documents)
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "Diving operations" in refusal, (
        "Reading the standard of a permit type with none configured gave "
        + str(refusal)
        + ". A permit type with no standard has no requirements to screen against, and is "
        "refused with a ValueError naming it."
    )
    for permit_type in config.PERMIT_STANDARD_DOCUMENTS:
        again = read_permit_standard(permit_type, documents)
        assert again == invented_standards["standards"][permit_type], (
            "The standard of "
            + permit_type
            + " reads differently the second time. A standard is read from its sections, the "
            "same way every time."
        )
    dig_standard = config.PERMIT_STANDARD_DOCUMENTS[config.EXCAVATION]
    refusal = None
    try:
        read_permit_standard(config.EXCAVATION,
                             [document for document in documents
                              if document["document_name"] != dig_standard])
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and dig_standard in refusal, (
        "Reading the excavation standard from documents that do not include it gave "
        + str(refusal)
        + ". A standard missing from the documents read leaves its permit type with nothing to "
        "screen against, so it is refused with a ValueError naming the document."
    )
    assessment = "Risk assessment. A task risk assessment for the job is attached to the request."
    flammable = ("Flammable gas. The flammable gas reading at the work point is at most 6 percent "
                 "of the lower explosive limit.")
    proved = "Proved dead. The isolation is proved dead at the point of work."
    broken = {
        config.HOT_WORK: (config.CONFINED_SPACE_ENTRY, ["1 " + assessment, "2 " + flammable]),
        config.CONFINED_SPACE_ENTRY: (config.CONFINED_SPACE_ENTRY,
                                      ["1 " + assessment, "2 " + flammable,
                                       "3 " + flammable.replace("6 percent", "9 percent")]),
        config.ELECTRICAL_ISOLATION: (config.ELECTRICAL_ISOLATION,
                                      ["1 " + assessment, "2 " + proved, "4 " + flammable]),
        config.WORK_AT_HEIGHT: (config.WORK_AT_HEIGHT,
                                ["1 " + assessment, "2 Noise level. The noise at the work "
                                                    "position is at most 85 decibels."]),
    }
    broken_documents = []
    for permit_type, (stated_type, requirement_lines) in broken.items():
        broken_documents.append({
            "document_name": config.PERMIT_STANDARD_DOCUMENTS[permit_type],
            "document_text": (
                "HEXLEY CHEMICALS\nDocument Standard 98 | Issue 1 | Owner: Process Safety | "
                "Review cycle: every twelve months\n1. Purpose and scope\nThis standard is laid "
                "out wrongly on purpose.\n2. What this permit covers\nThe permit type is "
                + stated_type
                + ".\n3. Who requests\nThe performing authority who leads the job.\n4. "
                "Requirements\nEvery requirement below must be Met, in this order:\n"
                + "\n".join(requirement_lines)
                + "\nA requirement is never met by a sentence in a statement, and is never "
                "waived.\n"),
        })
    broken_documents.append({
        "document_name": config.PERMIT_STANDARD_DOCUMENTS[config.EXCAVATION],
        "document_text": (
            "HEXLEY CHEMICALS\nDocument Standard 98 | Issue 1 | Owner: Process Safety | Review "
            "cycle: every twelve months\n1. Purpose and scope\nThis standard is laid out wrongly "
            "on purpose.\n2. What this permit covers\nThe permit type is Excavation.\n3. Who "
            "requests\nThe performing authority who leads the job.\n"),
    })
    meanings = {
        config.HOT_WORK: "states another permit type",
        config.CONFINED_SPACE_ENTRY: "states the flammable gas requirement twice",
        config.ELECTRICAL_ISOLATION: "numbers its requirements 1, 2 and 4",
        config.WORK_AT_HEIGHT: "states a requirement the screening cannot read",
        config.EXCAVATION: "has no requirements section",
    }
    for permit_type, meaning in meanings.items():
        document_name = config.PERMIT_STANDARD_DOCUMENTS[permit_type]
        refusal = None
        try:
            read_permit_standard(permit_type, broken_documents)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and document_name in refusal, (
            "A standard that "
            + meaning
            + " was read, giving "
            + str(refusal)
            + ". A standard that cannot be read as the Permit Rules lay it out would screen "
            "requests against requirements nobody wrote, so it is refused with a ValueError "
            "naming the document."
        )


def test_requirements_met_at_boundary(live_files):
    from decimal import Decimal

    from config import config
    from tools.request_data_tools import fetch_request
    from tools.screening_tools import screen_request

    assert live_files["problem"] is None, live_files["problem"]
    requests = live_files["requests"]
    templates = {}
    for permit_reference in requests[config.PERMIT_REFERENCE_COLUMN]:
        request = fetch_request(requests, permit_reference)
        if (request[config.PERMIT_TYPE_COLUMN] not in templates
                and request[config.LINKED_PERMIT_TYPE_COLUMN] == config.NO_LINKED_PERMIT):
            templates[request[config.PERMIT_TYPE_COLUMN]] = request
    lowest, highest, threshold = config.MINIMUM_RULE, config.MAXIMUM_RULE, config.THRESHOLD_RULE
    hot, entry = config.HOT_WORK, config.CONFINED_SPACE_ENTRY
    isolation, height, dig = config.ELECTRICAL_ISOLATION, config.WORK_AT_HEIGHT, config.EXCAVATION
    invented = {
        hot: [(config.FLAMMABLE_GAS_REQUIREMENT, "Flammable gas", {highest: Decimal("6")}),
              (config.OXYGEN_REQUIREMENT, "Oxygen",
               {lowest: Decimal("19.4"), highest: Decimal("22.8")}),
              (config.COMBUSTIBLES_CLEARED_REQUIREMENT, "Combustibles cleared",
               {lowest: Decimal("11.2")}),
              (config.FIRE_WATCH_REQUIREMENT, "Fire watch", {lowest: Decimal("50")})],
        entry: [(config.HYDROGEN_SULPHIDE_REQUIREMENT, "Hydrogen sulphide",
                 {highest: Decimal("4.7")}),
                (config.CARBON_MONOXIDE_REQUIREMENT, "Carbon monoxide",
                 {highest: Decimal("17.3")})],
        isolation: [(config.CIRCUIT_EARTHS_REQUIREMENT, "Circuit earths",
                     {threshold: Decimal("700")})],
        height: [(config.WIND_SPEED_REQUIREMENT, "Wind speed", {highest: Decimal("9.8")})],
        dig: [(config.TRENCH_SUPPORT_REQUIREMENT, "Trench support", {threshold: Decimal("1.3")})],
    }
    flammable, oxygen = "flammable_gas_lel_percent", config.OXYGEN_COLUMN
    cleared, watch = "combustibles_cleared_radius_m", "fire_watch_minutes"
    sulphide, monoxide = "hydrogen_sulphide_ppm", "carbon_monoxide_ppm"
    voltage, earths = config.SYSTEM_VOLTAGE_COLUMN, config.CIRCUIT_EARTHS_COLUMN
    location, wind = config.WORK_LOCATION_COLUMN, config.WIND_SPEED_COLUMN
    depth, support = config.EXCAVATION_DEPTH_COLUMN, config.TRENCH_SUPPORT_COLUMN
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    answered_no, outdoors = config.BOOLEAN_NO, config.OUTDOORS
    cases = [
        (hot, "Flammable gas", {flammable: 6}, met),
        (hot, "Flammable gas", {flammable: 7}, not_met),
        (hot, "Oxygen", {oxygen: 19.4}, met), (hot, "Oxygen", {oxygen: 19.3}, not_met),
        (hot, "Oxygen", {oxygen: 22.8}, met), (hot, "Oxygen", {oxygen: 22.9}, not_met),
        (hot, "Combustibles cleared", {cleared: 11.2}, met),
        (hot, "Combustibles cleared", {cleared: 11.1}, not_met),
        (hot, "Fire watch", {watch: 50}, met), (hot, "Fire watch", {watch: 49}, not_met),
        (entry, "Hydrogen sulphide", {sulphide: 4.7}, met),
        (entry, "Hydrogen sulphide", {sulphide: 4.8}, not_met),
        (entry, "Carbon monoxide", {monoxide: 17.3}, met),
        (entry, "Carbon monoxide", {monoxide: 17.4}, not_met),
        (isolation, "Circuit earths", {voltage: 700, earths: answered_no}, met),
        (isolation, "Circuit earths", {voltage: 701, earths: answered_no}, not_met),
        (height, "Wind speed", {location: outdoors, wind: 9.8}, met),
        (height, "Wind speed", {location: outdoors, wind: 9.9}, not_met),
        (dig, "Trench support", {depth: 1.3, support: answered_no}, met),
        (dig, "Trench support", {depth: 1.4, support: answered_no}, not_met),
    ]
    assert set(invented) <= set(templates), (
        "The live queue holds unlinked requests of the permit types "
        + str(sorted(templates))
        + ", where it holds one of every permit type."
    )
    wrong = []
    for permit_type, requirement_name, answers, expected in cases:
        standard = {"permit_type": permit_type,
                    "document_name": "Invented-Boundary-Standard.pdf",
                    "requirements": [{"requirement_kind": kind, "requirement_name": name,
                                      "requirement_sentence": "An invented requirement: " + name
                                                              + ".",
                                      "rules": rules}
                                     for kind, name, rules in invented[permit_type]]}
        request = dict(templates[permit_type])
        request.update({config.PERMIT_REFERENCE_COLUMN: "PTW0950101",
                        config.CONCURRENT_LIVE_PERMITS_COLUMN: 0,
                        config.CONFLICTING_LIVE_PERMITS_COLUMN: 0})
        request.update(answers)
        screening = screen_request(request, {permit_type: standard})
        assert isinstance(screening, dict) and isinstance(screening.get("requirements"), list), (
            "screen_request returned "
            + str(screening)[:200]
            + ", where it returns the screening with one result for each requirement."
        )
        results = {requirement.get("requirement_name"): requirement.get("result")
                   for requirement in screening["requirements"] if isinstance(requirement, dict)}
        if results.get(requirement_name) != expected:
            wrong.append(permit_type + " " + requirement_name + " with " + str(answers) + " is "
                         + str(results.get(requirement_name)) + ", not " + expected)
    assert len(wrong) == 0, (
        "Under standards whose figures no Hexley standard uses, the screening misplaces "
        "boundaries: "
        + "; ".join(wrong[:5])
        + ". A maximum is met at the figure itself and a minimum likewise, a band at either of "
        "its figures, and a follow-on question is not asked at its threshold, every reading "
        "compared as an exact decimal - an oxygen reading of 22.8 against a band up to 22.8, or "
        "a depth of 1.3 against 1.3, is within it - and every figure comes from the standard "
        "handed over, never from a figure written into the code."
    )


def test_follow_on_questions_ask_only_answers_needed(live_files):
    from decimal import Decimal

    from config import config
    from tools.request_data_tools import fetch_request
    from tools.screening_tools import screen_request

    assert live_files["problem"] is None, live_files["problem"]
    requests = live_files["requests"]
    template = None
    for permit_reference in requests[config.PERMIT_REFERENCE_COLUMN]:
        request = fetch_request(requests, permit_reference)
        if template is None and request[config.PERMIT_TYPE_COLUMN] == config.WORK_AT_HEIGHT:
            template = request
    assert template is not None, "The live queue holds no work at height request."
    standard = {"permit_type": config.WORK_AT_HEIGHT,
                "document_name": "Invented-Follow-On-Standard.pdf",
                "requirements": []}
    for kind, name, rules in [
            (config.CIRCUIT_EARTHS_REQUIREMENT, "Circuit earths",
             {config.THRESHOLD_RULE: Decimal("700")}),
            (config.SCAFFOLD_INSPECTION_REQUIREMENT, "Scaffold inspection", {}),
            (config.WIND_SPEED_REQUIREMENT, "Wind speed", {config.MAXIMUM_RULE: Decimal("9.8")}),
            (config.TRENCH_SUPPORT_REQUIREMENT, "Trench support",
             {config.THRESHOLD_RULE: Decimal("1.3")}),
            (config.RISK_ASSESSMENT_REQUIREMENT, "Risk assessment", {}),
            (config.OXYGEN_REQUIREMENT, "Oxygen",
             {config.MINIMUM_RULE: Decimal("19.4"), config.MAXIMUM_RULE: Decimal("22.8")})]:
        standard["requirements"].append({"requirement_kind": kind, "requirement_name": name,
                                         "requirement_sentence": ("An invented requirement: "
                                                                  + name + "."),
                                         "rules": rules})
    voltage, earths = config.SYSTEM_VOLTAGE_COLUMN, config.CIRCUIT_EARTHS_COLUMN
    access, inspected = config.ACCESS_METHOD_COLUMN, config.SCAFFOLD_INSPECTED_COLUMN
    location, wind = config.WORK_LOCATION_COLUMN, config.WIND_SPEED_COLUMN
    depth, support = config.EXCAVATION_DEPTH_COLUMN, config.TRENCH_SUPPORT_COLUMN
    risk, oxygen = "risk_assessment_attached", config.OXYGEN_COLUMN
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    undocumented = config.REQUIREMENT_NOT_DOCUMENTED
    cases = [
        ("Circuit earths", {voltage: 415, earths: None}, met, []),
        ("Circuit earths", {voltage: 700, earths: answered_no}, met, []),
        ("Circuit earths", {voltage: 701, earths: answered_yes}, met, []),
        ("Circuit earths", {voltage: 701, earths: answered_no}, not_met, []),
        ("Circuit earths", {voltage: 3300, earths: None}, undocumented, [earths]),
        ("Circuit earths", {voltage: None, earths: answered_yes}, undocumented, [voltage]),
        ("Circuit earths", {voltage: None, earths: None}, undocumented, [voltage]),
        ("Scaffold inspection", {access: config.LADDER, inspected: None}, met, []),
        ("Scaffold inspection", {access: config.MOBILE_ELEVATING_WORK_PLATFORM,
                                 inspected: answered_no}, met, []),
        ("Scaffold inspection", {access: config.SCAFFOLD, inspected: answered_yes}, met, []),
        ("Scaffold inspection", {access: config.SCAFFOLD, inspected: answered_no}, not_met, []),
        ("Scaffold inspection", {access: config.SCAFFOLD, inspected: None}, undocumented,
         [inspected]),
        ("Scaffold inspection", {access: None, inspected: answered_yes}, undocumented, [access]),
        ("Wind speed", {location: config.INDOORS, wind: None}, met, []),
        ("Wind speed", {location: config.INDOORS, wind: 30.0}, met, []),
        ("Wind speed", {location: config.OUTDOORS, wind: 9.8}, met, []),
        ("Wind speed", {location: config.OUTDOORS, wind: 12.5}, not_met, []),
        ("Wind speed", {location: config.OUTDOORS, wind: None}, undocumented, [wind]),
        ("Wind speed", {location: None, wind: 3.0}, undocumented, [location]),
        ("Trench support", {depth: 0.9, support: None}, met, []),
        ("Trench support", {depth: 1.3, support: answered_no}, met, []),
        ("Trench support", {depth: 2.5, support: answered_yes}, met, []),
        ("Trench support", {depth: 2.5, support: answered_no}, not_met, []),
        ("Trench support", {depth: 1.4, support: None}, undocumented, [support]),
        ("Trench support", {depth: None, support: answered_yes}, undocumented, [depth]),
        ("Risk assessment", {risk: answered_yes}, met, []),
        ("Risk assessment", {risk: answered_no}, not_met, []),
        ("Risk assessment", {risk: None}, undocumented, [risk]),
        ("Oxygen", {oxygen: None}, undocumented, [oxygen]),
    ]
    wrong = []
    for requirement_name, answers, expected, expected_blanks in cases:
        request = dict(template)
        request.update({config.PERMIT_REFERENCE_COLUMN: "PTW0950102",
                        config.LINKED_PERMIT_TYPE_COLUMN: config.NO_LINKED_PERMIT,
                        config.CONCURRENT_LIVE_PERMITS_COLUMN: 0,
                        config.CONFLICTING_LIVE_PERMITS_COLUMN: 0})
        request.update(answers)
        screening = screen_request(request, {config.WORK_AT_HEIGHT: standard})
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and all(isinstance(requirement, dict)
                        for requirement in screening["requirements"])), (
            "screen_request returned "
            + str(screening)[:200]
            + ", where it returns the screening with one result for each of the standard's "
            "requirements, in the standard's order."
        )
        found = [requirement for requirement in screening["requirements"]
                 if requirement.get("requirement_name") == requirement_name]
        if (len(found) != 1 or found[0].get("result") != expected
                or list(found[0].get("blank_answers", [])) != expected_blanks):
            wrong.append(requirement_name + " with " + str(answers) + " gives "
                         + str([(requirement.get("result"), requirement.get("blank_answers"))
                                for requirement in found])
                         + ", not " + expected + " asking for " + str(expected_blanks))
    assert len(wrong) == 0, (
        "The follow-on questions are screened wrongly: "
        + "; ".join(wrong[:4])
        + ". Circuit earths are asked only above the standard's voltage, scaffold inspection "
        "only from a scaffold, the wind only outdoors and trench support only deeper than the "
        "standard's depth: below the line each is Met whatever else the request says; past it "
        "Met on Yes or within the figure, Not met otherwise, and Not documented, asking for the "
        "follow-on answer alone, when it is blank; and Not documented, asking for the first "
        "answer alone, when that is blank. A yes-or-no requirement or a reading left blank asks "
        "for that answer."
    )


def test_linked_request_screened_under_both_standards(live_files):
    from decimal import Decimal

    from config import config
    from tools.request_data_tools import fetch_request
    from tools.screening_tools import screen_request

    assert live_files["problem"] is None, live_files["problem"]
    requests = live_files["requests"]
    template = fetch_request(requests, requests[config.PERMIT_REFERENCE_COLUMN].iloc[0])
    hot, entry = config.HOT_WORK, config.CONFINED_SPACE_ENTRY
    standards = {
        hot: {"permit_type": hot, "document_name": "Invented-Hot-Work-Standard.pdf",
              "requirements": [
                  {"requirement_kind": config.RISK_ASSESSMENT_REQUIREMENT,
                   "requirement_name": "Risk assessment",
                   "requirement_sentence": ("A task risk assessment for the job is attached to "
                                            "the request."), "rules": {}},
                  {"requirement_kind": config.FLAMMABLE_GAS_REQUIREMENT,
                   "requirement_name": "Flammable gas",
                   "requirement_sentence": ("The flammable gas reading at the work point is at "
                                            "most 8 percent of the lower explosive limit."),
                   "rules": {config.MAXIMUM_RULE: Decimal("8")}},
                  {"requirement_kind": config.OXYGEN_REQUIREMENT, "requirement_name": "Oxygen",
                   "requirement_sentence": ("The oxygen reading is at least 19.4 percent and at "
                                            "most 22.8 percent."),
                   "rules": {config.MINIMUM_RULE: Decimal("19.4"),
                             config.MAXIMUM_RULE: Decimal("22.8")}}]},
        entry: {"permit_type": entry, "document_name": "Invented-Entry-Standard.pdf",
                "requirements": [
                    {"requirement_kind": config.FLAMMABLE_GAS_REQUIREMENT,
                     "requirement_name": "Flammable gas",
                     "requirement_sentence": ("The flammable gas reading at the work point is at "
                                              "most 4 percent of the lower explosive limit."),
                     "rules": {config.MAXIMUM_RULE: Decimal("4")}},
                    {"requirement_kind": config.HYDROGEN_SULPHIDE_REQUIREMENT,
                     "requirement_name": "Hydrogen sulphide",
                     "requirement_sentence": "The hydrogen sulphide reading is at most 4.7 ppm.",
                     "rules": {config.MAXIMUM_RULE: Decimal("4.7")}},
                    {"requirement_kind": config.RESCUE_PLAN_REQUIREMENT,
                     "requirement_name": "Rescue plan",
                     "requirement_sentence": ("A rescue plan is in place with a standby person "
                                              "at the entry."), "rules": {}}]}}
    answers = {config.PERMIT_REFERENCE_COLUMN: "PTW0950103",
               config.PROCESS_UNIT_STATUS_COLUMN: config.SHUT_DOWN,
               config.CONCURRENT_LIVE_PERMITS_COLUMN: 0,
               config.CONFLICTING_LIVE_PERMITS_COLUMN: 0,
               "risk_assessment_attached": config.BOOLEAN_YES, "flammable_gas_lel_percent": 6,
               config.OXYGEN_COLUMN: 20.9, "hydrogen_sulphide_ppm": 1.0,
               "rescue_plan_in_place": config.BOOLEAN_YES}
    hot_document = standards[hot]["document_name"]
    entry_document = standards[entry]["document_name"]
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    cases = [
        ("a hot work request linked to a confined space entry", hot, entry,
         [hot, entry],
         [(hot, hot_document, "Risk assessment", met), (hot, hot_document, "Flammable gas", met),
          (hot, hot_document, "Oxygen", met), (entry, entry_document, "Flammable gas", not_met),
          (entry, entry_document, "Hydrogen sulphide", met),
          (entry, entry_document, "Rescue plan", met)],
         [5, 1, 0], config.DECISION_RETURN),
        ("a confined space entry linked to hot work", entry, hot, [entry, hot],
         [(entry, entry_document, "Flammable gas", not_met),
          (entry, entry_document, "Hydrogen sulphide", met),
          (entry, entry_document, "Rescue plan", met),
          (hot, hot_document, "Risk assessment", met), (hot, hot_document, "Flammable gas", met),
          (hot, hot_document, "Oxygen", met)],
         [5, 1, 0], config.DECISION_RETURN),
        ("the same hot work request with no linked permit", hot, config.NO_LINKED_PERMIT, [hot],
         [(hot, hot_document, "Risk assessment", met), (hot, hot_document, "Flammable gas", met),
          (hot, hot_document, "Oxygen", met)],
         [3, 0, 0], config.DECISION_ISSUE)]
    for meaning, permit_type, linked_type, permit_types, expected, counts, assessment in cases:
        request = dict(template, **answers)
        request[config.PERMIT_TYPE_COLUMN] = permit_type
        request[config.LINKED_PERMIT_TYPE_COLUMN] = linked_type
        screening = screen_request(request, standards)
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and all(isinstance(requirement, dict)
                        for requirement in screening["requirements"])), (
            "screen_request returned "
            + str(screening)[:200]
            + " for "
            + meaning
            + ", where it returns the screening with one result for each requirement."
        )
        got = [(requirement.get("permit_type"), requirement.get("standard_document"),
                requirement.get("requirement_name"), requirement.get("result"))
               for requirement in screening["requirements"]]
        got_counts = [screening.get(column_name)
                      for column_name in config.REQUIREMENT_COUNT_RESULTS]
        assert (got == expected and got_counts == counts
                and screening.get("assessment") == assessment
                and list(screening.get("permit_types", [])) == permit_types
                and screening.get("permit_reference") == "PTW0950103"), (
            "For "
            + meaning
            + " reading 6 percent of the lower explosive limit, the screening gives "
            + str(got)
            + " with counts "
            + str(got_counts)
            + ", the assessment "
            + str(screening.get("assessment"))
            + " and the permit types "
            + str(screening.get("permit_types"))
            + ", where the Permit Rules give "
            + str(expected)
            + ", "
            + str(counts)
            + ", "
            + assessment
            + " and "
            + str(permit_types)
            + ". A linked request is screened under its own standard first and then under its "
            "linked type's, each requirement keeping the permit type and the document it came "
            "from: 6 percent passes the hot work standard's 8 and fails the confined space "
            "entry standard's 4, so the request goes back to its requester."
        )


def test_simops_triggers_follow_register_and_rules(live_files, tmp_path):
    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.request_cleaning import clean_register_records
    from tools.request_data_tools import fetch_request
    from tools.screening_tools import count_live_permits, find_simops_triggers

    assert live_files["problem"] is None, live_files["problem"]
    requests = live_files["requests"]
    template = fetch_request(requests, requests[config.PERMIT_REFERENCE_COLUMN].iloc[0])
    raw_register = load_raw_dataset(config.LIVE_PERMIT_REGISTER_FILE)
    written = [("PTW-51301", "reactor house a", "Confined Space Permit", "No link",
                "Vessel inspection", "Live"),
               ("51302", "REACTOR HOUSE A", "Excavation", "nil", "Trial hole", "LIVE"),
               ("PTW0051303", "Reactor House A", "EI", "No linked permit", "Motor maintenance",
                "live"),
               ("PTW0051304", "Reactor House A", "Work at height", "Hot Work Permit", "Painting",
                "Live"),
               ("PTW0051305", "Reactor House A", "HW", "NIL", "Welding", "SUSPENDED"),
               ("PTW0051306", "Reactor House B", "CSE", "No link", "Tank cleaning", "Live")]
    rows = pd.concat([raw_register.head(1)] * len(written), ignore_index=True)
    for position, values in enumerate(written):
        for column_name, value in zip([config.PERMIT_REFERENCE_COLUMN, config.PLANT_AREA_COLUMN,
                                       config.PERMIT_TYPE_COLUMN,
                                       config.LINKED_PERMIT_TYPE_COLUMN, "work_activity",
                                       config.PERMIT_STATUS_COLUMN], values, strict=True):
            rows.loc[position, column_name] = value
    register_path = str(tmp_path / "planted_register.csv")
    rows.to_csv(register_path, index=False)
    try:
        register = clean_register_records(load_raw_dataset(register_path))
    except Exception as error:
        register = error
    assert isinstance(register, pd.DataFrame) and len(register) == len(written), (
        "clean_register_records gave "
        + repr(register)[:200]
        + " for six planted permits written in other capitals, where it returns them cleaned."
    )
    hot, entry = config.HOT_WORK, config.CONFINED_SPACE_ENTRY
    isolation, height, dig = config.ELECTRICAL_ISOLATION, config.WORK_AT_HEIGHT, config.EXCAVATION
    unlinked = config.NO_LINKED_PERMIT
    first, second, third, fourth = "PTW0051301", "PTW0051302", "PTW0051303", "PTW0051304"
    count_cases = [
        ("hot work in Reactor House A", hot, unlinked, config.REACTOR_HOUSE_A,
         [first, second, third, fourth], [first, second]),
        ("an electrical isolation linked to hot work in Reactor House A", isolation, hot,
         config.REACTOR_HOUSE_A, [first, second, third, fourth], [first, second]),
        ("work at height in Reactor House A", height, unlinked, config.REACTOR_HOUSE_A,
         [first, second, third, fourth], []),
        ("a confined space entry in Reactor House A", entry, unlinked, config.REACTOR_HOUSE_A,
         [first, second, third, fourth], [fourth]),
        ("an excavation in Reactor House B", dig, unlinked, config.REACTOR_HOUSE_B,
         ["PTW0051306"], [])]
    wrong = []
    for meaning, permit_type, linked_type, plant_area, live, conflicting in count_cases:
        request = dict(template)
        request.update({config.PERMIT_REFERENCE_COLUMN: "PTW0950104",
                        config.PERMIT_TYPE_COLUMN: permit_type,
                        config.LINKED_PERMIT_TYPE_COLUMN: linked_type,
                        config.PLANT_AREA_COLUMN: plant_area})
        counted = count_live_permits(register, request)
        expected = {config.CONCURRENT_LIVE_PERMITS_COLUMN: len(live),
                    config.CONFLICTING_LIVE_PERMITS_COLUMN: len(conflicting),
                    "live_permits": live, "conflicting_permits": conflicting}
        got = ({key: counted.get(key) for key in expected} if isinstance(counted, dict)
               else counted)
        if got != expected:
            wrong.append(meaning + " counts " + str(got) + ", not " + str(expected))
    assert len(wrong) == 0, (
        "The permits live beside a request are counted wrongly from a register whose plant "
        "areas and statuses are written in other capitals: "
        + "; ".join(wrong[:3])
        + ". Every permit live in the request's plant area is counted, in the order of its "
        "reference, and those whose permit type or linked type conflicts with the request's own "
        "or linked type - hot work with confined space entry and with excavation, each way "
        "round - are its conflicting permits; a suspended permit is never counted, and a permit "
        "in another plant area is not beside the work."
    )
    refusal = None
    try:
        count_live_permits(register, dict(template, **{config.PERMIT_REFERENCE_COLUMN:
                                                        "PTW0950105",
                                                        config.PLANT_AREA_COLUMN: None}))
    except ValueError as error:
        refusal = str(error)
    except Exception as error:
        refusal = type(error).__name__ + " instead of a ValueError"
    assert refusal is not None and "PTW0950105" in refusal, (
        "Counting the permits beside a request that gives no plant area gave "
        + str(refusal)
        + ". With no plant area there is no register to count from, so it is refused with a "
        "ValueError naming the request."
    )
    conflicting_trigger = config.CONFLICTING_PERMIT_TRIGGER
    live_process, high_voltage = config.LIVE_PROCESS_TRIGGER, config.HIGH_VOLTAGE_TRIGGER
    operating, shut, no_plant = config.OPERATING, config.SHUT_DOWN, config.NO_PROCESS_PLANT
    trigger_cases = [
        (hot, unlinked, shut, None, 0, []),
        (hot, unlinked, shut, None, 1, [conflicting_trigger]),
        (hot, unlinked, operating, None, 0, [live_process]),
        (hot, unlinked, None, None, 0, [live_process]),
        (hot, unlinked, no_plant, None, 0, []),
        (entry, unlinked, operating, None, 0, [live_process]),
        (height, unlinked, operating, None, 0, []),
        (height, entry, operating, None, 0, [live_process]),
        (isolation, unlinked, shut, 1000, 0, []),
        (isolation, unlinked, shut, 1001, 0, [high_voltage]),
        (isolation, unlinked, shut, None, 0, [high_voltage]),
        (height, isolation, shut, 11000, 0, [high_voltage]),
        (hot, unlinked, shut, 11000, 0, []),
        (isolation, hot, None, None, 2, [conflicting_trigger, live_process, high_voltage])]
    wrong = []
    for permit_type, linked_type, unit_status, voltage, conflicts, expected in trigger_cases:
        request = dict(template)
        request.update({config.PERMIT_REFERENCE_COLUMN: "PTW0950106",
                        config.PERMIT_TYPE_COLUMN: permit_type,
                        config.LINKED_PERMIT_TYPE_COLUMN: linked_type,
                        config.PROCESS_UNIT_STATUS_COLUMN: unit_status,
                        config.SYSTEM_VOLTAGE_COLUMN: voltage,
                        config.CONCURRENT_LIVE_PERMITS_COLUMN: conflicts + 1,
                        config.CONFLICTING_LIVE_PERMITS_COLUMN: conflicts})
        triggers = find_simops_triggers(request)
        if not isinstance(triggers, list) or triggers != expected:
            wrong.append(permit_type + " linked to " + linked_type + " on a unit "
                         + str(unit_status) + " at " + str(voltage) + " volts beside "
                         + str(conflicts) + " conflicting permits meets " + str(triggers)
                         + ", not " + str(expected))
    assert len(wrong) == 0, (
        "The SIMOPS triggers are found wrongly: "
        + "; ".join(wrong[:4])
        + ". A conflicting live permit triggers; so does hot work or confined space entry, as "
        "the permit type or the linked type, on a unit neither shut down nor without process "
        "plant; and an electrical isolation, either way, of a system above 1,000 volts. A blank "
        "unit status counts as operating and a blank voltage as above 1,000 volts, and the "
        "triggers are listed in that order."
    )


def test_assessment_decisions_and_worked_example(live_files, tmp_path):
    from decimal import Decimal

    import pandas as pd

    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.request_cleaning import clean_request_records
    from tools.request_data_tools import fetch_request
    from tools.screening_tools import screen_request

    assert live_files["problem"] is None, live_files["problem"]
    requests = live_files["requests"]
    template = fetch_request(requests, requests[config.PERMIT_REFERENCE_COLUMN].iloc[0])
    hot, isolation = config.HOT_WORK, config.ELECTRICAL_ISOLATION
    standards = {
        hot: {"permit_type": hot, "document_name": "Invented-Assessment-Standard.pdf",
              "requirements": [
                  {"requirement_kind": config.RISK_ASSESSMENT_REQUIREMENT,
                   "requirement_name": "Risk assessment",
                   "requirement_sentence": ("A task risk assessment for the job is attached to "
                                            "the request."), "rules": {}},
                  {"requirement_kind": config.FLAMMABLE_GAS_REQUIREMENT,
                   "requirement_name": "Flammable gas",
                   "requirement_sentence": ("The flammable gas reading at the work point is at "
                                            "most 6 percent of the lower explosive limit."),
                   "rules": {config.MAXIMUM_RULE: Decimal("6")}}]},
        isolation: {"permit_type": isolation, "document_name": "Invented-Assessment-Standard.pdf",
                    "requirements": [
                        {"requirement_kind": config.PROVED_DEAD_REQUIREMENT,
                         "requirement_name": "Proved dead",
                         "requirement_sentence": ("The isolation is proved dead at the point of "
                                                  "work."), "rules": {}},
                        {"requirement_kind": config.CIRCUIT_EARTHS_REQUIREMENT,
                         "requirement_name": "Circuit earths",
                         "requirement_sentence": ("On a system above 700 volts, circuit earths "
                                                  "are applied at the point of work."),
                         "rules": {config.THRESHOLD_RULE: Decimal("700")}}]}}
    risk, flammable = "risk_assessment_attached", "flammable_gas_lel_percent"
    unit, conflicts = config.PROCESS_UNIT_STATUS_COLUMN, config.CONFLICTING_LIVE_PERMITS_COLUMN
    voltage, earths = config.SYSTEM_VOLTAGE_COLUMN, config.CIRCUIT_EARTHS_COLUMN
    answered_yes, answered_no = config.BOOLEAN_YES, config.BOOLEAN_NO
    issue, give_back, refer = config.DECISION_ISSUE, config.DECISION_RETURN, config.DECISION_REFER
    conflicting_trigger = config.CONFLICTING_PERMIT_TRIGGER
    live_process, high_voltage = config.LIVE_PROCESS_TRIGGER, config.HIGH_VOLTAGE_TRIGGER
    base = {config.LINKED_PERMIT_TYPE_COLUMN: config.NO_LINKED_PERMIT, risk: answered_yes,
            flammable: 2, unit: config.SHUT_DOWN, config.CONCURRENT_LIVE_PERMITS_COLUMN: 0,
            conflicts: 0, "isolation_proved_dead": answered_yes, voltage: 415, earths: None}
    cases = [
        (hot, {}, issue, [2, 0, 0], []),
        (hot, {conflicts: 1, config.CONCURRENT_LIVE_PERMITS_COLUMN: 1}, refer, [2, 0, 0],
         [conflicting_trigger]),
        (hot, {unit: config.OPERATING}, refer, [2, 0, 0], [live_process]),
        (hot, {risk: answered_no}, give_back, [1, 1, 0], []),
        (hot, {flammable: 9}, give_back, [1, 1, 0], []),
        (hot, {risk: None, unit: config.OPERATING}, give_back, [1, 0, 1], [live_process]),
        (isolation, {}, issue, [2, 0, 0], []),
        (isolation, {voltage: 3300, earths: answered_yes}, refer, [2, 0, 0], [high_voltage]),
        (isolation, {voltage: None}, give_back, [1, 0, 1], [high_voltage]),
    ]
    for permit_type, answers, expected_assessment, expected_counts, expected_triggers in cases:
        request = dict(template)
        request[config.PERMIT_REFERENCE_COLUMN] = "PTW0950107"
        request[config.PERMIT_TYPE_COLUMN] = permit_type
        request.update(base)
        request.update(answers)
        screening = screen_request(request, standards)
        assert isinstance(screening, dict), (
            "screen_request returned "
            + type(screening).__name__
            + ", where it returns the screening of the request."
        )
        counts = [screening.get(column_name) for column_name in config.REQUIREMENT_COUNT_RESULTS]
        assert (screening.get("assessment") == expected_assessment
                and counts == expected_counts
                and list(screening.get("simops_triggers", [])) == expected_triggers
                and list(screening.get("decisions_open", []))
                == config.DECISIONS_OPEN[expected_assessment]
                and screening.get("permit_reference") == "PTW0950107"), (
            "A "
            + permit_type.lower()
            + " request with "
            + str(answers)
            + " on a request that otherwise meets every requirement and triggers nothing is "
            "assessed "
            + str(screening.get("assessment"))
            + " with counts "
            + str(counts)
            + ", the SIMOPS triggers "
            + str(screening.get("simops_triggers"))
            + " and the decisions open "
            + str(screening.get("decisions_open"))
            + ", where the Control of Work Policy gives "
            + expected_assessment
            + ", "
            + str(expected_counts)
            + ", "
            + str(expected_triggers)
            + " and "
            + str(config.DECISIONS_OPEN[expected_assessment])
            + ". Any requirement Not met or Not documented makes the assessment RETURN TO "
            "REQUESTER before any SIMOPS trigger counts, though the triggers are still found; "
            "otherwise a trigger makes it REFER FOR SIMOPS REVIEW; ISSUE PERMIT alone leaves "
            "ISSUE PERMIT open."
        )
    written = live_files["raw_requests"].head(1).copy()
    for column_name, value in [(config.PERMIT_REFERENCE_COLUMN, "41999"),
                               (config.REVISION_NUMBER_COLUMN, "1"),
                               ("request_channel", "Scanned paper"),
                               (config.PERMIT_TYPE_COLUMN, "HW"),
                               (config.LINKED_PERMIT_TYPE_COLUMN, "NIL"),
                               (config.WORK_LOCATION_COLUMN, "OUT")]:
        written[column_name] = value
    worked_standard = {"permit_type": hot, "document_name": "Worked-Example-Standard.pdf",
                       "requirements": [
                           {"requirement_kind": config.FLAMMABLE_GAS_REQUIREMENT,
                            "requirement_name": "Flammable gas",
                            "requirement_sentence": ("The flammable gas reading at the work point "
                                                     "is at most 8 percent of the lower "
                                                     "explosive limit."),
                            "rules": {config.MAXIMUM_RULE: Decimal("8")}},
                           {"requirement_kind": config.OXYGEN_REQUIREMENT,
                            "requirement_name": "Oxygen",
                            "requirement_sentence": ("The oxygen reading is at least 19.0 percent "
                                                     "and at most 23.0 percent."),
                            "rules": {config.MINIMUM_RULE: Decimal("19.0"),
                                      config.MAXIMUM_RULE: Decimal("23.0")}},
                           {"requirement_kind": config.HYDROGEN_SULPHIDE_REQUIREMENT,
                            "requirement_name": "Hydrogen sulphide",
                            "requirement_sentence": ("The hydrogen sulphide reading is at most 3 "
                                                     "ppm."),
                            "rules": {config.MAXIMUM_RULE: Decimal("3")}},
                           {"requirement_kind": config.WIND_SPEED_REQUIREMENT,
                            "requirement_name": "Wind speed",
                            "requirement_sentence": ("Outdoors, the wind at the work position is "
                                                     "at most 10 metres per second."),
                            "rules": {config.MAXIMUM_RULE: Decimal("10")}}]}
    oxygen, sulphide = config.OXYGEN_COLUMN, "hydrogen_sulphide_ppm"
    wind = config.WIND_SPEED_COLUMN
    outcomes = {}
    for label, sent in [("as sent", {}),
                        ("with the oxygen written 19.0% and the wind 22 mph",
                         {oxygen: "19.0%", wind: "22 mph"}),
                        ("with the unit written S/D, the gas 8.5%LEL, the oxygen 19 % O2, the "
                         "hydrogen sulphide 4.33 mg/m3 and the wind 10.0 m/s",
                         {unit: "S/D", flammable: "8.5%LEL", oxygen: "19 % O2",
                          sulphide: "4.33 mg/m3", wind: "10.0 m/s"})]:
        variant = written.copy()
        for column_name, value in [(unit, "OPER"), (flammable, "8 % LEL"), (oxygen, "18.9 % O2"),
                                   (sulphide, "4.3 mg/m3"), (wind, "illegible")]:
            variant[column_name] = sent.get(column_name, value)
        sent_path = str(tmp_path / "worked_example.csv")
        variant.to_csv(sent_path, index=False)
        cleaned = clean_request_records(load_raw_dataset(sent_path), config.FORM_VALUE_MAPPINGS,
                                        config.LIVE_WHOLE_NUMBER_PATTERNS)
        assert isinstance(cleaned, pd.DataFrame) and "PTW0041999" in set(
                cleaned.get(config.PERMIT_REFERENCE_COLUMN, [])), (
            "clean_request_records returned "
            + str(cleaned)[:200]
            + " for the worked example written 41999, where it returns the cleaned request as "
            "PTW0041999."
        )
        cleaned_request = fetch_request(cleaned, "PTW0041999")
        cleaned_request.update({config.CONCURRENT_LIVE_PERMITS_COLUMN: 0, conflicts: 0})
        screening = screen_request(cleaned_request, {hot: worked_standard})
        assert (isinstance(screening, dict) and isinstance(screening.get("requirements"), list)
                and all(isinstance(requirement, dict)
                        for requirement in screening["requirements"])), (
            "screen_request returned "
            + str(screening)[:200]
            + " for the worked example, where it returns its screening."
        )
        outcomes[label] = (screening.get("assessment"),
                           [requirement.get("result") for requirement in screening["requirements"]],
                           [list(requirement.get("blank_answers", []))
                            for requirement in screening["requirements"]],
                           list(screening.get("simops_triggers", [])),
                           cleaned_request[flammable], cleaned_request[oxygen],
                           cleaned_request[sulphide], cleaned_request[wind])
    met, not_met = config.REQUIREMENT_MET, config.REQUIREMENT_NOT_MET
    undocumented = config.REQUIREMENT_NOT_DOCUMENTED
    expected_outcomes = {
        "as sent": (give_back, [met, not_met, met, undocumented], [[], [], [], [wind]],
                    [live_process], 8, 18.9, 3.0, None),
        "with the oxygen written 19.0% and the wind 22 mph": (
            refer, [met, met, met, met], [[], [], [], []], [live_process], 8, 19.0, 3.0, 9.8),
        "with the unit written S/D, the gas 8.5%LEL, the oxygen 19 % O2, the hydrogen sulphide "
        "4.33 mg/m3 and the wind 10.0 m/s": (
            give_back, [not_met, met, met, met], [[], [], [], []], [], 9, 19.0, 3.0, 10.0),
    }
    assert outcomes == expected_outcomes, (
        "The worked example of the Permit Rules, section 13, screens as "
        + str(outcomes)
        + " where the Rules give "
        + str(expected_outcomes)
        + ": a flammable gas reading of 8 % LEL is exactly 8, at the limit, so Met; oxygen of "
        "18.9 % O2 is under 19.0, so Not met; 4.3 mg/m3 of hydrogen sulphide is 4.3 divided by "
        "1.42, 3.028, rounded half up once to 3.0 ppm, at the limit, so Met; a wind written "
        "illegible is blank, so Not documented, asking for the wind alone; and the unit written "
        "OPER is operating, so the live process unit trigger applies too, but the requirements "
        "come first. With the oxygen at 19.0 and a wind of 22 mph, 9.8 metres per second, every "
        "requirement is Met and the trigger makes it a referral; 8.5%LEL rounds half up to 9."
    )


def test_agents_called_once_in_order(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    listed = saved_run["calls_at_gate"]
    assert (listed["status"] == 200 and isinstance(listed["body"], dict)
            and {"calls", "call_count", "expected_call_count"} <= set(listed["body"])
            and all("agent_name" in call for call in listed["body"]["calls"])), (
        "Reading the run's model calls at the pause returned "
        + str(listed["status"])
        + " with "
        + str(listed["body"])[:300]
        + ", where the audit trail lists every call under the agent that made it, with their "
        "count and the count expected."
    )
    at_gate = listed["body"]
    reviewing = [config.AGENT_NAME_PLANNER, config.AGENT_NAME_REQUIREMENTS,
                 config.AGENT_NAME_EVIDENCE, config.AGENT_NAME_RISK, config.AGENT_NAME_WRITER]
    called = [call["agent_name"] for call in at_gate["calls"]]
    assert at_gate["call_count"] == len(at_gate["calls"]) == config.LLM_CALLS_BEFORE_DECISION, (
        "Reaching the Area Authority recorded a call count of "
        + str(at_gate["call_count"])
        + " and listed "
        + str(len(at_gate["calls"]))
        + " language model calls where the workflow allows "
        + str(config.LLM_CALLS_BEFORE_DECISION)
        + ", one per reviewing agent. More means an agent ran twice; fewer, that one never ran."
    )
    assert called == reviewing, (
        "The agents ran in the order "
        + str(called)
        + ". Each one reads what the one before it concluded: the requirements are explained "
        "against the plan, the documents are reviewed against it, the figures are weighed with "
        "the plan in hand, and the recommendation draws on all of them."
    )
    assert at_gate["expected_call_count"] == config.LLM_CALLS_BEFORE_DECISION, (
        "Before any decision the service expected "
        + str(at_gate["expected_call_count"])
        + " calls. Until a decision is taken only the reviewing agents' calls are expected."
    )
    assert saved_run["at_gate"].get("llm_call_count") == config.LLM_CALLS_BEFORE_DECISION, (
        "The review record at the pause counts "
        + str(saved_run["at_gate"].get("llm_call_count"))
        + " calls where five were made."
    )


def test_model_calls_recorded_with_tokens(saved_run):
    from config import config
    from persistence.database import session_factory
    from persistence.models import LlmCall

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    final_calls = saved_run["final_calls"]
    assert final_calls["status"] == 200 and isinstance(final_calls["body"], dict), (
        "Reading the run's model calls after the decision returned "
        + str(final_calls["status"])
        + " with "
        + str(final_calls["body"])[:300]
        + "."
    )
    calls = final_calls["body"]["calls"]
    called = [call.get("agent_name") for call in calls]
    added_agents = [config.AGENT_NAME_PERMIT_WRITER, config.AGENT_NAME_BRIEFING_WRITER]
    expected_count = config.LLM_CALLS_BEFORE_DECISION + len(added_agents)
    assert (called[config.LLM_CALLS_BEFORE_DECISION:] == added_agents
            and final_calls["body"]["call_count"] == len(calls) == expected_count
            and final_calls["body"]["expected_call_count"]
            == config.LLM_CALLS_BY_DECISION[config.DECISION_ISSUE]), (
        "After the permit was issued the run lists the calls "
        + str(called)
        + " ("
        + str(final_calls["body"].get("call_count"))
        + " of "
        + str(final_calls["body"].get("expected_call_count"))
        + " expected) where an issue adds the permit document and then the work party briefing "
        "to the five reviewing calls."
    )
    for call in calls:
        tokens = [call.get("prompt_tokens"), call.get("response_tokens"), call.get("total_tokens")]
        assert (all(isinstance(count, int) and count > 0 for count in tokens)
                and tokens[2] >= tokens[0] + tokens[1]
                and isinstance(call.get("latency_milliseconds"), int)
                and call["latency_milliseconds"] >= 0
                and isinstance(call.get("model_name"), str) and call["model_name"]), (
            "The call of "
            + str(call.get("agent_name"))
            + " is recorded as "
            + str({key: call.get(key) for key in ["model_name", "prompt_tokens", "response_tokens",
                                                  "total_tokens", "latency_milliseconds"]})
            + ". Every call is recorded with the model that answered, the tokens it was sent, "
            "the tokens it wrote and the total the model counted, and the milliseconds it "
            "took."
        )
    assert (final_calls["body"]["total_tokens"] == sum(call["total_tokens"] for call in calls)
            and final_calls["body"]["total_latency_milliseconds"]
            == sum(call["latency_milliseconds"] for call in calls)), (
        "The run's totals of "
        + str(final_calls["body"]["total_tokens"])
        + " tokens and "
        + str(final_calls["body"]["total_latency_milliseconds"])
        + " milliseconds are not the sums of its calls'."
    )
    with session_factory() as session:
        stored = session.query(LlmCall).filter(
            LlmCall.run_id == saved_run["at_gate"]["run_id"]).all()
    assert len(stored) == expected_count and sorted(
            row.agent_name for row in stored) == sorted(called), (
        "The llm_calls table holds "
        + str(len(stored))
        + " calls for the run, by "
        + str(sorted(row.agent_name for row in stored))
        + ", where the run made "
        + str(expected_count)
        + ", by "
        + str(sorted(called))
        + ". Each call is recorded once, under the run that made it."
    )


def test_recommendation_among_decisions_open(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    detail = saved_run["detail"]
    assert at_gate.get("decisions_open") == detail["screening"]["decisions_open"], (
        "The review offers the decisions "
        + str(at_gate.get("decisions_open"))
        + " where the screening of "
        + saved_run["permit_reference"]
        + " leaves "
        + str(detail["screening"]["decisions_open"])
        + " open."
    )
    assert at_gate.get("recommended_decision") in at_gate.get("decisions_open", []), (
        "The review recommends "
        + repr(at_gate.get("recommended_decision"))
        + ", which is not among the decisions open "
        + str(at_gate.get("decisions_open"))
        + ". A recommendation the screening does not allow could never be taken."
    )
    write_ups = {key: str(at_gate.get(key) or "") for key in
                 ["review_plan_text", "requirements_analysis_text", "evidence_review_text",
                  "prediction_analysis_text", "recommendation_text"]}
    short = [key for key, text in write_ups.items() if len(text.split()) < 8]
    assert len(short) == 0, (
        "The review stores "
        + str(short)
        + " with fewer than eight words. Every reviewing agent's write-up is kept with the "
        "review for the Area Authority to read."
    )
    assert len(set(write_ups.values())) == len(write_ups), (
        "Two of the review's write-ups are the same text. Each agent writes its own part of "
        "the review."
    )
    assert (at_gate.get("coordinator_request") == saved_run["coordinator_request"]
            and at_gate.get("coordinator_id") == saved_run["opener"]["username"]), (
        "The review records the request "
        + repr(at_gate.get("coordinator_request"))
        + " opened by "
        + repr(at_gate.get("coordinator_id"))
        + " where "
        + saved_run["opener"]["username"]
        + " asked "
        + repr(saved_run["coordinator_request"])
        + "."
    )
    assert at_gate.get("status") == config.REVIEW_STATUS_AWAITING, (
        "The review was stored as "
        + repr(at_gate.get("status"))
        + " at the pause, where it waits as "
        + config.REVIEW_STATUS_AWAITING
        + "."
    )


def test_decorated_reply_read_whole():
    from agentic.agents import parse_labelled_response

    labels = ["PRIORITY", "CHECKS", "RATIONALE"]
    decorated = ("Here is the plan you asked for.\n"
                 "**PRIORITY:** The flammable gas requirement of both standards.\n"
                 "## Checks: 1. Confirm the gas reading against the gas test record.\n"
                 "2. Confirm the rescue plan names a standby person.\n"
                 "- RATIONALE: The request decides; the documents are evidence.\n"
                 "rationale: The answers are read as the Permit Rules set out.")
    parsed = parse_labelled_response(decorated, labels, "review_planner_agent")
    assert (isinstance(parsed, dict) and sorted(parsed) == sorted(labels)
            and all(isinstance(value, str) for value in parsed.values())), (
        "Parsing a decorated reply returned "
        + str(parsed)[:300]
        + " where it returns exactly the fields "
        + str(labels)
        + ", each with its text."
    )
    assert (parsed["PRIORITY"] == "The flammable gas requirement of both standards."
            and parsed["CHECKS"] == ("1. Confirm the gas reading against the gas test record.\n"
                                     "2. Confirm the rescue plan names a standby person.")
            and "The answers are read as the Permit Rules set out." in parsed["RATIONALE"]
            and "The request decides" in parsed["RATIONALE"]), (
        "A reply with labels in bold or as headings, a field running over two lines and a "
        "label written twice was read as "
        + str(parsed)
        + ". Decoration around a label is not part of it, a line with no label belongs to the "
        "field above it, and a label written again adds to its field."
    )
    for reply, missing in [("PRIORITY: The gas.\nCHECKS: The rescue plan.", "RATIONALE"),
                           ("PRIORITY: The gas.\nCHECKS:\nRATIONALE: Because.", "CHECKS")]:
        refusal = None
        try:
            parse_labelled_response(reply, labels, "review_planner_agent")
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
        assert refusal is not None and missing in refusal and "review_planner_agent" in refusal, (
            "A reply that leaves "
            + missing
            + " out or empty gave "
            + str(refusal)
            + ". A reply missing a field is refused with a ValueError naming the agent and the "
            "field, never stored with a gap."
        )


def test_issue_writes_permit_document_and_briefing(saved_run, live_files):
    from config import config
    from tools.permit_document_tools import build_permit_details
    from tools.request_data_tools import fetch_request

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert live_files["problem"] is None, live_files["problem"]
    final = saved_run["final_record"]
    assert final["status"] == 200 and len(final["body"]) == 1, (
        "The issued review could not be read back: "
        + str(final["body"])[:300]
    )
    record = final["body"][0]
    request = fetch_request(live_files["requests"], saved_run["permit_reference"])
    built = build_permit_details(request, saved_run["detail"]["screening"])
    details = record.get("document_details") or {}
    assert (isinstance(built, dict) and details == built
            and details.get(config.PERMIT_REFERENCE_COLUMN) == saved_run["permit_reference"]
            and details.get(config.PERMIT_TYPE_COLUMN) == request[config.PERMIT_TYPE_COLUMN]
            and details.get(config.EQUIPMENT_TAG_COLUMN) == request[config.EQUIPMENT_TAG_COLUMN]), (
        "The permit document carries the details "
        + str(details)
        + " where build_permit_details gives "
        + str(built)
        + " for the request and its screening. The details are added from the request and the "
        "screening, never written by the model, and name the request, its permit type and the "
        "equipment it covers, with each standard's controls as the request answers them."
    )
    permit_document = str(record.get("permit_document_text") or "")
    briefing = str(record.get("work_party_briefing_text") or "")
    assert (len(permit_document.split()) >= 8 and len(briefing.split()) >= 8
            and permit_document != briefing), (
        "The issue stored a permit document of "
        + str(len(permit_document.split()))
        + " words and a work party briefing of "
        + str(len(briefing.split()))
        + ". An issue writes the permit document to the performing authority and then a work "
        "party briefing for the people who do the job, two different texts."
    )
    assert record.get("simops_brief") is None and record.get("status") == (
            config.REVIEW_STATUS_ISSUED), (
        "The issued review is stored as "
        + repr(record.get("status"))
        + " with a SIMOPS brief of "
        + str(record.get("simops_brief"))[:120]
        + ". An issue is recorded as "
        + config.REVIEW_STATUS_ISSUED
        + " and sends nothing to the SIMOPS panel."
    )


def test_workflow_nodes_and_decision_paths(tmp_path):
    from agentic.state import PermitReviewState
    from config import config

    try:
        from agentic.graph import build_workflow

        compiled = build_workflow(str(tmp_path / "workflow_check.db"))
    except Exception as error:
        compiled = error
    builder = getattr(compiled, "builder", None)
    assert all(hasattr(builder, name) for name in ["nodes", "branches", "edges"]) and callable(
            getattr(compiled, "get_state", None)), (
        "Building the workflow gave "
        + repr(compiled)[:200]
        + ", where it returns the compiled workflow, built from its nodes, edges and "
        "conditional branches and kept in a checkpoint store."
    )
    node_names = sorted(str(name) for name in builder.nodes)
    expected_nodes = sorted(["load_permit_context", "plan_permit_review",
                             "assess_permit_requirements", "analyse_method_statements",
                             "interpret_model_predictions", "prepare_permit_recommendation",
                             "await_area_authority_decision", "draft_permit_document",
                             "draft_work_party_briefing", "record_permit_outcome"])
    assert node_names == expected_nodes, (
        "The workflow declares the nodes "
        + str(node_names)
        + " where the design has "
        + str(expected_nodes)
        + "."
    )
    destinations = {}
    for gate in ["await_area_authority_decision", "draft_permit_document"]:
        branches = builder.branches.get(gate, {})
        assert len(branches) == 1, (
            "The node "
            + gate
            + " has "
            + str(len(branches))
            + " conditional branches where it has one, choosing the path from the decision."
        )
        branch = list(branches.values())[0]
        decisions = config.DECISIONS
        if gate == "draft_permit_document":
            decisions = [config.DECISION_ISSUE, config.DECISION_RETURN]
        for decision in decisions:
            state = {key: None for key in PermitReviewState.__annotations__}
            state["review_decision"] = {"decision": decision, "decided_by": "routing.check",
                                        "decided_by_role": config.ROLE_AREA_AUTHORITY,
                                        "decided_by_area_group": config.PROCESS_UNITS_GROUP,
                                        "decision_reason": "Routing is being checked."}
            routed = branch.path.invoke(state)
            if isinstance(routed, (list, tuple)) and len(routed) == 1:
                routed = routed[0]
            destinations[(gate, decision)] = branch.ends[routed] if branch.ends else routed
    expected_paths = {
        ("await_area_authority_decision", config.DECISION_ISSUE): "draft_permit_document",
        ("await_area_authority_decision", config.DECISION_RETURN): "draft_permit_document",
        ("await_area_authority_decision", config.DECISION_REFER): "record_permit_outcome",
        ("draft_permit_document", config.DECISION_ISSUE): "draft_work_party_briefing",
        ("draft_permit_document", config.DECISION_RETURN): "record_permit_outcome",
    }
    assert destinations == expected_paths, (
        "The decisions route to "
        + str(destinations)
        + ". An issue writes the permit document and then the work party briefing; a return "
        "writes the rework request alone; a referral writes nothing and goes straight to the "
        "record, which adds the SIMOPS brief."
    )
    edges = {(str(source), str(target)) for source, target in builder.edges}
    edges |= {(route[0], str(target)) for route, target in destinations.items()}
    reached = {"__start__"}
    grew = True
    while grew:
        grew = False
        for source, target in edges:
            if source in reached and target not in reached:
                reached.add(target)
                grew = True
    unreachable = sorted(set(node_names) - reached)
    assert len(unreachable) == 0, (
        "These nodes cannot be reached from the start: "
        + str(unreachable)
        + ". A node with no way in never runs, whatever it contains."
    )
    for source, target, reason in [
            ("__start__", "load_permit_context",
             "every review starts from the tools' findings, before any agent runs"),
            ("prepare_permit_recommendation", "await_area_authority_decision",
             "the review is stored before the run waits for an Area Authority"),
            ("draft_work_party_briefing", "record_permit_outcome",
             "what was written has to reach the record"),
            ("record_permit_outcome", "__end__",
             "the run ends once the outcome is recorded")]:
        assert (source, target) in edges, (
            "Nothing leads from " + source + " to " + target + ": " + reason + "."
        )


def test_run_waits_for_appointed_area_authority(saved_run):
    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    at_gate = saved_run["at_gate"]
    undecided = ["decision", "decided_by", "decided_by_role", "decided_by_area_group",
                 "decision_reason", "permit_document_text", "document_details",
                 "work_party_briefing_text", "simops_brief"]
    written = [field for field in undecided if at_gate.get(field) is not None]
    assert len(written) == 0 and at_gate.get("status") == config.REVIEW_STATUS_AWAITING, (
        "At the pause the review already holds "
        + str(written)
        + " with the status "
        + repr(at_gate.get("status"))
        + ". Nothing is decided, written or briefed until an Area Authority decides."
    )
    opener, outsider = saved_run["opener"], saved_run["outsider"]
    plant_area = saved_run["detail"]["request"][config.PLANT_AREA_COLUMN]
    for refused_key, expected_status, expected_refusal, meaning in [
            ("own_decision", 409, config.OWN_REVIEW_MESSAGE.format(
                decided_by=opener["username"], permit_reference=saved_run["permit_reference"]),
             "The Area Authority who opened the review, deciding it,"),
            ("outside_decision", 403, config.AREA_GROUP_REFUSED_MESSAGE.format(
                username=outsider["username"], area_group=outsider["area_group"],
                plant_area=plant_area, permit_reference=saved_run["permit_reference"]),
             "An Area Authority appointed for " + str(outsider["area_group"])
             + ", deciding a request in " + str(plant_area) + ",")]:
        refused = saved_run[refused_key]
        assert (refused["status"] == expected_status and isinstance(refused["body"], dict)
                and refused["body"].get("detail") == expected_refusal), (
            meaning
            + " was answered "
            + str(refused["status"])
            + " with "
            + str(refused["body"])[:300]
            + ", where the service refuses with "
            + str(expected_status)
            + " and the configuration's words: "
            + repr(expected_refusal)
            + ". The Control of Work Policy has an Area Authority appointed for the plant area, "
            "who did not open the review, decide it."
        )
        assert (refused["calls_before"] == refused["calls_after"]
                == config.LLM_CALLS_BEFORE_DECISION), (
            "The refused decision moved the call count from "
            + str(refused["calls_before"])
            + " to "
            + str(refused["calls_after"])
            + ". A decision is checked before the run resumes, so a refused one spends no call."
        )


def test_run_resumes_from_checkpoint(saved_run):
    import os

    from config import config

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    stored_run = saved_run["run_record"]
    assert stored_run["problem"] is None, stored_run["problem"]
    expected_calls = config.LLM_CALLS_BY_DECISION[config.DECISION_ISSUE]
    assert (stored_run["status"] == config.AGENT_RUN_STATUS_COMPLETE
            and stored_run["llm_call_count"] == expected_calls), (
        "After the permit was issued the run is recorded as "
        + str(stored_run["status"])
        + " with "
        + str(stored_run["llm_call_count"])
        + " calls, where it closes "
        + config.AGENT_RUN_STATUS_COMPLETE
        + " with "
        + str(expected_calls)
        + "."
    )
    assert os.path.isfile(config.CHECKPOINT_DATABASE_FILE), (
        "There is no checkpoint at "
        + config.CHECKPOINT_DATABASE_FILE
        + ". A paused run waits on disk, so a decision taken later, or after the service "
        "restarts, finishes it."
    )
    try:
        from agentic.graph import build_workflow

        snapshot = build_workflow(config.CHECKPOINT_DATABASE_FILE).get_state(
            {"configurable": {"thread_id": saved_run["at_gate"]["thread_id"]}})
    except Exception as error:
        snapshot = error
    assert isinstance(getattr(snapshot, "values", None), dict) and hasattr(snapshot, "next"), (
        "Reading the run back through a workflow built afresh on "
        + config.CHECKPOINT_DATABASE_FILE
        + " gave "
        + repr(snapshot)[:200]
        + ", where it gives the run's state and what it waits for."
    )
    values = snapshot.values
    decision = values.get("review_decision") or {}
    assert (len(snapshot.next) == 0 and decision.get("decision") == config.DECISION_ISSUE
            and decision.get("decided_by") == saved_run["decider"]["username"]
            and decision.get("decided_by_area_group") == saved_run["decider"]["area_group"]
            and values.get("llm_call_count") == expected_calls), (
        "A workflow built afresh on the checkpoint file reads the run as waiting for "
        + str(list(snapshot.next))
        + " with the decision "
        + str(decision)
        + " and "
        + str(values.get("llm_call_count"))
        + " calls. The run was resumed from that file and finished there, with the Area "
        "Authority's decision and area group kept in its state."
    )
    assert (values.get("review_id") == saved_run["at_gate"]["review_id"]
            and values.get("permit_reference") == saved_run["permit_reference"]
            and stored_run["thread_id"] == saved_run["at_gate"]["thread_id"]), (
        "The checkpoint's state belongs to "
        + str(values.get("review_id"))
        + " and "
        + str(values.get("permit_reference"))
        + ", not to the review that was decided."
    )


def test_decision_once_refusals_before_calls(saved_run, service_client):
    from config import config
    from persistence.database import session_factory
    from persistence.models import ReviewDecision
    from tools.decision_tools import find_decision_refusal, find_review_conflict

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    for name, meaning in [("short_reason", "a reason of five characters"),
                          ("unknown_decision", "a decision that is none of the three"),
                          ("short_request", "a request of two characters")]:
        assert saved_run[name]["status"] == 422, (
            "Sending "
            + meaning
            + " was answered "
            + str(saved_run[name]["status"])
            + " with "
            + str(saved_run[name]["body"])[:200]
            + ". The request's schema refuses it with 422 before anything is looked up."
        )
    second = saved_run["second_review"]
    expected_second = config.REQUEST_ALREADY_REVIEWED_MESSAGE.format(
        revision_number=saved_run["revision_number"],
        permit_reference=saved_run["permit_reference"])
    assert (second["status"] == 409 and second["body"].get("detail") == expected_second
            and saved_run["run_record"].get("runs_opened") == 1), (
        "Opening a second review of "
        + saved_run["permit_reference"]
        + " was answered "
        + str(second["status"])
        + " with "
        + str(second["body"])[:200]
        + ", and this run of the suite recorded "
        + str(saved_run["run_record"].get("runs_opened"))
        + " runs of the request. A revision of a request in the queue is reviewed once, and a "
        "second review is refused before a run starts: "
        + repr(expected_second)
    )
    decided = saved_run["decided"]["body"]
    decider = saved_run["decider"]
    assert (decided.get("decision") == config.DECISION_ISSUE
            and decided.get("decided_by") == decider["username"]
            and decided.get("decided_by_role") == config.ROLE_AREA_AUTHORITY
            and decided.get("decided_by_area_group") == decider["area_group"]
            and decided.get("decision_reason") == saved_run["decision_reason"]
            and decided.get("status") == config.REVIEW_STATUS_ISSUED), (
        "The issue came back recorded as "
        + str({key: decided.get(key) for key in ["decision", "decided_by", "decided_by_role",
                                                 "decided_by_area_group", "decision_reason",
                                                 "status"]})
        + ", which is not the decision the Area Authority sent."
    )
    review_id = saved_run["at_gate"]["review_id"]
    again = service_client["client"].post(
        "/reviews/" + review_id + "/decision",
        json={"decision": config.DECISION_ISSUE, "decision_reason": saved_run["decision_reason"]},
        headers=decider["headers"])
    expected_again = config.ALREADY_DECIDED_MESSAGE.format(
        permit_reference=saved_run["permit_reference"])
    assert again.status_code == 409 and again.json().get("detail") == expected_again, (
        "Deciding the review a second time was answered "
        + str(again.status_code)
        + " with "
        + again.text[:200]
        + ". A review is decided once: "
        + repr(expected_again)
    )
    with session_factory() as session:
        stored = session.query(ReviewDecision).filter(ReviewDecision.review_id == review_id).all()
    assert len(stored) == 1 and stored[0].decision == config.DECISION_ISSUE, (
        "The review_decisions table holds "
        + str(len(stored))
        + " decisions for the review where it holds the one decision taken."
    )
    reference = "PTW0950107"
    request = {config.PERMIT_REFERENCE_COLUMN: reference,
               config.PLANT_AREA_COLUMN: config.REACTOR_HOUSE_A}
    sound = {"decision": config.DECISION_ISSUE, "decided_by": "imogen",
             "decided_by_role": config.ROLE_AREA_AUTHORITY,
             "decided_by_area_group": config.PROCESS_UNITS_GROUP,
             "decision_reason": "Every requirement is met on the request."}
    issued_screening = {"permit_reference": reference, "assessment": config.DECISION_ISSUE,
                        "decisions_open": config.DECISIONS_OPEN[config.DECISION_ISSUE]}
    cases = [
        ("an issue of a request assessed RETURN TO REQUESTER", sound,
         dict(issued_screening, assessment=config.DECISION_RETURN,
              decisions_open=config.DECISIONS_OPEN[config.DECISION_RETURN]),
         config.DECISION_NOT_OPEN_MESSAGE.format(
             decision=config.DECISION_ISSUE, permit_reference=reference,
             reason=config.ISSUE_CLOSED_REASONS[config.DECISION_RETURN])),
        ("an issue of a request assessed REFER FOR SIMOPS REVIEW", sound,
         dict(issued_screening, assessment=config.DECISION_REFER,
              decisions_open=config.DECISIONS_OPEN[config.DECISION_REFER]),
         config.DECISION_NOT_OPEN_MESSAGE.format(
             decision=config.DECISION_ISSUE, permit_reference=reference,
             reason=config.ISSUE_CLOSED_REASONS[config.DECISION_REFER])),
        ("a decision with no reason", dict(sound, decision_reason="  "), issued_screening,
         config.DECISION_FIELDS_MISSING_MESSAGE.format(permit_reference=reference,
                                                       fields="decision_reason")),
        ("a decision that is none of the three", dict(sound, decision="PROCEED"),
         issued_screening, config.UNKNOWN_DECISION_MESSAGE.format(
             decision="PROCEED", permit_reference=reference,
             decisions=", ".join(config.DECISIONS))),
        ("a decision by a Permit Coordinator",
         dict(sound, decided_by_role=config.ROLE_PERMIT_COORDINATOR), issued_screening,
         config.DECIDING_ROLE_MESSAGE.format(
             role=config.ROLE_PERMIT_COORDINATOR, permit_reference=reference,
             authority_role=config.ROLE_AREA_AUTHORITY)),
        ("a decision by an Area Authority appointed for the storage areas",
         dict(sound, decided_by="kofi", decided_by_area_group=config.STORAGE_AREAS_GROUP),
         issued_screening, config.NOT_APPOINTED_MESSAGE.format(
             decided_by="kofi", area_group=config.STORAGE_AREAS_GROUP,
             plant_area=config.REACTOR_HOUSE_A, permit_reference=reference)),
        ("a decision by the Area Authority who opened the review", dict(sound, decided_by="dara"),
         issued_screening, config.OWN_REVIEW_MESSAGE.format(decided_by="dara",
                                                            permit_reference=reference)),
        ("a referral of a request assessed ISSUE PERMIT",
         dict(sound, decision=config.DECISION_REFER), issued_screening, None),
        ("a sound issue", sound, issued_screening, None),
    ]
    for meaning, review_decision, screening, expected_refusal in cases:
        refusal = find_decision_refusal(review_decision, request, screening, "dara")
        assert refusal == expected_refusal, (
            "find_decision_refusal answered "
            + meaning
            + " with "
            + repr(refusal)
            + " where it answers "
            + repr(expected_refusal)
            + ". Every decision the workflow's gate would refuse is named before the run "
            "resumes, in the configuration's words, so no call is spent on it."
        )
    other = dict(sound, decision=config.DECISION_RETURN, decided_by="dara")
    for meaning, review_status, pending, expected_conflict in [
            ("a decision on a review already issued", config.REVIEW_STATUS_ISSUED, None,
             config.ALREADY_DECIDED_MESSAGE.format(permit_reference=reference)),
            ("an issue while dara's return is still being recorded",
             config.REVIEW_STATUS_AWAITING, other,
             config.DECISION_IN_PROGRESS_MESSAGE.format(
                 pending_decided_by="dara", pending_decision=config.DECISION_RETURN,
                 permit_reference=reference)),
            ("the same decision sent again to finish it", config.REVIEW_STATUS_AWAITING, sound,
             None),
            ("a decision on a review awaiting one", config.REVIEW_STATUS_AWAITING, None, None)]:
        conflict = find_review_conflict(review_status, pending, sound, reference)
        assert conflict == expected_conflict, (
            "find_review_conflict answered "
            + meaning
            + " with "
            + repr(conflict)
            + " where it answers "
            + repr(expected_conflict)
            + ". A review is decided once, and a decision still being recorded is finished only "
            "by the same decision sent again."
        )


def test_stored_review_matches_tools(saved_run, saved_models, permit_standards, live_files):
    from config import config
    from tools.prediction_tools import (describe_expected_work_hours,
                                        estimate_expected_work_hours,
                                        estimate_suspension_likelihood)
    from tools.request_data_tools import fetch_request
    from tools.screening_tools import count_live_permits, screen_request

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    record = saved_run["final_record"]["body"][0]
    detail = saved_run["detail"]
    screening = detail["screening"]
    register_count = detail["register_count"]
    requirement_keys = ["permit_type", "standard_document", "requirement_kind",
                        "requirement_name", "requirement_sentence", "result", "blank_answers"]
    pairs = {
        "requirements": ([{key: requirement.get(key) for key in requirement_keys}
                          for requirement in record.get("requirements") or []],
                         [{key: requirement.get(key) for key in requirement_keys}
                          for requirement in screening["requirements"]]),
        "assessment": (record.get("assessment"), screening["assessment"]),
        "decisions_open": (record.get("decisions_open"), screening["decisions_open"]),
        "simops_triggers": (record.get("simops_triggers"), screening["simops_triggers"]),
        "live_permits": (record.get("live_permits"), register_count["live_permits"]),
        "conflicting_permits": (record.get("conflicting_permits"),
                                register_count["conflicting_permits"]),
        "evidence": (record.get("evidence"), detail["evidence_findings"]),
        "disagreements": (record.get("disagreements"), detail["disagreements"]),
        "suspension_likelihood": (record.get("suspension_likelihood"),
                                  detail["likelihood"]["suspension_likelihood"]),
        "likelihood_band": (record.get("likelihood_band"),
                            detail["likelihood"]["likelihood_band"]),
        "expected_work_hours": (record.get("expected_work_hours"),
                                detail["work_hours"]["expected_work_hours"]),
        "planned_work_hours": (record.get("planned_work_hours"),
                               detail["work_hours"]["planned_work_hours"]),
        "extension_expected": (record.get("extension_expected"),
                               detail["work_hours"]["extension_expected"]),
        "work_hours_text": (record.get("work_hours_text"),
                            describe_expected_work_hours(detail["work_hours"])),
        "permit_type": (record.get("permit_type"),
                        detail["request"][config.PERMIT_TYPE_COLUMN]),
        "linked_permit_type": (record.get("linked_permit_type"),
                               detail["request"][config.LINKED_PERMIT_TYPE_COLUMN]),
        "plant_area": (record.get("plant_area"), detail["request"][config.PLANT_AREA_COLUMN]),
        "revision_number": (record.get("revision_number"), saved_run["revision_number"]),
    }
    for count_column in config.REQUIREMENT_COUNT_RESULTS:
        pairs[count_column] = (record.get(count_column), screening[count_column])
    differing = {name: pair for name, pair in pairs.items() if pair[0] != pair[1]}
    assert len(differing) == 0, (
        "The stored review differs from the tools' findings for "
        + saved_run["permit_reference"]
        + " in "
        + str({name: (str(pair[0])[:80], str(pair[1])[:80]) for name, pair in
               differing.items()})
        + ". The review records the screening under both standards, the permits live beside "
        "the work, the statement evidence, the flags, the likelihood and the expected work "
        "hours the tools worked out, never the agents' retelling of them."
    )
    request = fetch_request(live_files["requests"], saved_run["permit_reference"])
    own_count = count_live_permits(live_files["register"], request)
    for count_column in [config.CONCURRENT_LIVE_PERMITS_COLUMN,
                         config.CONFLICTING_LIVE_PERMITS_COLUMN]:
        request[count_column] = own_count[count_column]
    own_screening = screen_request(request, permit_standards["standards"])
    likelihood = estimate_suspension_likelihood(
        saved_models["suspension"], saved_models["suspension_schema"], request, own_screening)
    work_hours = estimate_expected_work_hours(
        saved_models["work_hours"], saved_models["work_hours_schema"], request, own_screening)
    assert (isinstance(own_screening, dict) and isinstance(likelihood, dict)
            and isinstance(work_hours, dict)), (
        "Worked out again for "
        + saved_run["permit_reference"]
        + ", the screening, the likelihood and the expected work hours came back as "
        + type(own_screening).__name__
        + ", "
        + type(likelihood).__name__
        + " and "
        + type(work_hours).__name__
        + ", where each is a record of its findings."
    )
    assert (likelihood.get("suspension_likelihood") == record.get("suspension_likelihood")
            and work_hours.get("expected_work_hours") == record.get("expected_work_hours")
            and own_screening.get("assessment") == record.get("assessment")
            and own_screening.get("simops_triggers") == record.get("simops_triggers")
            and own_count.get("live_permits") == record.get("live_permits")), (
        "Worked out again from the live files, the saved models and the request's own "
        "standards, "
        + saved_run["permit_reference"]
        + " is assessed "
        + str(own_screening.get("assessment"))
        + " with the triggers "
        + str(own_screening.get("simops_triggers"))
        + ", the permits live "
        + str(own_count.get("live_permits"))
        + ", a likelihood of "
        + str(likelihood.get("suspension_likelihood"))
        + " and "
        + str(work_hours.get("expected_work_hours"))
        + " expected work hours, where the review stored "
        + str(record.get("assessment"))
        + ", "
        + str(record.get("simops_triggers"))
        + ", "
        + str(record.get("live_permits"))
        + ", "
        + str(record.get("suspension_likelihood"))
        + " and "
        + str(record.get("expected_work_hours"))
        + "."
    )


def test_sign_in_and_out_decide_access(service_client):
    from access.password_security import hash_password
    from config import config
    from persistence.database import session_factory
    from persistence.models import UserAccount
    from persistence.repository import fetch_user_account

    assert service_client["problem"] is None, service_client["problem"]
    client = service_client["client"]
    coordinators = [account for account in config.SEED_USER_ACCOUNTS
                    if account["role"] == config.ROLE_PERMIT_COORDINATOR]
    account = coordinators[1]
    other_account = coordinators[2]
    credentials = {"username": account["username"], "password": account["password"]}
    signed_in = client.post("/auth/signin", json=credentials)
    body = signed_in.json() if signed_in.status_code == 200 else {}
    assert ({"username", "display_name", "role", "area_group", "session_token"} <= set(body)
            and len(str(body["session_token"]).strip()) > 0), (
        "Signing in as "
        + account["username"]
        + " with the password the configuration seeds returned "
        + str(signed_in.status_code)
        + " with "
        + signed_in.text[:200]
        + ", where a sign-in answers with the user's name, display name, role, area group and a "
        "session token."
    )
    assert (body["username"] == account["username"] and body["role"] == account["role"]
            and body["display_name"] == account["display_name"]
            and body["area_group"] == account["area_group"]), (
        "Signing in as "
        + account["username"]
        + " answered with the name "
        + repr(body["username"])
        + ", the display name "
        + repr(body["display_name"])
        + ", the role "
        + repr(body["role"])
        + " and the area group "
        + repr(body["area_group"])
        + ", which are not those of the account the configuration seeds."
    )
    wrong = client.post("/auth/signin", json={"username": account["username"],
                                              "password": "wrong-" + account["password"]})
    stranger = client.post("/auth/signin", json={"username": "no-such-person",
                                                 "password": account["password"]})
    assert (wrong.status_code == 401 and stranger.status_code == 401
            and wrong.json() == stranger.json() == {"detail": config.SIGN_IN_FAILED_MESSAGE}), (
        "A wrong password was answered "
        + str(wrong.status_code)
        + " with "
        + wrong.text[:200]
        + " and an unknown username "
        + str(stranger.status_code)
        + " with "
        + stranger.text[:200]
        + ". Both are refused with 401 and the same words, "
        + repr(config.SIGN_IN_FAILED_MESSAGE)
        + ", so a stranger cannot tell which usernames exist."
    )
    parameters = {"permit_type": config.QUEUE_PERMIT_TYPE_ALL}
    first_headers = {config.SESSION_TOKEN_HEADER: body["session_token"]}
    accepted = client.get("/queue", params=parameters, headers=first_headers)
    assert accepted.status_code == 200, (
        "The token just given to "
        + account["username"]
        + " was answered "
        + str(accepted.status_code)
        + " by the queue. A fresh session token opens what its role is permitted."
    )
    for meaning, headers in [("with no token", None),
                             ("with a token nobody was given",
                              {config.SESSION_TOKEN_HEADER: "not-a-token-anyone-was-given"})]:
        refused = client.get("/queue", params=parameters, headers=headers)
        assert (refused.status_code == 401
                and refused.json() == {"detail": config.SIGN_IN_REQUIRED_MESSAGE}), (
            "The queue answered a request "
            + meaning
            + " with "
            + str(refused.status_code)
            + " and "
            + refused.text[:200]
            + ". Only a signed-in user is served; anyone else is refused with 401 and "
            + repr(config.SIGN_IN_REQUIRED_MESSAGE)
            + "."
        )
    again = client.post("/auth/signin", json=credentials)
    second_token = again.json().get("session_token") if again.status_code == 200 else None
    second_headers = {config.SESSION_TOKEN_HEADER: str(second_token)}
    replaced = client.get("/queue", params=parameters, headers=first_headers)
    current = client.get("/queue", params=parameters, headers=second_headers)
    assert (again.status_code == 200 and replaced.status_code == 401
            and current.status_code == 200), (
        "After "
        + account["username"]
        + " signed in a second time, answered "
        + str(again.status_code)
        + ", the first token got "
        + str(replaced.status_code)
        + " from the queue and the second "
        + str(current.status_code)
        + ". An account holds one session: signing in somewhere else ends the session before "
        "it."
    )
    signed_out = client.post("/auth/signout", headers=second_headers)
    after = client.get("/queue", params=parameters, headers=second_headers)
    twice = client.post("/auth/signout", headers=second_headers)
    assert (signed_out.status_code == 200
            and signed_out.json() == {"username": account["username"], "signed_out": True}
            and after.status_code == 401 and twice.status_code == 401), (
        "Signing out answered "
        + str(signed_out.status_code)
        + " with "
        + signed_out.text[:200]
        + "; the token then got "
        + str(after.status_code)
        + " from the queue and "
        + str(twice.status_code)
        + " when it signed out again. Signing out answers with the username and signed_out "
        "true, and the token it ends opens nothing afterwards."
    )
    with session_factory() as session:
        stored = fetch_user_account(session, UserAccount.username == account["username"])
        other = fetch_user_account(session, UserAccount.username == other_account["username"])
    missing = sorted({"password_hash", "password_salt"}
                     - (set(stored if isinstance(stored, dict) else {})
                        & set(other if isinstance(other, dict) else {})))
    assert len(missing) == 0, (
        "fetch_user_account read "
        + account["username"]
        + " and "
        + other_account["username"]
        + " back without "
        + str(missing)
        + ", so how their passwords are kept cannot be read."
    )
    assert (stored["password_hash"] != account["password"]
            and stored["password_hash"] == hash_password(account["password"],
                                                         stored["password_salt"])), (
        "The password stored for "
        + account["username"]
        + " is not the salted hash of the password. A password kept as written is read by "
        "anyone who can open the database."
    )
    assert (stored["password_salt"] != other["password_salt"]
            and stored["password_hash"] != other["password_hash"]), (
        account["username"]
        + " and "
        + other_account["username"]
        + " share a password and are stored with the same salt or the same hash, so one "
        "cracked hash opens both accounts."
    )


def test_roles_reach_permissions_and_area_groups(service_client, staff, live_files):
    from fastapi import HTTPException

    from access.permissions import authorize_plant_area
    from config import config
    from persistence.database import session_factory
    from persistence.models import AgentRun

    assert service_client["problem"] is None, service_client["problem"]
    assert staff["problem"] is None, staff["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    client = service_client["client"]
    signed_in = {}
    for role_name, signed in [
            (config.ROLE_AREA_AUTHORITY,
             staff["authorities"].get(config.PROCESS_UNITS_GROUP, [{}, {}])[1]),
            (config.ROLE_PERMIT_COORDINATOR, staff["coordinator"]),
            (config.ROLE_CONTROL_ROOM_OPERATOR, staff["operator"])]:
        assert {"username", "role", "area_group", "pages", "permissions", "headers"} <= set(
                signed), (
            "A signed-in "
            + role_name
            + " came back as "
            + str({key: value for key, value in signed.items() if key != "headers"})[:200]
            + ", where a sign-in answers with the username, role, area group and the pages "
            "and permissions of the user's role."
        )
        assert (signed["role"] == role_name
                and sorted(signed["pages"]) == sorted(config.ROLE_PAGE_ACCESS[role_name])
                and sorted(signed["permissions"]) == sorted(config.ROLE_PERMISSIONS[role_name])), (
            "Signing in as a "
            + role_name
            + " answered with the role "
            + str(signed["role"])
            + ", the pages "
            + str(signed["pages"])
            + " and the permissions "
            + str(signed["permissions"])
            + ", where the configuration grants that role the pages "
            + str(config.ROLE_PAGE_ACCESS[role_name])
            + " and the permissions "
            + str(config.ROLE_PERMISSIONS[role_name])
            + "."
        )
        signed_in[role_name] = (signed, signed["headers"])
    unknown_review = config.REVIEW_ID_PREFIX + "DOES-NOT-EXIST"
    probes = [
        (config.PERMISSION_QUEUE, "GET", "/queue", {"permit_type": config.QUEUE_PERMIT_TYPE_ALL},
         None, 200),
        (config.PERMISSION_QUEUE, "GET", "/queue/PTW0000000", None, None, 404),
        (config.PERMISSION_REVIEW, "POST", "/reviews", None,
         {"permit_reference": "PTW0000000",
          "coordinator_request": "Are the requirements met on the request?"}, 404),
        (config.PERMISSION_DECISION, "POST", "/reviews/" + unknown_review + "/decision", None,
         {"decision": config.DECISION_REFER,
          "decision_reason": "A SIMOPS trigger applies to this request."}, 404),
        (config.PERMISSION_AUDIT, "GET", "/reviews",
         {"status_value": [config.REVIEW_STATUS_AWAITING]}, None, 200),
        (config.PERMISSION_AUDIT, "GET", "/reviews/" + unknown_review + "/calls", None, None,
         404),
        (config.PERMISSION_AUDIT, "GET", "/audit/chat-turns", {"turn_limit": 1}, None, 200),
        (config.PERMISSION_CHAT, "GET", "/chat/turns", {"turn_limit": 1}, None, 200),
    ]
    for role_name, (_account, role_headers) in signed_in.items():
        for permission, method, path, parameters, payload, status_when_granted in probes:
            reply = client.request(method, path, params=parameters, json=payload,
                                   headers=role_headers)
            if permission in config.ROLE_PERMISSIONS[role_name]:
                assert reply.status_code == status_when_granted, (
                    "A signed-in "
                    + role_name
                    + " calling "
                    + method
                    + " "
                    + path
                    + " got "
                    + str(reply.status_code)
                    + " with "
                    + reply.text[:200]
                    + " where it gets "
                    + str(status_when_granted)
                    + ". The "
                    + permission
                    + " permission is granted to that role, so the guard lets it through and "
                    "the endpoint answers: a request or a review that does not exist is answered "
                    "404, before any language model call."
                )
            else:
                expected_refusal = config.PERMISSION_REFUSED_MESSAGE.format(
                    role=role_name, permission=permission)
                assert (reply.status_code == 403
                        and reply.json().get("detail") == expected_refusal), (
                    "A signed-in "
                    + role_name
                    + " calling "
                    + method
                    + " "
                    + path
                    + " got "
                    + str(reply.status_code)
                    + " with "
                    + reply.text[:200]
                    + ". The "
                    + permission
                    + " permission is not granted to that role, so the guard refuses it with 403 "
                    "and the configuration's words, "
                    + repr(expected_refusal)
                    + ", whether or not the thing asked for exists."
                )
    authority, authority_headers = signed_in[config.ROLE_AREA_AUTHORITY]
    own_areas = config.AREA_GROUPS[authority["area_group"]]
    requests = live_files["requests"]
    outside = requests[~requests[config.PLANT_AREA_COLUMN].isin(own_areas)].head(1)
    inside = requests[requests[config.PLANT_AREA_COLUMN].isin(own_areas)].head(1)
    assert len(outside) == 1 and len(inside) == 1, (
        "The live queue holds no request inside, or none outside, the plant areas of "
        + authority["area_group"]
        + ", where it holds both."
    )
    outside_reference = outside[config.PERMIT_REFERENCE_COLUMN].iloc[0]
    outside_area = outside[config.PLANT_AREA_COLUMN].iloc[0]
    with session_factory() as session:
        runs_before = session.query(AgentRun).count()
    refused = client.post("/reviews", json={"permit_reference": outside_reference,
                                            "coordinator_request": "Is every requirement met?"},
                          headers=authority_headers)
    with session_factory() as session:
        runs_after = session.query(AgentRun).count()
    expected_refusal = config.AREA_GROUP_REFUSED_MESSAGE.format(
        username=authority["username"], area_group=authority["area_group"],
        plant_area=outside_area, permit_reference=outside_reference)
    assert (refused.status_code == 403 and refused.json().get("detail") == expected_refusal
            and runs_after == runs_before), (
        "An Area Authority appointed for "
        + authority["area_group"]
        + " opening a review of "
        + outside_reference
        + ", in "
        + outside_area
        + ", was answered "
        + str(refused.status_code)
        + " with "
        + refused.text[:200]
        + ", and the runs went from "
        + str(runs_before)
        + " to "
        + str(runs_after)
        + ". An Area Authority acts only on the plant areas of its own group, so it is refused "
        "with 403 and "
        + repr(expected_refusal)
        + " before any run starts."
    )
    coordinator_account = signed_in[config.ROLE_PERMIT_COORDINATOR][0]
    outcomes = {}
    for meaning, user, plant_area, reference in [
            ("an Area Authority inside its group", authority,
             inside[config.PLANT_AREA_COLUMN].iloc[0],
             inside[config.PERMIT_REFERENCE_COLUMN].iloc[0]),
            ("an Area Authority outside its group", authority, outside_area, outside_reference),
            ("a Permit Coordinator anywhere", coordinator_account, outside_area,
             outside_reference)]:
        try:
            outcomes[meaning] = authorize_plant_area(user, plant_area, reference)
        except HTTPException as error:
            outcomes[meaning] = (error.status_code, error.detail)
        except Exception as error:
            outcomes[meaning] = type(error).__name__ + ": " + str(error)[:120]
    expected_outcomes = {"an Area Authority inside its group": None,
                         "an Area Authority outside its group": (403, expected_refusal),
                         "a Permit Coordinator anywhere": None}
    assert outcomes == expected_outcomes, (
        "authorize_plant_area answered "
        + str(outcomes)
        + " where it answers "
        + str(expected_outcomes)
        + ". It lets an Area Authority act within the plant areas of its group and refuses it "
        "with 403 outside them, and a Permit Coordinator, who belongs to no group, is not held "
        "to one."
    )


def test_queue_order_and_rows_match_detail(service_client, staff, live_files, saved_models,
                                           permit_standards):
    from config import config
    from tools.prediction_tools import (estimate_expected_work_hours,
                                        estimate_suspension_likelihood)
    from tools.request_data_tools import fetch_request, rank_request_queue
    from tools.screening_tools import count_live_permits, screen_request

    assert staff["problem"] is None, staff["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    assert saved_models["problem"] is None, saved_models["problem"]
    assert permit_standards["problem"] is None, permit_standards["problem"]
    entry, unlinked = config.CONFINED_SPACE_ENTRY, config.NO_LINKED_PERMIT
    planted = live_files["requests"].head(5).copy()
    planted[config.PERMIT_REFERENCE_COLUMN] = ["PTW0500004", "PTW0500002", "PTW0500001",
                                               "PTW0500003", "PTW0500000"]
    planted[config.PERMIT_TYPE_COLUMN] = [config.WORK_AT_HEIGHT, entry, config.HOT_WORK, entry,
                                          config.ELECTRICAL_ISOLATION]
    planted[config.LINKED_PERMIT_TYPE_COLUMN] = [unlinked, unlinked, entry, config.HOT_WORK,
                                                 unlinked]
    ranked = rank_request_queue(planted)
    ranked = list(ranked) if isinstance(ranked, (list, tuple)) else ranked
    assert ranked == ["PTW0500001", "PTW0500002", "PTW0500003", "PTW0500000", "PTW0500004"], (
        "rank_request_queue put planted requests - work at height PTW0500004, a confined space "
        "entry PTW0500002, hot work PTW0500001 linked to a confined space entry, a confined "
        "space entry PTW0500003 linked to hot work and an electrical isolation PTW0500000 - in "
        "the order "
        + str(ranked)
        + ". Requests whose permit type or linked type is a confined space entry come first, the "
        "rest after them, and each group runs by permit reference."
    )
    client = service_client["client"]
    headers = staff["coordinator"]["headers"]
    reply = client.get("/queue", params={"permit_type": config.QUEUE_PERMIT_TYPE_ALL},
                       headers=headers)
    assert reply.status_code == 200 and isinstance(reply.json(), list), (
        "The queue returned "
        + str(reply.status_code)
        + " with "
        + reply.text[:200]
        + " to a signed-in Permit Coordinator."
    )
    everything = reply.json()
    missing = sorted({"queue_position", "permit_reference", "revision_number", "permit_type",
                      "linked_permit_type", "plant_area", "request_channel", "request_status",
                      "document_count", "requirements_met_count", "requirements_not_met_count",
                      "requirements_not_documented_count", "live_permit_count",
                      "conflicting_permit_count", "simops_trigger_count", "assessment",
                      "suspension_likelihood", "likelihood_band", "expected_work_hours",
                      "extension_expected", "disagreement_count"}
                     - set(everything[0] if everything else {}))
    assert len(missing) == 0, (
        "A queue row carries no "
        + str(missing)
        + ", so the queue cannot be ordered, filtered or read."
    )
    references = [row["permit_reference"] for row in everything]
    requests = live_files["requests"]
    open_references = list(requests[config.PERMIT_REFERENCE_COLUMN])
    assert len(set(references)) == len(references) and sorted(references) == sorted(
            open_references), (
        "The queue holds "
        + str(len(references))
        + " rows for "
        + str(len(set(references)))
        + " requests, where the open permit requests, read through the cleaning, number "
        + str(len(open_references))
        + ". Every open request is in the queue exactly once, at its latest revision."
    )
    entries = set(requests.loc[(requests[config.PERMIT_TYPE_COLUMN] == entry)
                               | (requests[config.LINKED_PERMIT_TYPE_COLUMN] == entry),
                               config.PERMIT_REFERENCE_COLUMN])
    flags = [row["permit_reference"] in entries for row in everything]
    first_other = flags.index(False) if False in flags else len(flags)
    first_block, other_block = references[:first_other], references[first_other:]
    assert (not any(flags[first_other:]) and len(entries) > 0
            and set(first_block) == entries
            and first_block == sorted(first_block)
            and other_block == sorted(other_block)), (
        "The queue opens with "
        + str(first_block[:6])
        + " and then runs "
        + str(other_block[:6])
        + " ..., where the open requests for a confined space entry, as the permit type or the "
        "linked type, are "
        + str(sorted(entries))
        + ". They come first and then the rest, each group in the order of its permit "
        "references, so the Permit Coordinators work the queue from the top."
    )
    assert [row["queue_position"] for row in everything] == list(range(1, len(everything) + 1)), (
        "The queue positions run "
        + str([row["queue_position"] for row in everything][:10])
        + " ... where they count from 1 down the queue."
    )
    checked_against_tools = [references[0], references[len(references) // 2], references[-1]]
    requirement_keys = ["permit_type", "standard_document", "requirement_kind",
                        "requirement_name", "requirement_sentence", "result", "blank_answers"]
    differing = {}
    for row in everything:
        detail_reply = client.get("/queue/" + row["permit_reference"], headers=headers)
        detail = detail_reply.json() if detail_reply.status_code == 200 else {}
        described = detail.get("request", {})
        shown = detail.get("screening", {})
        shown_count = detail.get("register_count", {})
        shown_likelihood = detail.get("likelihood", {})
        shown_hours = detail.get("work_hours", {})
        expected = {
            "queue_position": detail.get("queue_position"),
            "request_status": detail.get("request_status"),
            "permit_type": described.get(config.PERMIT_TYPE_COLUMN),
            "linked_permit_type": described.get(config.LINKED_PERMIT_TYPE_COLUMN),
            "plant_area": described.get(config.PLANT_AREA_COLUMN),
            "request_channel": described.get("request_channel"),
            "document_count": len(detail.get("request_documents", [])),
            "requirements_met_count": shown.get(config.REQUIREMENTS_MET_COUNT_COLUMN),
            "requirements_not_met_count": shown.get(config.REQUIREMENTS_NOT_MET_COUNT_COLUMN),
            "requirements_not_documented_count": shown.get(
                config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN),
            "live_permit_count": shown_count.get(config.CONCURRENT_LIVE_PERMITS_COLUMN),
            "conflicting_permit_count": shown_count.get(config.CONFLICTING_LIVE_PERMITS_COLUMN),
            "simops_trigger_count": len(shown.get("simops_triggers", [])),
            "assessment": shown.get("assessment"),
            "suspension_likelihood": shown_likelihood.get("suspension_likelihood"),
            "likelihood_band": shown_likelihood.get("likelihood_band"),
            "expected_work_hours": shown_hours.get("expected_work_hours"),
            "extension_expected": shown_hours.get("extension_expected"),
            "disagreement_count": len(detail.get("disagreements", [])),
        }
        for key, value in expected.items():
            if row[key] != value:
                differing[row["permit_reference"]] = (key + " is " + str(row[key])[:60]
                                                      + " where its detail gives "
                                                      + str(value)[:60])
        if str(row["revision_number"]) != described.get(config.REVISION_NUMBER_COLUMN):
            differing[row["permit_reference"]] = ("revision_number is "
                                                  + str(row["revision_number"])
                                                  + " where its detail gives "
                                                  + str(described.get(
                                                      config.REVISION_NUMBER_COLUMN)))
        if row["permit_reference"] in checked_against_tools:
            request = fetch_request(requests, row["permit_reference"])
            register_count = count_live_permits(live_files["register"], request)
            for count_column in [config.CONCURRENT_LIVE_PERMITS_COLUMN,
                                 config.CONFLICTING_LIVE_PERMITS_COLUMN]:
                request[count_column] = register_count[count_column]
            screening = screen_request(request, permit_standards["standards"])
            assert isinstance(screening, dict) and isinstance(
                    screening.get("requirements"), list), (
                "screen_request returned "
                + str(screening)[:200]
                + " for "
                + row["permit_reference"]
                + ", where it returns the request's screening."
            )
            for key in (["permit_reference", "permit_types", "simops_triggers", "assessment",
                         "decisions_open"] + list(config.REQUIREMENT_COUNT_RESULTS)):
                if shown.get(key) != screening.get(key):
                    differing[row["permit_reference"]] = ("the detail's " + key
                                                          + " is not the tools' own")
            for key in ["live_permits", "conflicting_permits"]:
                if shown_count.get(key) != register_count.get(key):
                    differing[row["permit_reference"]] = ("the detail's " + key
                                                          + " is not the register's count")
            own_requirements = [{key: requirement.get(key) for key in requirement_keys}
                                for requirement in screening["requirements"]
                                if isinstance(requirement, dict)]
            shown_requirements = [{key: requirement.get(key) for key in requirement_keys}
                                  for requirement in shown.get("requirements", [])
                                  if isinstance(requirement, dict)]
            if shown_requirements != own_requirements:
                differing[row["permit_reference"]] = ("the detail's requirements are not "
                                                      "the tools'")
            likelihood = estimate_suspension_likelihood(
                saved_models["suspension"], saved_models["suspension_schema"], request, screening)
            work_hours = estimate_expected_work_hours(
                saved_models["work_hours"], saved_models["work_hours_schema"], request, screening)
            if shown_likelihood != likelihood or shown_hours != work_hours:
                differing[row["permit_reference"]] = ("the detail's figures are not the saved "
                                                      "models'")
    assert len(differing) == 0, (
        str(len(differing))
        + " rows of the queue differ from their detail or from the tools, for example "
        + str(dict(list(differing.items())[:3]))
        + ". Each row shows what its detail shows: its request's own screening under both its "
        "standards, the permits live beside it in the register, the figures the saved models "
        "give it, the documents attached to it and the statement sentences flagged in them."
    )
    for permit_type in config.QUEUE_PERMIT_TYPE_OPTIONS[1:]:
        filtered_reply = client.get("/queue", params={"permit_type": permit_type},
                                    headers=headers)
        filtered = filtered_reply.json() if filtered_reply.status_code == 200 else []
        assert len(filtered) > 0 and filtered == [
                row for row in everything
                if permit_type in (row["permit_type"], row["linked_permit_type"])], (
            "Filtering the queue to "
            + permit_type
            + " did not return exactly the requests of the full queue whose permit type or "
            "linked type is "
            + permit_type
            + ", in the same order and with the same positions."
        )
    for meaning, parameters in [("a permit type Hexley has no standard for",
                                 {"permit_type": "Diving operations"}),
                                ("no permit type at all", None)]:
        refused = client.get("/queue", params=parameters, headers=headers)
        assert refused.status_code == 422, (
            "Asking for the queue with "
            + meaning
            + " returned "
            + str(refused.status_code)
            + " rather than 422. The filter takes a permit type the configuration names, or all "
            "of them, and nothing else."
        )
    unknown = client.get("/queue/PTW0000000", headers=headers)
    expected_unknown = config.UNKNOWN_REQUEST_MESSAGE.format(permit_reference="PTW0000000")
    assert unknown.status_code == 404 and unknown.json().get("detail") == expected_unknown, (
        "The detail of PTW0000000, which is not an open request, returned "
        + str(unknown.status_code)
        + " with "
        + unknown.text[:200]
        + " where it is 404 with "
        + repr(expected_unknown)
        + "."
    )


def test_chat_answers_from_documents_with_memory(chat_replies, knowledge_collection, tmp_path):
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from chat.conversation_memory import recall_conversation
    from config import config
    from knowledge.vector_index import search_documents
    from persistence.database import session_factory
    from persistence.models import Base
    from persistence.repository import insert_chat_turn

    assert chat_replies["problem"] is None, chat_replies["problem"]
    assert knowledge_collection["problem"] is None, knowledge_collection["problem"]
    for name, meaning in [("standard", "Asking the permit assistant what the hot work permit "
                                       "standard asks for"),
                          ("turns", "Reading back the Permit Coordinator's own chat turns")]:
        assert chat_replies[name]["status"] == 200, (
            meaning
            + " returned "
            + str(chat_replies[name]["status"])
            + " with "
            + str(chat_replies[name]["body"])[:300]
            + "."
        )
    reply = chat_replies["standard"]["body"]
    missing = sorted({"question_type", "permit_reference", "plant_area", "permit_type", "answer",
                      "source_documents", "was_refused"} - set(reply))
    assert len(missing) == 0, (
        "The assistant's reply carries no "
        + str(missing)
        + ", so what it answered and where from cannot be read."
    )
    assert (reply["was_refused"] is False and reply["question_type"] in config.CHAT_ANSWERED_TYPES
            and reply["permit_reference"] is None and reply["plant_area"] is None), (
        "Asking what the hot work permit standard asks for was answered as "
        + str(reply["question_type"])
        + " about the request "
        + str(reply["permit_reference"])
        + " and the plant area "
        + str(reply["plant_area"])
        + (" and refused" if reply["was_refused"] else "")
        + ". The standard answers it, so the question is answered from the documents, names no "
        "request or plant area and is not refused."
    )
    folder = {name for name in os.listdir(config.DOCS_DIRECTORY) if name.lower().endswith(".pdf")}
    cited = reply["source_documents"]
    searched = search_documents(knowledge_collection["collection"],
                                [chat_replies["standard_question"]])
    grounding = (searched[0][0].get("document_name")
                 if isinstance(searched, list) and len(searched) > 0
                 and isinstance(searched[0], list) and len(searched[0]) > 0
                 and isinstance(searched[0][0], dict) else None)
    assert len(cited) > 0 and set(cited) <= folder and grounding in cited, (
        "The answer cites "
        + str(cited)
        + ". An answer names the documents it came from, each of them in the document folder, "
        "and among them the one holding the passage search_documents finds closest to the "
        "question, "
        + repr(grounding)
        + "; an answer that cannot is an answer from general knowledge."
    )
    assert not set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS), (
        "The answer cites "
        + str(sorted(set(cited) & set(config.CHAT_EXCLUDED_DOCUMENTS)))
        + ", which the assistant never quotes: it records past reviews of named requests, not "
        "what the standards say."
    )
    assert len(reply["answer"].strip()) > 0 and reply["answer"] != config.CHAT_REFUSAL_MESSAGE, (
        "The answer given was "
        + repr(reply["answer"][:120])
        + ", which answers nothing."
    )
    turns = chat_replies["turns"]["body"]
    missing = sorted({"turn_id", "username", "user_role", "question_text", "question_type",
                      "permit_reference", "plant_area", "permit_type", "answer_text",
                      "source_documents", "closest_distance", "was_refused"}
                     - set(turns[0] if len(turns) > 0 else {}))
    assert len(turns) > 0 and len(missing) == 0, (
        "The Permit Coordinator's chat turns read back as "
        + str(len(turns))
        + " turns"
        + ("" if len(turns) == 0 else ", the first without " + str(missing))
        + ", where each turn read back carries its identifier, who asked and in which role, "
        "the question, its type, its request, plant area and permit type, the answer, the "
        "documents it cites, its closest distance and whether it was refused."
    )
    asked = [turn for turn in turns if turn["question_text"] == chat_replies["standard_question"]
             and turn["turn_id"] > chat_replies["latest_turn_before"]]
    assert len(asked) > 0, (
        "The Permit Coordinator's latest turns read back as "
        + str([turn["question_text"] for turn in turns])
        + ", which do not include the question just answered, so the audit trail and the chat "
        "page lose it."
    )
    latest = asked[0]
    assert (latest["question_type"] == reply["question_type"] and latest["was_refused"] is False
            and latest["answer_text"] == reply["answer"]
            and latest["permit_reference"] is None
            and latest["username"] == chat_replies["asker"]
            and latest["user_role"] == config.ROLE_PERMIT_COORDINATOR), (
        "The stored turn is "
        + str({key: latest[key] for key in ["username", "user_role", "question_type",
                                            "permit_reference", "was_refused"]})
        + ", which is not who asked or how the question was just answered."
    )
    assert latest["closest_distance"] <= config.CHAT_MAX_RELEVANCE_DISTANCE, (
        "An answered question is stored with its closest passage "
        + str(latest["closest_distance"])
        + " away, beyond the "
        + str(config.CHAT_MAX_RELEVANCE_DISTANCE)
        + " within which an answer counts as found in the documents."
    )
    assert all(name in latest["source_documents"] for name in cited), (
        "The stored turn cites "
        + repr(latest["source_documents"][:300])
        + " where the answer cited "
        + str(cited)
        + ". The audit trail shows where each answer came from."
    )
    follow_up = "And what documentation does it ask the performing authority to provide?"
    with session_factory() as session:
        remembered = recall_conversation(session, chat_replies["asker"], follow_up)
        someone_else = recall_conversation(session, chat_replies["someone_else"], follow_up)
    assert all(isinstance(memory, dict)
               and isinstance(memory.get("history_text"), str)
               and isinstance(memory.get("search_text"), str)
               for memory in [remembered, someone_else]), (
        "recall_conversation returned "
        + str(remembered)[:200]
        + ", where it returns the history_text the next prompt is told and the search_text "
        "searched for it."
    )
    assert (chat_replies["standard_question"] in remembered["history_text"]
            and reply["answer"] in remembered["history_text"]), (
        "The conversation recalled for the Permit Coordinator's next question does not hold the "
        "question just asked and its answer, so a follow-up such as "
        + repr(follow_up)
        + " would be read with no idea what it follows."
    )
    assert (follow_up in remembered["search_text"]
            and chat_replies["standard_question"] in remembered["search_text"]), (
        "The text searched for the follow-up is "
        + repr(remembered["search_text"][:200])
        + ". It carries the new question and the question asked just before it."
    )
    assert (chat_replies["standard_question"] not in someone_else["history_text"]
            and chat_replies["standard_question"] not in someone_else["search_text"]), (
        "The conversation recalled for a Control Room Operator carries the Permit Coordinator's "
        "question "
        + repr(chat_replies["standard_question"])
        + ". Each user's memory is their own last exchanges."
    )
    planted_turns = [
        (config.CHAT_TYPE_PERMIT_STANDARD, "What does the excavation standard ask for first?",
         "A task risk assessment for the job is attached to the request, as the excavation "
         "standard states.", None, None, None),
        (config.CHAT_TYPE_REQUEST_STATUS, "What is the status of PTW0500001?",
         "REV-PLANTED00001, request PTW0500001 revision 1 (Hot work) in Boiler House, opened by "
         "marcus: issued.", "PTW0500001", None, None),
        (config.CHAT_TYPE_CONTROL_OF_WORK, "Who decides a review?",
         "An Area Authority appointed for the plant area, who did not open the review, decides "
         "it.", None, None, None),
        (config.CHAT_TYPE_PLANT_AREA_FIGURE,
         "How many hot work permits have been worked in Boiler House?",
         "27 hot work permits have been worked in Boiler House.", None, config.BOILER_HOUSE,
         config.HOT_WORK),
    ]
    planted_engine = create_engine("sqlite:///" + str(tmp_path / "planted_chat.db"))
    Base.metadata.create_all(planted_engine)
    with sessionmaker(bind=planted_engine)() as session:
        for (question_type, question_text, answer_text, permit_reference, plant_area,
             permit_type) in planted_turns:
            insert_chat_turn(session, {
                "username": chat_replies["asker"], "user_role": config.ROLE_PERMIT_COORDINATOR,
                "question_text": question_text, "question_type": question_type,
                "permit_reference": permit_reference, "plant_area": plant_area,
                "permit_type": permit_type, "answer_text": answer_text, "source_documents": "",
                "closest_distance": 0.5, "was_refused": False})
        recalled = recall_conversation(session, chat_replies["asker"], "And in Gatehouse?")
    planted_engine.dispose()
    assert isinstance(recalled, dict) and isinstance(recalled.get("history_text"), str), (
        "recall_conversation returned "
        + str(recalled)[:200]
        + " after four planted exchanges, where it returns the history_text of the last ones."
    )
    history = recalled["history_text"]
    kept = planted_turns[-config.CHAT_HISTORY_TURN_LIMIT:]
    dropped = planted_turns[:-config.CHAT_HISTORY_TURN_LIMIT]
    positions = [history.find(turn[1]) for turn in kept]
    assert (min(positions) >= 0 and positions == sorted(positions)
            and all(turn[1] not in history for turn in dropped)), (
        "After four planted exchanges, the conversation recalled is "
        + repr(history[:500])
        + ". It holds the last "
        + str(config.CHAT_HISTORY_TURN_LIMIT)
        + " exchanges, oldest first, and nothing before them."
    )
    carried = [fact for fact in ["REV-PLANTED00001", "marcus", "27 hot work permits"]
               if fact in history]
    record_turns = [turn for turn in kept if turn[0] in config.CHAT_RECORD_TYPES]
    assert (len(carried) == 0
            and history.count(config.CHAT_RECORD_ANSWER_IN_HISTORY) == len(record_turns)
            and all(turn[2] in history for turn in kept if turn not in record_turns)), (
        "After a Permit Coordinator was told a request's status and a plant area figure, the "
        "conversation recalled for the next question carries "
        + str(carried)
        + " and "
        + str(history.count(config.CHAT_RECORD_ANSWER_IN_HISTORY))
        + " times "
        + repr(config.CHAT_RECORD_ANSWER_IN_HISTORY)
        + ". The recalled conversation goes into the next prompt, and nothing from the records "
        "is ever put in front of the language model: an answer from the records is replaced by "
        "those words, while an answer from the documents is replayed as given."
    )


def test_plant_area_figures_and_operator_hears_status(saved_run, staff, live_files,
                                                      processed_frames, tmp_path):
    import pandas as pd
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from chat.record_answer import answer_plant_area_figure, answer_request_record
    from config import config
    from persistence.database import session_factory
    from persistence.models import AgentRun, Base, PermitReview
    from tools.request_data_tools import describe_record, fetch_request

    assert saved_run["problem"] is None, saved_run["problem"]
    assert saved_run["decision_problem"] is None, saved_run["decision_problem"]
    assert saved_run["heard_at_gate"]["problem"] is None, saved_run["heard_at_gate"]["problem"]
    assert saved_run["heard_after"]["problem"] is None, saved_run["heard_after"]["problem"]
    assert live_files["problem"] is None, live_files["problem"]
    assert processed_frames["problem"] is None, processed_frames["problem"]
    requests = live_files["requests"]
    reference = saved_run["permit_reference"]
    described = describe_record(fetch_request(requests, reference))
    permit_types = described[config.PERMIT_TYPE_COLUMN]
    if described[config.LINKED_PERMIT_TYPE_COLUMN] != config.NO_LINKED_PERMIT:
        permit_types = config.CHAT_LINKED_PERMIT_TEMPLATE.format(
            permit_type=described[config.PERMIT_TYPE_COLUMN],
            linked_permit_type=described[config.LINKED_PERMIT_TYPE_COLUMN])
    status_lines = {status: config.CHAT_OPERATOR_STATUS_TEMPLATE.format(
        permit_reference=reference, permit_types=permit_types,
        plant_area=described[config.PLANT_AREA_COLUMN],
        status=config.REQUEST_STATUS_PHRASES[status])
        for status in [config.REVIEW_STATUS_AWAITING, config.REVIEW_STATUS_ISSUED]}
    issued_work = config.CHAT_ISSUED_WORK_TEMPLATE.format(
        equipment_tag=described[config.EQUIPMENT_TAG_COLUMN],
        work_party_size=described["work_party_size"])
    at_gate = str(saved_run["heard_at_gate"]["operator_status"])
    after = str(saved_run["heard_after"]["operator_status"])
    assert (at_gate == status_lines[config.REVIEW_STATUS_AWAITING]
            and after == status_lines[config.REVIEW_STATUS_ISSUED] + " " + issued_work), (
        "A Control Room Operator asking about "
        + reference
        + " was told "
        + repr(at_gate[:300])
        + " while the review waited and "
        + repr(after[:300])
        + " after the permit was issued. An operator hears the request's permit types, plant "
        "area and status, first "
        + repr(status_lines[config.REVIEW_STATUS_AWAITING])
        + " and nothing more, and then "
        + repr(status_lines[config.REVIEW_STATUS_ISSUED] + " " + issued_work)
        + ", the equipment and the size of the work party the control room must know of."
    )
    for moment, answer in [
            ("while the review waited", saved_run["heard_at_gate"]["operator_prediction"]),
            ("after the permit was issued", saved_run["heard_after"]["operator_prediction"])]:
        assert answer == config.OUTCOME_NOT_PREDICTED_LINE, (
            "A Control Room Operator asking whether "
            + reference
            + " will be suspended "
            + moment
            + " was told "
            + repr(str(answer)[:300])
            + " where they hear exactly "
            + repr(config.OUTCOME_NOT_PREDICTED_LINE)
            + ". The control room is never told a prediction."
        )
    record = saved_run["final_record"]["body"][0]
    decider = saved_run["decider"]
    review_line = config.CHAT_REVIEW_TEMPLATE.format(
        review_id=saved_run["at_gate"]["review_id"], permit_reference=reference,
        revision_number=saved_run["revision_number"], permit_types=permit_types,
        plant_area=described[config.PLANT_AREA_COLUMN],
        coordinator_id=saved_run["opener"]["username"],
        status=config.REQUEST_STATUS_PHRASES[config.REVIEW_STATUS_ISSUED])
    figures_line = config.CHAT_FIGURES_TEMPLATE.format(
        suspension_likelihood=format(record["suspension_likelihood"],
                                     "." + str(config.LIKELIHOOD_DECIMALS) + "f"),
        likelihood_band=record["likelihood_band"], work_hours_text=record["work_hours_text"])
    decided_line = config.CHAT_DECIDED_TEMPLATE.format(
        decided_by=decider["username"], decided_by_role=config.ROLE_AREA_AUTHORITY,
        decided_by_area_group=decider["area_group"], decision=config.DECISION_ISSUE,
        decision_reason=saved_run["decision_reason"])
    waiting_line = config.CHAT_AWAITING_DECISION_TEMPLATE.format(
        recommended_decision=saved_run["at_gate"]["recommended_decision"])
    team_lines = str(saved_run["heard_after"]["team"]).split("\n")
    waiting_lines = str(saved_run["heard_at_gate"]["team"]).split("\n")
    assert (team_lines[0] == review_line and figures_line in team_lines
            and decided_line in team_lines and waiting_line in waiting_lines), (
        "The permit team asking about "
        + reference
        + " was told "
        + repr(str(saved_run["heard_at_gate"]["team"])[:300])
        + " while the review waited and "
        + repr(str(saved_run["heard_after"]["team"])[:500])
        + " after the permit was issued. Area Authorities and Permit Coordinators hear, in the "
        "configuration's words, the review "
        + repr(review_line[:160])
        + " first, its figures "
        + repr(figures_line[:160])
        + ", what it recommends while it waits and the decision once taken."
    )
    with session_factory() as session:
        stored_review = session.get(PermitReview, saved_run["at_gate"]["review_id"])
        stored_run = session.get(AgentRun, saved_run["at_gate"]["run_id"])
        assert stored_review is not None and stored_run is not None, (
            "The review "
            + saved_run["at_gate"]["review_id"]
            + " or its run "
            + saved_run["at_gate"]["run_id"]
            + " is not in the database, where every review and run is stored."
        )
        review_values = {column.name: getattr(stored_review, column.name)
                         for column in PermitReview.__table__.columns}
        run_values = {column.name: getattr(stored_run, column.name)
                      for column in AgentRun.__table__.columns}
    others = [fetch_request(requests, permit_reference)
              for permit_reference in requests[config.PERMIT_REFERENCE_COLUMN]
              if permit_reference != reference]
    revised = [other for other in others if other[config.REVISION_NUMBER_COLUMN] >= 2][:1]
    linked = [other for other in others
              if other[config.LINKED_PERMIT_TYPE_COLUMN] != config.NO_LINKED_PERMIT
              and other not in revised][:1]
    unlinked = [other for other in others
                if other[config.LINKED_PERMIT_TYPE_COLUMN] == config.NO_LINKED_PERMIT
                and other not in revised][:5]
    assert len(revised) == 1 and len(linked) == 1 and len(unlinked) == 5, (
        "The open requests hold "
        + str(len(revised))
        + " revised, "
        + str(len(linked))
        + " linked and "
        + str(len(unlinked))
        + " unlinked requests besides the one reviewed, where the supplied files hold a revised "
        "request, a linked one and many unlinked."
    )
    issued, returned = config.REVIEW_STATUS_ISSUED, config.REVIEW_STATUS_RETURNED
    referred, awaiting = config.REVIEW_STATUS_REFERRED, config.REVIEW_STATUS_AWAITING
    cases = [(unlinked[0], issued, 0), (linked[0], issued, 0), (unlinked[1], returned, 0),
             (unlinked[2], referred, 0), (unlinked[3], awaiting, 0), (unlinked[4], None, 0),
             (revised[0], returned, 1)]
    planted_engine = create_engine("sqlite:///" + str(tmp_path / "planted_reviews.db"))
    Base.metadata.create_all(planted_engine)
    told = []
    with sessionmaker(bind=planted_engine)() as planted:
        for position, (picked, status, revisions_back) in enumerate(cases, start=1):
            if status is None:
                continue
            picked_reference = picked[config.PERMIT_REFERENCE_COLUMN]
            document_details = None
            if status == issued:
                document_details = {
                    config.PERMIT_REFERENCE_COLUMN: picked_reference,
                    config.EQUIPMENT_TAG_COLUMN: picked[config.EQUIPMENT_TAG_COLUMN],
                    "work_party_size": picked["work_party_size"]}
            planted.add(AgentRun(**dict(
                run_values, run_id="planted-run-" + str(position),
                thread_id="planted-thread-" + str(position), permit_reference=picked_reference)))
            planted.add(PermitReview(**dict(
                review_values, review_id="REV-PLANTED" + str(position).zfill(5),
                run_id="planted-run-" + str(position), permit_reference=picked_reference,
                revision_number=int(picked[config.REVISION_NUMBER_COLUMN]) - revisions_back,
                permit_type=picked[config.PERMIT_TYPE_COLUMN],
                linked_permit_type=picked[config.LINKED_PERMIT_TYPE_COLUMN],
                plant_area=picked[config.PLANT_AREA_COLUMN], status=status,
                document_details=document_details)))
        planted.flush()
        for picked, status, revisions_back in cases:
            picked_reference = picked[config.PERMIT_REFERENCE_COLUMN]
            picked_described = describe_record(picked)
            picked_types = picked_described[config.PERMIT_TYPE_COLUMN]
            if picked_described[config.LINKED_PERMIT_TYPE_COLUMN] != config.NO_LINKED_PERMIT:
                picked_types = config.CHAT_LINKED_PERMIT_TEMPLATE.format(
                    permit_type=picked_described[config.PERMIT_TYPE_COLUMN],
                    linked_permit_type=picked_described[config.LINKED_PERMIT_TYPE_COLUMN])
            heard_status = config.REQUEST_STATUS_AWAITING_REVIEW
            if status is not None and revisions_back == 0:
                heard_status = status
            expected = config.CHAT_OPERATOR_STATUS_TEMPLATE.format(
                permit_reference=picked_reference, permit_types=picked_types,
                plant_area=picked_described[config.PLANT_AREA_COLUMN],
                status=config.REQUEST_STATUS_PHRASES[heard_status])
            if heard_status == issued:
                expected = expected + " " + config.CHAT_ISSUED_WORK_TEMPLATE.format(
                    equipment_tag=picked_described[config.EQUIPMENT_TAG_COLUMN],
                    work_party_size=picked_described["work_party_size"])
            told.append((
                "a Control Room Operator asking about "
                + picked_reference
                + " with a planted review "
                + str(status)
                + (" of an earlier revision" if revisions_back else ""),
                answer_request_record(planted, staff["operator"], requests,
                                      config.CHAT_TYPE_REQUEST_STATUS, picked_reference),
                expected))
        unreviewed = unlinked[4][config.PERMIT_REFERENCE_COLUMN]
        told.extend([
            ("a Control Room Operator naming no request",
             answer_request_record(planted, staff["operator"], requests,
                                   config.CHAT_TYPE_REQUEST_STATUS, None),
             config.CHAT_REQUEST_NOT_NAMED_MESSAGE),
            ("a Control Room Operator naming PTW0099999, which is not an open request",
             answer_request_record(planted, staff["operator"], requests,
                                   config.CHAT_TYPE_REQUEST_STATUS, "PTW0099999"),
             config.CHAT_NO_REQUEST_TEMPLATE.format(permit_reference="PTW0099999")),
            ("a Control Room Operator asking how long " + unreviewed + " will take",
             answer_request_record(planted, staff["operator"], requests,
                                   config.CHAT_TYPE_OUTCOME_PREDICTION, unreviewed),
             config.OUTCOME_NOT_PREDICTED_LINE),
            ("a Permit Coordinator naming " + unreviewed + ", which nobody has reviewed",
             answer_request_record(planted, staff["coordinator"], requests,
                                   config.CHAT_TYPE_OUTCOME_PREDICTION, unreviewed),
             config.CHAT_NO_REVIEW_TEMPLATE.format(permit_reference=unreviewed)),
        ])
        refusal = None
        try:
            answer_request_record(planted, {"username": "visitor", "role": "Auditor"},
                                  requests, config.CHAT_TYPE_REQUEST_STATUS, reference)
        except ValueError as error:
            refusal = str(error)
        except Exception as error:
            refusal = type(error).__name__ + " instead of a ValueError"
    planted_engine.dispose()
    for meaning, answer, expected in told:
        assert answer == expected, (
            "answer_request_record told "
            + meaning
            + " "
            + repr(str(answer)[:300])
            + " where it answers exactly "
            + repr(expected)
            + ". An operator hears a request's permit types, plant area and status at its latest "
            "revision, and once its permit is issued the equipment tag and the size of the "
            "work party - never a prediction."
        )
    assert refusal is not None and "Auditor" in refusal, (
        "answer_request_record, asked by a role that is neither the control room nor the permit "
        "team, gave "
        + str(refusal)
        + ". Only those roles hear about a request from the records; any other is refused with "
        "a ValueError naming the role."
    )
    template = processed_frames[config.CLEAN_PERMIT_HISTORY_EVAL_FILE].head(1)
    planted_rows = []
    hot, entry = config.HOT_WORK, config.CONFINED_SPACE_ENTRY
    height, isolation = config.WORK_AT_HEIGHT, config.ELECTRICAL_ISOLATION
    for plant_area, permit_type, linked_type, numbers in [
            (config.BOILER_HOUSE, hot, config.NO_LINKED_PERMIT, [1, 2, 3, 4, 5, 1]),
            (config.GATEHOUSE, hot, config.NO_LINKED_PERMIT, [11, 12, 13, 14, 11, 12]),
            (config.BOILER_HOUSE, entry, config.NO_LINKED_PERMIT, [21, 22, 23, 24, 25, 26, 27]),
            (config.LABORATORY, height, config.NO_LINKED_PERMIT, [41, 42, 43, 44]),
            (config.LABORATORY, hot, height, [45, 46, 47]),
            (config.FIRE_STATION, isolation, config.NO_LINKED_PERMIT, [51, 52, 53, 54, 55])]:
        for number in numbers:
            row = template.copy()
            row[config.PLANT_AREA_COLUMN] = plant_area
            row[config.PERMIT_TYPE_COLUMN] = permit_type
            row[config.LINKED_PERMIT_TYPE_COLUMN] = linked_type
            row[config.PERMIT_REFERENCE_COLUMN] = "PTW0" + str(900000 + number)
            planted_rows.append(row)
    planted_history = pd.concat(planted_rows, ignore_index=True)
    figure = config.CHAT_PLANT_AREA_FIGURE_TEMPLATE
    figure_cases = [
        ("hot work in Boiler House, six rows of five permits", config.BOILER_HOUSE, hot,
         figure.format(permit_count=5, permit_type="hot work", plant_area=config.BOILER_HOUSE)),
        ("hot work in Gatehouse, six rows of four permits", config.GATEHOUSE, hot,
         config.GROUP_SIZE_LINE),
        ("confined space entry in Boiler House, seven permits", config.BOILER_HOUSE, entry,
         figure.format(permit_count=7, permit_type="confined space entry",
                       plant_area=config.BOILER_HOUSE)),
        ("work at height in Laboratory, four permits and three hot work permits linked to it",
         config.LABORATORY, height, config.GROUP_SIZE_LINE),
        ("hot work in Laboratory, three permits", config.LABORATORY, hot, config.GROUP_SIZE_LINE),
        ("electrical isolation in Fire Station, five permits", config.FIRE_STATION, isolation,
         figure.format(permit_count=5, permit_type="electrical isolation",
                       plant_area=config.FIRE_STATION)),
        ("excavation in Gatehouse, none", config.GATEHOUSE, config.EXCAVATION,
         config.GROUP_SIZE_LINE),
        ("no plant area", None, hot, config.CHAT_PLANT_AREA_NOT_NAMED_MESSAGE),
        ("no permit type", config.BOILER_HOUSE, None, config.CHAT_PLANT_AREA_NOT_NAMED_MESSAGE),
    ]
    for meaning, plant_area, permit_type, expected in figure_cases:
        answer = answer_plant_area_figure(planted_history, plant_area, permit_type)
        assert answer == expected, (
            "Asked about "
            + meaning
            + ", answer_plant_area_figure gave "
            + repr(str(answer)[:300])
            + " where it answers exactly "
            + repr(expected)
            + ". A plant area figure counts the different permits of that permit type worked in "
            "that plant area, never one only linked to the type, and is given only where at "
            "least "
            + str(config.GROUP_SIZE_MINIMUM)
            + " permits make it up, for everyone who asks: below that the contractors could be "
            "told apart, so the answer is the group-size line."
        )
