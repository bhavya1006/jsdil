"""A person's own last exchanges, as history to read and words to search with."""
from config import config
from persistence.models import ChatTurn
from persistence.repository import fetch_chat_turns


def recall_conversation(session, username, question_text):
    """The asker's last few turns, oldest first; answers from the records never reach a prompt."""
    turns = list(reversed(fetch_chat_turns(session, ChatTurn.username == username,
                                           config.CHAT_HISTORY_TURN_LIMIT)))
    if turns:
        exchanges = []
        for turn in turns:
            answer = (config.CHAT_RECORD_ANSWER_IN_HISTORY
                      if turn["question_type"] in config.CHAT_RECORD_TYPES
                      else turn["answer_text"])
            exchanges.append(f"Question: {turn['question_text']}\nAnswer: {answer}")
        history_text = "\n\n".join(exchanges)
    else:
        history_text = config.CHAT_FIRST_QUESTION_TEXT
    previous_question = turns[-1]["question_text"] if turns else ""
    search_text = (previous_question + " " + question_text).strip()
    return {"history_text": history_text, "search_text": search_text,
            "previous_question": previous_question}
