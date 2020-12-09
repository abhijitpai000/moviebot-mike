import json
import random


def bot_response(intent_detected):
    """
    Responds to user input based on intent detected.

    Parameters
    ----------
        intent_detected: str.

    Returns
    -------
        response (str)
    """
    with open("backend/database/data.json", "r") as f:
        data = json.load(f)

    for i in data["intents"]:
        if intent_detected == i["intent"]:
            response = random.choice(i["response"])
            return response
