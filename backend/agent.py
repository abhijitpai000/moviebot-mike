"""
Handles Bot Response based on user intent detected.

"""





# Local Imports.
from backend.conversation import bot_response
from backend.watch_next import get_recommendations, get_movie_plot


def response_handler(user_input_tokens, intent_detected, confidence, unk_percent):
    """
    Handle Bot Responses based on Intent Detected.

    Parameters
    ----------
        intent_detected: str.
        user_input_tokens: list.
        confidence: int.
        unk_percent: int.

    Returns
    -------
        response
    """
    if (confidence < 50) or (unk_percent == 100):
        response = "i'm having trouble processing the request, i'm still learning new words. try 'movies like movie-title' for movie recommendations"
    else:
        if intent_detected == "movie_plot":
            response = get_movie_plot(user_input_tokens)
        elif intent_detected == "watch_next":
            response = get_recommendations(user_input_tokens)
        else:
            response = bot_response(intent_detected)

    bot_return_message = {"response": response,
                          "intent_detected": intent_detected,
                          "confidence": confidence,
                          "unk_percent": unk_percent}
    return bot_return_message
