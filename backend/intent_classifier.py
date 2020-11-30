"""
Takes user input and classifies intent for further action.

"""
import torch
import numpy as np
import json
from nltk.tokenize import word_tokenize

# Local Imports.
from backend.model import NeuralNet
from backend.config import parameters_path


def process_user_input(user_input, vocab, word_to_idx):
    """
    Processes user input for intent classification.

    Parameters
    ----------
        user_input: str
        vocab: list
        word_to_idx: dict

    Returns
    ------
        bow_vec, unk_percent.
    """
    user_input_tokens = word_tokenize(user_input)
    user_input_tokens = [w.lower() for w in user_input_tokens]

    bow_vec = np.zeros(len(word_to_idx), dtype=np.float)
    unk_counter = 0

    for word in user_input_tokens:
        if word not in vocab:
            word = "unk"
            unk_counter += 1
            bow_vec[word_to_idx[word]] += 1
        else:
            bow_vec[word_to_idx[word]] += 1

    unk_percent = unk_counter / len(user_input_tokens)
    return bow_vec, unk_percent


def classify_user_input(user_input):
    """
    Classifies User Input.

    Parameters
    ---------
        user_input: str

    Returns
    -------
        intent_detected, confidence
    """

    parameters = torch.load(parameters_path)
    model_state = parameters["model_state"]
    input_size = parameters["input_size"]
    hidden_size = parameters["hidden_size"]
    output_size = parameters["output_size"]
    vocab = parameters["vocab"]
    word_to_idx = parameters["word_to_idx"]
    intents = parameters["intents"]

    model = NeuralNet(input_size=input_size,
                      hidden_size=hidden_size,
                      output_size=output_size)

    model.load_state_dict(model_state)
    model.eval()

    bow_vec, unk_percent = process_user_input(user_input=user_input,
                                              vocab=vocab,
                                              word_to_idx=word_to_idx)

    bow_vec = bow_vec.reshape(1, bow_vec.shape[0])
    bow_vec = torch.from_numpy(bow_vec)
    out = model(bow_vec.float())

    # intent.
    _, predicted_idx = torch.max(out, dim=1)
    intent_detected = intents[predicted_idx.item()]

    # confidence.
    probabilities = torch.softmax(out, dim=1)
    confidence = probabilities[0][predicted_idx.item()].item()
    confidence = round(confidence*100, 2)
    return intent_detected, confidence







