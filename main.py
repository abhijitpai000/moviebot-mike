from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# NLU.
from backend.intent_classifier import classify_user_input
from backend.agent import response_handler


app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
def homepage(request: Request):
    return templates.TemplateResponse("homepage.html", context={
        "request": request
    })


class Chat(BaseModel):
    """
    Validating the POST sent from client.
    """
    user_input: str


@app.post("/chat")
def item_rec(chat: Chat):
    """
    Generates similar products using input in the 'Find Similar Products' section.
    """
    user_input = chat.user_input
    user_input_tokens, intent_detected, confidence = classify_user_input(user_input=user_input)
    response = response_handler(intent_detected=intent_detected, user_input_tokens=user_input_tokens)
    return {
        "bot_response": response
    }
