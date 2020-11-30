from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()
templates = Jinja2Templates(directory="frontend/templates")

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
def homepage(request: Request):
    return templates.TemplateResponse("homepage.html", context={
        "request": request
    })
