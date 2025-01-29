from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from spacy import load

app = FastAPI()

# Load SpaCy model
nlp = load("en_core_web_sm")

# Enable CORS for frontend integration
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve HTML templates
templates = Jinja2Templates(directory="templates")

@app.post("/ner/")
async def named_entity_recognition(text: str = Form(...)):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return {"entities": entities}

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("templates/index.html", "r") as file:
        return file.read()
