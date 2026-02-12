from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import mlflow
from pathlib import Path

from chatbot.model_loader import ModelManager
from chatbot.feedback_store import save_feedback
from chatbot.models.routes import router as model_router


# =====================================================
# PROJECT PATH CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent          # chatbot/
PROJECT_ROOT = BASE_DIR.parent                     # gpu_train/
MLRUNS_PATH = PROJECT_ROOT / "mlruns"

MLRUNS_PATH.mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000")


# =====================================================
# FASTAPI APP INIT
# =====================================================

app = FastAPI()

# Register router (export endpoint)
app.include_router(model_router)

# Static + Templates (relative to chatbot folder)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

model_manager = ModelManager()


# =====================================================
# REQUEST MODELS
# =====================================================

class ChatRequest(BaseModel):
    model_name: str
    version: int
    prompt: str


class FeedbackRequest(BaseModel):
    model_name: str
    version: int
    prompt: str
    response: str
    rating: int
    comment: str
    latency: float
    input_tokens: int
    output_tokens: int


# =====================================================
# ROUTES
# =====================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
def chat(data: ChatRequest):
    model_manager.load_version(data.model_name, data.version)
    result = model_manager.generate(data.prompt)
    return result


@app.post("/feedback")
def feedback(data: FeedbackRequest):

    with mlflow.start_run(run_name="manual_evaluation"):

        mlflow.log_param("model_name", data.model_name)
        mlflow.log_param("version", data.version)

        mlflow.log_metric("rating", data.rating)
        mlflow.log_metric("latency", data.latency)
        mlflow.log_metric("input_tokens", data.input_tokens)
        mlflow.log_metric("output_tokens", data.output_tokens)

        mlflow.log_text(data.prompt, "prompt.txt")
        mlflow.log_text(data.response, "response.txt")

    save_feedback(data.dict())

    return {"status": "saved"}


@app.get("/versions/{model_name}")
def get_versions(model_name: str):
    versions = model_manager.list_versions(model_name)
    return {"versions": versions}


@app.get("/models")
def list_models():
    models = []
    for item in model_manager.MLRUNS_PATH.iterdir():
        if item.is_dir():
            models.append(item.name)
    return {"models": sorted(models)}
