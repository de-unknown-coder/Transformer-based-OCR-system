import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.inference import run_inference

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/web", StaticFiles(directory="web"), name="web")


@app.get("/")
def root():
    return {"status": "OCR API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{file.filename}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = run_inference(file_path)

        os.remove(file_path)

        return {"text": text}

    except Exception as e:

        if os.path.exists(file_path):
            os.remove(file_path)

        return {"error": str(e)}