from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from prediction import predict, load, preprocess
import uvicorn
import shutil
import uuid
import zipfile
import os
from pathlib import Path
import subprocess

app = FastAPI()

# Load model once at startup
model, inv_map = load()


# GLOBAL TEMP DIRECTORY FOR UPLOADS

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



# 1. SINGLE IMAGE PREDICTION

@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    temp_name = f"{uuid.uuid4()}_{file.filename}"
    tmp_path = os.path.join(UPLOAD_DIR, temp_name)

    # Save file
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # Predict
    result = predict(tmp_path)

    # Cleanup
    os.remove(tmp_path)

    return JSONResponse(result)

# 2. BATCH PREDICTION (ZIP OF IMAGES)

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    temp_zip_name = f"{uuid.uuid4()}.zip"
    tmp_zip_path = os.path.join(UPLOAD_DIR, temp_zip_name)

    # Save zip file
    with open(tmp_zip_path, "wb") as f:
        f.write(await file.read())

    # Extract zip
    extract_dir = os.path.join(UPLOAD_DIR, temp_zip_name + "_unzipped")
    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    results = {}
    for p in Path(extract_dir).rglob("*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            results[p.name] = predict(str(p))

    # Cleanup
    shutil.rmtree(extract_dir)
    os.remove(tmp_zip_path)

    return results



# 3. RETRAINING ENDPOINT

@app.post("/retrain")
async def retrain(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    temp_zip_name = f"{uuid.uuid4()}.zip"
    tmp_zip_path = os.path.join(UPLOAD_DIR, temp_zip_name)

    with open(tmp_zip_path, "wb") as f:
        f.write(await file.read())

    if background_tasks:
        background_tasks.add_task(run_retrain, tmp_zip_path)
        return {"status": "retrain_started"}
    else:
        run_retrain(tmp_zip_path)
        return {"status": "retrain_complete"}


def run_retrain(zip_path):
    extract_dir = zip_path + "_unzipped"

    # Extract retraining dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # CALL TRAINING SCRIPT 
    try:
        subprocess.run([
            "python", "-u", "src/train_cli.py",
            "--train_dir", extract_dir
        ], check=True)
    finally:
        # Cleanup
        shutil.rmtree(extract_dir)
        os.remove(zip_path)



# 4. MODEL UPTIME API

@app.get("/model/uptime")
def uptime():
    return {
        "model": "plantdisease_cnn",
        "version": "v1",
        "loaded": True
    }


# START SERVER
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
