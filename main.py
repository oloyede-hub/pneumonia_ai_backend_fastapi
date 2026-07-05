from fastapi import FastAPI, File, UploadFile
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import shutil
import uuid
import os
from services.supabase_storage import check_connection

from services.predictor import predict_xray

load_dotenv()


os.makedirs("static/gradcam",exist_ok=True)
# Instantiating the fast api package
app = FastAPI()

# Until cors middleware is added no response will be delivered to the browser
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    # Rename the file and create a folder
    filename = f"{uuid.uuid4()}_{file.filename}"

    filepath = f"uploads/{filename}"

    os.makedirs("uploads", exist_ok=True)

    # Write the new uploaded file into the new file
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())


    result = predict_xray(filepath)

    return result


check_connection()


if __name__ == '__main__':
    # Run the server with uvicorn
    uvicorn.run("main:app", host="localhost", port=7000, reload=True)
