from contextlib import asynccontextmanager
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
import uvicorn

from services.predictor import InvalidImage, predict_xray
from services.supabase_storage import get_supabase
from utils.load_model import load_model_config
from utils.storage_utils import StorageError

load_dotenv()
logger = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 10 * 1024 * 1024


class MemoryMultipartParser(MultiPartParser):
    # Support Starlette versions before and after the attribute rename.
    spool_max_size = MAX_REQUEST_BYTES + 1
    max_file_size = MAX_REQUEST_BYTES + 1


@asynccontextmanager
async def lifespan(app):
    get_supabase()
    app.state.model_config = await run_in_threadpool(load_model_config, "vgg16")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post(
    "/predict",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                }
            },
        }
    },
)
async def prediction(request: Request):
    if (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        != "multipart/form-data"
    ):
        raise HTTPException(415, "Use multipart/form-data with a file field.")
    too_large = False

    async def limited_stream():
        nonlocal too_large
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_REQUEST_BYTES:
                too_large = True
                raise MultiPartException("Upload exceeds the 10 MiB request limit.")
            yield chunk

    parser = MemoryMultipartParser(
        request.headers, limited_stream(), max_files=1, max_fields=0
    )
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        raise HTTPException(413 if too_large else 400, str(exc)) from exc
    try:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(422, "A file field is required.")
        data = await file.read()
    finally:
        await form.close()
    if not data:
        raise HTTPException(400, "Image is empty.")
    try:
        return await run_in_threadpool(
            predict_xray, data, request.app.state.model_config
        )
    except InvalidImage as exc:
        raise HTTPException(400, str(exc)) from exc
    except StorageError as exc:
        logger.exception("Image upload failed")
        raise HTTPException(502, str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=7000, reload=True)
