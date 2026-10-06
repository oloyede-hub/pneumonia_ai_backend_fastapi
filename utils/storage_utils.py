import logging
import os
import uuid

from services.supabase_storage import get_supabase

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    pass


def upload_image(original_bytes: bytes, overlay_bytes: bytes):
    image_id = uuid.uuid4().hex
    paths = [f"originals/{image_id}.png", f"gradcam/{image_id}.png"]
    uploaded = []
    try:
        bucket = get_supabase().storage.from_(os.getenv("SUPABASE_BUCKET", "uploads"))
        for path, data in zip(paths, (original_bytes, overlay_bytes)):
            bucket.upload(
                path=path,
                file=data,
                file_options={"content-type": "image/png", "upsert": "false"},
            )
            uploaded.append(path)
        return {
            "original_url": bucket.get_public_url(paths[0]),
            "gradcam_url": bucket.get_public_url(paths[1]),
        }
    except Exception as exc:
        if uploaded:
            try:
                bucket.remove(uploaded)
            except Exception:
                logger.exception("Failed to clean up incomplete image upload")
        raise StorageError("Image storage is temporarily unavailable.") from exc
