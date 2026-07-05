import os
import uuid
from pathlib import Path
from services.supabase_storage import supabase


BUCKET_NAME = "uploads"



def get_public_url(storage_path: str):
    return (
        supabase
        .storage
        .from_(BUCKET_NAME)
        .get_public_url(storage_path)
    )



def upload_file(local_file_path: str, destination_path: str = ""):

    with open(local_file_path, "rb") as file:
        supabase.storage.from_(BUCKET_NAME).upload(
            path=destination_path,
            file=file,
            file_options={
                "upsert": "false"
            }
        )

    return supabase.storage.from_(BUCKET_NAME).get_public_url(destination_path)


def upload_image(filepath, gradcam_base):
    # Generate a unique ID for the image
    image_id = str(uuid.uuid4())

    # Define storage paths
    original_storage_path = f"originals/{image_id}.jpg"
    overlay_storage_path = f"gradcam/{image_id}.png"

    # Upload the original image
    original_url = upload_file(filepath, original_storage_path)

    # Upload the gradcam image
    gradcam_url = upload_file(gradcam_base, overlay_storage_path)

    return {
        "original_url": original_url,
        "gradcam_url": gradcam_url
    }
