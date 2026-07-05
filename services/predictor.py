import os
import uuid

import cv2
import numpy as np
import tensorflow as tf

from utils.storage_utils import upload_file, upload_image


from .gradcam import (make_gradcam_heatmap, save_gradcam)

model = tf.keras.models.load_model("model/penumonia1.keras")

preprocess_input = tf.keras.applications.vgg16.preprocess_input



def analyze_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.6,
    top_k: int = 5
):
    # Pixels above threshold
    mask = heatmap >= threshold

    activated_pixels = int(np.sum(mask))
    total_pixels = heatmap.size

    activated_percentage = (
        activated_pixels / total_pixels
    ) * 100

    # Get indices of highest activations
    flat_indices = np.argsort(
        heatmap.ravel()
    )[::-1][:top_k]

    top_activations = []

    for idx in flat_indices:
        y, x = np.unravel_index(
            idx,
            heatmap.shape
        )

        top_activations.append({
            "x": int(x),
            "y": int(y),
            "activation": float(
                heatmap[y, x]
            )
        })

    return {
        "threshold": threshold,
        "activatedPercentage": round(
            activated_percentage,
            2
        ),
        "topActivations": top_activations
    }


def build_gradcam_response(
    heatmap: np.ndarray,
):
    heatmap_224 = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    analysis = analyze_heatmap(
        heatmap
    )


    return {
        "surface": {
            "width": heatmap_224.shape[1],
            "height": heatmap_224.shape[0],
            "values": heatmap_224.tolist()
        },
        "raw":{
            "width": heatmap.shape[1],
            "height": heatmap.shape[0],
            "values": heatmap.tolist(),
        },
        "analysis": analysis,
        "statistics": {
                "maxActivation": float(np.max(heatmap)),
                "minActivation": float(np.min(heatmap)),
                "meanActivation": float(np.mean(heatmap)),
                "stdActivation": float(np.std(heatmap))
            },
    }

def compute_heatmap_statistics(
    heatmap: np.ndarray
):
    return {
        "maxActivation": float(np.max(heatmap)),
        "meanActivation": float(np.mean(heatmap)),
        "activatedPixels": int(np.sum(heatmap > 0.5)),
        "activatedPercentage":
            float(np.mean(heatmap > 0.5) * 100)
    }

def predict_xray(filepath):
    # Image preprocessing
    original = cv2.imread(filepath)
    

    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb, (224, 224))

    img_array = np.expand_dims(resized, axis=0)

    img_array = preprocess_input(img_array.astype(np.float32))

    # Pass the image to the model
    prediction = float(model.predict(img_array, verbose=0)[0][0])


    # Predictiion to return a norma
    if prediction >= 0.5:
        label = "Pneumonia"
        confidence = float(prediction) * 100
    else:
        label = "Normal"
        confidence = (1 - float(prediction)) * 100

    # Make a heat map
    heatmap = make_gradcam_heatmap(img_array, model)

    gradcam_base = save_gradcam(original, heatmap)

    images = upload_image(filepath, gradcam_base)

    os.remove(filepath)
    os.remove(gradcam_base)
    gradcam = build_gradcam_response(
        heatmap
    )

    return  {
    "prediction": {
        "label": label,
        "confidence": confidence,
        "probability": {
            "Normal": round(1 - prediction, 4),
            "Pneumonia": round(prediction, 4),
        }
    },
    "images": images,
    "gradcam": gradcam
    }

