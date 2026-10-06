# Pneumonia Detection Backend

FastAPI backend for binary chest X-ray classification with a pneumonia-evidence Grad-CAM overlay and Supabase image storage.

## Run

Use Python with the packages in requirements.txt:

```sh
pip install -r requirements.txt
uvicorn main:app --host localhost --port 7000
```

Configure .env with SUPABASE_URL and SUPABASE_SECRET_KEY. SUPABASE_BUCKET is optional and defaults to uploads. The existing public-URL response requires a bucket configured for public reads. Credentials stay on the server.

ResNet50 is selected by default. Set MODEL_NAME=resnet50 explicitly in .env, or MODEL_NAME=vgg16 to select VGG16. ResNet50 downloads resnet50_shortcut_aware_final.keras from Buore/RESNET50-CHEST-CLASSIFIER; VGG16 downloads near_final__vgg16_pneumonia.keras from Buore/VGG16-CHEST-CLASSIFIER. Hugging Face caches the downloaded model.

Switch architectures by changing MODEL_NAME in .env and restarting the backend. The model is selected once at startup; changing the file while the server is running does not reload the model.

Image dimensions come from the saved model input. The published ResNet50 model already embeds RGB-to-BGR conversion and ImageNet channel mean subtraction, so the backend supplies raw RGB float32 pixels in the 0–255 range. Do not additionally call preprocess_input or divide by 255. VGG16 uses external Keras VGG16 preprocessing. These settings are specific to the configured model files and must be rechecked when replacing weights. ResNet50 Grad-CAM targets conv5_block3_out and preserves both the embedded preprocessing and residual graph connections.

GET /ping returns {"ping":"pong"}. POST /predict accepts a multipart file field. Interactive API documentation is at /docs.

## Image handling

Requests, decoded images, and overlays stay in memory. Neither uploads/ nor static/gradcam/ is created or written, and multipart files below the request limit do not spill to disk. Existing files in those directories are left untouched.

The limit is 10 MiB for the whole multipart request (including headers/boundaries), and 20 megapixels for decoded images. JPEG, PNG and WebP are accepted. Images are oriented using EXIF before both inference and rendering, then uploaded directly as lossless PNG bytes with matching MIME types. The original upload is normalized to RGB and metadata is discarded. Supabase still stores both images remotely. A failed second upload triggers best-effort removal of the first.

The response preserves the existing frontend contract:

```json
{
  "prediction": {
    "label": "Pneumonia",
    "confidence": 90.0,
    "probability": {"Normal": 0.1, "Pneumonia": 0.9}
  },
  "images": {"original_url": "...", "gradcam_url": "..."}
}
```

Invalid images return 400, oversized requests 413, unsupported request types 415, missing file fields 422, and storage failures 502.

## Grad-CAM and performance

The explanation targets the pneumonia score before sigmoid (the logit), avoiding zero gradients when a confident probability rounds to 1. The original model and its prediction probabilities are unchanged. Normal predictions return the uncolored image in gradcam_url for compatibility; they do not display a pneumonia overlay. The gradcam metadata reports target, available, status (available, not_applicable, or no_positive_activation), and lung_localization_validated=false. A positive prediction without positive spatial activation also returns the uncolored image and explicitly reports the unavailable explanation. Inference uses training=False, with one shared forward pass for prediction and gradients. The feature and classifier graphs are built once on startup, and a consistency check compares their prediction with the loaded model. A lock serializes model inference within each worker.

Scalar heatmaps are resized to the same oriented image coordinates before applying the colormap. Opacity follows activation, so zero-activation pixels keep their original appearance. All-zero maps remain finite.

Grad-CAM shows model evidence; it is not lung segmentation. Persistent armpit activation can indicate learned shortcuts, a preprocessing mismatch with training, or coarse localization. This implementation does not shift or mask activations to simulate lung localization. Confirm the training preprocessing and class mapping (sigmoid output is assumed to mean Pneumonia), and evaluate representative X-rays with lung annotations before claiming localization accuracy. No training data, lung segmentation weights, or clinical localization benchmark is included here.

The supported architecture has a single call to a nested backbone, optional input preprocessing, and a sequential classifier head ending in a single Dense sigmoid output. The configured VGG16 and ResNet50 target layers must have a connected graph tail. Changing model architecture requires rechecking these assumptions. Startup compares Grad-CAM predictions against direct model inference on black and deterministic random images.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests cover pneumonia targeting, sigmoid saturation, normal-overlay suppression, prediction consistency, zero-gradient maps, overlay geometry, image validation, multipart memory behavior and limits, and remote-upload rollback. Storage tests use mocks and do not send images remotely.

Grad-CAM reference: https://keras.io/examples/vision/grad_cam/
Supabase upload reference: https://supabase.com/docs/reference/python/storage-from-upload
