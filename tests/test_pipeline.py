import unittest
import asyncio
from io import BytesIO
from unittest.mock import patch, Mock

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi.testclient import TestClient

import main
from services.gradcam import GradCAM, render_gradcam
from services.predictor import decode_image, InvalidImage, predict_xray
from threading import Lock
from utils.storage_utils import upload_image, StorageError
from utils.load_model import load_model_config, MODELS


class PipelineTests(unittest.TestCase):
    def test_startup_architecture_selection(self):
        for configured, expected in ((None, "resnet50"), ("resnet50", "resnet50"),
                                     (" VGG16 ", "vgg16")):
            environment = {} if configured is None else {"MODEL_NAME": configured}
            config = {"backbone": expected}
            with patch.dict("os.environ", environment, clear=True), patch(
                    "main.get_supabase"), patch("main.load_model_config", return_value=config) as load:
                async def startup():
                    async with main.lifespan(main.app):
                        self.assertIs(main.app.state.model_config, config)
                asyncio.run(startup())
                load.assert_called_once_with(expected)

    def test_embedded_preprocessing_and_residual_connections(self):
        inputs = tf.keras.Input((4, 4, 3))
        branch = tf.keras.layers.Conv2D(3, 1, name="conv")(inputs)
        residual = tf.keras.layers.Add()([inputs, branch])
        features = tf.keras.layers.Activation("relu", name="conv5_block3_out")(residual)
        backbone = tf.keras.Model(inputs, features, name="resnet50")
        # A saved backbone can retain calls from an earlier training graph.
        backbone(tf.keras.Input((4, 4, 3)))
        raw = tf.keras.Input((4, 4, 3))
        prepared = tf.keras.applications.resnet.preprocess_input(raw)
        pooled = tf.keras.layers.GlobalAveragePooling2D()(backbone(prepared))
        output = tf.keras.layers.Dense(1, activation="sigmoid")(pooled)
        model = tf.keras.Model(raw, output)
        backbone.get_layer("conv").set_weights([
            np.zeros((1, 1, 3, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)
        ])
        model.layers[-1].set_weights([
            np.full((3, 1), 0.001, dtype=np.float32), np.zeros(1, dtype=np.float32)
        ])
        cam = GradCAM(model, "resnet50", "conv5_block3_out")
        sample = np.full((1, 4, 4, 3), [200, 180, 160], dtype=np.float32)
        probability, heatmap = cam(sample)
        self.assertAlmostEqual(probability, float(model(sample)[0, 0]), places=6)
        np.testing.assert_allclose(cam.prefix(sample),
                                   tf.keras.applications.resnet.preprocess_input(sample.copy()))
        np.testing.assert_allclose(heatmap, 1)
        np.testing.assert_array_equal(MODELS["resnet50"]["preprocess"](sample), sample)

    def test_invalid_model_selection(self):
        with patch("utils.load_model.hf_hub_download") as download:
            with self.assertRaisesRegex(ValueError, "Unsupported MODEL_NAME"):
                load_model_config("unknown")
            download.assert_not_called()

    def test_binary_target_and_inference(self):
        inputs = tf.keras.Input((4, 4, 1))
        conv = tf.keras.layers.Conv2D(1, 1, use_bias=False, name="conv")(inputs)
        backbone = tf.keras.Model(inputs, conv, name="backbone")
        model = tf.keras.Sequential([backbone, tf.keras.layers.GlobalAveragePooling2D(),
                                     tf.keras.layers.Dense(1, activation="sigmoid")])
        model(np.ones((1, 4, 4, 1)))
        backbone.get_layer("conv").set_weights([np.ones((1, 1, 1, 1))])
        cam = GradCAM(model, "backbone", "conv")
        for weight in (1.0, -1.0, 100.0):
            model.layers[-1].set_weights([np.array([[weight]]), np.zeros(1)])
            sample = np.ones((1, 4, 4, 1), dtype=np.float32)
            probability, heatmap = cam(sample)
            self.assertAlmostEqual(probability, float(model(sample)[0, 0]), places=6)
            np.testing.assert_allclose(heatmap, 1 if weight > 0 else 0)
        model.layers[-1].set_weights([np.zeros((1, 1)), np.zeros(1)])
        _, heatmap = cam(sample)
        np.testing.assert_array_equal(heatmap, 0)

    def test_overlay_status(self):
        data = BytesIO()
        Image.new("RGB", (12, 8), (100, 100, 100)).save(data, format="PNG")
        for probability, active, status in [(0.1, True, "not_applicable"),
                                             (1.0, True, "available"),
                                             (0.9, False, "no_positive_activation")]:
            config = {"preprocess": lambda x: x, "inference_lock": Lock(),
                      "gradcam": lambda x: (probability, np.full((2, 2), float(active)))}
            with patch("services.predictor.upload_image", return_value={}) as upload:
                result = predict_xray(data.getvalue(), config)
                self.assertEqual(result["gradcam"]["status"], status)
                self.assertEqual(result["gradcam"]["available"], status == "available")
                original, overlay = upload.call_args.args
                if status != "available":
                    self.assertEqual(original, overlay)
                else:
                    self.assertNotEqual(original, overlay)

    def test_overlay_geometry_and_zero_activation(self):
        original = np.full((40, 80, 3), 100, np.uint8)
        np.testing.assert_array_equal(render_gradcam(original, np.zeros((2, 4))), original)
        heatmap = np.zeros((2, 4), np.float32)
        heatmap[0, 0] = 1
        overlay = render_gradcam(original, heatmap)
        self.assertTrue(np.any(overlay[0, 0] != original[0, 0]))
        np.testing.assert_array_equal(overlay[-1, -1], original[-1, -1])

    def test_decode(self):
        data = BytesIO()
        Image.new("RGB", (12, 8), (200, 100, 20)).save(data, format="PNG")
        decoded = decode_image(data.getvalue())
        self.assertEqual(decoded.shape, (8, 12, 3))
        np.testing.assert_array_equal(decoded[0, 0], [200, 100, 20])
        with self.assertRaises(InvalidImage):
            decode_image(b"not an image")

    def test_storage_bytes_and_rollback(self):
        bucket = Mock()
        client = Mock()
        client.storage.from_.return_value = bucket
        with patch("utils.storage_utils.get_supabase", return_value=client):
            result = upload_image(b"original", b"overlay")
            self.assertEqual(set(result), {"original_url", "gradcam_url"})
            self.assertEqual(bucket.upload.call_args_list[0].kwargs["file"], b"original")
            bucket.upload.side_effect = [None, RuntimeError("failed")]
            with self.assertRaises(StorageError):
                upload_image(b"original", b"overlay")
            bucket.remove.assert_called_once()

    def test_request_limits_and_no_disk_spooling(self):
        client = TestClient(main.app)
        main.app.state.model_config = {}
        with patch("main.predict_xray", return_value={"ok": True}), patch(
                "tempfile.SpooledTemporaryFile.rollover", side_effect=AssertionError("disk write")):
            response = client.post("/predict", files={"file": ("x.png", b"x" * (2 * 1024 * 1024))})
            self.assertEqual(response.status_code, 200)
            response = client.post("/predict", files={"file": ("x.png", b"x" * main.MAX_REQUEST_BYTES)})
            self.assertEqual(response.status_code, 413)
        self.assertEqual(client.post("/predict", files={"file": ("x.png", b"")}).status_code, 400)
        self.assertEqual(client.post("/predict", files={"other": ("x.png", b"x")}).status_code, 422)
        self.assertEqual(client.post("/predict", content=b"x").status_code, 415)


if __name__ == "__main__":
    unittest.main()
