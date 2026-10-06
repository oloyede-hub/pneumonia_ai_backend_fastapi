"""Pneumonia-logit Grad-CAM without filesystem output."""

import cv2
import numpy as np
import tensorflow as tf


class GradCAM:
    def __init__(self, model, backbone_name, last_conv_name):
        backbone = model.get_layer(backbone_name)
        target = backbone.get_layer(last_conv_name)
        # backbone.input belongs to its internal graph. The inbound node holds
        # the outer-model tensor, including saved preprocessing operations.
        # Saved backbones may retain nodes from earlier training graphs. Only
        # consider calls that participate in this classifier's active graph.
        graph_model = getattr(model, "_functional", None) or model
        graph_nodes = {
            node for nodes in graph_model._nodes_by_depth.values() for node in nodes
        }
        backbone_nodes = [node for node in backbone._inbound_nodes if node in graph_nodes]
        if len(backbone_nodes) != 1:
            raise ValueError("Expected the backbone to be called exactly once.")
        backbone_inputs = backbone_nodes[0].input_tensors
        if len(backbone_inputs) != 1:
            raise ValueError("Expected a single backbone input.")
        self.prefix = tf.keras.Model(model.inputs[0], backbone_inputs[0])
        self.features = tf.keras.Model(backbone.inputs[0], target.output)
        # Preserve graph connections rather than replaying residual layers.
        self.tail = (
            tf.keras.layers.Identity()
            if target.output is backbone.output
            else tf.keras.Model(target.output, backbone.output)
        )
        layers = [
            layer
            for layer in model.layers
            if not isinstance(layer, tf.keras.layers.InputLayer)
        ]
        backbone_index = layers.index(backbone)
        self.output_layer = layers[-1]
        if (
            not isinstance(self.output_layer, tf.keras.layers.Dense)
            or self.output_layer.units != 1
            or self.output_layer.get_config()["activation"] != "sigmoid"
        ):
            raise ValueError("Grad-CAM requires a single Dense sigmoid output.")
        self.head = (
            tf.keras.Sequential(layers[backbone_index + 1:-1])
            if len(layers) > backbone_index + 2
            else tf.keras.layers.Identity()
        )

    def __call__(self, img_array):
        inputs = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            prepared = self.prefix(inputs, training=False)
            features = self.features(prepared, training=False)
            tape.watch(features)
            hidden = self.head(self.tail(features, training=False), training=False)
            # Use the actual final-layer weights without mutating the loaded model.
            logits = tf.linalg.matmul(hidden, self.output_layer.kernel)
            if self.output_layer.use_bias:
                logits = tf.nn.bias_add(logits, self.output_layer.bias)
            predictions = self.output_layer.activation(logits)
            probability = predictions[:, 0]
            # Explain pneumonia evidence before sigmoid saturation destroys gradients.
            score = logits[:, 0]
        gradients = tape.gradient(score, features)
        if gradients is None:
            raise RuntimeError("Grad-CAM is disconnected from the classifier.")
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heatmap = tf.nn.relu(tf.reduce_sum(features * weights, axis=-1))[0]
        heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
        tf.debugging.assert_all_finite(predictions, "Non-finite prediction")
        tf.debugging.assert_all_finite(heatmap, "Non-finite Grad-CAM")
        return float(probability[0]), heatmap.numpy()


def render_gradcam(original_image, heatmap, alpha=0.4):
    """Resize scalar activation before coloring; zero activation stays unchanged."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    height, width = original_image.shape[:2]
    activation = cv2.resize(
        np.asarray(heatmap, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    activation = np.clip(np.nan_to_num(activation), 0, 1)
    colors = cv2.applyColorMap(np.uint8(255 * activation), cv2.COLORMAP_JET)
    opacity = (alpha * activation)[..., None]
    return np.rint(original_image * (1 - opacity) + colors * opacity).astype(np.uint8)
