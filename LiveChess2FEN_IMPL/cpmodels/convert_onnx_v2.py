import tf2onnx
import tensorflow as tf
from keras.models import load_model

# Load the updated Keras model
model = load_model("../data/models/ResNet152V2.keras")

# Create a concrete function with explicit batch dimensions
input_spec = tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input_layer')

# Convert the model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[input_spec],
    output_path="../data/models/ResNet152V2_103.onnx"
)

print("ONNX model exported successfully with explicit batch dimensions!")

