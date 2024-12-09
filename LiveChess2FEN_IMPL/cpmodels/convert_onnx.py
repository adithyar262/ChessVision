from keras.api.models import load_model
import tf2onnx
import onnx
import tensorflow as tf
import keras

model = load_model("models/ResNet152.keras")
input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input_layer')]

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, output_path="models/ResNet152V2_3.onnx")
