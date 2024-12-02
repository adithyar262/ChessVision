from keras.api.models import load_model
import tf2onnx
import onnx
import tensorflow as tf
import keras

model = load_model("../../LiveChess2FEN_IMPL/lc2fen/detectboard/models/laps_model.h5")
model.summary()

# # Rebuild the model with a dynamic batch size
# input_shape = (None, 21, 21, 1)
# new_input = keras.api.Input(shape=(21, 21, 1), batch_size=None)
# new_output = model(new_input)
# updated_model = keras.api.Model(inputs=new_input, outputs=new_output)
#
# # Save the updated model
# updated_model.save("../../LiveChess2FEN_IMPL/lc2fen/detectboard/models/updated_laps_model.h5")

# model = load_model("models/EfficientNetB7.keras")
# input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='digit')]
#
# onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, output_path="models/EfficientNetB7.onnx")