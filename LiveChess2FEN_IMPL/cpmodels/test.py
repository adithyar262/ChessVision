import tensorflow as tf
import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

print("GPU Devices:", tf.config.list_physical_devices('GPU'))