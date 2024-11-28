import math
from typing import List

import tensorflow as tf
from keras.api.preprocessing.image import ImageDataGenerator
from keras.api.preprocessing.image import Iterator
from keras.api.utils import Sequence
import numpy as np

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

import os
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.api.models import Model
from keras.api.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    concatenate,
    GlobalMaxPooling2D,
)
from keras.api.optimizers import Adam

def build_model(base_model: Model) -> Model:
    """
    Build the model from a pretrained base model.

    :param base_model: Base model from keras applications.

    :return: The model to train.
    """
    layers = base_model.output
    avg_pool = GlobalAveragePooling2D()(layers)
    max_pool = GlobalMaxPooling2D()(layers)
    layers = concatenate([avg_pool, max_pool])

    layers = BatchNormalization()(layers)
    layers = Dense(1024, activation="relu")(layers)
    layers = BatchNormalization()(layers)
    layers = Dropout(0.5)(layers)
    layers = Dense(512, activation="relu")(layers)
    layers = BatchNormalization()(layers)
    layers = Dropout(0.5)(layers)
    preds = Dense(13, activation="softmax")(layers)

    model = Model(inputs=base_model.input, outputs=preds)

    return model


def data_generators(
        preprocessing_func,
        target_size: tuple[int, int],
        batch_size: int,
        train_path: str = "../data/dataset/train/",
        validation_path: str = "../data/dataset/validation/",
):
    """
    Return the train and validation generators.

    :param preprocessing_func: Preprocessing function for base model.
    :param target_size: Dimensions to which all images will be resized.
    :param batch_size: Size of the batches of data.
    :param train_path: Path to the train folder.
    :param validation_path: Path to the validation folder.

    :return: Train and validation generators.
    """

    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        rotation_range=20,  # Randomly rotate images by 0 to 20 degrees
        width_shift_range=0.2,  # Shift images horizontally by 0 to 20% of width
        height_shift_range=0.2,  # Shift images vertically by 0 to 20% of height
        shear_range=0.15,  # Shear transformation
        zoom_range=0.15,  # Zoom in or out by 0 to 15%
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode="nearest",  # Fill pixels after rotation or shift
        dtype="float32"
    )

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    # No augmentation for validation data
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        dtype="float32"
    )

    val_gen = val_datagen.flow_from_directory(
        validation_path,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen


def train_model(
        model,
        epochs,
        train_generator,
        validation_generator,
        callbacks,
        use_weights=False,
        class_weights=None,
        workers=5,
):
    """Train the model."""
    steps_per_epoch = math.ceil(train_generator.samples / train_generator.batch_size)
    validation_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        use_multiprocessing=False,
        workers=workers,
        class_weight=class_weights if use_weights else None,
    )
    return history


def model_callbacks(
    early_stopping_patience: int,
    model_checkpoint_path: str,
    reducelr_factor: float,
    reducelr_patience: int,
    monitored_metric: str = "val_accuracy",
    mode: str = "max",
    min_delta: float = 0.0,
    tensorboard_log_dir: str = None,
) -> list:
    """
    Initialize and return a list of Keras callbacks for model training.
    """

    # Ensure the model checkpoint directory exists
    checkpoint_dir = os.path.dirname(model_checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    early_stopping = EarlyStopping(
        monitor=monitored_metric,
        mode=mode,
        verbose=1,
        patience=early_stopping_patience,
        restore_best_weights=True,
        min_delta=min_delta,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=model_checkpoint_path,
        monitor=monitored_metric,
        mode=mode,
        verbose=1,
        save_best_only=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor=monitored_metric,
        mode=mode,
        factor=reducelr_factor,
        patience=reducelr_patience,
        verbose=1,
        min_delta=min_delta,
        min_lr=1e-6,  # Set a minimum learning rate
    )

    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    if tensorboard_log_dir:
        tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir)
        callbacks.append(tensorboard_callback)

    return callbacks


def plot_model_history(history, accuracy_savedir, loss_savedir):
    """
    Plot the model history (accuracy and loss).
    """

    # Summarize history for accuracy
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Training", "Validation"], loc="upper left")
        plt.grid(True)
        plt.savefig(accuracy_savedir)
        plt.close()
    else:
        print("Accuracy metrics not found in history.")

    # Summarize history for loss
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Training", "Validation"], loc="upper left")
        plt.grid(True)
        plt.savefig(loss_savedir)
        plt.close()
    else:
        print("Loss metrics not found in history.")


def evaluate_model(model, test_generator):
    """
    Evaluate the model on the test data and print the results.

    :param model: Model to evaluate.
    :param test_generator: Generator with which to test the model.
    """
    scores = model.evaluate(test_generator, verbose=1)
    for metric_name, score in zip(model.metrics_names, scores):
        print(f"{metric_name}: {score}")
