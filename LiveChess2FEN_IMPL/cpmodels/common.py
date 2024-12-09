import math
from typing import List

import tensorflow as tf
from keras.api import layers, Sequential
from keras.api.utils import Sequence, image_dataset_from_directory
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

    # Create training dataset
    train_ds = image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=target_size,
        shuffle=True,
        seed=123,
    )

    # Create validation dataset
    val_ds = image_dataset_from_directory(
        validation_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False,
    )

    # Apply preprocessing function
    train_ds = train_ds.map(lambda x, y: (preprocessing_func(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocessing_func(x), y))

    # Define data augmentation pipeline
    data_augmentation = Sequential([
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomZoom(0.15),
        layers.RandomFlip("horizontal"),
    ])

    # Apply data augmentation to the training dataset
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch datasets for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


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
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
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
