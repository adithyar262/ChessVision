"""This module is responsible for training the ConvNeXtTiny model."""

from keras.api.applications import ConvNeXtTiny
from keras.api.applications.convnext import preprocess_input
from keras.api.models import load_model
from keras.api.optimizers import Adam

from common import (
    build_model,
    data_generators,
    train_model,
    plot_model_history,
    evaluate_model,
    model_callbacks,
)


def train_chesspiece_model():
    """Train the chess-piece model based on ConvNeXtTiny."""
    base_model = ConvNeXtTiny(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    # First train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    model = build_model(base_model)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_generator, validation_generator = data_generators(
        preprocess_input, (224, 224), 32
    )

    callbacks = model_callbacks(
        early_stopping_patience=5,
        model_checkpoint_path="./models/ConvNeXtTiny_pre.keras",
        reducelr_factor=0.1,
        reducelr_patience=5,
        tensorboard_log_dir="./logs/pretraining",
    )

    history = train_model(
        model,
        epochs=20,
        train_generator=train_generator,
        validation_generator=validation_generator,
        callbacks=callbacks,
        use_weights=False,
        workers=5,
    )

    plot_model_history(
        history,
        "./models/ConvNeXtTiny_pre_acc.png",
        "./models/ConvNeXtTiny_pre_loss.png",
    )
    evaluate_model(model, validation_generator)

    # Unfreeze the last 20% of layers for further training
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * 0.8)
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True

    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = model_callbacks(
        early_stopping_patience=10,
        model_checkpoint_path="./models/ConvNeXtTiny.keras",
        reducelr_factor=0.5,
        reducelr_patience=5,
        tensorboard_log_dir="./logs/fine_tuning",
    )

    history = train_model(
        model,
        epochs=50,
        train_generator=train_generator,
        validation_generator=validation_generator,
        callbacks=callbacks,
        use_weights=False,
        workers=5,
    )

    plot_model_history(
        history, "./models/ConvNeXtTiny_acc.png", "./models/ConvNeXtTiny_loss.png"
    )
    evaluate_model(model, validation_generator)

    model.save("./models/ConvNeXtTiny_last.keras")


if __name__ == "__main__":
    train_chesspiece_model()
