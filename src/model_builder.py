from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow import keras

from config import IMG_SIZE

def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # этап 1: замораживаем

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model

def unfreeze_for_finetune(model, base_model, unfreeze_last_n=30):
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model