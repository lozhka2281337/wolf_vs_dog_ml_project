import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint 

from data import get_train_val_generators
from model_builder import build_model, unfreeze_for_finetune
import config as cfg

def main():
    tf.random.set_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    train_data, val_data = get_train_val_generators(cfg.TRAIN_DIR)

    model, base_model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            cfg.MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n[1/2], Тренировка головы...")
    
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=cfg.EPOCHS_HEAD,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[2/2] Тонкая настройка...")
    model = unfreeze_for_finetune(model, base_model, unfreeze_last_n=30)
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=cfg.EPOCHS_FINE,
        callbacks=callbacks,
        verbose=1
    ) 

    model.save(cfg.MODEL_PATH)
    print(f"\nМодель сохранена: {cfg.MODEL_PATH}")


if __name__ == "__main__":
    main()