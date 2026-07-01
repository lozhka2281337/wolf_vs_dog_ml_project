from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config as cfg

# генератор данных

def get_train_val_generators(train_dir):
    train_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=cfg.VALIDATION_SPLIT,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True
    )

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=cfg.SEED
    )

    val_data = train_gen.flow_from_directory(
        train_dir,
        target_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=cfg.SEED
    )

    return train_data, val_data

def get_test_generator(test_dir):
    test_gen = ImageDataGenerator(rescale=1./255)

    test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=cfg.IMG_SIZE,
    batch_size=cfg.BATCH_SIZE,
    class_mode="binary",
    shuffle=False
    )

    return test_data