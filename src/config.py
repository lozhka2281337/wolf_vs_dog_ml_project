import os

MAIN_DIR = "dataset"
TRAIN_DIR = os.path.join(MAIN_DIR, "train")
TEST_DIR = os.path.join(MAIN_DIR, "test")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "dog_wolf_best.keras")

IMG_SIZE = (224, 224)          # для MobileNetV2 обычно 224x224
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EPOCHS_HEAD = 10               # сначала обучаем "голову"
EPOCHS_FINE = 10               # потом немного fine-tuning
SEED = 42