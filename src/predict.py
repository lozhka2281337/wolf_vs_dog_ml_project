import argparse
import numpy as np

from tensorflow import keras

import config as cfg

CLASS_NAMES = ["dog", "wolf"]

def predict_image(image_path, threshold=0.5):
    model = keras.models.load_model(cfg.MODEL_PATH)

    img = keras.utils.load_img(image_path, target_size=cfg.IMG_SIZE)
    arr = keras.utils.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])
    pred_idx = 1 if prob >= threshold else 0
    pred_name = CLASS_NAMES[pred_idx]

    print(f"\nФайл: {image_path}")
    print(f"Вероятность класса '{CLASS_NAMES[1]}': {prob:.4f}")
    print(f"Предсказание: {pred_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    predict_image(args.image, args.threshold)