import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import config as cfg
from data import get_test_generator

def main():
    test_data = get_test_generator(cfg.TEST_DIR)
    model = keras.models.load_model(cfg.MODEL_PATH)

    loss, acc = model.evaluate(test_data, verbose=1)
    print(f"Тест потерь: {loss:.4f}")
    print(f"Тест точности: {acc:.4f}")

    test_data.reset()
    y_prob = model.predict(test_data, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_data.classes

    idx_to_class = {v: k for k, v in test_data.class_indices.items()}
    target_names = [idx_to_class[0], idx_to_class[1]]

    print("\nОтчет о классификации")
    print(classification_report(y_true, y_pred, target_names=target_names))

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    ).plot(cmap="Blues", values_format="d")
    
    plt.title("Матрица ошибок (тестовая выборка)")
    plt.show()

if __name__ == "__main__":
    main()