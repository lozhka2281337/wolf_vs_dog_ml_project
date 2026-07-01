# Dog vs Wolf Classifier 🐶🐺

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)

Классификатор изображений собака / волк на базе TensorFlow + Keras с использованием Transfer Learning (MobileNetV2).

Проект разделён на отдельные сценарии:

- train.py — обучение модели  
- evaluate.py — тестирование и метрики  
- predict.py — использование модели на новых изображениях  

---

## ✨ Features проекта ✨ 

- Обучение в 2 этапа:
  - обучение классификационной головы
  - fine-tuning верхних слоёв 
- Аугментация изображений
- Сохранение лучшей модели 
- Метрики качества:
  - Accuracy / Loss
  - Precision / Recall / F1
  - Confusion Matrix
- Предсказание для одного изображения

---

## 🗂️ Структура проекта
```
dog-wolf-classifier/
│
├─ dataset/
│  ├─ train/
│  │  ├─ dog/
│  │  └─ wolf/
│  └─ test/
│     ├─ dog/
│     └─ wolf/
│
├─ model/
│  └─ dog_wolf_best.keras
│
├─ src/
│  ├─ config.py
│  ├─ data.py
│  ├─ model_builder.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ predict.py
│
├─ requirements.txt
└─ README.md
```
---

## ⚙️ Установка
```bash
git clone https://github.com/lozhka2281337/wolf_vs_dog_ml_project
cd wolf_vs_dog_ml_project
python -m venv .venv
```
Windows
```bash
.venv\Scripts\activate
```
Linux / macOS
```bash
source .venv/bin/activate
```


```bash
pip install -r requirements.txt
```
---

## 🧠 Данные

Ижображения раскиданы по папкам классов:
```bash
dataset/
  train/
    dog/
    wolf/
  test/
    dog/
    wolf/
```
> Названия папок (`dog`, `wolf`) используются как имена классов.

---

## 🚀 Запуск

### 1) Обучение модели
```bash
python src/train.py
```
Что делает скрипт:

- создаёт `train/validation` генераторы
- обучает голову модели
- запускает fine-tuning
- сохраняет лучшую модель в `model/dog_wolf_best.keras`

---

### 2) Оценка на тестовой выборке
```bash
python src/evaluate.py
```

Выводит:

- `Тестовые ошибки`
- `Тестовая точность`
- `отчет о классификации`
- `матрица ошибок`

---

### 3) Предсказание на одном изображении
```bash
python src/predict.py --image "path/to/image.jpg" --threshold 0.5
```
