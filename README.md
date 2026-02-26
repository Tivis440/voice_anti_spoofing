# Voice Anti-Spoofing (Article-Based, no log-Mel)

Проект полностью переведен на признаки из статьи и **не использует log-Mel/CNN**.

## Что используется

Для каждой пары `sample` (проверяемый файл) и `reference` (эталонный real) считаются признаки:
- `mfcc_euclidean`
- `mfcc_euclidean_norm`
- `mfcc_cosine`
- `spectral_centroid_diff`
- `energy_ratio_noise_to_signal`

Далее обучается бинарный классификатор `real(0) / fake(1)` на `scikit-learn` (`StandardScaler + LogisticRegression`).

## Формат сплитов

`data/splits/train.csv`, `val.csv`, `test.csv`:
- обязательные колонки: `path`, `label`
- опционально: `ref_path` (явный путь к эталону для fake)

Если `ref_path` не задан для fake, код ищет эталон в `data/raw/real` по совпадающему `stem` имени файла.

## Установка

```bash
pip install -r requirements.txt
```

## Обучение

```bash
python3 -m src.train --config configs/config.yaml --root .
```

Артефакты:
- `logs/best_model.pkl`
- `logs/metrics.json`

## Инференс через GUI

```bash
python3 -m src.gui
```

В GUI нужно выбрать:
- `Sample audio` — проверяемый файл
- `Reference real audio` — эталонный real голос
- `Model (.pkl)` — `logs/best_model.pkl`

## Структура

- `src/features.py` — признаки из статьи
- `src/dataset.py` — сбор feature matrix из CSV
- `src/model.py` — sklearn-модель
- `src/train.py` — обучение/валидация/сохранение
- `src/inference.py` — предсказание по паре sample/reference
- `src/gui.py` — Tkinter GUI
- `configs/config.yaml` — параметры аудио/признаков/обучения

## Важно

Текущий подход по статье требует **эталонного real-файла** для сравнения.
Без эталона корректная оценка по этой методике невозможна.
