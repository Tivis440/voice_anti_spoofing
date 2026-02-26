# Voice Anti-Spoofing (Article Features, Multi-class)

Проект детектирует:
- `real`
- `fake_<engine>` (например `fake_coqui`, `fake_f5`, ...)

Инференс работает по **одному файлу**.

## Признаки (без log-mel)

Используются группы признаков из статьи в однофайловом варианте:
- MFCC-статистики (`mean/std`) и динамика (`delta`, `delta2`)
- спектральные признаки: centroid, bandwidth, rolloff, flatness
- энергетические и временные признаки: RMS, ZCR, energy ratio proxy

## Структура данных

Рекомендуемая:
- `data/raw/real/.../*.wav`
- `data/raw/fake/coqui/.../*.wav`
- `data/raw/fake/f5/.../*.wav`

Сплиты: `data/splits/train.csv`, `val.csv`, `test.csv`

Минимум колонок в CSV:
- `path`

Опционально:
- `class_name` (если нет — класс выводится из пути: `real`, `fake_<engine>`)
- `label` (используется только как fallback)

## Установка

```bash
pip install -r requirements.txt
```

## Обучение

```bash
python3 -m src.train --config configs/config.yaml --root .
```

Сохраняется:
- `logs/best_model.pt`
- `logs/metrics.json`

## Инференс GUI

```bash
python3 -m src.gui
```

На выходе:
- итог `REAL/FAKE`
- предсказанный класс (`real` или `fake_<engine>`)
- вероятности по top-классам

## Замечания

1. Для качества обязательно делайте сплиты без утечки по голосу/паре (`pair_id`/speaker-level split).
2. Для мультикласса балансируйте количество файлов по движкам.
