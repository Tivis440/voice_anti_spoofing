# Voice Anti-Spoofing 

Проект детектирует:
- `real`
- `fake_<engine>` (например `fake_coqui`, `fake_f5`, ...)

Инференс работает по **одному файлу**.

## Признаки 

Используются группы признаков из статьи в однофайловом варианте:
- MFCC-статистики (`mean/std`) и динамика (`delta`, `delta2`)
- спектральные признаки: centroid, bandwidth, rolloff, flatness
- энергетические и временные признаки: RMS, ZCR, energy ratio proxy

## Структура данных

Рекомендуемая:
- `data/raw/real/.../*.wav` и `data/raw/fake/<engine>/.../*.wav`
- `data/real/.../*.wav` и `data/fake/<engine>/.../*.wav`

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


## Быстрый запуск (пошагово)

```bash
cd "/Volumes/Tivis data/Project archive/voice_anti_spoofing"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Собрать сплиты из папок `data/raw/...` или `data/...`:

```bash
python3 scripts/make_splits.py --root . --max-per-class 2000
```

Проверить сплиты:

```bash
python3 scripts/validate_dataset.py \
  --root . \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --test data/splits/test.csv
```

Запустить обучение:

```bash
python3 -m src.train --config configs/config.yaml --root .
```

После обучения появятся:
- `logs/best_model.pt`
- `logs/metrics.json`
- `logs/training_history.json`
- `logs/training_plot.png`

Запуск интерфейсов:

```bash
python3 scripts/project_tui.py
python3 -m src.gui
```

## Исследование влияния количества признаков

```bash
python3 scripts/feature_count_study.py --root . --config configs/config.yaml --mfcc-values 5,8,13,16,20,24
```

Скрипт прогоняет несколько значений `n_mfcc`, для каждого случая обучает модель и сохраняет:
- `logs/feature_count_study.csv`
- `logs/feature_count_study.json`
- `logs/feature_count_study.png`

`feature_count` в отчете — это реальное количество признаков, которое получилось после извлечения.

## Исследование влияния количества эпох

```bash
python3 scripts/epoch_count_study.py --root . --config configs/config.yaml --epoch-values 2,4,6,8,10,12,16,20
```

Скрипт фиксирует набор признаков из конфига, меняет только `epochs`, обучает модель для каждого значения и сохраняет:
- `logs/epoch_count_study.csv`
- `logs/epoch_count_study.json`
- `logs/epoch_count_study.png`

В отчете `best_epoch` показывает эпоху, на которой было достигнуто лучшее значение `val_macro_f1`.

## Исследование влияния hidden_dim

```bash
python3 scripts/hidden_dim_study.py --root . --config configs/config.yaml --hidden-dim-values 32,64,96,128,192,256
```

Скрипт фиксирует признаки и число эпох из конфига, меняет только размер первого скрытого слоя и сохраняет:
- `logs/hidden_dim_study.csv`
- `logs/hidden_dim_study.json`
- `logs/hidden_dim_study.png`

## Исследование влияния dropout

```bash
python3 scripts/dropout_study.py --root . --config configs/config.yaml --dropout-values 0.0,0.1,0.2,0.3,0.4,0.5,0.6
```

Скрипт фиксирует признаки, число эпох и hidden_dim из конфига, меняет только dropout и сохраняет:
- `logs/dropout_study.csv`
- `logs/dropout_study.json`
- `logs/dropout_study.png`

## Исследование влияния learning_rate

```bash
python3 scripts/learning_rate_study.py --root . --config configs/config.yaml --lr-values 0.0001,0.0003,0.0005,0.0007,0.001,0.002,0.003
```

Скрипт фиксирует признаки, epochs, hidden_dim и dropout из конфига, меняет только шаг обучения и сохраняет:
- `logs/learning_rate_study.csv`
- `logs/learning_rate_study.json`
- `logs/learning_rate_study.png`

## Обучение

```bash
python3 -m src.train --config configs/config.yaml --root .
```

Сохраняется:
- `logs/best_model.pt`
- `logs/metrics.json`
- `logs/training_history.json`
- `logs/training_plot.png`

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

## Шаблон CSV и валидация выборки

Сгенерировать шаблон CSV:

```bash
python3 scripts/validate_dataset.py --write-template data/splits/template.csv
```

Проверить готовые `train/val/test`:

```bash
python3 scripts/validate_dataset.py \
  --root . \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --test data/splits/test.csv
```

Скрипт проверяет:
- обязательные колонки: `path,class_name,speaker_id,utt_id`
- существование файлов
- дубли внутри сплита
- утечки между сплитами (`speaker_id`, `(speaker_id,utt_id)`, `path`)
- дисбаланс классов (по умолчанию порог 10%)
