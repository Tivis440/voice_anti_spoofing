# Voice Anti-Spoofing

Классификатор реальный vs синтетический голос на PyTorch (log-Mel + CNN).

## Структура

```
voice_anti_spoofing/
├── configs/config.yaml   # параметры (sample_rate, segment_length, n_mels, batch_size, lr, epochs)
├── data/
│   ├── raw/              # wav: real/ и fake/
│   └── splits/           # train.csv, val.csv, test.csv (колонки path, label; 0=Real, 1=Fake)
├── src/
│   ├── features.py       # log-Mel спектрограмма
│   ├── dataset.py        # VoiceDataset (на лету спектрограммы)
│   ├── model.py          # CNNClassifier
│   └── train.py          # обучение/валидация
├── logs/                 # best_model.pt, last_model.pt, метрики
└── scripts/create_dummy_data.py  # тестовые wav для быстрого запуска
```

## Запуск

```bash
cd voice_anti_spoofing
pip install -r requirements.txt

# Создать тестовые wav (опционально)
python scripts/create_dummy_data.py

# Обучение (из корня проекта)
python -m src.train --config configs/config.yaml --root .
```

Чекпоинты: `logs/best_model.pt`, `logs/last_model.pt`.

## Разметка CSV

В `data/splits/train.csv` (и val/test) — колонки:

- `path` — путь к wav (относительно `--root` или абсолютный)
- `label` — 0 = Real, 1 = Fake

Для теста на unseen generator можно добавить колонку `generator_id` и фильтровать по ней при создании сплитов.
