<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

# 🎙️ Voice Anti-Spoofing

**Детекция синтетического голоса** — бинарный классификатор «реальный vs фейковый» голос на PyTorch.  
Log-Mel спектрограммы + лёгкая CNN: готовый пайплайн для экспериментов и расширения под unseen generators.

---

## Что это

Голосовые клоны и TTS становятся настолько качественными, что отличить их от живого голоса на слух сложно. Этот репозиторий — **baseline-модель** для детекции спуфинга по аудио: по короткому сегменту (например, 2 сек) модель предсказывает класс **Real** (0) или **Fake** (1).

- 🧠 **Модель:** 3 блока Conv2D (BatchNorm, ReLU, MaxPool) → Global Pooling → FC → Softmax  
- 📊 **Вход:** нормализованная log-Mel спектрограмма (on-the-fly из wav в датасете)  
- ⚙️ Всё настраивается через `configs/config.yaml`  
- 🔬 Удобно расширять под **unseen generator** тесты (добавить `generator_id` в разметку и сплиты)

---

## Быстрый старт

```bash
git clone https://github.com/YOUR_USERNAME/voice_anti_spoofing.git
cd voice_anti_spoofing
pip install -r requirements.txt
```

**Проверить пайплайн на тестовых данных:**

```bash
# Сгенерировать тестовые wav (real/fake)
python scripts/create_dummy_data.py

# Запустить обучение
python -m src.train --config configs/config.yaml --root .
```

Чекпоинты сохраняются в `logs/`: `best_model.pt`, `last_model.pt`.

**Графический интерфейс** (проверка одного wav-файла):

```bash
python -m src.gui
```

Откроется окно: выберите WAV-файл и модель (.pt), нажмите «Проверить» — получите ответ «Реальный голос» или «Синтетический голос» и вероятности.

---

## Структура проекта

```
voice_anti_spoofing/
├── configs/
│   └── config.yaml          # sample_rate, segment_length, n_mels, batch_size, lr, epochs
├── data/
│   ├── raw/
│   │   ├── real/             # wav — реальный голос
│   │   └── fake/             # wav — синтетика
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv          # колонки: path, label (0=Real, 1=Fake)
├── src/
│   ├── features.py           # log-Mel спектрограмма (librosa)
│   ├── dataset.py            # VoiceDataset — нарезка сегментов + спектрограммы на лету
│   ├── model.py              # CNNClassifier
│   ├── inference.py          # предсказание по одному wav (для GUI и CLI)
│   ├── train.py              # цикл обучения/валидации, сохранение чекпоинтов
│   └── gui.py                # графический интерфейс (Tkinter)
├── scripts/
│   └── create_dummy_data.py  # генерация тестовых wav
├── logs/                     # best_model.pt, last_model.pt
├── requirements.txt
└── README.md
```

---

## Конфигурация

Основные параметры в `configs/config.yaml`:

| Параметр | Описание | Пример |
|----------|----------|--------|
| `sample_rate` | Частота дискретизации | 16000 |
| `segment_length` | Длина сегмента в секундах | 2.0 |
| `n_mels` | Число Mel-фильтров | 80 |
| `batch_size` | Размер батча | 32 |
| `learning_rate` | Learning rate для Adam | 0.001 |
| `epochs` | Число эпох | 30 |

Пути к CSV сплитов и папке с логами задаются в том же файле.

---

## Формат данных

**CSV** (train/val/test) — две колонки:

| path | label |
|------|--------|
| `data/raw/real/sample1.wav` | 0 |
| `data/raw/fake/sample1.wav` | 1 |

- `path` — путь к wav (относительно `--root` или абсолютный).  
- `label` — **0** = Real, **1** = Fake.

Для сценария **unseen generator**: добавьте колонку `generator_id` и при формировании сплитов исключайте нужные генераторы из train, оставляя их для val/test.

---

## Зависимости

- Python 3.8+
- PyTorch ≥ 2.0
- librosa, pandas, PyYAML, scipy

Подробно: `requirements.txt`.

---

## Лицензия

MIT (или укажите свою).
