# Background Replacer

🚀 **Background Replacer** — Python-пакет для сегментации фона на изображениях с использованием PyTorch Lightning и Hydra. Проект оформлен как воспроизводимый ML-пайплайн с DVC, логированием и удобной CLI-структурой (`train.py`, `infer.py`).

## 📦 Установка

1. Клонируй репозиторий:

```bash
git clone https://github.com/yourname/background-replacer.git
cd background-replacer
```

2. Установи зависимости через [Poetry](https://python-poetry.org/):

```bash
poetry install
```

3. Установи [DVC](https://dvc.org/) и подтяни данные:

```bash
poetry run dvc pull
```

4. Настрой [pre-commit](https://pre-commit.com/):

```bash
poetry run pre-commit install
```

## 🏗️ Структура проекта

```
background-replacer/
├── background_replacer/
│   ├── train.py              # Точка входа для обучения
│   ├── infer.py              # Точка входа для инференса
│   ├── models/               # Модели (UNet и др.)
│   ├── data/                 # DataModule и препроцессинг
│   ├── utils/                # Утилиты: логгирование, трансформации
├── configs/                  # Конфиги Hydra
│   └── config.yaml
├── data/                     # Каталог с raw/ и processed/ данными
├── logs/                     # TensorBoard логи
├── outputs/                  # Модели и результаты
├── dvc.yaml
├── pyproject.toml
└── README.md
```

## ⚙️ Использование

### 🔧 Обучение модели

```bash
poetry run python background_replacer/train.py
```

Все параметры задаются в `configs/config.yaml`. Поддерживается логгирование в TensorBoard (`logs/`).

### 🔎 Инференс

```bash
poetry run python background_replacer/infer.py
```

Результаты сохраняются в `.csv`, путь задаётся в конфиге `cfg.infer.output_file`.

## 🔍 Конфигурация (Hydra)

Файл `configs/config.yaml` определяет:
- пути до данных,
- гиперпараметры модели,
- параметры обучения и инференса.

Пример:

```yaml
data:
  train_path: data/train
  val_path: data/val
  batch_size: 8

train:
  epochs: 20
  lr: 0.0003

infer:
  input_path: data/test
  model_path: outputs/model.pth
  output_file: outputs/inference.csv
```
