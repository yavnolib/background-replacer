FROM ubuntu:22.04

# Обновление системы и установка базовых зависимостей
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip curl git ffmpeg libsm6 libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
RUN curl -sSL https://install.python-poetry.org | python3.11 -

# Добавляем Poetry в PATH
ENV PATH="/root/.local/bin:$PATH"

# Установка рабочей директории
WORKDIR /app

# Копируем файлы зависимостей проекта
COPY background_replacer /app/color_checker
COPY docs /app/
COPY notebooks /app/
COPY poetry.lock /app
COPY pyproject.toml /app
COPY README.md /app

# Устанавливаем зависимости
RUN cd /app && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# Открываем доступ к консоли
CMD ["tail", "-f", "/dev/null"]
