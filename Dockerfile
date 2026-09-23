FROM python:3.11-slim

# Устанавливаем curl для healthcheck и gcc для сборки зависимостей pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем зависимости (слой кэшируется)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и папку с тасками
COPY . .

# Создаем папку для БД, если её нет
RUN mkdir -p /app/data

EXPOSE 8501

# Проверка здоровья контейнера
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Запуск Streamlit. 
# --server.address 0.0.0.0 обязателен, чтобы сайт был доступен снаружи контейнера
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.enableCORS", "false"]

