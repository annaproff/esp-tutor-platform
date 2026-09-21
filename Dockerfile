FROM python:3.11-slim

# Устанавливаем системные библиотеки (критично для psycopg2) и curl для проверки здоровья
RUN apt-get update && apt-get install -y libpq-dev gcc curl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Проверка здоровья для Timeweb
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ИСПРАВЛЕННЫЙ ЗАПУСК: убран enableCORS, добавлен headless=true
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
