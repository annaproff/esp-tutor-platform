FROM python:3.11-slim

# Устанавливаем системные библиотеки для psycopg2 и curl (для проверки здоровья)
RUN apt-get update && apt-get install -y libpq-dev gcc curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Healthcheck для Timeweb (даем 40 секунд на старт, чтобы успели загрузиться все библиотеки)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Запуск без лишних флагов, всё настроено в .streamlit/config.toml
ENTRYPOINT ["streamlit", "run", "app.py"]
