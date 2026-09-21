FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Проверка здоровья
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Запуск Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
