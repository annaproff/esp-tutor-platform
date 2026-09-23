FROM python:3.11-slim

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Streamlit по умолчанию слушает порт 8501
EXPOSE 8501

# Запуск приложения
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
