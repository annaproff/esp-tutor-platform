import streamlit as st
import os
import re
import yaml
import json
import pandas as pd
from datetime import datetime
from openai import OpenAI
from urllib.parse import urlparse
import psycopg2

# ==========================================
# 1. НАСТРОЙКИ И ПОДКЛЮЧЕНИЯ
# ==========================================
st.set_page_config(page_title="AI English Tutor", page_icon="🎓", layout="wide")

# Получаем секреты из переменных окружения
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY", "ВСТАВЬТЕ_СЮДА_КЛЮЧ_ЯНДЕКСА_ДЛЯ_ТЕСТА")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "ВСТАВЬТЕ_СЮДА_FOLDER_ID")
SUPABASE_URI = os.environ.get("SUPABASE_URI", "ВСТАВЬТЕ_СЮДА_SUPABASE_URI")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123") # Пароль для админки

# Клиент для YandexGPT
client = OpenAI(
    api_key=YANDEX_API_KEY,
    base_url="https://llm.api.cloud.yandex.net/foundationModels/v1"
)

def get_db_connection():
    parsed = urlparse(SUPABASE_URI)
    return psycopg2.connect(
        dbname=parsed.path[1:], user=parsed.username, password=parsed.password,
        host=parsed.hostname, port=parsed.port or 5432
    )

# ==========================================
# 2. ВАЛИДАЦИЯ И ВХОД
# ==========================================
def is_valid_russian_name(name):
    return bool(re.match(r"^[А-Яа-яЁё\s-]+$", name)) and len(name.split()) >= 2

if "student_name" not in st.session_state and "is_admin" not in st.session_state:
    st.title(" Добро пожаловать в AI Tutor Platform")
    st.markdown("Выберите режим входа:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨‍ Я студент", use_container_width=True):
            st.session_state.mode = "student"
            st.rerun()
    with col2:
        if st.button("👩‍🏫 Я преподаватель", use_container_width=True):
            st.session_state.mode = "admin"
            st.rerun()

# --- РЕЖИМ СТУДЕНТА ---
if st.session_state.get("mode") == "student" and "student_name" not in st.session_state:
    st.subheader("Вход для студента")
    with st.form("login_form"):
        fio = st.text_input("Фамилия и Имя (на русском)", placeholder="Иванов Иван")
        group = st.text_input("Номер группы", placeholder="101")
        submitted = st.form_submit_button("Начать занятие")
        
        if submitted:
            if not is_valid_russian_name(fio):
                st.error("Пожалуйста, введите Фамилию и Имя на русском языке.")
            elif not group:
                st.error("Пожалуйста, укажите номер группы.")
            else:
                st.session_state.student_name = fio.strip()
                st.session_state.student_group = group.strip()
                st.session_state.current_step = 0
                st.session_state.answers = {}
                st.rerun()

elif st.session_state.get("mode") == "student" and "student_name" in st.session_state:
    with st.sidebar:
        st.success(f" {st.session_state.student_name}")
        st.caption(f"Группа: {st.session_state.student_group}")
        if st.button("Выйти"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    # Загрузка YAML
    TASK_FILE_PATH = "tasks/tech_habits.yaml"
    try:
        with open(TASK_FILE_PATH, "r", encoding="utf-8") as f:
            task_data = yaml.safe_load(f)
        steps = task_data.get("steps", [])
        question_steps = [s for s in steps if s.get("type") == "question"]
    except FileNotFoundError:
        st.error(f"Файл задания не найден: {TASK_FILE_PATH}")
        st.stop()

    st.title(f"Задание: {task_data.get('meta', {}).get('title', 'Interview')}")
    st.markdown("---")

    step_idx = st.session_state.get("current_step", 0)

    if step_idx < len(question_steps):
        current_q = question_steps[step_idx]
        st.subheader(f"Вопрос {step_idx + 1} из {len(question_steps)}")
        st.markdown(f"**{current_q.get('topic', '')}**")
        st.write(current_q.get("say"))
        
        user_answer = st.text_area("Ваш ответ (на английском):", height=150, key=f"answer_{step_idx}")
        
        if st.button("Отправить ответ", type="primary"):
            if len(user_answer.split()) < 10:
                st.warning("Пожалуйста, напишите чуть подробнее (минимум 2-3 предложения).")
            elif re.search(r'[а-яА-Я]', user_answer) and len(re.findall(r'[а-яА-Я]', user_answer)) > len(user_answer) * 0.3:
                st.warning("Please try to answer in English — simple English is totally fine.")
            else:
                st.session_state.answers[current_q["id"]] = user_answer
                st.session_state.current_step = step_idx + 1
                
                # Сохраняем черновик в БД
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO sessions (full_name, group_name, task_id, current_step, answers)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (st.session_state.student_name, st.session_state.student_group, 
                          task_data["meta"]["id"], step_idx + 1, json.dumps(st.session_state.answers, ensure_ascii=False)))
                    st.session_state.session_id = cur.fetchone()[0]
                    conn.commit(); cur.close(); conn.close()
                except Exception as e:
                    st.error(f"Ошибка БД: {e}")
                st.rerun()
    else:
        # ЗАВЕРШЕНИЕ И ОЦЕНКА
        st.subheader("Отлично! Вы ответили на все вопросы.")
        if "final_report" not in st.session_state:
            st.markdown("Нейросеть анализирует ваши ответы... ⏳")
            answers_text = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(st.session_state.answers.values())])
            prompt = f"""Ты экспертный тьютор. Оцени ответы студента (B1-B2) по теме технологий.
            Критерии: Accuracy (макс 20) и Fluency (макс 10).
            ОТВЕТЫ: {answers_text}
            Верни СТРОГО JSON: {{"accuracy": <int>, "fluency": <int>, "total": <int>, "feedback": "<текст>"}}"""
            
            try:
                response = client.chat.completions.create(
                    model=f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
                    messages=[{"role": "user", "text": prompt}], temperature=0.2)
                llm_response = response.choices[0].message.content
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                grade_data = json.loads(json_match.group()) if json_match else {"error": "JSON parse error"}
                st.session_state.final_report = grade_data
            except Exception as e:
                st.session_state.final_report = {"error": str(e)}
            st.rerun()
        else:
            grade = st.session_state.final_report
            if "error" in grade: st.error(f"Ошибка LLM: {grade['error']}")
            else:
                st.success("Анализ завершен!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{grade.get('accuracy', 0)}/20")
                col2.metric("Fluency", f"{grade.get('fluency', 0)}/10")
                col3.metric("Total", f"{grade.get('total', 0)}/30")
                st.markdown(f"**Feedback:** {grade.get('feedback', '')}")
                
                # Обновляем статус в БД
                try:
                    conn = get_db_connection(); cur = conn.cursor()
                    cur.execute("UPDATE sessions SET status = 'completed', grade_json = %s, completed_at = NOW() WHERE id = %s",
                                (json.dumps(grade, ensure_ascii=False), st.session_state.session_id))
                    conn.commit(); cur.close(); conn.close()
                except Exception as e: st.error(f"Ошибка обновления БД: {e}")

# --- РЕЖИМ АДМИНА (ПРЕПОДАВАТЕЛЬ) ---
elif st.session_state.get("mode") == "admin":
    if "admin_auth" not in st.session_state:
        st.subheader("🔒 Вход для преподавателя")
        pwd = st.text_input("Введите пароль администратора", type="password")
        if st.button("Войти"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_auth = True
                st.rerun()
            else: st.error("Неверный пароль")
    else:
        st.title("‍🏫 Панель преподавателя (Дашборд)")
        if st.button("Выйти из админки"):
            del st.session_state.admin_auth; st.rerun()
        
        try:
            conn = get_db_connection()
            df = pd.read_sql("""
                SELECT full_name, group_name, task_id, 
                       (grade_json->>'accuracy')::int as accuracy,
                       (grade_json->>'fluency')::int as fluency,
                       (grade_json->>'total')::int as total,
                       status, created_at
                FROM sessions ORDER BY group_name, full_name
            """, conn)
            conn.close()
            
            if df.empty:
                st.info("Пока нет данных от студентов.")
            else:
                # ВАРИАНТ В: Pivot Table (Сводная)
                st.subheader(" Сводная таблица (Pivot)")
                pivot = df.pivot_table(index=['full_name', 'group_name'], columns='task_id', values='total', aggfunc='first')
                st.dataframe(pivot.fillna("—"), use_container_width=True)
                
                st.markdown("---")
                
                # ВАРИАНТ В: Плоская таблица для экспорта
                st.subheader(" Экспорт в Excel (Плоская таблица)")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Скачать CSV/Excel", csv, "student_results.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Ошибка подключения к БД: {e}")
