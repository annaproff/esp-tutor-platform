import streamlit as st
import os
import re
import yaml
import psycopg2
from openai import OpenAI
from urllib.parse import urlparse
import json

# ==========================================
# 1. НАСТРОЙКИ И ПОДКЛЮЧЕНИЯ
# ==========================================
st.set_page_config(page_title="AI English Tutor", page_icon="🎓", layout="centered")

# Получаем секреты из переменных окружения (мы настроим их в Render на следующем шаге)
# Для теста можно вписать их прямо сюда, но потом обязательно уберите!
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY", "ВСТАВЬТЕ_СЮДА_КЛЮЧ_ЯНДЕКСА_ДЛЯ_ТЕСТА")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "ВСТАВЬТЕ_СЮДА_FOLDER_ID")
SUPABASE_URI = os.environ.get("SUPABASE_URI", "ВСТАВЬТЕ_СЮДА_SUPABASE_URI")

# Клиент для YandexGPT (используем библиотеку openai как универсальный пульт)
client = OpenAI(
    api_key=YANDEX_API_KEY,
    base_url="https://llm.api.cloud.yandex.net/foundationModels/v1"
)

# Функция подключения к базе данных
def get_db_connection():
    # Парсим строку подключения Supabase для psycopg2
    parsed = urlparse(SUPABASE_URI)
    return psycopg2.connect(
        dbname=parsed.path[1:],
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port or 5432
    )

# ==========================================
# 2. ВАЛИДАЦИЯ И ВХОД СТУДЕНТА
# ==========================================
def is_valid_russian_name(name):
    # Разрешаем только кириллицу, пробелы и дефис
    return bool(re.match(r"^[А-Яа-яЁё\s-]+$", name)) and len(name.split()) >= 2

if "student_name" not in st.session_state:
    st.title("🎓 Добро пожаловать в AI Tutor")
    st.markdown("Пожалуйста, представьтесь, чтобы мы могли сохранить ваш прогресс.")
    
    with st.form("login_form"):
        fio = st.text_input("Фамилия и Имя (на русском)", placeholder="Иванов Иван")
        group = st.text_input("Номер группы", placeholder="101")
        submitted = st.form_submit_button("Начать занятие")
        
        if submitted:
            if not is_valid_russian_name(fio):
                st.error("Пожалуйста, введите Фамилию и Имя на русском языке (кириллицей).")
            elif not group:
                st.error("Пожалуйста, укажите номер группы.")
            else:
                st.session_state.student_name = fio.strip()
                st.session_state.student_group = group.strip()
                st.session_state.current_step = 0
                st.session_state.answers = {}
                st.rerun()
else:
    # Сайдбар с профилем студента
    with st.sidebar:
        st.success(f"👤 {st.session_state.student_name}")
        st.caption(f"Группа: {st.session_state.student_group}")
        if st.button("Выйти / Сменить пользователя"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ==========================================
    # 3. ЗАГРУЗКА ЗАДАНИЯ ИЗ YAML
    # ==========================================
    # Путь к файлу задания (убедитесь, что он лежит в папке tasks на GitHub)
    TASK_FILE_PATH = "tasks/tech_habits.yaml"
    
    try:
        with open(TASK_FILE_PATH, "r", encoding="utf-8") as f:
            task_data = yaml.safe_load(f)
        steps = task_data.get("steps", [])
        # Фильтруем только шаги с вопросами
        question_steps = [s for s in steps if s.get("type") == "question"]
    except FileNotFoundError:
        st.error(f"Файл задания не найден: {TASK_FILE_PATH}. Проверьте репозиторий GitHub.")
        st.stop()

    st.title(f"Задание: {task_data.get('meta', {}).get('title', 'Interview')}")
    st.markdown("---")

    # ==========================================
    # 4. ЛОГИКА ДИАЛОГА (ИТЕРАЦИИ)
    # ==========================================
    step_idx = st.session_state.get("current_step", 0)

    if step_idx < len(question_steps):
        current_q = question_steps[step_idx]
        
        st.subheader(f"Вопрос {step_idx + 1} из {len(question_steps)}")
        st.markdown(f"**{current_q.get('topic', '')}**")
        st.write(current_q.get("say"))
        
        user_answer = st.text_area("Ваш ответ (на английском):", height=150, key=f"answer_{step_idx}")
        
        if st.button("Отправить ответ", type="primary"):
            # Валидация на стороне кода (идея Тарасова/Маджитова)
            if len(user_answer.split()) < 10:
                st.warning("Пожалуйста, напишите чуть подробнее (минимум 2-3 предложения).")
            elif re.search(r'[а-яА-Я]', user_answer) and len(re.findall(r'[а-яА-Я]', user_answer)) > len(user_answer) * 0.3:
                st.warning("Please try to answer in English — simple English is totally fine.")
            else:
                # Сохраняем ответ
                st.session_state.answers[current_q["id"]] = user_answer
                st.session_state.current_step = step_idx + 1
                
                # Сохраняем в БД (черновик сессии)
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
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    st.error(f"Ошибка сохранения в БД: {e}")
                
                st.rerun()
    else:
        # ==========================================
        # 5. ЗАВЕРШЕНИЕ И ВЫЗОВ LLM
        # ==========================================
        st.subheader("Отлично! Вы ответили на все вопросы.")
        st.markdown("Нейросеть анализирует ваши ответы. Это займет около 20 секунд...")
        
        if "final_report" not in st.session_state:
            # Формируем промпт для оценки (упрощенная версия для MVP)
            answers_text = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(st.session_state.answers.values())])
            
            prompt = f"""
            Ты экспертный тьютор по английскому языку. Оцени ответы студента (уровень B1-B2) по теме технологий.
            Критерии: Accuracy (макс 20) и Fluency (макс 10).
            
            ОТВЕТЫ СТУДЕНТА:
            {answers_text}
            
            Верни СТРОГО JSON без markdown:
            {{"accuracy": <int>, "fluency": <int>, "total": <int>, "feedback": "<текст на английском>"}}
            """
            
            try:
                response = client.chat.completions.create(
                    model=f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
                    messages=[{"role": "user", "text": prompt}],
                    temperature=0.2
                )
                llm_response = response.choices[0].message.content
                
                # Пытаемся распарсить JSON (с защитой от markdown-оберток)
                import re
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    grade_data = json.loads(json_match.group())
                    st.session_state.final_report = grade_data
                else:
                    st.session_state.final_report = {"error": "Не удалось распарсить ответ модели"}
            except Exception as e:
                st.session_state.final_report = {"error": str(e)}
            st.rerun()
        else:
            grade = st.session_state.final_report
            if "error" in grade:
                st.error(f"Ошибка LLM: {grade['error']}")
            else:
                st.success("Анализ завершен!")
                st.markdown(f"### Ваши результаты")
                st.metric("Accuracy", f"{grade.get('accuracy', 0)}/20")
                st.metric("Fluency", f"{grade.get('fluency', 0)}/10")
                st.metric("Total", f"{grade.get('total', 0)}/30")
                st.markdown(f"**Feedback:** {grade.get('feedback', '')}")
                
                # Обновляем статус в БД
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE sessions 
                        SET status = 'completed', grade_json = %s 
                        WHERE id = %s
                    """, (json.dumps(grade, ensure_ascii=False), st.session_state.session_id))
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    st.error(f"Ошибка обновления БД: {e}")
