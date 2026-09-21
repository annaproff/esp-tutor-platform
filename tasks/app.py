import streamlit as st
import os
import re
import yaml
import json
import hashlib
import pandas as pd
from datetime import datetime
from openai import OpenAI
from urllib.parse import urlparse
import psycopg2
import time

# ==========================================
# 1. НАСТРОЙКИ И ПОДКЛЮЧЕНИЯ
# ==========================================
st.set_page_config(page_title="AI English Tutor", page_icon="🎓", layout="wide")

YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY", "ВСТАВЬТЕ_СЮДА_КЛЮЧ_ЯНДЕКСА_ДЛЯ_ТЕСТА")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "ВСТАВЬТЕ_СЮДА_FOLDER_ID")
SUPABASE_URI = os.environ.get("SUPABASE_URI", "ВСТАВЬТЕ_СЮДА_SUPABASE_URI")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

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
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def is_valid_russian_name(name):
    return bool(re.match(r"^[А-Яа-яЁё\s-]+$", name)) and len(name.split()) >= 2

def count_tokens_in_response(response):
    """Подсчет токенов из ответа модели"""
    try:
        usage = response.usage
        if usage:
            return {
                "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                "total_tokens": getattr(usage, 'total_tokens', 0) or 0
            }
    except:
        pass
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def call_llm_with_retry(prompt, max_retries=2, temperature=0.2):
    """Вызов LLM с ретраем при ошибке"""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
                messages=[{"role": "user", "text": prompt}],
                temperature=temperature
            )
            content = response.choices[0].message.content
            tokens = count_tokens_in_response(response)
            return content, tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM failed after {max_retries} retries: {last_error}")

def extract_json_with_retry(text, prompt_for_retry=None, max_retries=1):
    """Парсинг JSON с ретраем"""
    for attempt in range(max_retries + 1):
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON object found")
        except Exception as e:
            if attempt < max_retries and prompt_for_retry:
                retry_prompt = prompt_for_retry + "\n\nВАЖНО: верни ТОЛЬКО валидный JSON без пояснений."
                text, _ = call_llm_with_retry(retry_prompt, max_retries=0, temperature=0.0)
            else:
                raise ValueError(f"JSON parse error: {e}")
    return None

def compute_sha256(content):
    """Вычисление SHA-256 хеша"""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()

def validate_answer(answer, min_words=15, russian_threshold=0.3):
    """Валидация ответа с флагами"""
    flags = []
    warnings = []
    word_count = len(answer.split())
    if word_count < min_words:
        flags.append("too_short_accepted")
        warnings.append(f"Пожалуйста, напишите чуть подробнее (минимум {min_words} слов).")
    cyrillic_chars = len(re.findall(r'[а-яА-Я]', answer))
    if cyrillic_chars > len(answer) * russian_threshold:
        flags.append("russian_accepted")
        warnings.append("Please try to answer in English — simple English is totally fine.")
    if len(answer) < 20:
        flags.append("fast_answer")
    return flags, warnings

# ==========================================
# 3. РАБОТА С ПРОФИЛЕМ СТУДЕНТА (НОВОЕ!)
# ==========================================
def get_or_create_student(full_name, group_name):
    """Находит студента в БД или создаёт нового. Возвращает (student_id, profile_dict)."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Ищем существующего
        cur.execute(
            "SELECT id, profile_json FROM students WHERE full_name = %s AND group_name = %s",
            (full_name, group_name)
        )
        row = cur.fetchone()
        if row:
            student_id, profile = row
            profile = profile if isinstance(profile, dict) else {}
        else:
            # Создаём нового
            cur.execute(
                "INSERT INTO students (full_name, group_name) VALUES (%s, %s) RETURNING id",
                (full_name, group_name)
            )
            student_id = cur.fetchone()[0]
            profile = {}
            conn.commit()
        return student_id, profile
    finally:
        cur.close()
        conn.close()

def save_student_profile(student_id, profile_notes):
    """Сохраняет profile_notes из оценки в профиль студента."""
    if not profile_notes:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Загружаем текущий профиль
        cur.execute("SELECT profile_json FROM students WHERE id = %s", (student_id,))
        row = cur.fetchone()
        current_profile = row[0] if row and row[0] else {}
        if isinstance(current_profile, str):
            current_profile = json.loads(current_profile)
        
        # Обновляем/дополняем профиль новыми заметками
        current_profile.update(profile_notes)
        
        cur.execute(
            "UPDATE students SET profile_json = %s WHERE id = %s",
            (json.dumps(current_profile, ensure_ascii=False), student_id)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def format_profile_for_prompt(profile):
    """Форматирует профиль в текст для промпта."""
    if not profile:
        return "У студента нет предыдущих заданий — это первая попытка."
    lines = ["Previous profile notes from past tasks:"]
    if profile.get("recurring_errors"):
        lines.append("- Recurring errors: " + "; ".join(profile["recurring_errors"]))
    if profile.get("resolved_since_last"):
        lines.append("- Resolved since last task: " + "; ".join(profile["resolved_since_last"]))
    if profile.get("estimated_level"):
        lines.append("- Estimated level: " + profile["estimated_level"])
    if profile.get("vocabulary_gaps"):
        lines.append("- Vocabulary gaps: " + "; ".join(profile["vocabulary_gaps"]))
    return "\n".join(lines)

# ==========================================
# 4. ВХОД
# ==========================================
if "student_name" not in st.session_state and "admin_auth" not in st.session_state:
    st.title("🎓 Добро пожаловать в AI Tutor Platform")
    st.markdown("Выберите режим входа:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎓 Я студент", use_container_width=True):
            st.session_state.mode = "student"
            st.rerun()
    with col2:
        if st.button("👩‍🏫 Я преподаватель", use_container_width=True):
            st.session_state.mode = "admin"
            st.rerun()

# ==========================================
# 5. РЕЖИМ СТУДЕНТА
# ==========================================
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
                fio_clean = fio.strip()
                group_clean = group.strip()
                st.session_state.student_name = fio_clean
                st.session_state.student_group = group_clean
                # Получаем или создаём студента + загружаем профиль
                student_id, profile = get_or_create_student(fio_clean, group_clean)
                st.session_state.student_id = student_id
                st.session_state.student_profile = profile
                # Инициализация сессии задания
                st.session_state.current_step = 0
                st.session_state.answers = {}
                st.session_state.flags = []
                st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                st.session_state.session_id = None  # будет создан при первом ответе
                st.rerun()

elif st.session_state.get("mode") == "student" and "student_name" in st.session_state:
    with st.sidebar:
        st.success(f"👤 {st.session_state.student_name}")
        st.caption(f"Группа: {st.session_state.student_group}")
        # Показываем профиль, если есть
        profile = st.session_state.get("student_profile", {})
        if profile:
            level = profile.get("estimated_level", "—")
            st.caption(f"Уровень: {level}")
        if st.button("Выйти"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Загрузка YAML
    TASK_FILE_PATH = "tasks/tech_habits.yaml"
    try:
        with open(TASK_FILE_PATH, "r", encoding="utf-8") as f:
            task_data = yaml.safe_load(f)
        steps = task_data.get("steps", [])
        question_steps = [s for s in steps if s.get("type") == "question"]
        settings = task_data.get("settings", {}).get("answers", {})
        min_words = settings.get("min_words", 15)
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
            flags, warnings = validate_answer(user_answer, min_words=min_words)
            if warnings:
                for w in warnings:
                    st.warning(w)
                if st.button("Всё равно отправить", key="force_send"):
                    st.session_state.flags.extend(flags)
                    st.session_state.answers[current_q["id"]] = user_answer
                    st.session_state.current_step = step_idx + 1
                    
                    # ИСПРАВЛЕНИЕ БАГА 1: INSERT только при первом ответе, UPDATE при последующих
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        if st.session_state.session_id is None:
                            cur.execute("""
                                INSERT INTO sessions (full_name, group_name, task_id, current_step, answers, flags)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                RETURNING id
                            """, (st.session_state.student_name, st.session_state.student_group, 
                                  task_data["meta"]["id"], step_idx + 1, 
                                  json.dumps(st.session_state.answers, ensure_ascii=False),
                                  json.dumps(st.session_state.flags, ensure_ascii=False)))
                            st.session_state.session_id = cur.fetchone()[0]
                        else:
                            cur.execute("""
                                UPDATE sessions 
                                SET current_step = %s, answers = %s, flags = %s
                                WHERE id = %s
                            """, (step_idx + 1, 
                                  json.dumps(st.session_state.answers, ensure_ascii=False),
                                  json.dumps(st.session_state.flags, ensure_ascii=False),
                                  st.session_state.session_id))
                        conn.commit()
                        cur.close()
                        conn.close()
                    except Exception as e:
                        st.error(f"Ошибка БД: {e}")
                    st.rerun()
            else:
                st.session_state.answers[current_q["id"]] = user_answer
                st.session_state.current_step = step_idx + 1
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    if st.session_state.session_id is None:
                        cur.execute("""
                            INSERT INTO sessions (full_name, group_name, task_id, current_step, answers)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id
                        """, (st.session_state.student_name, st.session_state.student_group, 
                              task_data["meta"]["id"], step_idx + 1, 
                              json.dumps(st.session_state.answers, ensure_ascii=False)))
                        st.session_state.session_id = cur.fetchone()[0]
                    else:
                        cur.execute("""
                            UPDATE sessions 
                            SET current_step = %s, answers = %s
                            WHERE id = %s
                        """, (step_idx + 1, 
                              json.dumps(st.session_state.answers, ensure_ascii=False),
                              st.session_state.session_id))
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    st.error(f"Ошибка БД: {e}")
                st.rerun()
    else:
        # ЗАВЕРШЕНИЕ И ОЦЕНКА
        st.subheader("Отлично! Вы ответили на все вопросы.")
        answers_text = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(st.session_state.answers.values())])
        
        if "grade_data" not in st.session_state:
            st.markdown("Нейросеть анализирует ваши ответы... ")
            # Формируем блок профиля для промпта
            profile_block = format_profile_for_prompt(st.session_state.get("student_profile", {}))
            grade_prompt_template = task_data.get("prompts", {}).get("grade", "")
            grade_prompt = grade_prompt_template.format(
                answers_numbered=answers_text,
                profile_block=profile_block
            )
            try:
                llm_response, tokens = call_llm_with_retry(grade_prompt, max_retries=2, temperature=0.2)
                st.session_state.token_usage["prompt_tokens"] += tokens["prompt_tokens"]
                st.session_state.token_usage["completion_tokens"] += tokens["completion_tokens"]
                st.session_state.token_usage["total_tokens"] += tokens["total_tokens"]
                grade_data = extract_json_with_retry(llm_response, prompt_for_retry=grade_prompt, max_retries=1)
                st.session_state.grade_data = grade_data
            except Exception as e:
                st.session_state.grade_data = {"error": str(e)}
            st.rerun()
        elif "final_report" not in st.session_state:
            grade = st.session_state.grade_data
            if "error" in grade:
                st.error(f"Ошибка LLM: {grade['error']}")
            else:
                st.success("Оценка завершена!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{grade.get('accuracy', 0)}/20")
                col2.metric("Fluency", f"{grade.get('fluency', 0)}/10")
                col3.metric("Total", f"{grade.get('total', 0)}/30")
                strengths = "\n".join([f"• {s}" for s in grade.get("strengths", [])])
                improvements = "\n".join([f"• {imp}" for imp in grade.get("improvements", [])])
                errors = "\n".join([f"• ❌ *{e.get('original', '')}* → ✅ {e.get('corrected', '')}\n  _{e.get('explanation', '')}_" 
                                   for e in grade.get("errors", [])])
                st.markdown(f"**Strengths:**\n{strengths}")
                st.markdown(f"**Areas for Improvement:**\n{improvements}")
                st.markdown(f"**Corrections:**\n{errors}")
                st.markdown("---")
                st.markdown("Всё ли понятно? Нажмите кнопку ниже, чтобы получить итоговый отчет.")
                if st.button("Yes! Получить отчет", type="primary"):
                    st.markdown("Генерируем итоговый отчет... ⏳")
                    report_prompt_template = task_data.get("prompts", {}).get("report", "")
                    report_prompt = report_prompt_template.format(
                        answers_numbered=answers_text,
                        grade_json=json.dumps(grade, ensure_ascii=False),
                        accuracy=grade.get("accuracy", 0),
                        fluency=grade.get("fluency", 0),
                        total=grade.get("total", 0),
                        strengths=strengths,
                        improvements=improvements
                    )
                    try:
                        report_md, tokens = call_llm_with_retry(report_prompt, max_retries=1, temperature=0.2)
                        st.session_state.token_usage["prompt_tokens"] += tokens["prompt_tokens"]
                        st.session_state.token_usage["completion_tokens"] += tokens["completion_tokens"]
                        st.session_state.token_usage["total_tokens"] += tokens["total_tokens"]
                        report_hash = compute_sha256(report_md)
                        st.session_state.final_report = report_md
                        st.session_state.report_hash = report_hash
                    except Exception as e:
                        st.session_state.final_report = f"Ошибка генерации отчета: {e}"
                        st.session_state.report_hash = None
                    st.rerun()
        else:
            st.success("Отчет готов!")
            st.markdown(st.session_state.final_report)
            st.download_button(
                "Скачать отчет (Markdown)",
                st.session_state.final_report,
                file_name=f"report_{st.session_state.student_name.replace(' ', '_')}.md",
                mime="text/markdown"
            )
            # ИСПРАВЛЕНИЕ БАГА 2: добавлен answers в UPDATE + сохранение профиля
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("""
                    UPDATE sessions 
                    SET status = 'completed', 
                        grade_json = %s, 
                        report_hash = %s,
                        token_usage = %s,
                        answers = %s,
                        completed_at = NOW()
                    WHERE id = %s
                """, (
                    json.dumps(st.session_state.grade_data, ensure_ascii=False),
                    st.session_state.report_hash,
                    st.session_state.token_usage["total_tokens"],
                    json.dumps(st.session_state.answers, ensure_ascii=False),
                    st.session_state.session_id
                ))
                conn.commit()
                cur.close()
                conn.close()
                # СОХРАНЕНИЕ ПРОФИЛЯ для будущих заданий
                profile_notes = st.session_state.grade_data.get("profile_notes", {})
                if profile_notes:
                    save_student_profile(st.session_state.student_id, profile_notes)
                    st.session_state.student_profile.update(profile_notes)
                st.success("Результаты сохранены! Профиль обновлён для будущих заданий.")
            except Exception as e:
                st.error(f"Ошибка обновления БД: {e}")

# ==========================================
# 6. РЕЖИМ АДМИНА (ПРЕПОДАВАТЕЛЬ)
# ==========================================
elif st.session_state.get("mode") == "admin":
    if "admin_auth" not in st.session_state:
        st.subheader("🔒 Вход для преподавателя")
        pwd = st.text_input("Введите пароль администратора", type="password")
        if st.button("Войти"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_auth = True
                st.rerun()
            else:
                st.error("Неверный пароль")
    else:
        st.title("👩‍ Панель преподавателя (Дашборд)")
        if st.button("Выйти из админки"):
            del st.session_state.admin_auth
            st.rerun()
        try:
            conn = get_db_connection()
            df = pd.read_sql("""
                SELECT id, full_name, group_name, task_id, 
                       (grade_json->>'accuracy')::int as accuracy,
                       (grade_json->>'fluency')::int as fluency,
                       (grade_json->>'total')::int as total,
                       status, flags, token_usage, report_hash, answers, created_at, completed_at
                FROM sessions ORDER BY group_name, full_name, created_at
            """, conn)
            conn.close()
            if df.empty:
                st.info("Пока нет данных от студентов.")
            else:
                st.subheader("📊 Сводная таблица (Pivot)")
                pivot = df.pivot_table(index=['full_name', 'group_name'], columns='task_id', values='total', aggfunc='first')
                st.dataframe(pivot.fillna("—"), use_container_width=True)
                st.markdown("---")
                st.subheader("🔍 Детальный просмотр сессий")
                for idx, row in df.iterrows():
                    with st.expander(f"{row['full_name']} ({row['group_name']}) - {row['task_id']} - {row['status']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{row['accuracy']}/20")
                            st.metric("Fluency", f"{row['fluency']}/10")
                            st.metric("Total", f"{row['total']}/30")
                        with col2:
                            st.write(f"**Статус:** {row['status']}")
                            st.write(f"**Токены:** {row['token_usage']}")
                            st.write(f"**Флаги:** {row['flags']}")
                            st.write(f"**Хеш отчета:** `{row['report_hash'][:16]}...`" if row['report_hash'] else "Нет хеша")
                        st.write(f"**Начато:** {row['created_at']}")
                        st.write(f"**Завершено:** {row['completed_at']}")
                        # ИСПРАВЛЕНИЕ БАГА 3: показ сырых ответов студента
                        st.markdown("**📝 Сырые ответы студента:**")
                        try:
                            answers = json.loads(row['answers']) if row['answers'] else {}
                            for q_id, answer in answers.items():
                                st.markdown(f"**{q_id}:** {answer}")
                        except:
                            st.write("Не удалось загрузить ответы")
                st.markdown("---")
                st.subheader(" Экспорт в Excel (Плоская таблица)")
                st.dataframe(df.drop(columns=['answers']), use_container_width=True)
                csv = df.drop(columns=['answers']).to_csv(index=False).encode('utf-8')
                st.download_button("Скачать CSV/Excel", csv, "student_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Ошибка подключения к БД: {e}")
