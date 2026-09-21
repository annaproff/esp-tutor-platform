import streamlit as st
import os
import re
import yaml
import json
import hashlib
import pandas as pd
import sqlite3
import uuid
from datetime import datetime
from openai import OpenAI
import time

# ==========================================
# 1. НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ БД (SQLite)
# ==========================================
st.set_page_config(page_title="AI English Tutor", page_icon="🎓", layout="wide")

YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    st.error("Missing environment variables: YANDEX_API_KEY, YANDEX_FOLDER_ID")
    st.stop()

client = OpenAI(
    api_key=YANDEX_API_KEY,
    base_url="https://llm.api.cloud.yandex.net/foundationModels/v1"
)

DB_NAME = "tutor.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            full_name TEXT UNIQUE NOT NULL,
            profile_json TEXT DEFAULT '{}'
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            task_id TEXT NOT NULL,
            current_step INTEGER DEFAULT 0,
            answers TEXT DEFAULT '{}',
            grade_json TEXT,
            report_hash TEXT,
            token_usage INTEGER DEFAULT 0,
            status TEXT DEFAULT 'in_progress',
            flags TEXT DEFAULT '[]',
            completed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Инициализируем БД при старте
init_db()

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def is_valid_russian_name(name):
    return bool(re.match(r"^[А-Яа-яЁё\s-]+$", name)) and len(name.split()) >= 2

def count_tokens_in_response(response):
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
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()

def validate_answer(answer, min_words=15, russian_threshold=0.3):
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
# 3. РАБОТА С ПРОФИЛЕМ СТУДЕНТА (SQLite)
# ==========================================
def get_or_create_student(full_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, profile_json FROM students WHERE full_name = ?", (full_name,))
    row = cursor.fetchone()
    
    if row:
        student_id = row["id"]
        profile = json.loads(row["profile_json"]) if row["profile_json"] else {}
    else:
        student_id = str(uuid.uuid4())
        profile = {}
        cursor.execute(
            "INSERT INTO students (id, full_name, profile_json) VALUES (?, ?, ?)",
            (student_id, full_name, json.dumps(profile, ensure_ascii=False))
        )
        conn.commit()
    
    conn.close()
    return student_id, profile

def save_student_profile(student_id, profile_notes):
    if not profile_notes:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT profile_json FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    
    if row:
        current_profile = json.loads(row["profile_json"]) if row["profile_json"] else {}
        current_profile.update(profile_notes)
        cursor.execute(
            "UPDATE students SET profile_json = ? WHERE id = ?",
            (json.dumps(current_profile, ensure_ascii=False), student_id)
        )
        conn.commit()
    conn.close()

def format_profile_for_prompt(profile):
    if not profile:
        return "У студента нет предыдущих заданий — это первая попытка."
    lines = ["Previous profile notes from past tasks:"]
    for key in ["recurring_errors", "resolved_since_last", "estimated_level", "vocabulary_gaps", "grammar_gaps", "writing_gaps", "academic_vocabulary_gaps"]:
        if profile.get(key):
            val = profile[key]
            if isinstance(val, list):
                lines.append(f"- {key.replace('_', ' ').title()}: " + "; ".join(val))
            else:
                lines.append(f"- {key.replace('_', ' ').title()}: {val}")
    return "\n".join(lines)

def check_attempts(full_name, task_id, max_attempts=1):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) as count FROM sessions WHERE full_name = ? AND task_id = ? AND status = 'completed'",
        (full_name, task_id)
    )
    count = cursor.fetchone()["count"]
    conn.close()
    return count < max_attempts

def format_prompt_with_context(prompt_template, context_dict):
    try:
        return prompt_template.format(**context_dict)
    except KeyError as e:
        st.warning(f"Missing variable in prompt: {e}")
        return prompt_template

# ==========================================
# 4. ОБРАБОТЧИКИ ШАГОВ
# ==========================================
def render_gate_step(step):
    st.markdown(step.get("say", ""))
    buttons = step.get("buttons", ["Start"])
    if st.button(buttons[0], type="primary"):
        return True
    return False

def render_question_step(step, min_words=15):
    st.markdown(f"**{step.get('topic', '')}**")
    st.write(step.get("say"))
    user_answer = st.text_area("Ваш ответ (на английском):", height=150, key=f"answer_{step['id']}")
    
    if st.button("Отправить ответ", type="primary"):
        flags, warnings = validate_answer(user_answer, min_words=min_words)
        if warnings:
            for w in warnings:
                st.warning(w)
            if st.button("Всё равно отправить", key="force_send"):
                return user_answer, flags
        else:
            return user_answer, []
    return None, []

def render_multiple_choice_step(step):
    st.markdown(f"**{step.get('topic', '')}**")
    st.write(step.get("question"))
    options = step.get("options", {})
    option_labels = list(options.keys())
    selected = st.radio("Выберите ответ:", option_labels, key=f"mc_{step['id']}")
    
    if st.button("Отправить ответ", type="primary"):
        return {
            "selected": options[selected],
            "correct": step.get("correct"),
            "is_correct": options[selected] == step.get("correct")
        }
    return None

def render_matching_step(step):
    st.write(step.get("say"))
    correct_mapping = step.get("correct", {})
    terms = list(correct_mapping.keys())
    
    st.markdown("**Термины:**")
    for i, term in enumerate(terms, 1):
        st.write(f"{i}. {term}")
    st.markdown("**Определения:**")
    for letter, definition in zip("ABCDEFGH", list(correct_mapping.values())):
        st.write(f"{letter}. {definition}")
    
    user_mapping = {}
    for term in terms:
        user_mapping[term] = st.selectbox(f"Сопоставьте: {term}", [""] + list("ABCDEFGH"), key=f"match_{step['id']}_{term}")
    
    if st.button("Отправить ответы", type="primary"):
        return {term: {"user_answer": user_mapping[term], "correct_answer": correct_mapping[term], "is_correct": user_mapping[term] == correct_mapping[term]} for term in terms}
    return None

def render_llm_step(step, task_data, answers, profile, student_context):
    prompt_template = task_data.get("prompts", {}).get(step.get("prompt", ""), "")
    context = {
        "student_context": json.dumps(student_context, ensure_ascii=False) if isinstance(student_context, dict) else str(student_context),
        "profile_block": format_profile_for_prompt(profile),
        "answers": json.dumps(answers, ensure_ascii=False),
    }
    for key, value in answers.items():
        context[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    for key, value in student_context.items():
        if key not in context:
            context[key] = str(value)
    
    prompt = format_prompt_with_context(prompt_template, context)
    
    with st.spinner("🧠 Нейросеть анализирует ваш ответ (это может занять 10-20 секунд)..."):
        try:
            response, tokens = call_llm_with_retry(prompt, max_retries=2, temperature=step.get("temperature", 0.2))
            try:
                result = extract_json_with_retry(response, prompt_for_retry=prompt, max_retries=1)
                return result, tokens
            except:
                return {"raw_response": response}, tokens
        except Exception as e:
            return {"error": str(e)}, {"total_tokens": 0}

def render_message_step(step):
    st.markdown(step.get("say", ""))
    buttons = step.get("buttons", ["Continue"])
    for btn in buttons:
        if st.button(btn, key=f"msg_{step['id']}_{btn}"):
            return btn
    return None

# ==========================================
# 5. УНИВЕРСАЛЬНОЕ ОТОБРАЖЕНИЕ МЕТРИК
# ==========================================
def display_grade_metrics(grade, max_score=None):
    metrics = {
        "Accuracy": (grade.get("accuracy"), 20),
        "Fluency": (grade.get("fluency"), 10),
        "MC Score": (grade.get("mc_score"), 10),
        "Task Achievement": (grade.get("task_achievement"), 5),
        "Coherence": (grade.get("coherence"), 5),
        "Lexical Resource": (grade.get("lexical_resource"), 5),
        "Grammar": (grade.get("grammar"), 5),
    }
    total = grade.get("total")
    active_metrics = [(name, val, mx) for name, (val, mx) in metrics.items() if val is not None]
    
    if active_metrics:
        cols = st.columns(len(active_metrics))
        for col, (name, val, mx) in zip(cols, active_metrics):
            col.metric(name, f"{val}/{mx}")
    
    if total is not None:
        max_display = max_score if max_score else 30
        st.metric("Total", f"{total}/{max_display}")

# ==========================================
# 6. ВХОД
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
# 7. РЕЖИМ СТУДЕНТА
# ==========================================
if st.session_state.get("mode") == "student" and "student_name" not in st.session_state:
    st.subheader("Вход для студента")
    with st.form("login_form"):
        fio = st.text_input("Фамилия и Имя (на русском)", placeholder="Иванов Иван")
        submitted = st.form_submit_button("Начать занятие")
        if submitted:
            if not is_valid_russian_name(fio):
                st.error("Пожалуйста, введите Фамилию и Имя на русском языке.")
            else:
                fio_clean = fio.strip()
                st.session_state.student_name = fio_clean
                student_id, profile = get_or_create_student(fio_clean)
                st.session_state.student_id = student_id
                st.session_state.student_profile = profile
                st.session_state.current_step_idx = 0
                st.session_state.answers = {}
                st.session_state.flags = []
                st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                st.session_state.session_id = None
                st.session_state.llm_results = {}
                st.rerun()

elif st.session_state.get("mode") == "student" and "student_name" in st.session_state:
    with st.sidebar:
        st.success(f"👤 {st.session_state.student_name}")
        profile = st.session_state.get("student_profile", {})
        if profile:
            level = profile.get("estimated_level", "—")
            st.caption(f"Уровень: {level}")
        if st.button("Выйти"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    TASKS_DIR = "tasks"
    try:
        task_files = sorted([f for f in os.listdir(TASKS_DIR) if f.endswith('.yaml')])
    except FileNotFoundError:
        st.error(f"Папка {TASKS_DIR} не найдена!")
        st.stop()

    if not task_files:
        st.error("Нет доступных заданий!")
        st.stop()

    if "selected_task" not in st.session_state:
        st.title("Выберите задание:")
        selected_task = st.selectbox("Доступные задания:", task_files, format_func=lambda x: x.replace('.yaml', '').replace('_', ' ').title())
        if st.button("Начать задание"):
            st.session_state.selected_task = selected_task
            st.rerun()
    else:
        TASK_FILE_PATH = os.path.join(TASKS_DIR, st.session_state.selected_task)
        try:
            with open(TASK_FILE_PATH, "r", encoding="utf-8") as f:
                task_data = yaml.safe_load(f)
            steps = task_data.get("steps", [])
            settings = task_data.get("settings", {}).get("answers", {})
            min_words = settings.get("min_words", 15)
            max_attempts = task_data.get("settings", {}).get("attempts", {}).get("max", 1)
            max_score = task_data.get("meta", {}).get("max_score", 30)
        except FileNotFoundError:
            st.error(f"Файл задания не найден: {TASK_FILE_PATH}")
            st.stop()

        task_id = task_data.get("meta", {}).get("id", "unknown")
        if not check_attempts(st.session_state.student_name, task_id, max_attempts):
            st.error(f"Вы уже выполнили это задание (лимит: {max_attempts} попытка).")
            if st.button("Выбрать другое задание"):
                del st.session_state.selected_task
                st.rerun()
            st.stop()

        st.title(f"Задание: {task_data.get('meta', {}).get('title', 'Interview')}")
        st.markdown("---")

        step_idx = st.session_state.get("current_step_idx", 0)
        
        if step_idx < len(steps):
            current_step = steps[step_idx]
            step_type = current_step.get("type", "question")
            
            if step_type == "gate":
                if render_gate_step(current_step):
                    st.session_state.current_step_idx = step_idx + 1
                    st.rerun()
            
            elif step_type == "question":
                answer, flags = render_question_step(current_step, min_words=min_words)
                if answer is not None:
                    st.session_state.answers[current_step["id"]] = answer
                    st.session_state.flags.extend(flags)
                    st.session_state.current_step_idx = step_idx + 1
                    
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        if st.session_state.session_id is None:
                            new_id = str(uuid.uuid4())
                            cursor.execute("""
                                INSERT INTO sessions (id, full_name, task_id, current_step, answers, flags, status)
                                VALUES (?, ?, ?, ?, ?, ?, 'in_progress')
                            """, (new_id, st.session_state.student_name, task_id, step_idx + 1,
                                  json.dumps(st.session_state.answers, ensure_ascii=False),
                                  json.dumps(st.session_state.flags, ensure_ascii=False)))
                            st.session_state.session_id = new_id
                        else:
                            cursor.execute("""
                                UPDATE sessions SET current_step = ?, answers = ?, flags = ? WHERE id = ?
                            """, (step_idx + 1,
                                  json.dumps(st.session_state.answers, ensure_ascii=False),
                                  json.dumps(st.session_state.flags, ensure_ascii=False),
                                  st.session_state.session_id))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        st.error(f"Ошибка БД: {e}")
                    st.rerun()
            
            elif step_type == "multiple_choice":
                result = render_multiple_choice_step(current_step)
                if result is not None:
                    st.session_state.answers[current_step["id"]] = result
                    st.session_state.current_step_idx = step_idx + 1
                    st.rerun()
            
            elif step_type == "matching":
                result = render_matching_step(current_step)
                if result is not None:
                    st.session_state.answers[current_step["id"]] = result
                    st.session_state.current_step_idx = step_idx + 1
                    st.rerun()
            
            elif step_type == "llm":
                student_context = task_data.get("student_context", {})
                result, tokens = render_llm_step(current_step, task_data, st.session_state.answers, st.session_state.get("student_profile", {}), student_context)
                st.session_state.llm_results[current_step["id"]] = result
                st.session_state.token_usage["prompt_tokens"] += tokens.get("prompt_tokens", 0)
                st.session_state.token_usage["completion_tokens"] += tokens.get("completion_tokens", 0)
                st.session_state.token_usage["total_tokens"] += tokens.get("total_tokens", 0)
                st.session_state.current_step_idx = step_idx + 1
                st.rerun()
            
            elif step_type == "message":
                clicked = render_message_step(current_step)
                if clicked:
                    st.session_state.answers[current_step["id"]] = clicked
                    st.session_state.current_step_idx = step_idx + 1
                    st.rerun()
            
            elif step_type == "action":
                st.success("Задание завершено! Формируем отчет...")
                st.session_state.current_step_idx = step_idx + 1
                st.rerun()
        else:
            st.subheader("Отлично! Все шаги пройдены.")
            
            if "final_report" not in st.session_state:
                st.markdown("Генерируем итоговый отчет... ⏳")
                report_prompt_template = task_data.get("prompts", {}).get("report", "")
                student_context = task_data.get("student_context", {})
                
                context = {
                    "student_context": json.dumps(student_context, ensure_ascii=False) if isinstance(student_context, dict) else str(student_context),
                    "profile_block": format_profile_for_prompt(st.session_state.get("student_profile", {})),
                    "answers": json.dumps(st.session_state.answers, ensure_ascii=False),
                }
                for key, value in st.session_state.answers.items():
                    context[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                for key, value in st.session_state.llm_results.items():
                    if isinstance(value, dict):
                        context[key] = json.dumps(value, ensure_ascii=False)
                        for k, v in value.items():
                            if k not in context:
                                context[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    else:
                        context[key] = str(value)
                for key, value in student_context.items():
                    if key not in context:
                        context[key] = str(value)
                
                report_prompt = format_prompt_with_context(report_prompt_template, context)
                
                try:
                    report_md, tokens = call_llm_with_retry(report_prompt, max_retries=1, temperature=0.2)
                    st.session_state.token_usage["prompt_tokens"] += tokens["prompt_tokens"]
                    st.session_state.token_usage["completion_tokens"] += tokens["completion_tokens"]
                    st.session_state.token_usage["total_tokens"] += tokens["total_tokens"]
                    report_hash = compute_sha256(report_md)
                    
                    teacher_meta = f"\n---\n### Teacher Meta (скрыто от студента)\n- **Flags:** {', '.join(st.session_state.flags) if st.session_state.flags else 'none'}\n- **Token usage:** {st.session_state.token_usage['total_tokens']}\n- **Report hash:** {report_hash}\n- **Completed at:** {datetime.now().isoformat()}\n"
                    st.session_state.final_report = report_md + teacher_meta
                    st.session_state.report_hash = report_hash
                except Exception as e:
                    st.session_state.final_report = f"Ошибка генерации отчета: {e}"
                    st.session_state.report_hash = None
                st.rerun()
            else:
                st.success("Отчет готов!")
                report_body = st.session_state.final_report.split("---")[0]
                st.markdown(report_body)
                st.download_button("📥 Скачать отчет (Markdown)", report_body, file_name=f"report_{st.session_state.student_name.replace(' ', '_')}.md", mime="text/markdown")
                
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE sessions SET status = 'completed', grade_json = ?, report_hash = ?, token_usage = ?, answers = ?, completed_at = ? WHERE id = ?
                    """, (
                        json.dumps(st.session_state.llm_results, ensure_ascii=False),
                        st.session_state.report_hash,
                        st.session_state.token_usage["total_tokens"],
                        json.dumps(st.session_state.answers, ensure_ascii=False),
                        datetime.now().isoformat(),
                        st.session_state.session_id
                    ))
                    conn.commit()
                    conn.close()
                    
                    profile_notes = None
                    for result in st.session_state.llm_results.values():
                        if isinstance(result, dict) and "profile_notes" in result:
                            profile_notes = result["profile_notes"]
                            break
                    
                    if profile_notes:
                        save_student_profile(st.session_state.student_id, profile_notes)
                        st.session_state.student_profile.update(profile_notes)
                    st.success("✅ Результаты сохранены! Профиль обновлён.")
                except Exception as e:
                    st.error(f"Ошибка обновления БД: {e}")

# ==========================================
# 8. РЕЖИМ АДМИНА
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
        st.title("👩‍🏫 Панель преподавателя (Дашборд)")
        if st.button("Выйти из админки"):
            del st.session_state.admin_auth
            st.rerun()
        try:
            conn = get_db_connection()
            df = pd.read_sql_query("SELECT * FROM sessions ORDER BY full_name, created_at", conn)
            conn.close()
            
            if df.empty:
                st.info("ℹ️ Пока нет данных от студентов.")
            else:
                # Распаковка JSON полей для Pandas
                if 'grade_json' in df.columns:
                    df['grade_json'] = df['grade_json'].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else x)
                    df['accuracy'] = df['grade_json'].apply(lambda x: x.get('accuracy') if isinstance(x, dict) else None)
                    df['fluency'] = df['grade_json'].apply(lambda x: x.get('fluency') if isinstance(x, dict) else None)
                    df['mc_score'] = df['grade_json'].apply(lambda x: x.get('mc_score') if isinstance(x, dict) else None)
                    df['task_achievement'] = df['grade_json'].apply(lambda x: x.get('task_achievement') if isinstance(x, dict) else None)
                    df['coherence'] = df['grade_json'].apply(lambda x: x.get('coherence') if isinstance(x, dict) else None)
                    df['lexical_resource'] = df['grade_json'].apply(lambda x: x.get('lexical_resource') if isinstance(x, dict) else None)
                    df['grammar'] = df['grade_json'].apply(lambda x: x.get('grammar') if isinstance(x, dict) else None)
                    df['total'] = df['grade_json'].apply(lambda x: x.get('total') if isinstance(x, dict) else None)
                
                st.subheader("📊 Сводная таблица (Pivot)")
                if 'total' in df.columns and df['total'].notna().any():
                    pivot_values = 'total'
                elif 'mc_score' in df.columns and df['mc_score'].notna().any():
                    pivot_values = 'mc_score'
                else:
                    df['essay_total'] = (df['task_achievement'].fillna(0).astype(float) + df['coherence'].fillna(0).astype(float) + df['lexical_resource'].fillna(0).astype(float) + df['grammar'].fillna(0).astype(float))
                    pivot_values = 'essay_total'
                
                pivot = df.pivot_table(index=['full_name'], columns='task_id', values=pivot_values, aggfunc='first')
                st.dataframe(pivot.fillna("—"), use_container_width=True)
                
                st.markdown("---")
                st.subheader("🔍 Детальный просмотр сессий")
                for idx, row in df.iterrows():
                    with st.expander(f"{row.get('full_name')} - {row.get('task_id')} - {row.get('status')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            if pd.notna(row.get('accuracy')): st.metric("Accuracy", f"{row['accuracy']}/20")
                            if pd.notna(row.get('fluency')): st.metric("Fluency", f"{row['fluency']}/10")
                            if pd.notna(row.get('mc_score')): st.metric("MC Score", f"{row['mc_score']}/10")
                            if pd.notna(row.get('total')): st.metric("Total", f"{row['total']}")
                        with col2:
                            st.write(f"**Статус:** {row.get('status')}")
                            st.write(f"**Токены:** {row.get('token_usage')}")
                            st.write(f"**Хеш:** `{row.get('report_hash')[:16]}...`" if pd.notna(row.get('report_hash')) else "Нет хеша")
                        
                        st.markdown("**📝 Сырые ответы:**")
                        try:
                            answers = json.loads(row['answers']) if pd.notna(row['answers']) and isinstance(row['answers'], str) else row['answers']
                            if isinstance(answers, dict):
                                for q_id, answer in answers.items():
                                    st.markdown(f"**{q_id}:** {answer}")
                        except:
                            st.write("Не удалось загрузить")
                
                st.markdown("---")
                st.subheader("📥 Экспорт в Excel")
                export_df = df.drop(columns=['answers', 'grade_json'], errors='ignore')
                st.dataframe(export_df, use_container_width=True)
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button("Скачать CSV", csv, "student_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Ошибка: {e}")
