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

init_db()

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
            json_match = re.search(r'{.*}', text, re.DOTALL)
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

# FIX 2: Убрана жесткая валидация, принимаем любые ответы для апробации
def validate_answer(answer, min_words=15, russian_threshold=0.3):
    return [], []

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
    for key in ["recurring_errors", "resolved_since_last", "estimated_level", "vocabulary_gaps", "grammar_gaps"]:
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
        # FIX 2: Возвращаем ответ без проверок
        return user_answer, []
    return None, []

# FIX 1: Исправлен рендеринг вариантов ответов (теперь видно текст)
def render_multiple_choice_step(step):
    st.markdown(f"**{step.get('topic', '')}**")
    st.write(step.get("question"))
    options = step.get("options", {})
    # Формируем красивые лейблы: "A: is deployed"
    display_labels = [f"{k}: {v}" for k, v in options.items()]
    selected_label = st.radio("Выберите ответ:", display_labels, key=f"mc_{step['id']}")
    
    if st.button("Отправить ответ", type="primary"):
        # Извлекаем букву (A, B, C, D) из лейбла
        selected_key = selected_label.split(":")[0].strip()
        return {
            "selected": options[selected_key],
            "correct": step.get("correct"),
            "is_correct": selected_key == step.get("correct")
        }
    return None

# FIX 3: Новый рендерер для динамического теста по лексике
def render_dynamic_mcq_step(step, vocab_context):
    st.markdown(f"**{step.get('topic', '')}**")
    st.write("Based on your answers, here is your personalized vocabulary test. Choose the correct option for each question.")

    mcq_list = vocab_context.get("vocab_mcq", [])
    if not mcq_list:
        st.error("Vocabulary questions were not generated. Please contact the teacher.")
        return None

    answers = {}
    for q in mcq_list:
        st.markdown(f"**{q['id']}. {q['question']}**")
        options = q.get("options", {})
        display_labels = [f"{k}: {v}" for k, v in options.items()]
        selected_label = st.radio(f"Select for Q{q['id']}", display_labels, key=f"dyn_mc_{step['id']}_{q['id']}")
        answers[q['id']] = {
            "selected_label": selected_label,
            "correct": q.get("correct")
        }

    if st.button("Submit Vocabulary Test", type="primary"):
        score = 0
        details = []
        for q in mcq_list:
            user_data = answers[q['id']]
            selected_key = user_data["selected_label"].split(":")[0].strip()
            is_correct = selected_key == q.get("correct")
            if is_correct:
                score += 1
            details.append({
                "id": q['id'],
                "student": selected_key,
                "correct": q.get("correct"),
                "is_correct": is_correct
            })
        
        # Возвращаем в формате, который ожидает остальная часть приложения (как будто это оценка от LLM)
        return {"vocab_score": score, "details": details}
    return None

def render_message_step(step):
    st.markdown(step.get("say", ""))
    buttons = step.get("buttons", ["Continue"])
    for btn in buttons:
        if st.button(btn, key=f"msg_{step['id']}_{btn}"):
            return btn
    return None

# ==================== MAIN APP ====================

if "student_name" not in st.session_state and "admin_auth" not in st.session_state:
    st.title(" Welcome to AI Tutor Platform")
    st.markdown("Choose your mode:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 I'm a student", use_container_width=True):
            st.session_state.mode = "student"
            st.rerun()
    with col2:
        if st.button("👩‍🏫 I'm a teacher", use_container_width=True):
            st.session_state.mode = "admin"
            st.rerun()

if st.session_state.get("mode") == "student" and "student_name" not in st.session_state:
    st.subheader("Student Login")
    with st.form("login_form"):
        fio = st.text_input("Surname and Name (in Russian)", placeholder="Иванов Иван")
        submitted = st.form_submit_button("Start session")
        if submitted:
            if not is_valid_russian_name(fio):
                st.error("Please enter Surname and Name in Russian.")
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
            st.caption(f"Level: {level}")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    TASKS_DIR = "tasks"
    UNITS_DIR = os.path.join(TASKS_DIR, "units")
    TEMPLATE_FILE = os.path.join(TASKS_DIR, "template.yaml")
    
    try:
        unit_files = sorted([f for f in os.listdir(UNITS_DIR) if f.endswith('.json')])
    except FileNotFoundError:
        st.error(f"Units directory not found: {UNITS_DIR}")
        st.stop()
    
    if not unit_files:
        st.error("No available units!")
        st.stop()
    
    if "selected_unit" not in st.session_state:
        st.title("Choose your unit:")
        selected_unit = st.selectbox("Available units:", unit_files, format_func=lambda x: x.replace('.json', '').replace('_', ' ').title())
        if st.button("Start unit"):
            st.session_state.selected_unit = selected_unit
            st.rerun()
    else:
        UNIT_FILE_PATH = os.path.join(UNITS_DIR, st.session_state.selected_unit)
        
        try:
            with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
                task_data = yaml.safe_load(f)
            with open(UNIT_FILE_PATH, "r", encoding="utf-8") as f:
                unit_config = json.load(f)
            
            # Подстановка переменных
            task_data['meta']['id'] = task_data['meta']['id'].replace('{{UNIT_NUMBER}}', unit_config['unit_number'])
            task_data['meta']['title'] = task_data['meta']['title'].replace('{{UNIT_TITLE}}', unit_config['title'])
            task_data['student_context']['topic'] = task_data['student_context']['topic'].replace('{{UNIT_TOPIC}}', unit_config['topic'])
            
            steps = task_data.get("steps", [])
            settings = task_data.get("settings", {}).get("answers", {})
            min_words = settings.get("min_words", 15)
            max_attempts = task_data.get("settings", {}).get("attempts", {}).get("max", 1)
        except FileNotFoundError:
            st.error(f"Template or unit file not found!")
            st.stop()
        
        task_id = task_data.get("meta", {}).get("id", "unknown")
        
        if not check_attempts(st.session_state.student_name, task_id, max_attempts):
            st.error(f"You have already completed this unit (limit: {max_attempts} attempt).")
            if st.button("Choose another unit"):
                del st.session_state.selected_unit
                st.rerun()
            st.stop()
        
        st.title(f"Unit: {task_data.get('meta', {}).get('title', 'Comprehensive Tech English')}")
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
                    # ... (сохранение в БД, как было)
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
                        st.error(f"Database error: {e}")
                    st.rerun()
            
            elif step_type == "multiple_choice":
                result = render_multiple_choice_step(current_step)
                if result is not None:
                    st.session_state.answers[current_step["id"]] = result
                    st.session_state.current_step_idx = step_idx + 1
                    st.rerun()
            
            # FIX 3: Обработка нового типа шага для лексики
            elif step_type == "dynamic_mcq":
                vocab_context = st.session_state.llm_results.get("vocab_context", {})
                result = render_dynamic_mcq_step(current_step, vocab_context)
                if result is not None:
                    st.session_state.answers[current_step["id"]] = result
                    st.session_state.llm_results["vocab_grade"] = result # Сохраняем как оценку
                    st.session_state.current_step_idx = step_idx + 1
                    st.rerun()
            
            elif step_type == "llm":
                student_context = task_data.get("student_context", {})
                # Передаем unit_config в промпт
                context_for_llm = {
                    "student_context": json.dumps(student_context, ensure_ascii=False),
                    "profile_block": format_profile_for_prompt(st.session_state.get("student_profile", {})),
                    "answers": json.dumps(st.session_state.answers, ensure_ascii=False),
                    "unit_topic": unit_config.get("topic", ""),
                    "vocab_seed": ", ".join(unit_config.get("vocab_seed", []))
                }
                
                # Добавляем ответы на вопросы v_q1, v_q2 для промпта генерации
                if current_step.get("id") == "v_generate":
                    context_for_llm["v_q1"] = st.session_state.answers.get("v_q1", "")
                    context_for_llm["v_q2"] = st.session_state.answers.get("v_q2", "")

                prompt_template = task_data.get("prompts", {}).get(current_step.get("prompt", ""), "")
                prompt = format_prompt_with_context(prompt_template, context_for_llm)
                
                with st.spinner(" AI is analyzing..."):
                    try:
                        response, tokens = call_llm_with_retry(prompt, max_retries=2, temperature=current_step.get("temperature", 0.2))
                        try:
                            result = extract_json_with_retry(response, prompt_for_retry=prompt, max_retries=1)
                            st.session_state.llm_results[current_step["id"]] = result
                            st.session_state.token_usage["prompt_tokens"] += tokens.get("prompt_tokens", 0)
                            st.session_state.token_usage["completion_tokens"] += tokens.get("completion_tokens", 0)
                            st.session_state.token_usage["total_tokens"] += tokens.get("total_tokens", 0)
                            st.session_state.current_step_idx = step_idx + 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"JSON Parse Error: {e}")
                            st.write("Raw response:", response)
                    except Exception as e:
                        st.error(f"LLM Error: {e}")
            
            elif step_type == "message":
                clicked = render_message_step(current_step)
                if clicked:
                    st.session_state.answers[current_step["id"]] = clicked
                    st.session_state.current_step_idx = step_idx + 1
                    st.rerun()
            
            elif step_type == "action":
                st.success("Unit completed! Generating report...")
                st.session_state.current_step_idx = step_idx + 1
                st.rerun()
        
        else:
            st.subheader("Excellent! All steps completed.")
            if "final_report" not in st.session_state:
                st.markdown("Generating final report... ⏳")
                # ... (логика генерации финального отчета, как была в твоем коде)
                # Для краткости я не дублирую весь блок генерации отчета, он остается без изменений.
                # Главное, что он берет данные из st.session_state.llm_results и st.session_state.answers.
                st.write("Report generation logic here (unchanged from your working version).")
                st.rerun()
            else:
                st.success("Report is ready!")
                st.markdown(st.session_state.final_report)
                # ... (логика сохранения и скачивания)

elif st.session_state.get("mode") == "admin":
    # ... (код админки без изменений)
    st.write("Admin dashboard code here (unchanged).")
