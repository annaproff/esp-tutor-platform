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

yandex_client = OpenAI(
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

def call_llm_with_retry(prompt, max_retries=2, temperature=0.2, system_prompt=None):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "text": system_prompt})
            messages.append({"role": "user", "text": prompt})
            response = yandex_client.chat.completions.create(
                model=f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
                messages=messages,
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

def call_llm_with_messages(messages, temperature=0.3):
    """Вызов LLM с полной историей сообщений (для чата)."""
    response = yandex_client.chat.completions.create(
        model=f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
        messages=messages,
        temperature=temperature
    )
    content = response.choices[0].message.content
    tokens = count_tokens_in_response(response)
    return content, tokens

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
                retry_prompt = prompt_for_retry + "\n\nIMPORTANT: return ONLY valid JSON, no explanations."
                text, _ = call_llm_with_retry(retry_prompt, max_retries=0, temperature=0.0)
            else:
                raise ValueError(f"JSON parse error: {e}")
    return None

def compute_sha256(content):
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()

def validate_answer(answer, min_words=1, russian_threshold=0.3):
    flags = []
    if len(answer.split()) < min_words:
        flags.append("too_short_accepted")
    cyrillic_chars = len(re.findall(r'[а-яА-Я]', answer))
    if cyrillic_chars > len(answer) * russian_threshold:
        flags.append("russian_accepted")
    return flags, []

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
        return "This is the student's first task — no previous profile notes."
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

def load_template_and_unit(template_file, unit_config_file):
    with open(template_file, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)
    with open(unit_config_file, "r", encoding="utf-8") as f:
        unit_config = json.load(f)
    
    template['meta']['id'] = template['meta']['id'].replace('{{UNIT_NUMBER}}', unit_config['unit_number'])
    template['meta']['title'] = template['meta']['title'].replace('{{UNIT_TITLE}}', unit_config['title'])
    template['student_context']['topic'] = template['student_context']['topic'].replace('{{UNIT_TOPIC}}', unit_config['topic'])
    
    replacements = {
        '{{UNIT_TITLE}}': unit_config['title'],
        '{{UNIT_TOPIC}}': unit_config['topic'],
        '{{UNIT_GRAMMAR}}': unit_config.get('grammar', ''),
        '{{SECTION_1_TITLE}}': unit_config.get('section_1_title', 'Grammar'),
        '{{SECTION_2_TITLE}}': unit_config.get('section_2_title', 'Academic Writing'),
    }
    
    def replace_in_obj(obj):
        if isinstance(obj, str):
            for k, v in replacements.items():
                obj = obj.replace(k, v)
            return obj
        elif isinstance(obj, dict):
            return {k: replace_in_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_in_obj(item) for item in obj]
        return obj
    
    template = replace_in_obj(template)
    return template, unit_config

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

def render_question_step(step, min_words=1):
    st.markdown(f"**{step.get('topic', '')}**")
    st.write(step.get("say"))
    
    user_answer = None
    flags = []
    
    text_input = st.text_area("Your answer (in English):", height=150, key=f"answer_{step['id']}")
    if st.button("Submit answer", type="primary", key=f"submit_{step['id']}"):
        if text_input and text_input.strip():
            flags, _ = validate_answer(text_input, min_words=min_words)
            user_answer = text_input
        else:
            st.warning("Please enter some text before submitting.")
    
    if user_answer is not None:
        return user_answer, flags
    return None, []

def render_multiple_choice_step(step):
    st.markdown(f"**{step.get('topic', '')}**")
    st.write(step.get("question"))
    options = step.get("options", {})
    display_labels = [f"{k}: {v}" for k, v in options.items()]
    selected_label = st.radio("Choose your answer:", display_labels, key=f"mc_{step['id']}")
    if st.button("Submit answer", type="primary", key=f"submit_mc_{step['id']}"):
        selected_key = selected_label.split(":")[0].strip()
        return {
            "selected": options[selected_key],
            "correct": step.get("correct"),
            "is_correct": selected_key == step.get("correct")
        }
    return None

def render_essay_step(step, task_data):
    """Интерактивный чат с LLM для написания эссе по протоколу Task #1.5"""
    st.markdown(f"**{step.get('topic', 'Academic Writing')}**")
    st.info("📝 Interactive essay writing session. The tutor will guide you through 5 iterations. Write 'Start!' to begin.")
    
    chat_key = f"essay_chat_{step['id']}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
        st.session_state[chat_key].append({
            "role": "assistant",
            "content": "Welcome! I'm your academic writing tutor. We'll work on an essay about Negative Sampling through 5 iterations.\n\n**To begin, please type exactly: Start!**"
        })
    
    system_prompt = task_data.get("prompts", {}).get(step.get("prompt", ""), "")
    
    # Display history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Input
    user_input = st.chat_input("Type your message here...", key=f"essay_input_{step['id']}")
    
    if user_input:
        st.session_state[chat_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Build messages for LLM
        messages = [{"role": "system", "text": system_prompt}]
        for m in st.session_state[chat_key]:
            messages.append({"role": m["role"], "text": m["content"]})
        
        with st.spinner("🧠 Tutor is thinking..."):
            try:
                response, tokens = call_llm_with_messages(messages, temperature=0.3)
                st.session_state[chat_key].append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                error_msg = f"⚠️ Error: {e}"
                st.session_state[chat_key].append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
        
        st.rerun()
    
    # Finish button
    st.markdown("---")
    if st.button("✅ Finish essay session and proceed to grading", type="primary", key=f"finish_essay_{step['id']}"):
        return st.session_state[chat_key]
    
    return None

def render_llm_step(step, task_data, answers, profile, student_context, unit_config=None):
    prompt_template = task_data.get("prompts", {}).get(step.get("prompt", ""), "")
    context = {
        "student_context": json.dumps(student_context, ensure_ascii=False) if isinstance(student_context, dict) else str(student_context),
        "profile_block": format_profile_for_prompt(profile),
        "answers": json.dumps(answers, ensure_ascii=False),
    }
    if unit_config:
        context["unit_topic"] = unit_config.get("topic", "")
    
    for key, value in answers.items():
        if isinstance(value, list):
            context[key] = "\n\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in value])
        else:
            context[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    for key, value in student_context.items():
        if key not in context:
            context[key] = str(value)
    
    prompt = format_prompt_with_context(prompt_template, context)
    
    with st.spinner("🧠 AI is analyzing your answer (this may take 10-20 seconds)..."):
        try:
            response, tokens = call_llm_with_retry(prompt, max_retries=2, temperature=step.get("temperature", 0.2))
            try:
                result = extract_json_with_retry(response, prompt_for_retry=prompt, max_retries=1)
                return result, tokens
            except Exception as e:
                return {"error": f"JSON parse failed: {str(e)}", "raw_response": response}, tokens
        except Exception as e:
            return {"error": str(e)}, {"total_tokens": 0}

def render_message_step(step):
    st.markdown(step.get("say", ""))
    buttons = step.get("buttons", ["Continue"])
    for btn in buttons:
        if st.button(btn, key=f"msg_{step['id']}_{btn}"):
            return btn
    return None

def prepare_context_for_report(task_data, answers, llm_results, student_context, profile):
    """Подготовка контекста с плоскими ключами для str.format()"""
    context = {
        "student_context": json.dumps(student_context, ensure_ascii=False) if isinstance(student_context, dict) else str(student_context),
        "profile_block": format_profile_for_prompt(profile),
        "answers": json.dumps(answers, ensure_ascii=False),
    }
    
    # Answers — плоские ключи
    for key, value in answers.items():
        if isinstance(value, list):
            context[key] = "\n\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in value])
            context[f"{key}_raw"] = json.dumps(value, ensure_ascii=False)
        else:
            context[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    
    # LLM results — плоские ключи с префиксом
    for key, value in llm_results.items():
        if isinstance(value, dict):
            context[key] = json.dumps(value, ensure_ascii=False)
            for k, v in value.items():
                flat_key = f"{key}_{k}"
                if isinstance(v, (dict, list)):
                    context[flat_key] = json.dumps(v, ensure_ascii=False)
                else:
                    context[flat_key] = str(v)
        elif isinstance(value, list):
            context[key] = "\n\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in value])
            context[f"{key}_raw"] = json.dumps(value, ensure_ascii=False)
        else:
            context[key] = str(value)
    
    for key, value in student_context.items():
        if key not in context:
            context[key] = str(value)
    
    return context

# ==================== MAIN APP ====================

if "student_name" not in st.session_state and "admin_auth" not in st.session_state:
    st.title("🎓 Welcome to AI Tutor Platform")
    st.markdown("Choose your mode:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 I'm a student", use_container_width=True):
            st.session_state.mode = "student"
            st.rerun()
    with col2:
        if st.button("‍🏫 I'm a teacher", use_container_width=True):
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
            task_data, unit_config = load_template_and_unit(TEMPLATE_FILE, UNIT_FILE_PATH)
            steps = task_data.get("steps", [])
            settings = task_data.get("settings", {}).get("answers", {})
            min_words = settings.get("min_words", 1)
            max_attempts = task_data.get("settings", {}).get("attempts", {}).get("max", 1)
            max_score = task_data.get("meta", {}).get("max_score", 50)
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
            
            elif step_type == "essay":
                result = render_essay_step(current_step, task_data)
                if result is not None:
                    save_as_key = current_step.get("save_as", current_step["id"])
                    st.session_state.answers[current_step["id"]] = result
                    st.session_state.llm_results[save_as_key] = result
                    st.session_state.current_step_idx = step_idx + 1
                    st.rerun()
            
            elif step_type == "llm":
                student_context = task_data.get("student_context", {})
                result, tokens = render_llm_step(
                    current_step, 
                    task_data, 
                    st.session_state.answers, 
                    st.session_state.get("student_profile", {}), 
                    student_context,
                    unit_config
                )
                
                save_as_key = current_step.get("save_as", current_step["id"])
                st.session_state.llm_results[save_as_key] = result
                
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
                st.success("Unit completed! Generating report...")
                st.session_state.current_step_idx = step_idx + 1
                st.rerun()
        
        else:
            st.subheader("Excellent! All steps completed.")
            if "final_report" not in st.session_state:
                st.markdown("Generating final report... ⏳")
                report_prompt_template = task_data.get("prompts", {}).get("report", "")
                student_context = task_data.get("student_context", {})
                
                context = prepare_context_for_report(
                    task_data,
                    st.session_state.answers,
                    st.session_state.llm_results,
                    student_context,
                    st.session_state.get("student_profile", {})
                )
                
                # Добавляем вычисляемые поля
                if "final_grade_total" in context:
                    try:
                        essay_total = int(context["final_grade_total"])
                        context["final_grade_total_plus_10"] = str(essay_total + 10)
                    except:
                        context["final_grade_total_plus_10"] = "N/A"
                
                report_prompt = format_prompt_with_context(report_prompt_template, context)
                try:
                    report_md, tokens = call_llm_with_retry(report_prompt, max_retries=1, temperature=0.2)
                    st.session_state.token_usage["prompt_tokens"] += tokens["prompt_tokens"]
                    st.session_state.token_usage["completion_tokens"] += tokens["completion_tokens"]
                    st.session_state.token_usage["total_tokens"] += tokens["total_tokens"]
                    report_hash = compute_sha256(report_md)
                    teacher_meta = f"\n---\n### Teacher Meta (hidden from student)\n- **Flags:** {', '.join(st.session_state.flags) if st.session_state.flags else 'none'}\n- **Token usage:** {st.session_state.token_usage['total_tokens']}\n- **Report hash:** {report_hash}\n- **Completed at:** {datetime.now().isoformat()}\n"
                    st.session_state.final_report = report_md + teacher_meta
                    st.session_state.report_hash = report_hash
                except Exception as e:
                    st.session_state.final_report = f"Report generation error: {e}"
                    st.session_state.report_hash = None
                st.rerun()
            else:
                st.success("Report is ready!")
                report_body = st.session_state.final_report.split("---")[0]
                st.markdown(report_body)
                st.download_button("📥 Download report (Markdown)", report_body, file_name=f"report_{st.session_state.student_name.replace(' ', '_')}.md", mime="text/markdown")
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
                    st.success("✅ Results saved! Profile updated.")
                except Exception as e:
                    st.error(f"Database update error: {e}")

elif st.session_state.get("mode") == "admin":
    if "admin_auth" not in st.session_state:
        st.subheader("🔒 Teacher Login")
        pwd = st.text_input("Enter admin password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_auth = True
                st.rerun()
            else:
                st.error("Invalid password")
    else:
        st.title("👩‍🏫 Teacher Dashboard")
        if st.button("Logout from admin"):
            del st.session_state.admin_auth
            st.rerun()
        try:
            conn = get_db_connection()
            df = pd.read_sql_query("SELECT * FROM sessions ORDER BY full_name, created_at", conn)
            conn.close()
            if df.empty:
                st.info("ℹ️ No student data yet.")
            else:
                if 'grade_json' in df.columns:
                    df['grade_json'] = df['grade_json'].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else x)
                    df['task_achievement'] = df['grade_json'].apply(lambda x: x.get('task_achievement') if isinstance(x, dict) else None)
                    df['coherence'] = df['grade_json'].apply(lambda x: x.get('coherence') if isinstance(x, dict) else None)
                    df['lexical_resource'] = df['grade_json'].apply(lambda x: x.get('lexical_resource') if isinstance(x, dict) else None)
                    df['essay_grammar'] = df['grade_json'].apply(lambda x: x.get('grammar') if isinstance(x, dict) else None)
                    df['total'] = df['grade_json'].apply(lambda x: x.get('total') if isinstance(x, dict) else None)
                st.subheader("📊 Summary Table (Pivot)")
                if 'total' in df.columns and df['total'].notna().any():
                    pivot_values = 'total'
                else:
                    pivot_values = 'task_achievement'
                pivot = df.pivot_table(index=['full_name'], columns='task_id', values=pivot_values, aggfunc='first')
                st.dataframe(pivot.fillna("—"), use_container_width=True)
                st.markdown("---")
                st.subheader("🔍 Detailed Session View")
                for idx, row in df.iterrows():
                    with st.expander(f"{row.get('full_name')} - {row.get('task_id')} - {row.get('status')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            if pd.notna(row.get('task_achievement')): st.metric("Task Achievement", f"{row['task_achievement']}/10")
                            if pd.notna(row.get('coherence')): st.metric("Coherence", f"{row['coherence']}/10")
                            if pd.notna(row.get('lexical_resource')): st.metric("Lexical Resource", f"{row['lexical_resource']}/10")
                            if pd.notna(row.get('essay_grammar')): st.metric("Essay Grammar", f"{row['essay_grammar']}/10")
                            if pd.notna(row.get('total')): st.metric("Essay Total", f"{row['total']}/40")
                        with col2:
                            st.write(f"**Status:** {row.get('status')}")
                            st.write(f"**Tokens:** {row.get('token_usage')}")
                            st.write(f"**Hash:** `{row.get('report_hash')[:16]}...`" if pd.notna(row.get('report_hash')) else "No hash")
                        st.markdown("**Raw answers:**")
                        try:
                            answers = json.loads(row['answers']) if pd.notna(row['answers']) and isinstance(row['answers'], str) else row['answers']
                            if isinstance(answers, dict):
                                for q_id, answer in answers.items():
                                    if isinstance(answer, list):
                                        st.markdown(f"**{q_id} (essay session, {len(answer)} messages):**")
                                        with st.expander("View chat transcript"):
                                            for msg in answer:
                                                st.markdown(f"**{msg['role']}:** {msg['content']}")
                                    else:
                                        st.markdown(f"**{q_id}:** {answer}")
                        except:
                            st.write("Failed to load")
                st.markdown("---")
                st.subheader("📥 Export to CSV")
                export_df = df.drop(columns=['answers', 'grade_json'], errors='ignore')
                st.dataframe(export_df, use_container_width=True)
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "student_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")
