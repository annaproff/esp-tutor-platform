--- public/app.py (原始)


+++ public/app.py (修改后)
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
    project=YANDEX_FOLDER_ID,
    base_url="https://ai.api.cloud.yandex.net/v1"
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

def call_llm_with_retry(prompt, max_retries=2, temperature=0.2, model_name="yandexgpt-5-lite"):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=f"gpt://{YANDEX_FOLDER_ID}/{model_name}",
                messages=[{"role": "user", "content": prompt}],
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

def load_template_and_unit(template_file, unit_config_file):
    with open(template_file, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)
    with open(unit_config_file, "r", encoding="utf-8") as f:
        unit_config = json.load(f)

    template['meta']['id'] = template['meta']['id'].replace('{{UNIT_NUMBER}}', unit_config['unit_number'])
    template['meta']['title'] = template['meta']['title'].replace('{{UNIT_TITLE}}', unit_config['title'])
    template['meta']['student_context']['topic'] = template['meta']['student_context']['topic'].replace('{{UNIT_TOPIC}}', unit_config['topic'])

    if 'templates' in template and 'final_md' in template['templates']:
        template['templates']['final_md'] = template['templates']['final_md'].replace('{{UNIT_TITLE}}', unit_config['title'])

    return template, unit_config

def format_prompt_with_context(prompt_template, context_dict):
    try:
        return prompt_template.format(**context_dict)
    except KeyError as e:
        st.warning(f"Missing variable in prompt: {e}")
        return prompt_template

def flatten_answers(answers):
    flat = {}
    for key, value in answers.items():
        if isinstance(value, dict):
            for k, v in value.items():
                flat[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        else:
            flat[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return flat

def flatten_llm_results(llm_results):
    flat = {}
    for key, value in llm_results.items():
        if isinstance(value, dict):
            flat[key] = json.dumps(value, ensure_ascii=False)
            for k, v in value.items():
                flat[f"{key}_{k}"] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                if isinstance(v, (list, dict)):
                    flat[f"{key}_{k}"] = json.dumps(v, ensure_ascii=False)
        else:
            flat[key] = str(value)
    return flat

def render_gate_step(step):
    st.markdown(step.get("say", ""))
    buttons = step.get("buttons", ["Start"])
    if st.button(buttons[0], type="primary"):
        return True
    return False

def render_question_step(step, min_words=15):
    st.markdown(f"**{step.get('topic', '')}**")
    st.write(step.get("say"))
    user_answer = st.text_area("Your answer (in English):", height=150, key=f"answer_{step['id']}")
    if st.button("Submit answer", type="primary"):
        flags, warnings = validate_answer(user_answer, min_words=min_words)
        if warnings:
            for w in warnings:
                st.warning(w)
            if st.button("Submit anyway", key=f"force_send_{step['id']}"):
                return user_answer, flags
        else:
            return user_answer, []
    return None, []

def render_multiple_choice_step(step):
    st.markdown(f"**{step.get('topic', '')}**")
    st.write(step.get("question"))
    options = step.get("options", {})
    option_labels = list(options.keys())
    selected = st.radio("Choose your answer:", option_labels, key=f"mc_{step['id']}")
    if st.button("Submit answer", type="primary"):
        return {
            "selected": options[selected],
            "correct": step.get("correct"),
            "is_correct": options[selected] == step.get("correct")
        }
    return None

def render_llm_step(step, task_data, answers, profile, student_context, unit_config=None):
    prompt_template = task_data.get("prompts", {}).get(step.get("prompt", ""), "")
    model_name = step.get("model", "yandexgpt-5-lite")

    context = {
        "student_context": json.dumps(student_context, ensure_ascii=False) if isinstance(student_context, dict) else str(student_context),
        "profile_block": format_profile_for_prompt(profile),
    }

    if unit_config:
        context["unit_config"] = json.dumps(unit_config, ensure_ascii=False)

    flat_answers = flatten_answers(answers)
    context.update(flat_answers)
    context["answers"] = json.dumps(answers, ensure_ascii=False)
    context["answers_numbered"] = "\n".join([f"{i+1}. {v}" for i, v in enumerate(flat_answers.values())])

    for key, value in student_context.items():
        if key not in context:
            context[key] = str(value)

    llm_results = st.session_state.get("llm_results", {})
    flat_llm = flatten_llm_results(llm_results)
    context.update(flat_llm)

    prompt = format_prompt_with_context(prompt_template, context)

    with st.spinner("🧠 AI is analyzing your answer (this may take 10-20 seconds)..."):
        try:
            response, tokens = call_llm_with_retry(
                prompt,
                max_retries=2,
                temperature=step.get("temperature", 0.2),
                model_name=model_name
            )

            output_schema = step.get("output")
            if output_schema and output_schema in task_data.get("schemas", {}):
                try:
                    result = extract_json_with_retry(response, prompt_for_retry=prompt, max_retries=1)
                    st.success("✅ JSON parsed successfully")
                    return result, tokens
                except Exception as json_err:
                    st.error(f"⚠️ JSON parse error: {json_err}")
                    with st.expander("Raw response (click to see)"):
                        st.code(response[:1000])
                    return {"parse_error": str(json_err), "raw_response": response}, tokens
            else:
                st.success("✅ Text response received")
                return {"feedback_text": response}, tokens

        except Exception as e:
            st.error(f"❌ LLM call failed: {e}")
            return {"error": str(e)}, {"total_tokens": 0}

def render_message_step(step):
    st.markdown(step.get("say", ""))
    buttons = step.get("buttons", ["Continue"])
    for btn in buttons:
        if st.button(btn, key=f"msg_{step['id']}_{btn}"):
            return btn
    return None

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
        if st.button("👩‍🏫 I'm a teacher", use_container_width=True):
            st.session_state.mode = "admin"
            st.rerun()

if st.session_state.get("mode") == "student" and "student_name" not in st.session_state:
    st.subheader("👤 Student Login")
    full_name = st.text_input("Enter your full name (in Russian):")
    if st.button("Continue", type="primary"):
        if full_name and is_valid_russian_name(full_name):
            student_id, profile = get_or_create_student(full_name)
            st.session_state.student_name = full_name
            st.session_state.student_id = student_id
            st.session_state.student_profile = profile
            st.session_state.current_step_idx = 0
            st.session_state.answers = {}
            st.session_state.flags = []
            st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            st.session_state.session_id = None
            st.session_state.llm_results = {}
            st.rerun()
        elif full_name:
            st.error("Please enter a valid Russian full name (at least 2 words, Cyrillic letters only).")
        else:
            st.error("Please enter your name.")

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
                    df['total'] = df['grade_json'].apply(lambda x: x.get('total') if isinstance(x, dict) else None)

                st.subheader("Summary Table")
                if 'total' in df.columns and df['total'].notna().any():
                    pivot = df.pivot_table(index=['full_name'], columns='task_id', values='total', aggfunc='first')
                    st.dataframe(pivot.fillna("—"), use_container_width=True)

                st.markdown("---")
                st.subheader("Detailed Session View")
                for idx, row in df.iterrows():
                    with st.expander(f"{row.get('full_name')} - {row.get('task_id')} - {row.get('status')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            if pd.notna(row.get('total')):
                                st.metric("Total", f"{row['total']}")
                        with col2:
                            st.write(f"Status: {row.get('status')}")
                            st.write(f"Tokens: {row.get('token_usage')}")
                            st.write(f"Hash: `{row.get('report_hash')[:16]}...`" if pd.notna(row.get('report_hash')) else "No hash")
                        st.markdown("Raw answers:")
                        try:
                            answers = json.loads(row['answers']) if pd.notna(row['answers']) and isinstance(row['answers'], str) else row['answers']
                            if isinstance(answers, dict):
                                for q_id, answer in answers.items():
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
            min_words = settings.get("min_words", 15)
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

            elif step_type == "llm":
                student_context = task_data.get("meta", {}).get("student_context", {})
                result, tokens = render_llm_step(
                    current_step,
                    task_data,
                    st.session_state.answers,
                    st.session_state.get("student_profile", {}),
                    student_context,
                    unit_config
                )
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
                st.success("✅ Unit completed! Generating feedback...")
                st.session_state.current_step_idx = step_idx + 1
                st.rerun()

        else:
            st.subheader("Excellent! All steps completed.")

            if "final_report" not in st.session_state:
                st.markdown("Generating personalized feedback... ⏳")

                report_data = st.session_state.llm_results.get("report", {})

                if "feedback_text" in report_data:
                    feedback_text = report_data["feedback_text"]
                elif "raw_response" in report_data:
                    feedback_text = report_data["raw_response"]
                elif "text_response" in report_data:
                    feedback_text = report_data["text_response"]
                else:
                    feedback_text = str(report_data)

                report_hash = compute_sha256(feedback_text)
                teacher_meta = f"\n---\n### Teacher Meta (hidden from student)\n- **Flags:** {', '.join(st.session_state.flags) if st.session_state.flags else 'none'}\n- **Token usage:** {st.session_state.token_usage.get('total_tokens', 'N/A')}\n- **Report hash:** {report_hash}\n- **Completed at:** {datetime.now().isoformat()}\n"

                st.session_state.final_report = feedback_text + teacher_meta
                st.session_state.report_hash = report_hash
                st.rerun()
            else:
                st.success("✅ Feedback is ready!")

                report_parts = st.session_state.final_report.split("---\n### Teacher Meta")
                report_body = report_parts[0]
                teacher_meta = report_parts[1] if len(report_parts) > 1 else ""

                st.markdown("### 📋 Your Personalized Feedback")
                st.markdown(report_body)

                st.download_button(
                    label="📥 Download Feedback (Markdown)",
                    data=report_body,
                    file_name=f"feedback_{st.session_state.student_name.replace(' ', '_')}.md",
                    mime="text/markdown"
                )

                if teacher_meta:
                    with st.expander("👩‍🏫 Teacher Meta (hidden from student)"):
                        st.markdown(teacher_meta)

                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE sessions SET
                            status = 'completed',
                            grade_json = ?,
                            report_hash = ?,
                            token_usage = ?,
                            answers = ?,
                            completed_at = ?
                        WHERE id = ?
                    """, (
                        json.dumps(st.session_state.llm_results, ensure_ascii=False),
                        st.session_state.report_hash,
                        st.session_state.token_usage.get("total_tokens", 0),
                        json.dumps(st.session_state.answers, ensure_ascii=False),
                        datetime.now().isoformat(),
                        st.session_state.session_id
                    ))
                    conn.commit()

                    profile_notes = None
                    for result in st.session_state.llm_results.values():
                        if isinstance(result, dict) and "profile_notes" in result:
                            profile_notes = result["profile_notes"]
                            break

                    if profile_notes:
                        save_student_profile(st.session_state.student_id, profile_notes)
                        st.session_state.student_profile.update(profile_notes)

                    conn.close()
                    st.success("✅ Results saved to database! Profile updated.")

                except Exception as e:
                    st.error(f"Database update error: {e}")
