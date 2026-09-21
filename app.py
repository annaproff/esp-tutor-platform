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

YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")
SUPABASE_URI = os.environ.get("SUPABASE_URI")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID or not SUPABASE_URI:
    st.error("Missing environment variables. Check YANDEX_API_KEY, YANDEX_FOLDER_ID, SUPABASE_URI.")
    st.stop()

client = OpenAI(
    api_key=YANDEX_API_KEY,
    base_url="https://llm.api.cloud.yandex.net/foundationModels/v1"
)

def get_db_connection():
    parsed = urlparse(SUPABASE_URI)
    return psycopg2.connect(
        dbname=parsed.path[1:],
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port or 5432
    )

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
# 3. РАБОТА С ПРОФИЛЕМ СТУДЕНТА
# ==========================================
def get_or_create_student(full_name, group_name):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, profile_json FROM students WHERE full_name = %s AND group_name = %s",
            (full_name, group_name)
        )
        row = cur.fetchone()
        if row:
            student_id, profile = row
            profile = profile if isinstance(profile, dict) else {}
        else:
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
    if not profile_notes:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT profile_json FROM students WHERE id = %s", (student_id,))
        row = cur.fetchone()
        current_profile = row[0] if row and row[0] else {}
        if isinstance(current_profile, str):
            current_profile = json.loads(current_profile)
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
    if profile.get("grammar_gaps"):
        lines.append("- Grammar gaps: " + "; ".join(profile["grammar_gaps"]))
    if profile.get("writing_gaps"):
        lines.append("- Writing gaps: " + "; ".join(profile["writing_gaps"]))
    if profile.get("academic_vocabulary_gaps"):
        lines.append("- Academic vocabulary gaps: " + "; ".join(profile["academic_vocabulary_gaps"]))
    return "\n".join(lines)

def check_attempts(full_name, group_name, task_id, max_attempts=1):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM sessions WHERE full_name = %s AND group_name = %s AND task_id = %s AND status = 'completed'",
            (full_name, group_name, task_id)
        )
        count = cur.fetchone()[0]
        return count < max_attempts
    finally:
        cur.close()
        conn.close()

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
        selected_value = options[selected]
        correct_value = step.get("correct")
        is_correct = selected_value == correct_value
        return {
            "selected": selected_value,
            "correct": correct_value,
            "is_correct": is_correct
        }
    return None

def render_matching_step(step):
    st.write(step.get("say"))
    
    correct_mapping = step.get("correct", {})
    terms = list(correct_mapping.keys())
    definitions = list(correct_mapping.values())
    
    st.markdown("**Термины:**")
    for i, term in enumerate(terms, 1):
        st.write(f"{i}. {term}")
    
    st.markdown("**Определения:**")
    for letter, definition in zip("ABCDEFGH", definitions):
        st.write(f"{letter}. {definition}")
    
    user_mapping = {}
    for term in terms:
        user_mapping[term] = st.selectbox(
            f"Сопоставьте: {term}",
            [""] + list("ABCDEFGH"),
            key=f"match_{step['id']}_{term}"
        )
    
    if st.button("Отправить ответы", type="primary"):
        results = {}
        for term, user_answer in user_mapping.items():
            correct_answer = correct_mapping[term]
            results[term] = {
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": user_answer == correct_answer
            }
        return results
    return None

def render_llm_step(step, task_data, answers, profile, student_context):
    prompt_template = task_data.get("prompts", {}).get(step.get("prompt", ""), "")
    
    context = {
        "student_context": json.dumps(student_context, ensure_ascii=False) if isinstance(student_context, dict) else str(student_context),
        "profile_block": format_profile_for_prompt(profile),
        "answers": json.dumps(answers, ensure_ascii=False),
    }
    
    for key, value in answers.items():
        if isinstance(value, str):
            context[key] = value
        else:
            context[key] = json.dumps(value, ensure_ascii=False)
    
    for key, value in student_context.items():
        if key not in context:
            context[key] = str(value)
    
    prompt = format_prompt_with_context(prompt_template, context)
    
    st.markdown("Нейросеть анализирует... ⏳")
    try:
        response, tokens = call_llm_with_retry(
            prompt,
            max_retries=2,
            temperature=step.get("temperature", 0.2)
        )
        
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
    st.title(" Добро пожаловать в AI Tutor Platform")
    st.markdown("Выберите режим входа:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Я студент", use_container_width=True):
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
                student_id, profile = get_or_create_student(fio_clean, group_clean)
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
        st.success(f" {st.session_state.student_name}")
        st.caption(f"Группа: {st.session_state.student_group}")
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
        task_files = [f for f in os.listdir(TASKS_DIR) if f.endswith('.yaml')]
    except FileNotFoundError:
        st.error(f"Папка {TASKS_DIR} не найдена!")
        st.stop()

    if not task_files:
        st.error("Нет доступных заданий!")
        st.stop()

    if "selected_task" not in st.session_state:
        st.title("Выберите задание:")
        selected_task = st.selectbox(
            "Доступные задания:",
            task_files,
            format_func=lambda x: x.replace('.yaml', '').replace('_', ' ').title()
        )
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
        if not check_attempts(st.session_state.student_name, st.session_state.student_group, task_id, max_attempts):
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
                        cur = conn.cursor()
                        if st.session_state.session_id is None:
                            cur.execute("""
                                INSERT INTO sessions (full_name, group_name, task_id, current_step, answers, flags)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                RETURNING id
                            """, (st.session_state.student_name, st.session_state.student_group,
                                  task_id, step_idx + 1,
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
                result, tokens = render_llm_step(
                    current_step, task_data, st.session_state.answers,
                    st.session_state.get("student_profile", {}), student_context
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
                    if isinstance(value, str):
                        context[key] = value
                    else:
                        context[key] = json.dumps(value, ensure_ascii=False)
                
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
                    
                    teacher_meta = f"""
---
### Teacher Meta (скрыто от студента)
- **Flags:** {', '.join(st.session_state.flags) if st.session_state.flags else 'none'}
- **Token usage:** {st.session_state.token_usage['total_tokens']}
- **Report hash:** {report_hash}
- **Completed at:** {datetime.now().isoformat()}
"""
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
                st.download_button(
                    "Скачать отчет (Markdown)",
                    report_body,
                    file_name=f"report_{st.session_state.student_name.replace(' ', '_')}.md",
                    mime="text/markdown"
                )
                
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
                        json.dumps(st.session_state.llm_results, ensure_ascii=False),
                        st.session_state.report_hash,
                        st.session_state.token_usage["total_tokens"],
                        json.dumps(st.session_state.answers, ensure_ascii=False),
                        st.session_state.session_id
                    ))
                    conn.commit()
                    cur.close()
                    conn.close()
                    
                    profile_notes = None
                    for result in st.session_state.llm_results.values():
                        if isinstance(result, dict) and "profile_notes" in result:
                            profile_notes = result["profile_notes"]
                            break
                    
                    if profile_notes:
                        save_student_profile(st.session_state.student_id, profile_notes)
                        st.session_state.student_profile.update(profile_notes)
                    st.success("Результаты сохранены! Профиль обновлён для будущих заданий.")
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
                       (grade_json->>'mc_score')::int as mc_score,
                       (grade_json->>'task_achievement')::int as task_achievement,
                       (grade_json->>'coherence')::int as coherence,
                       (grade_json->>'lexical_resource')::int as lexical_resource,
                       (grade_json->>'grammar')::int as grammar,
                       (grade_json->>'total')::int as total,
                       status, flags, token_usage, report_hash, answers, created_at, completed_at
                FROM sessions ORDER BY group_name, full_name, created_at
            """, conn)
            conn.close()
            if df.empty:
                st.info("Пока нет данных от студентов.")
            else:
                st.subheader(" Сводная таблица (Pivot)")
                if 'total' in df.columns and df['total'].notna().any():
                    pivot_values = 'total'
                elif 'mc_score' in df.columns and df['mc_score'].notna().any():
                    pivot_values = 'mc_score'
                else:
                    df['essay_total'] = (
                        df['task_achievement'].fillna(0).astype(int) +
                        df['coherence'].fillna(0).astype(int) +
                        df['lexical_resource'].fillna(0).astype(int) +
                        df['grammar'].fillna(0).astype(int)
                    )
                    pivot_values = 'essay_total'
                
                pivot = df.pivot_table(index=['full_name', 'group_name'], columns='task_id', values=pivot_values, aggfunc='first')
                st.dataframe(pivot.fillna("—"), use_container_width=True)
                
                st.markdown("---")
                st.subheader("🔍 Детальный просмотр сессий")
                for idx, row in df.iterrows():
                    with st.expander(f"{row['full_name']} ({row['group_name']}) - {row['task_id']} - {row['status']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            if row['accuracy'] is not None:
                                st.metric("Accuracy", f"{row['accuracy']}/20")
                            if row['fluency'] is not None:
                                st.metric("Fluency", f"{row['fluency']}/10")
                            if row['mc_score'] is not None:
                                st.metric("MC Score", f"{row['mc_score']}/10")
                            if row['task_achievement'] is not None:
                                st.metric("Task Achievement", f"{row['task_achievement']}/5")
                            if row['coherence'] is not None:
                                st.metric("Coherence", f"{row['coherence']}/5")
                            if row['lexical_resource'] is not None:
                                st.metric("Lexical Resource", f"{row['lexical_resource']}/5")
                            if row['grammar'] is not None:
                                st.metric("Grammar", f"{row['grammar']}/5")
                            if row['total'] is not None:
                                st.metric("Total", f"{row['total']}")
                        with col2:
                            st.write(f"**Статус:** {row['status']}")
                            st.write(f"**Токены:** {row['token_usage']}")
                            st.write(f"**Флаги:** {row['flags']}")
                            st.write(f"**Хеш отчета:** `{row['report_hash'][:16]}...`" if row['report_hash'] else "Нет хеша")
                        st.write(f"**Начато:** {row['created_at']}")
                        st.write(f"**Завершено:** {row['completed_at']}")
                        st.markdown("**📝 Сырые ответы студента:**")
                        try:
                            answers = json.loads(row['answers']) if row['answers'] else {}
                            for q_id, answer in answers.items():
                                st.markdown(f"**{q_id}:** {answer}")
                        except:
                            st.write("Не удалось загрузить ответы")
                st.markdown("---")
                st.subheader("📥 Экспорт в Excel (Плоская таблица)")
                st.dataframe(df.drop(columns=['answers']), use_container_width=True)
                csv = df.drop(columns=['answers']).to_csv(index=False).encode('utf-8')
                st.download_button("Скачать CSV/Excel", csv, "student_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Ошибка подключения к БД: {e}")
