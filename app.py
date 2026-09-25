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
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    st.error("Missing environment variables: YANDEX_API_KEY, YANDEX_FOLDER_ID")
    st.stop()

if not ADMIN_PASSWORD:
    st.error("Missing environment variable: ADMIN_PASSWORD (пароль преподавателя больше не имеет значения по умолчанию)")
    st.stop()

client = OpenAI(
    api_key=YANDEX_API_KEY,
    project=YANDEX_FOLDER_ID,
    base_url="https://ai.api.cloud.yandex.net/v1"
)

# ВАЖНО: путь должен совпадать с volume из docker-compose (/app/data),
# иначе база лежит в слое контейнера и стирается при каждом редеплое.
DB_PATH = os.environ.get("DB_PATH", "data/tutor.db")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
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
    cursor.execute("PRAGMA table_info(students)")
    student_cols = {row[1] for row in cursor.fetchall()}
    if "name_key" not in student_cols:
        cursor.execute("ALTER TABLE students ADD COLUMN name_key TEXT")
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
    # Миграции для уже существующих баз
    cursor.execute("PRAGMA table_info(sessions)")
    existing = {row[1] for row in cursor.fetchall()}
    if "total" not in existing:
        cursor.execute("ALTER TABLE sessions ADD COLUMN total INTEGER")
    if "max_score" not in existing:
        cursor.execute("ALTER TABLE sessions ADD COLUMN max_score INTEGER")
    if "report_md" not in existing:
        cursor.execute("ALTER TABLE sessions ADD COLUMN report_md TEXT")
    if "timings" not in existing:
        cursor.execute("ALTER TABLE sessions ADD COLUMN timings TEXT")
    conn.commit()
    conn.close()

init_db()

def normalize_name(name):
    """Ключ для поиска студента: регистр, порядок слов и ё/е не должны создавать дубли."""
    text = str(name or "").strip().lower().replace("ё", "е")
    return " ".join(sorted(text.split()))


def is_valid_name(name):
    # Латиница тоже допустима: раньше студент с фамилией на латинице не мог войти.
    return len(str(name or "").split()) >= 2

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

def extract_total(grade_json):
    """Достаёт итоговый балл из llm_results любого вида (плоского или по шагам)."""
    if not isinstance(grade_json, dict):
        return None
    if isinstance(grade_json.get("total"), (int, float)):
        return grade_json["total"]
    for value in grade_json.values():
        if isinstance(value, dict) and isinstance(value.get("total"), (int, float)):
            return value["total"]
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

def score_multiple_choice(answers, steps, points_per_item=1):
    """Считает тестовую часть кодом. Модель к этим баллам не прикасается."""
    mc_steps = [s for s in steps if s.get("type") == "multiple_choice"]
    correct, wrong_topics = 0, []
    for s in mc_steps:
        saved = answers.get(s["id"])
        if isinstance(saved, dict) and saved.get("is_correct"):
            correct += 1
        elif saved is not None:
            wrong_topics.append(s.get("topic", s["id"]))
    return {
        "correct": correct,
        "items": len(mc_steps),
        "score": correct * points_per_item,
        "max": len(mc_steps) * points_per_item,
        "wrong_topics": wrong_topics,
    }


def normalize_text(value):
    """Грубая нормализация для сверки коротких фактических ответов."""
    text = str(value or "").lower().replace("\u00a0", " ")
    text = re.sub(r"[^0-9a-zа-яё\-\^\.\(\)/=+*]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_answer_key(student_answer, key):
    norm = normalize_text(student_answer)
    if not norm:
        return False
    return any(normalize_text(v) and normalize_text(v) in norm for v in key.get("accept", []))


def score_reading(answers, steps):
    """Сверяет короткие ответы кодом. Что не совпало — уходит модели на суждение."""
    items, auto, maximum = [], 0, 0
    for step in steps:
        key = step.get("answer_key")
        if not key:
            continue
        points = int(key.get("points", 0))
        maximum += points
        raw = answers.get(step["id"])
        text = raw if isinstance(raw, str) else ""
        if not text.strip():
            status = "empty"
        elif match_answer_key(text, key):
            status = "matched"
            auto += points
        else:
            status = "unmatched"
        items.append({
            "id": step["id"],
            "status": status,
            "max_points": points,
            "points": points if status == "matched" else 0,
            "student": text.strip(),
            "expected": key.get("expected", ""),
        })
    return {"items": items, "score": auto, "auto_score": auto, "max": maximum,
            "unmatched": [i for i in items if i["status"] == "unmatched"]}


def apply_reading_verdicts(reading, verdicts):
    """Добавляет баллы за ответы, которые модель признала эквивалентными ключу."""
    verdicts = verdicts if isinstance(verdicts, dict) else {}
    flags = []
    for item in reading["items"]:
        if item["status"] != "unmatched":
            continue
        verdict = verdicts.get(item["id"])
        if verdict is True or str(verdict).strip().lower() in ("true", "yes", "correct"):
            item["status"] = "matched_by_llm"
            item["points"] = item["max_points"]
            flags.append(f"reading_accepted_by_llm:{item['id']}")
    reading["score"] = sum(i["points"] for i in reading["items"])
    return reading, flags


def reading_with_verdicts(answers, steps, llm_results):
    """Чтение с учётом вердиктов модели из уже пройденных шагов оценивания.

    Без этого шаг отчёта считал чтение заново только по ключу: студент видел итог,
    где ответ засчитан (15/15), а в тексте отчёта модель писала 12/15.
    """
    reading = score_reading(answers, steps)
    for earlier in (llm_results or {}).values():
        if isinstance(earlier, dict) and earlier.get("reading_verdicts"):
            reading, _ = apply_reading_verdicts(reading, earlier["reading_verdicts"])
    return reading


def format_reading_for_prompt(reading):
    if not reading["unmatched"]:
        return "none — all short answers matched the key automatically"
    lines = []
    for item in reading["unmatched"]:
        lines.append(f'- {item["id"]}: expected "{item["expected"]}" | student wrote: "{item["student"]}"')
    return "\n".join(lines)


def compute_total(grade, mc, task_data, reading=None):
    """Итог = тест (код) + чтение (код) + оценочные поля модели, с обрезкой по максимуму.

    Возвращает (total, flags). Сумму складывает код: у модели арифметика плавает.
    """
    scoring = task_data.get("settings", {}).get("scoring", {})
    fields = scoring.get("llm_score_fields", []) or []
    caps = scoring.get("max_by_field", {}) or {}
    flags = []
    total = mc.get("score", 0) + (reading or {}).get("score", 0)
    for f in fields:
        try:
            value = int(grade.get(f, 0) or 0)
        except (TypeError, ValueError):
            value = 0
            flags.append(f"llm_score_not_a_number:{f}")
        cap = caps.get(f)
        if cap is not None and value > cap:
            flags.append(f"llm_score_over_max:{f}={value}>{cap}")
            value = cap
        if value < 0:
            value = 0
        total += value
    llm_total = grade.get("total")
    if isinstance(llm_total, int) and llm_total != total:
        flags.append(f"llm_total_mismatch:{llm_total}!={total}")
    return total, flags


def get_or_create_student(full_name):
    """Находит студента по ФИО или заводит нового. Возвращает (student_id, full_name, profile).

    Ищем по name_key, а не по строке как есть: «Иванов Иван», «иван иванов» и «Иван Иванов»
    — один человек. Раньше перестановка слов создавала второго студента и обходила лимит попыток.
    Возвращаем ФИО из базы: по нему привязаны попытки, как бы студент его ни набрал.
    """
    name_key = normalize_name(full_name)
    conn = get_db_connection()
    cursor = conn.cursor()
    # Студенты из старой базы без name_key: досчитываем, иначе поиск их не находит,
    # а вставка того же ФИО падает на UNIQUE.
    cursor.execute("SELECT id, full_name FROM students WHERE name_key IS NULL")
    for old in cursor.fetchall():
        cursor.execute("UPDATE students SET name_key = ? WHERE id = ?",
                       (normalize_name(old["full_name"]), old["id"]))
    cursor.execute("SELECT id, full_name, profile_json FROM students WHERE name_key = ?", (name_key,))
    row = cursor.fetchone()
    if row:
        student_id, stored_name = row["id"], row["full_name"]
        profile = json.loads(row["profile_json"]) if row["profile_json"] else {}
    else:
        student_id, stored_name, profile = str(uuid.uuid4()), " ".join(full_name.split()), {}
        cursor.execute(
            "INSERT INTO students (id, full_name, profile_json, name_key) VALUES (?, ?, '{}', ?)",
            (student_id, stored_name, name_key),
        )
    conn.commit()
    conn.close()
    return student_id, stored_name, profile


def allow_retry(session_id):
    """Преподаватель разрешает пройти юнит заново: попытка перестаёт считаться.

    Вход без пароля — значит, кто-то может зайти под чужим ФИО и сжечь чужую попытку.
    Запись не удаляется, только меняет статус, — ответы и отчёт остаются в истории.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET status = 'retry_allowed' WHERE id = ? AND status = 'completed'",
                   (session_id,))
    conn.commit()
    conn.close()


def find_in_progress_session(full_name, task_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, current_step, answers, flags FROM sessions
           WHERE full_name = ? AND task_id = ? AND status = 'in_progress'
           ORDER BY created_at DESC LIMIT 1""",
        (full_name, task_id),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def abandon_session(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET status = 'abandoned' WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


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

def resolve_ref(task_data, section, ref):
    """Находит промпт или схему по ссылке из шага.

    Шаблон пишет ссылки как "prompts.final_grade", а код раньше искал ключ
    с точкой внутри — и получал пустую строку. В модель уходил пустой запрос.
    Принимаем обе формы: "prompts.final_grade" и "final_grade".
    """
    if not ref:
        return None
    ref = str(ref)
    prefix = section + "."
    key = ref[len(prefix):] if ref.startswith(prefix) else ref
    return task_data.get(section, {}).get(key)


class _KeepMissing(dict):
    """Неизвестная переменная остаётся в тексте как есть и попадает в список missing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.missing = []

    def __missing__(self, key):
        self.missing.append(key)
        return "{" + key + "}"


def build_prompt(prompt_template, context_dict):
    """Подставляет переменные. Возвращает (prompt, missing).

    Раньше при одной отсутствующей переменной в модель уходил весь шаблон
    неотформатированным. Теперь подставляется всё, что есть, а пропуски видны.
    """
    safe = _KeepMissing(context_dict)
    prompt = prompt_template.format_map(safe)
    return prompt, sorted(set(safe.missing))


def format_prompt_with_context(prompt_template, context_dict):
    prompt, missing = build_prompt(prompt_template, context_dict)
    if missing:
        st.warning(f"Missing variables in prompt: {', '.join(missing)}")
    return prompt

def flatten_answers(answers):
    """Плоский словарь для подстановки в промпты.

    Ключи вложенных словарей префиксуются id шага: раньше selected/correct/is_correct
    от g1..g10 перезаписывали друг друга, и в промпт уходил только последний вопрос.
    """
    flat = {}
    for key, value in answers.items():
        if isinstance(value, dict):
            flat[key] = json.dumps(value, ensure_ascii=False)
            for k, v in value.items():
                flat[f"{key}_{k}"] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
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
    st.session_state.setdefault(f"shown_{step['id']}", time.time())
    st.markdown(f"**{step.get('topic', '')}**")
    st.write(step.get("say"))
    user_answer = st.text_area("Your answer (in English):", height=150, key=f"answer_{step['id']}")

    # Вложенная кнопка внутри if st.button(...) в Streamlit не срабатывает никогда:
    # при перерисовке внешняя кнопка снова False и внутренняя исчезает.
    # Поэтому состояние "показали предупреждение" держим в session_state.
    pending_key = f"pending_{step['id']}"
    forced_key = f"forced_{step['id']}"

    if st.button("Submit answer", type="primary", key=f"submit_{step['id']}"):
        st.session_state[pending_key] = True

    if st.session_state.get(pending_key):
        flags, warnings = validate_answer(user_answer, min_words=min_words)
        if warnings and not st.session_state.get(forced_key):
            for w in warnings:
                st.warning(w)
            if st.button("Submit anyway", key=f"force_send_{step['id']}"):
                st.session_state[forced_key] = True
                st.rerun()
            return None, []
        elapsed = round(time.time() - st.session_state.get(f"shown_{step['id']}", time.time()), 1)
        st.session_state.setdefault("timings", {})[step["id"]] = elapsed
        # Единственный доступный сигнал вставки из соседнего чата: скорость набора.
        if elapsed > 0 and len(user_answer) > 120 and len(user_answer) / elapsed > 8:
            flags.append(f"typing_too_fast:{step['id']}")
        st.session_state.pop(pending_key, None)
        st.session_state.pop(forced_key, None)
        return user_answer, flags
    return None, []

def render_multiple_choice_step(step):
    st.session_state.setdefault(f"shown_{step['id']}", time.time())
    st.markdown(f"**{step.get('topic', '')}**")
    st.write(step.get("question"))
    options = step.get("options", {})
    option_labels = list(options.keys())
    # Раньше студент видел голые буквы A/B/C/D без текста ответа.
    selected = st.radio(
        "Choose your answer:",
        option_labels,
        format_func=lambda label: f"{label}. {options[label]}",
        key=f"mc_{step['id']}",
    )
    if st.button("Submit answer", type="primary", key=f"mc_submit_{step['id']}"):
        # Сравниваем букву с буквой: раньше сравнивался текст варианта с буквой
        # из ключа, поэтому is_correct был False у всех и всегда.
        return {
            "selected": selected,
            "selected_text": options[selected],
            "correct": step.get("correct"),
            "is_correct": str(selected).strip().upper() == str(step.get("correct", "")).strip().upper(),
            "seconds": round(time.time() - st.session_state.get(f"shown_{step['id']}", time.time()), 1),
        }
    return None

def render_llm_step(step, task_data, answers, profile, student_context, unit_config=None, mc=None, reading=None):
    prompt_template = resolve_ref(task_data, "prompts", step.get("prompt"))
    if not prompt_template:
        st.error(f"Промпт '{step.get('prompt')}' не найден в шаблоне — шаг пропущен, модель не вызывалась.")
        return {"error": f"prompt not found: {step.get('prompt')}"}, {"total_tokens": 0}
    # Имя модели можно переопределить переменной окружения, не трогая шаблоны:
    # GRADING_MODEL для шагов оценивания, REPORT_MODEL для отчёта.
    env_key = "REPORT_MODEL" if step.get("id") == "report" else "GRADING_MODEL"
    model_name = os.environ.get(env_key) or step.get("model", "yandexgpt-5-lite")

    context = build_llm_context(
        answers, profile, student_context, unit_config, mc, reading,
        st.session_state.get("llm_results", {}),
    )
    prompt = format_prompt_with_context(prompt_template, context)

    with st.spinner("🧠 AI is analyzing your answer (this may take 10-20 seconds)..."):
        return _run_llm_step(step, task_data, prompt, model_name)


def build_llm_context(answers, profile, student_context, unit_config=None, mc=None, reading=None, llm_results=None):
    """Собирает все переменные для подстановки в промпт. Без Streamlit — чтобы
    промпты можно было прогонять из скрипта (smoke_llm.py) ровно так же, как в приложении."""
    context = {
        "student_context": json.dumps(student_context, ensure_ascii=False) if isinstance(student_context, dict) else str(student_context),
        "profile_block": format_profile_for_prompt(profile),
    }

    if unit_config:
        context["unit_config"] = json.dumps(unit_config, ensure_ascii=False)

    flat_answers = flatten_answers(answers)
    context.update(flat_answers)
    context["answers"] = json.dumps(answers, ensure_ascii=False)
    text_answers = [v for k, v in answers.items() if isinstance(v, str)]
    context["answers_numbered"] = "\n".join([f"{i+1}. {v}" for i, v in enumerate(text_answers)])

    mc = mc or {"score": 0, "max": 0, "correct": 0, "items": 0, "wrong_topics": []}
    context["mc_score"] = mc["score"]
    context["mc_max"] = mc["max"]
    context["mc_correct"] = mc["correct"]
    context["mc_items"] = mc["items"]
    context["mc_wrong_topics"] = ", ".join(mc["wrong_topics"]) if mc["wrong_topics"] else "none"

    reading = reading or {"score": 0, "max": 0, "items": [], "unmatched": []}
    context["reading_score"] = reading["score"]
    context["reading_max"] = reading["max"]
    context["reading_unmatched"] = format_reading_for_prompt(reading)

    for key, value in (student_context or {}).items():
        if key not in context:
            context[key] = str(value)

    context.update(flatten_llm_results(llm_results or {}))
    return context


def _run_llm_step(step, task_data, prompt, model_name):
    if True:  # отступ сохранён, чтобы диф остался читаемым
        try:
            response, tokens = call_llm_with_retry(
                prompt,
                max_retries=2,
                temperature=step.get("temperature", 0.2),
                model_name=model_name
            )

            output_schema = resolve_ref(task_data, "schemas", step.get("output"))
            if output_schema:
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
    full_name = st.text_input("Enter your full name (surname and first name):")
    if st.button("Continue", type="primary"):
        if is_valid_name(full_name):
            student_id, stored_name, profile = get_or_create_student(full_name)
            st.session_state.student_name = stored_name
            st.session_state.student_id = student_id
            st.session_state.student_profile = profile
            st.session_state.current_step_idx = 0
            st.session_state.answers = {}
            st.session_state.flags = []
            st.session_state.timings = {}
            st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            st.session_state.session_id = None
            st.session_state.llm_results = {}
            st.rerun()
        elif full_name:
            st.error("Please enter at least your surname and first name.")
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
                    # grade_json = {"final_grade": {...}, "report": {...}} — total лежит на уровень глубже,
                    # поэтому старое x.get('total') всегда возвращало None и таблица была пустой.
                    df['total'] = df.apply(
                        lambda row: row['total'] if pd.notna(row.get('total')) else extract_total(row['grade_json']),
                        axis=1
                    )

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
                        if pd.notna(row.get('timings')):
                            st.caption(f"Время на ответы, сек: {row['timings']}")
                        if pd.notna(row.get('flags')) and row['flags'] not in ('[]', ''):
                            st.caption(f"Флаги: {row['flags']}")
                        if pd.notna(row.get('report_md')):
                            # Не st.expander: вложенный в карточку сессии, он роняет страницу
                            # («Expanders may not be nested inside other expanders»).
                            st.markdown("**📄 Отчёт студента (как он его получил):**")
                            st.markdown(row['report_md'])
                        if row.get('status') == 'completed':
                            if st.button("🔄 Разрешить пройти заново", key=f"retry_{row['id']}"):
                                allow_retry(row['id'])
                                st.rerun()
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

        # Студент обновил вкладку или закрыл браузер — ответы уже в базе,
        # предлагаем продолжить с того же шага, а не проходить юнит заново.
        resume_key = f"resume_checked_{task_id}"
        if not st.session_state.get(resume_key) and st.session_state.session_id is None:
            unfinished = find_in_progress_session(st.session_state.student_name, task_id)
            if unfinished:
                st.warning("У вас есть незаконченная попытка по этому юниту.")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("↩️ Продолжить с того же места", type="primary"):
                        st.session_state.session_id = unfinished["id"]
                        st.session_state.answers = json.loads(unfinished["answers"] or "{}")
                        st.session_state.flags = json.loads(unfinished["flags"] or "[]")
                        st.session_state.current_step_idx = int(unfinished["current_step"] or 0)
                        st.session_state[resume_key] = True
                        st.rerun()
                with col_b:
                    if st.button("🗑 Начать заново"):
                        abandon_session(unfinished["id"])
                        st.session_state[resume_key] = True
                        st.rerun()
                st.stop()
            st.session_state[resume_key] = True
        # Лимит проверяем только до сдачи. Своя только что сданная попытка — не повод
        # прятать отчёт: иначе любая перерисовка (хотя бы кнопка «Download») заменяла
        # отчёт на «You have already completed this unit».
        if "final_report" not in st.session_state and not check_attempts(st.session_state.student_name, task_id, max_attempts):
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
                mc = score_multiple_choice(
                    st.session_state.answers,
                    steps,
                    task_data.get("settings", {}).get("scoring", {}).get("mc_points_per_item", 1),
                )
                reading = reading_with_verdicts(st.session_state.answers, steps, st.session_state.llm_results)
                st.session_state.mc = mc
                result, tokens = render_llm_step(
                    current_step,
                    task_data,
                    st.session_state.answers,
                    st.session_state.get("student_profile", {}),
                    student_context,
                    unit_config,
                    mc,
                    reading,
                )
                if isinstance(result, dict) and any(
                    f in result for f in task_data.get("settings", {}).get("scoring", {}).get("llm_score_fields", [])
                ):
                    reading, verdict_flags = apply_reading_verdicts(reading, result.get("reading_verdicts"))
                    total, score_flags = compute_total(result, mc, task_data, reading)
                    result["mc_score"] = mc["score"]
                    result["reading_score"] = reading["score"]
                    result["reading_detail"] = reading["items"]
                    result["total_llm"] = result.get("total")
                    result["total"] = total
                    st.session_state.flags.extend(verdict_flags + score_flags)
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

                report_body = st.session_state.final_report.split("---\n### Teacher Meta")[0]

                total_now = extract_total(st.session_state.llm_results)
                if total_now is not None:
                    st.metric("Your score", f"{total_now} / {max_score}")
                st.markdown("### 📋 Your Personalized Feedback")
                st.markdown(report_body)

                st.download_button(
                    label="📥 Download Feedback (Markdown)",
                    data=report_body,
                    file_name=f"feedback_{st.session_state.student_name.replace(' ', '_')}.md",
                    mime="text/markdown"
                )

                # Блок «Teacher Meta» здесь больше не выводится: это экран студента, и он видел
                # флаги (например typing_too_fast), токены и хеш. Преподаватель видит всё это
                # в карточке сессии в своей панели.

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
                            total = ?,
                            max_score = ?,
                            report_md = ?,
                            flags = ?,
                            completed_at = ?
                        WHERE id = ?
                    """, (
                        json.dumps(st.session_state.llm_results, ensure_ascii=False),
                        st.session_state.report_hash,
                        st.session_state.token_usage.get("total_tokens", 0),
                        json.dumps(st.session_state.answers, ensure_ascii=False),
                        extract_total(st.session_state.llm_results),
                        max_score,
                        report_body,
                        json.dumps(st.session_state.flags, ensure_ascii=False),
                        datetime.now().isoformat(),
                        st.session_state.session_id
                    ))
                    cursor.execute(
                        "UPDATE sessions SET timings = ? WHERE id = ?",
                        (json.dumps(st.session_state.get("timings", {}), ensure_ascii=False),
                         st.session_state.session_id),
                    )
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
