"""Проверка чистых функций из app.py без запуска Streamlit.

Вынимаем нужные def-ы через AST и выполняем их в изолированном пространстве имён.
Запуск:  python3 test_fixes.py   (из корня репозитория)
"""
import ast, json, re, secrets, pathlib, sqlite3, tempfile, uuid
import yaml

ROOT = pathlib.Path(__file__).resolve().parent
src = (ROOT / "app.py").read_text(encoding="utf-8")
tree = ast.parse(src)
wanted = {
    "score_multiple_choice", "compute_total", "extract_total", "flatten_answers",
    "normalize_text", "match_answer_key", "score_reading", "apply_reading_verdicts",
    "format_reading_for_prompt", "normalize_name", "is_valid_name",
    "resolve_ref", "build_prompt", "_KeepMissing", "reading_with_verdicts",
    "init_db", "get_or_create_student", "allow_retry", "check_attempts",
}
DB_FILE = pathlib.Path(tempfile.mkdtemp()) / "tutor.db"


def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


ns = {"json": json, "re": re, "secrets": secrets, "uuid": uuid, "get_db_connection": get_db_connection}
module = ast.Module(
    body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted],
    type_ignores=[],
)
exec(compile(module, "<app-subset>", "exec"), ns)
missing = wanted - set(ns)
assert not missing, f"не найдены функции: {missing}"

task = yaml.safe_load((ROOT / "tasks" / "template.yaml").read_text(encoding="utf-8"))
steps = task["steps"]

# --- 1. тест с вариантами считается кодом ---
answers = {}
for i, s in enumerate([x for x in steps if x.get("type") == "multiple_choice"]):
    ok = i < 7  # студент ответил верно на 7 из 10
    answers[s["id"]] = {"selected": s["correct"] if ok else "Z", "correct": s["correct"], "is_correct": ok}

mc = ns["score_multiple_choice"](answers, steps, 1)
assert (mc["correct"], mc["items"], mc["score"], mc["max"]) == (7, 10, 7, 10), mc
assert len(mc["wrong_topics"]) == 3, mc
print("1. тест с вариантами:", mc["score"], "/", mc["max"], "| ошибки в темах:", mc["wrong_topics"])

# --- 2. короткие ответы сверяются кодом, в разных формулировках ---
answers.update({
    "r1": "In 1986, as far as I remember",
    "r2": "more than 100 billion words per day",
    "r3": "P(wi)=1-sqrt(t/f(wi))",
    "r4": "приблизительно 10^-5",
    "r5": "they used 300 dimensions",
    "essay_write": "Word vectors are used in many systems ...",
})
reading = ns["score_reading"](answers, steps)
assert reading["score"] == reading["max"] == 15, reading
print("2. автопроверка чтения:", reading["score"], "/", reading["max"], "| на суждение модели:", len(reading["unmatched"]))

# --- 3. неузнанный ответ уходит модели, а не обнуляется молча ---
answers["r2"] = "ten to the eleventh power of words"
reading2 = ns["score_reading"](answers, steps)
assert reading2["score"] == 12 and len(reading2["unmatched"]) == 1, reading2
assert "r2" in ns["format_reading_for_prompt"](reading2)
reading2, vflags = ns["apply_reading_verdicts"](reading2, {"r2": True})
assert reading2["score"] == 15, reading2
assert vflags == ["reading_accepted_by_llm:r2"], vflags
print("3. вердикт модели по неузнанному ответу:", reading2["score"], "/ 15 | флаги:", vflags)

# отказ модели ничего не добавляет
reading3, _ = ns["apply_reading_verdicts"](ns["score_reading"](answers, steps), {"r2": False})
assert reading3["score"] == 12, reading3
print("   отказ модели:", reading3["score"], "/ 15")

# --- 4. итог считает код, завышенные баллы модели обрезаются ---
grade = {"essay_task_achievement": 99, "essay_language": 7, "total": 50}
total, flags = ns["compute_total"](grade, mc, task, reading2)
assert total == 7 + 15 + 15 + 7 == 44, total
assert any("over_max" in f for f in flags) and any("mismatch" in f for f in flags), flags
print("4. итог:", total, "| флаги:", flags)

# --- 5. мусор от модели не роняет подсчёт ---
total2, flags2 = ns["compute_total"]({"essay_language": "отлично"}, mc, task, reading2)
assert total2 == 7 + 15, total2
print("5. битый ответ модели:", total2, "| флаги:", flags2)

# --- 6. total находится на любом уровне вложенности ---
assert ns["extract_total"]({"final_grade": {"total": 44}, "report": {}}) == 44
assert ns["extract_total"]({"total": 33}) == 33
assert ns["extract_total"]({"report": {"feedback_text": "..."}}) is None
print("6. извлечение total из llm_results: ок")

# --- 7. ключи вариантов больше не перетирают друг друга ---
flat = ns["flatten_answers"](answers)
assert flat["g1_is_correct"] != flat["g10_is_correct"], "ключи всё ещё склеиваются"
assert flat["r1"].startswith("In 1986")
print("7. плоские ключи: g1_selected =", flat["g1_selected"], ", g10_selected =", flat["g10_selected"])

# --- 8. имена: перестановка, регистр и ё не создают дубль, латиница проходит ---
assert ns["normalize_name"]("Иванов  Иван") == ns["normalize_name"]("иван иванов")
assert ns["normalize_name"]("Алёна Петрова") == ns["normalize_name"]("петрова алена")
assert ns["is_valid_name"]("Anna Smirnova") and not ns["is_valid_name"]("Анна")
print("8. имена: перестановка, регистр и ё — один студент, латиница проходит")

# --- 9. промпт находится по ссылке из шаблона (раньше в модель уходила пустая строка) ---
for step_id in ("final_grade", "report"):
    step = next(x for x in steps if x["id"] == step_id)
    found = ns["resolve_ref"](task, "prompts", step["prompt"])
    assert found and len(found) > 200, f"промпт для {step_id} не найден по ссылке {step['prompt']}"
assert ns["resolve_ref"](task, "schemas", "schemas.final_grade")
assert ns["resolve_ref"](task, "prompts", "final_grade")  # короткая форма тоже работает
assert ns["resolve_ref"](task, "prompts", "prompts.nope") is None
print("9. промпты и схемы находятся по ссылкам из шаблона: ок")

# --- 10. отсутствующая переменная не превращает запрос в сырой шаблон ---
text, miss = ns["build_prompt"]("score {a}, lost {b}, json {{x}}", {"a": 5})
assert text == "score 5, lost {b}, json {x}" and miss == ["b"], (text, miss)
print("10. подстановка с пропуском:", repr(text), "| пропущено:", miss)

# --- 11. отчёт видит то же чтение, что вошло в итог ---
answers["r2"] = "ten to the eleventh power of words"
llm_results = {"final_grade": {"reading_verdicts": {"r2": True}, "total": 44}}
assert ns["score_reading"](answers, steps)["score"] == 12
assert ns["reading_with_verdicts"](answers, steps, llm_results)["score"] == 15
assert ns["reading_with_verdicts"](answers, steps, {})["score"] == 12
print("11. чтение для отчёта с вердиктом модели: 15 / 15 (без вердикта было бы 12)")

# --- 12. вход по ФИО: старая база не падает, разное написание — один студент ---
old = sqlite3.connect(DB_FILE)
old.execute("CREATE TABLE students (id TEXT PRIMARY KEY, full_name TEXT UNIQUE NOT NULL, profile_json TEXT DEFAULT '{}')")
old.execute("INSERT INTO students VALUES ('old-1', 'Иванов Иван', '{\"estimated_level\": \"B1\"}')")
old.commit()
old.close()
ns["init_db"]()
sid, name, profile = ns["get_or_create_student"]("иван  Иванов")  # другой регистр и порядок слов
assert (sid, name, profile) == ("old-1", "Иванов Иван", {"estimated_level": "B1"}), (sid, name, profile)
sid2, name2, _ = ns["get_or_create_student"]("Anna   Smirnova")
assert sid2 != sid and name2 == "Anna Smirnova"
assert ns["get_or_create_student"]("smirnova anna")[0] == sid2
print("12. вход по ФИО: старая запись найдена при другом написании, профиль на месте, дублей нет")

# --- 13. преподаватель разрешает пройти заново ---
conn = get_db_connection()
conn.execute("INSERT INTO sessions (id, full_name, task_id, status) VALUES ('s-1', 'Иванов Иван', 'unit_1', 'completed')")
conn.commit()
conn.close()
assert not ns["check_attempts"]("Иванов Иван", "unit_1", 1)
ns["allow_retry"]("s-1")
assert ns["check_attempts"]("Иванов Иван", "unit_1", 1)
status = get_db_connection().execute("SELECT status FROM sessions WHERE id = 's-1'").fetchone()[0]
assert status == "retry_allowed", status
print("13. сброс попытки: студент снова может пройти юнит, старая запись осталась в истории")

print("\nвсе проверки прошли")
