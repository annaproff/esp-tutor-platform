"""Проверка чистых функций из app.py без запуска Streamlit.

Вынимаем нужные def-ы через AST и выполняем их в изолированном пространстве имён.
Запуск:  python3 test_fixes.py   (из корня репозитория)
"""
import ast, io, json, re, secrets, pathlib, sqlite3, tempfile, uuid, wave
import requests
import yaml

ROOT = pathlib.Path(__file__).resolve().parent
src = (ROOT / "app.py").read_text(encoding="utf-8")
tree = ast.parse(src)
wanted = {
    "score_multiple_choice", "compute_total", "extract_total", "flatten_answers", "flatten_llm_results",
    "normalize_name", "is_valid_name", "resolve_ref", "build_prompt", "_KeepMissing",
    "scores_so_far", "score_llm_result", "mc_table", "fill_step_text", "build_llm_context",
    "init_db", "get_or_create_student", "allow_retry", "check_attempts", "find_in_progress_session",
    "split_wav", "transcribe_speech", "format_voice_stats", "STT_URL",
}
DB_FILE = pathlib.Path(tempfile.mkdtemp()) / "tutor.db"


def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


ns = {"json": json, "re": re, "secrets": secrets, "uuid": uuid, "get_db_connection": get_db_connection,
      "io": io, "wave": wave, "requests": requests}
module = ast.Module(
    body=[n for n in tree.body if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted)
          or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))],
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

# --- 2. итог складывается по всем частям, а не внутри одного шага ---
results = {}
dialogue = {"dialogue_accuracy": 7, "dialogue_fluency": 8, "dialogue_feedback": "Mind articles."}
ns["score_llm_result"](dialogue, "dialogue_eval", results, mc, task)
results["dialogue_eval"] = dialogue
vocab = {"vocabulary_md": "**portability** — портативность"}
assert ns["score_llm_result"](vocab, "vocab_gen", results, mc, task) == [] and "total" not in vocab
results["vocab_gen"] = vocab
writing = {"writing_grammar": 8, "writing_vocabulary": 6, "feedback_text": "Good.", "constraint_check": {}}
ns["score_llm_result"](writing, "writing_eval", results, mc, task)
results["writing_eval"] = writing
assert dialogue["step_score"] == 15 and writing["step_score"] == 14, (dialogue, writing)
assert writing["total"] == 7 + 15 + 14 == 36, writing
assert ns["extract_total"](results) == 36, "итог должен браться у последнего шага с оценкой"
print("2. итог по всем частям: тест 7 + диалог 15 + письмо 14 =", ns["extract_total"](results))

# --- 3. завышенные баллы модели обрезаются, мусор не роняет подсчёт ---
over = {"dialogue_accuracy": 99, "dialogue_fluency": "отлично"}
flags = ns["score_llm_result"](over, "dialogue_eval", {}, mc, task)
assert over["step_score"] == 10, over
assert any("over_max" in f for f in flags) and any("not_a_number" in f for f in flags), flags
print("3. 99 баллов обрезаны до 10, нечисло — 0 | флаги:", flags)

# --- 4. total находится на любом уровне вложенности ---
assert ns["extract_total"]({"total": 33}) == 33
assert ns["extract_total"]({"report": {"feedback_text": "..."}}) is None
print("4. извлечение total из llm_results: ок")

# --- 5. в отчёт уходят готовые баллы и настоящие вопросы теста ---
answers.update({"d1": "a", "d2": "b", "d3": "c", "d4": "d", "writing": "text"})
ctx = ns["build_llm_context"](answers, {}, task["meta"]["student_context"], None, mc, None, results, steps=steps)
report, report_missing = ns["build_prompt"](ns["resolve_ref"](task, "prompts", "prompts.report"), ctx)
assert not report_missing, report_missing
assert "ИТОГОВАЯ ОЦЕНКА: 36 / 50" in report and "Итого: 15/20" in report and "Итого: 14/20" in report
assert "The new educational software" in report and "WRONG" in report
print("5. отчёт: итог 36/50 и баллы частей от кода, таблица теста с вопросами")

# --- 6. текст для студента: результаты прошлых шагов подставлены, пропуски не видны ---
text_ctx = ns["build_llm_context"](answers, {}, {}, None, None, None, results)
writing_step = next(x for x in steps if x["id"] == "writing")
shown = ns["fill_step_text"](writing_step["say"], text_ctx)
assert "portability" in shown and "{" not in shown, shown
assert ns["fill_step_text"]("Слова: {vocab_gen_vocabulary_md}.", {}) == "Слова: ."
assert ns["fill_step_text"]("скобка } без пары {x}", {"x": 1}) == "скобка } без пары {x}"
print("6. задание на письмо показывает словарь; пустое значение — пусто, а не {скобки}")

# --- 7. ключи вариантов больше не перетирают друг друга ---
flat = ns["flatten_answers"](answers)
assert flat["g1_is_correct"] != flat["g10_is_correct"], "ключи всё ещё склеиваются"
print("7. плоские ключи: g1_selected =", flat["g1_selected"], ", g10_selected =", flat["g10_selected"])

# --- 8. имена: перестановка, регистр и ё не создают дубль, латиница проходит ---
assert ns["normalize_name"]("Иванов  Иван") == ns["normalize_name"]("иван иванов")
assert ns["normalize_name"]("Алёна Петрова") == ns["normalize_name"]("петрова алена")
assert ns["is_valid_name"]("Anna Smirnova") and not ns["is_valid_name"]("Анна")
print("8. имена: перестановка, регистр и ё — один студент, латиница проходит")

# --- 9. промпт находится по ссылке из шаблона (раньше в модель уходила пустая строка) ---
for step in [x for x in steps if x.get("type") == "llm"]:
    found = ns["resolve_ref"](task, "prompts", step["prompt"])
    assert found and len(found) > 200, f"промпт для {step['id']} не найден по ссылке {step['prompt']}"
assert ns["resolve_ref"](task, "schemas", "schemas.dialogue_eval")
assert ns["resolve_ref"](task, "prompts", "writing_eval")  # короткая форма тоже работает
assert ns["resolve_ref"](task, "prompts", "prompts.nope") is None
print("9. промпты и схемы находятся по ссылкам из шаблона: ок")

# --- 10. отсутствующая переменная не превращает запрос в сырой шаблон ---
text, miss = ns["build_prompt"]("score {a}, lost {b}, json {{x}}", {"a": 5})
assert text == "score 5, lost {b}, json {x}" and miss == ["b"], (text, miss)
print("10. подстановка с пропуском:", repr(text), "| пропущено:", miss)

# --- 11. вход по ФИО: старая база не падает, разное написание — один студент ---
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
print("11. вход по ФИО: старая запись найдена при другом написании, профиль на месте, дублей нет")

# --- 12. преподаватель разрешает пройти заново ---
conn = get_db_connection()
conn.execute("INSERT INTO sessions (id, full_name, task_id, status) VALUES ('s-1', 'Иванов Иван', 'unit_1', 'completed')")
conn.commit()
conn.close()
assert not ns["check_attempts"]("Иванов Иван", "unit_1", 1)
ns["allow_retry"]("s-1")
assert ns["check_attempts"]("Иванов Иван", "unit_1", 1)
status = get_db_connection().execute("SELECT status FROM sessions WHERE id = 's-1'").fetchone()[0]
assert status == "retry_allowed", status
print("12. сброс попытки: студент снова может пройти юнит, старая запись осталась в истории")

# --- 13. продолжение попытки возвращает и результаты модели ---
conn = get_db_connection()
conn.execute("INSERT INTO sessions (id, full_name, task_id, status, current_step, grade_json) "
             "VALUES ('s-2', 'Anna Smirnova', 'unit_vol3-1', 'in_progress', 17, ?)", (json.dumps(results),))
conn.commit()
conn.close()
unfinished = ns["find_in_progress_session"]("Anna Smirnova", "unit_vol3-1")
assert json.loads(unfinished["grade_json"])["vocab_gen"]["vocabulary_md"].startswith("**portability**")
print("13. после обновления вкладки словарь и баллы за диалог возвращаются из базы")

# --- 14. запись с микрофона: режется на куски до 29 с, уходит в SpeechKit без заголовка WAV ---
def make_wav(seconds, rate=16000, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x01" * channels * int(rate * seconds))
    return buf.getvalue()

sent = []
class Reply:
    def __init__(self, code, body): self.status_code, self.body, self.text = code, body, json.dumps(body)
    def json(self): return self.body
def fake_post(url, params, data, headers, timeout):
    sent.append({"url": url, "params": params, "size": len(data), "auth": headers["Authorization"]})
    return Reply(200, {"result": f"part {len(sent)}"})
text, seconds, error = ns["transcribe_speech"](make_wav(40), "KEY", post=fake_post)
assert error is None and text == "part 1 part 2" and seconds == 40.0, (text, seconds, error)
assert [x["size"] for x in sent] == [29 * 16000 * 2, 11 * 16000 * 2], sent  # чистый звук, без 44 байт заголовка
assert sent[0]["params"] == {"lang": "en-US", "format": "lpcm", "sampleRateHertz": 16000}, sent[0]
assert sent[0]["auth"] == "Api-Key KEY" and sent[0]["url"].endswith("/speech/v1/stt:recognize")
denied = lambda *a, **k: Reply(401, {"error_message": "Unauthorized"})
assert ns["transcribe_speech"](make_wav(3), "KEY", post=denied)[2].startswith("SpeechKit 401")
assert ns["transcribe_speech"](make_wav(3, channels=2), "KEY", post=fake_post)[2].startswith("bad audio")
print("14. 40 с записи → 2 запроса по 29 и 11 с, en-US, lpcm 16 кГц, Api-Key; отказ доступа и стерео — понятная ошибка")

# --- 15. модель видит, какие ответы надиктованы и в каком темпе ---
stats = ns["format_voice_stats"](steps, ["fast_answer", "voice:d1:24s:52w", "stt_error:d2"])
assert "d1: spoken, 24 s, 52 words (~130 words per minute)" in stats and "d2: typed, no recording" in stats, stats
print("15. для оценки беглости:", stats.splitlines()[0], "| d2–d4 — текстом")

print("\nвсе проверки прошли")
