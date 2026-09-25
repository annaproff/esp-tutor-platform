"""Прогон промптов оценивания без Streamlit — ровно тем же кодом, что в app.py.

Сухой режим (без ключа): собирает промпты для трёх тестовых студентов и проверяет,
что все переменные подставились.
    python3 smoke_llm.py --dry-run

Живой режим: вызывает YandexGPT, проверяет JSON, считает итог кодом, показывает разброс.
    export YANDEX_API_KEY=... YANDEX_FOLDER_ID=...
    python3 smoke_llm.py --runs 5
    python3 smoke_llm.py --runs 5 --model yandexgpt/latest --json-mode

Критерий прохождения (гейт перед запуском студентов):
    1. JSON валиден во всех прогонах;
    2. разброс итога у каждого студента не больше 2 баллов;
    3. weak < mid < strong по среднему.
"""
import argparse, ast, json, os, re, secrets, statistics, sys, time, pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parent

# --- берём функции прямо из app.py, чтобы не было второй копии логики ---
PURE = {
    "resolve_ref", "build_prompt", "_KeepMissing", "build_llm_context",
    "flatten_answers", "flatten_llm_results", "format_profile_for_prompt",
    "score_multiple_choice", "normalize_text", "match_answer_key", "score_reading",
    "apply_reading_verdicts", "format_reading_for_prompt", "compute_total",
}
tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in PURE]
ns = {"json": json, "re": re, "secrets": secrets}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<app>", "exec"), ns)
missing_fns = PURE - set(ns)
if missing_fns:
    sys.exit(f"В app.py не найдены функции: {missing_fns}")

REQUIRED_GRADE_KEYS = ["essay_task_achievement", "essay_language", "reading_verdicts",
                       "essay_strengths", "essay_improvements", "errors", "profile_notes"]


def build_answers(task, student):
    answers = {}
    mc_steps = [s for s in task["steps"] if s.get("type") == "multiple_choice"]
    for i, s in enumerate(mc_steps):
        ok = i < student["mc_correct"]
        answers[s["id"]] = {"selected": s["correct"] if ok else "Z", "correct": s["correct"], "is_correct": ok}
    answers.update(student["reading"])
    answers["essay_write"] = student["essay_write"]
    return answers


def parse_json(text):
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("в ответе нет JSON-объекта")
    return json.loads(match.group())


def make_client():
    from openai import OpenAI
    key, folder = os.environ.get("YANDEX_API_KEY"), os.environ.get("YANDEX_FOLDER_ID")
    if not key or not folder:
        sys.exit("Нужны YANDEX_API_KEY и YANDEX_FOLDER_ID (или запустите с --dry-run)")
    return OpenAI(api_key=key, project=folder, base_url="https://ai.api.cloud.yandex.net/v1"), folder


def call(client, folder, model, prompt, temperature, json_mode):
    kwargs = dict(model=f"gpt://{folder}/{model}", temperature=temperature,
                  messages=[{"role": "user", "content": prompt}])
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    started = time.time()
    resp = client.chat.completions.create(**kwargs)
    usage = getattr(resp, "usage", None)
    return resp.choices[0].message.content, round(time.time() - started, 1), getattr(usage, "total_tokens", 0) or 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="только собрать промпты, без вызовов")
    ap.add_argument("--runs", type=int, default=3, help="прогонов оценки на каждого студента")
    ap.add_argument("--student", default="all", choices=["all", "weak", "mid", "strong"])
    ap.add_argument("--model", default=None, help="переопределить модель шага final_grade")
    ap.add_argument("--json-mode", action="store_true", help="передать response_format=json_object")
    ap.add_argument("--show-prompt", action="store_true", help="напечатать собранный промпт")
    args = ap.parse_args()

    task = yaml.safe_load((ROOT / "tasks" / "template.yaml").read_text(encoding="utf-8"))
    unit = json.loads((ROOT / "tasks" / "units" / "unit_01.json").read_text(encoding="utf-8"))
    students = yaml.safe_load((ROOT / "tests_fixtures" / "students.yaml").read_text(encoding="utf-8"))
    steps = task["steps"]
    grade_step = next(s for s in steps if s["id"] == "final_grade")
    report_step = next(s for s in steps if s["id"] == "report")
    student_context = task["meta"]["student_context"]
    points = task["settings"]["scoring"].get("mc_points_per_item", 1)
    max_score = task["meta"]["max_score"]

    grade_tpl = ns["resolve_ref"](task, "prompts", grade_step["prompt"])
    report_tpl = ns["resolve_ref"](task, "prompts", report_step["prompt"])
    if not grade_tpl or not report_tpl:
        sys.exit("Промпт не найден по ссылке из шага — проверьте resolve_ref и шаблон")

    client = folder = None
    model = args.model or os.environ.get("GRADING_MODEL") or grade_step.get("model")
    if not args.dry_run:
        client, folder = make_client()
        print(f"Модель: {model} | json_mode: {args.json_mode} | прогонов: {args.runs}\n")

    names = ["weak", "mid", "strong"] if args.student == "all" else [args.student]
    summary, gate_ok = {}, True

    for name in names:
        answers = build_answers(task, students[name])
        mc = ns["score_multiple_choice"](answers, steps, points)
        reading = ns["score_reading"](answers, steps)
        ctx = ns["build_llm_context"](answers, {}, student_context, unit, mc, reading, {})
        prompt, missing = ns["build_prompt"](grade_tpl, ctx)

        print(f"=== {name}: тест {mc['score']}/{mc['max']}, чтение по ключу {reading['score']}/{reading['max']}, "
              f"на суждение модели: {[i['id'] for i in reading['unmatched']] or 'нет'}")
        if missing:
            gate_ok = False
            print(f"   ❌ не подставились переменные в final_grade: {missing}")
        else:
            print(f"   ✅ final_grade: все переменные на месте, промпт {len(prompt)} символов")
        if args.show_prompt:
            print("-" * 60 + "\n" + prompt + "\n" + "-" * 60)

        if args.dry_run:
            fake = {"final_grade": {"essay_task_achievement": 10, "essay_language": 7, "reading_feedback": "x",
                                    "essay_strengths": ["a"], "essay_improvements": ["b"]}}
            _, rep_missing = ns["build_prompt"](report_tpl, ns["build_llm_context"](
                answers, {}, student_context, unit, mc, reading, fake))
            if rep_missing:
                gate_ok = False
                print(f"   ❌ не подставились переменные в report: {rep_missing}")
            else:
                print("   ✅ report: все переменные на месте")
            continue

        totals = []
        for run in range(1, args.runs + 1):
            try:
                raw, secs, tokens = call(client, folder, model, prompt, grade_step.get("temperature", 0), args.json_mode)
                grade = parse_json(raw)
            except Exception as err:
                gate_ok = False
                print(f"   прогон {run}: ❌ {type(err).__name__}: {err}")
                continue
            absent = [k for k in REQUIRED_GRADE_KEYS if k not in grade]
            r, _ = ns["apply_reading_verdicts"](json.loads(json.dumps(reading)), grade.get("reading_verdicts"))
            total, flags = ns["compute_total"](grade, mc, task, r)
            totals.append(total)
            mark = "⚠️" if absent else "✅"
            print(f"   прогон {run}: {mark} эссе {grade.get('essay_task_achievement')}+{grade.get('essay_language')} "
                  f"| чтение {r['score']} | итог {total}/{max_score} | {secs}с, {tokens} ток."
                  + (f" | нет полей: {absent}" if absent else "") + (f" | флаги: {flags}" if flags else ""))
            if absent:
                gate_ok = False
        if totals:
            spread = max(totals) - min(totals)
            summary[name] = statistics.mean(totals)
            ok = spread <= 2
            gate_ok &= ok
            print(f"   разброс итога: {spread} {'✅' if ok else '❌ больше 2 баллов — рубрику надо уточнять'}")
        print()

    if not args.dry_run and len(summary) == 3:
        ordered = summary["weak"] < summary["mid"] < summary["strong"]
        gate_ok &= ordered
        print(f"Средние: weak {summary['weak']:.1f} < mid {summary['mid']:.1f} < strong {summary['strong']:.1f} — "
              + ("✅ рубрика различает уровни" if ordered else "❌ порядок нарушен"))

    print("\nГЕЙТ:", "✅ ПРОЙДЕН" if gate_ok else "❌ НЕ ПРОЙДЕН")
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
