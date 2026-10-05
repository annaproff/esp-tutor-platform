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
    "flatten_answers", "flatten_llm_results", "extract_total", "mc_table",
    "score_multiple_choice", "compute_total", "scores_so_far", "score_llm_result", "format_voice_stats",
}

tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in PURE]
ns = {"json": json, "re": re, "secrets": secrets}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<app>", "exec"), ns)

missing_fns = PURE - set(ns)
if missing_fns:
    sys.exit(f"В app.py не найдены функции: {missing_fns}")

REQUIRED_GRADE_KEYS_DIALOGUE = ["dialogue_accuracy", "dialogue_fluency", "dialogue_feedback", "errors"]
REQUIRED_GRADE_KEYS_WRITING = ["writing_grammar", "writing_vocabulary", "feedback_text", "constraint_check", "errors"]

def build_answers(task, student):
    answers = {}
    mc_steps = [s for s in task["steps"] if s.get("type") == "multiple_choice"]
    for i, s in enumerate(mc_steps):
        ok = i < student["mc_correct"]
        answers[s["id"]] = {
            "selected": s["correct"] if ok else "Z",
            "correct": s["correct"],
            "is_correct": ok
        }
    # Dialogue answers
    answers.update(student["dialogue"])
    # Writing answer
    answers["writing"] = student["writing"]
    return answers

def parse_json(text):
    match = re.search(r"{.*}", text or "", re.DOTALL)
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
    kwargs = dict(
        model=f"gpt://{folder}/{model}",
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
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
    ap.add_argument("--model", default=None, help="переопределить модель шагов оценивания")
    ap.add_argument("--json-mode", action="store_true", help="передать response_format=json_object")
    ap.add_argument("--show-prompt", action="store_true", help="напечатать собранный промпт")
    args = ap.parse_args()

    task = yaml.safe_load((ROOT / "tasks" / "template.yaml").read_text(encoding="utf-8"))
    unit_file = sorted((ROOT / "tasks" / "units").glob("*.json"))[0]  # как в app.py: первый юнит из папки
    unit = json.loads(unit_file.read_text(encoding="utf-8"))
    students = yaml.safe_load((ROOT / "tests_fixtures" / "students.yaml").read_text(encoding="utf-8"))

    steps = task["steps"]
    dialogue_eval_step = next(s for s in steps if s["id"] == "dialogue_eval")
    writing_eval_step = next(s for s in steps if s["id"] == "writing_eval")
    report_step = next(s for s in steps if s["id"] == "report")

    student_context = task["meta"]["student_context"]
    points = task["settings"]["scoring"].get("mc_points_per_item", 1)
    max_score = task["meta"]["max_score"]

    dialogue_tpl = ns["resolve_ref"](task, "prompts", dialogue_eval_step["prompt"])
    writing_tpl = ns["resolve_ref"](task, "prompts", writing_eval_step["prompt"])
    report_tpl = ns["resolve_ref"](task, "prompts", report_step["prompt"])

    if not dialogue_tpl or not writing_tpl or not report_tpl:
        sys.exit("Промпт не найден по ссылке из шага — проверьте resolve_ref и шаблон")

    client = folder = None
    model = args.model or os.environ.get("GRADING_MODEL") or dialogue_eval_step.get("model")

    if not args.dry_run:
        client, folder = make_client()
        print(f"Модель: {model} | json_mode: {args.json_mode} | прогонов: {args.runs}\n")

    names = ["weak", "mid", "strong"] if args.student == "all" else [args.student]
    summary, gate_ok = {}, True

    for name in names:
        answers = build_answers(task, students[name])
        mc = ns["score_multiple_choice"](answers, steps, points)
        
        # Для dialogue_eval нужен только контекст диалога
        # Как в app.py: шаги и флаги нужны для {voice_stats}; тестовые студенты отвечают текстом
        ctx_dialogue = ns["build_llm_context"](answers, {}, student_context, unit, mc, None, {}, steps=steps, flags=[])
        prompt_dialogue, missing_dialogue = ns["build_prompt"](dialogue_tpl, ctx_dialogue)
        
        print(f"=== {name}: тест {mc['score']}/{mc['max']}")
        
        if missing_dialogue:
            gate_ok = False
            print(f"   ❌ не подставились переменные в dialogue_eval: {missing_dialogue}")
        else:
            print(f"   ✅ dialogue_eval: все переменные на месте, промпт {len(prompt_dialogue)} символов")
        
        if args.show_prompt:
            print("-" * 60 + "\n" + prompt_dialogue + "\n" + "-" * 60)

        if args.dry_run:
            # Имитируем результаты для проверки report
            fake_dialogue = {
                "dialogue_accuracy": 7,
                "dialogue_fluency": 8,
                "dialogue_feedback": "Good effort",
                "corrected_dialogue": "...",
                "errors": []
            }
            fake_vocab = {
                "vocabulary_list": [{"word": "test", "translation": "тест", "definition": "...", "example": "..."}],
                "vocabulary_md": "**test** — тест\nDefinition: ...\nExample: ..."
            }
            fake_writing = {
                "writing_grammar": 6,
                "writing_vocabulary": 7,
                "feedback_text": "Good work",
                "constraint_check": {"passive_count": 2, "relative_clause_count": 2, "vocabulary_used": ["test"]},
                "errors": []
            }
            
            # Баллы шагов считаются тем же кодом, что в app.py
            results = {}
            for step_id, fake in [("dialogue_eval", fake_dialogue), ("vocab_gen", fake_vocab), ("writing_eval", fake_writing)]:
                ns["score_llm_result"](fake, step_id, results, mc, task)
                results[step_id] = fake
            ctx_report = ns["build_llm_context"](answers, {}, student_context, unit, mc, None, results, steps=steps)
            report_prompt, rep_missing = ns["build_prompt"](report_tpl, ctx_report)
            if rep_missing:
                gate_ok = False
                print(f"   ❌ не подставились переменные в report: {rep_missing}")
            else:
                total_line = next((l.strip() for l in report_prompt.splitlines() if "ИТОГОВАЯ" in l), "")
                print(f"   ✅ report: все переменные на месте | {total_line}")
            # Тексты, которые видит студент: словарь, отзывы — тоже должны подставиться
            ctx_text = ns["build_llm_context"](answers, {}, student_context, unit, None, None, results)
            for step in steps:
                if "{" in str(step.get("say", "")):
                    _, text_missing = ns["build_prompt"](step["say"], ctx_text)
                    if text_missing:
                        gate_ok = False
                        print(f"   ❌ текст шага {step['id']}: не подставилось {text_missing}")
                    else:
                        print(f"   ✅ текст шага {step['id']}: подстановки на месте")
            continue

        # Живой прогон: dialogue_eval -> vocab_gen -> writing_eval
        totals = []
        for run in range(1, args.runs + 1):
            try:
                # 1. Dialogue evaluation
                raw_d, secs_d, tokens_d = call(client, folder, model, prompt_dialogue, 
                                               dialogue_eval_step.get("temperature", 0), args.json_mode)
                dialogue_result = parse_json(raw_d)
                
                absent_d = [k for k in REQUIRED_GRADE_KEYS_DIALOGUE if k not in dialogue_result]
                
                # 2. Vocab generation (используем те же ответы)
                vocab_tpl = ns["resolve_ref"](task, "prompts", "vocab_gen")
                ctx_vocab = ns["build_llm_context"](answers, {}, student_context, unit, mc, None, {})
                prompt_vocab, _ = ns["build_prompt"](vocab_tpl, ctx_vocab)
                raw_v, secs_v, tokens_v = call(client, folder, model, prompt_vocab,
                                               0.3, args.json_mode)
                vocab_result = parse_json(raw_v)
                
                # 3. Writing evaluation
                writing_tpl = ns["resolve_ref"](task, "prompts", writing_eval_step["prompt"])
                ctx_writing = ns["build_llm_context"](
                    answers, {}, student_context, unit, mc, None,
                    {"dialogue_eval": dialogue_result, "vocab_gen": vocab_result}
                )
                prompt_writing, _ = ns["build_prompt"](writing_tpl, ctx_writing)
                raw_w, secs_w, tokens_w = call(client, folder, model, prompt_writing,
                                               writing_eval_step.get("temperature", 0), args.json_mode)
                writing_result = parse_json(raw_w)
                
                absent_w = [k for k in REQUIRED_GRADE_KEYS_WRITING if k not in writing_result]
                
                # Считаем итог
                combined_grade = {**dialogue_result, **writing_result}
                total, flags = ns["compute_total"](combined_grade, mc, task, None)
                totals.append(total)
                
                mark_d = "⚠️" if absent_d else "✅"
                mark_w = "⚠️" if absent_w else "✅"
                
                print(f"   прогон {run}:")
                print(f"     {mark_d} диалог: accuracy={dialogue_result.get('dialogue_accuracy')}, "
                      f"fluency={dialogue_result.get('dialogue_fluency')} | {secs_d}с, {tokens_d} ток."
                      + (f" | нет полей: {absent_d}" if absent_d else ""))
                print(f"     {mark_w} письмо: grammar={writing_result.get('writing_grammar')}, "
                      f"vocab={writing_result.get('writing_vocabulary')} | {secs_w}с, {tokens_w} ток."
                      + (f" | нет полей: {absent_w}" if absent_w else ""))
                print(f"     итог: {total}/{max_score} | флаги: {flags}")
                
                if absent_d or absent_w:
                    gate_ok = False
                    
            except Exception as err:
                gate_ok = False
                print(f"   прогон {run}: ❌ {type(err).__name__}: {err}")
                continue

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