import argparse
import json
import sys
import time
from pathlib import Path


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def emit_chat(result):
    print("\nLocalBrain>")
    for line in result["answer"].splitlines():
        print(line)
    if result.get("note"):
        print("\nหมายเหตุ: " + result["note"])
    if result.get("sources"):
        print("\nแหล่งข้อมูล:")
        for source in result["sources"]:
            print(f"[{source['id']}] {source['title'][:80]} — {source['source'][:180]}")
    print()


def parser():
    root = argparse.ArgumentParser(
        prog="localbrain", description="Train from scratch; retrieve local knowledge; use explicit tools.")
    root.add_argument("--home", default="workspace", help="Knowledge/tools workspace")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect RAM, CPU and test CUDA")
    sub.add_parser("tools", help="List available explicit tool calls")
    sub.add_parser("stats", help="Knowledge store counts")
    prepare = sub.add_parser("prepare", help="Train local tokenizer and encode a UTF-8 corpus")
    prepare.add_argument("source")
    prepare.add_argument("--output", default="workspace/dataset")
    prepare.add_argument("--config", default="configs/default.json")
    instruction = sub.add_parser("prepare-instructions", help="Prepare deduplicated assistant-only train/val/test data")
    instruction.add_argument("source")
    instruction.add_argument("--output", default="workspace/dataset-instruction")
    instruction.add_argument("--config", default="configs/instruction.json")
    instruction.add_argument("--tokenizer", help="Reuse this tokenizer.json (e.g. from build-corpus)")
    tasks = sub.add_parser("make-code-tasks", help="Generate verified coding tasks as SFT question/answer JSONL")
    tasks.add_argument("--count", type=int, default=60000)
    tasks.add_argument("--output", default="workspace/raw-sft/code-tasks.jsonl")
    tasks.add_argument("--seed", type=int, default=1)
    tasks.add_argument("--languages", nargs="+", default=["python", "javascript"])
    corpus = sub.add_parser("build-corpus", help="Tokenizer + token stream from code/English parquet shards")
    corpus.add_argument("--code", nargs="+", required=True, help="github-code-clean parquet files")
    corpus.add_argument("--english", nargs="*", default=[], help="FineWeb-Edu parquet files")
    corpus.add_argument("--output", default="workspace/dataset-code")
    corpus.add_argument("--vocab-size", type=int, default=16384)
    corpus.add_argument("--max-tokens", type=int, default=600_000_000)
    corpus.add_argument("--tokenizer", help="Reuse this tokenizer.json instead of training a new one "
                                            "(keeps earlier datasets compatible)")
    anneal = sub.add_parser("build-anneal", help="High-quality data mix for the final LR-decay phase")
    anneal.add_argument("--tokenizer", required=True)
    anneal.add_argument("--code", nargs="+", required=True)
    anneal.add_argument("--pyedu", nargs="+", required=True)
    anneal.add_argument("--english", nargs="+", required=True)
    anneal.add_argument("--output", default="workspace/dataset-anneal")
    anneal.add_argument("--max-tokens", type=int, default=80_000_000)
    anneal.add_argument("--personal", help="JSONL from collect-personal (your own code, kept local)")
    anneal.add_argument("--extra", action="append", default=[], metavar="NAME=PATH",
                        help="Extra training source JSONL of {path,text} docs, e.g. --extra offensive=D:\\abs\\offensive.jsonl (repeatable)")
    anneal.add_argument("--extra-share", type=float, default=0.1, help="Character share for each --extra source in the final anneal mix")
    personal = sub.add_parser("collect-personal", help="Collect your own code (secrets redacted) for private training")
    personal.add_argument("folders", nargs="+")
    personal.add_argument("--output", default="workspace/personal/personal-code.jsonl")
    collect_offensive = sub.add_parser("collect-offensive", help="Collect offensive-security documents (malware analysis, attack toolkits) as JSONL")
    collect_offensive.add_argument("--root", required=True, help="Folder of topic subdirectories with markdown notes")
    collect_offensive.add_argument("--output", default="workspace/raw-offensive/offensive.jsonl")
    train = sub.add_parser("train", help="Train new random weights or resume a checkpoint")
    train.add_argument("--data", default="workspace/dataset")
    train.add_argument("--run", default="workspace/run")
    train.add_argument("--config", default="configs/default.json")
    train.add_argument("--steps", type=int)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"])
    train.add_argument("--resume", action="store_true")
    train.add_argument("--init", help="Start from another run's checkpoint weights (e.g. pretrain -> SFT)")
    generate = sub.add_parser("generate", help="Raw text generation from your checkpoint")
    generate.add_argument("prompt")
    generate.add_argument("--run", default="workspace/run")
    generate.add_argument("--max-tokens", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    add = sub.add_parser("ingest", help="Add a local document/folder or public URL")
    add.add_argument("source")
    search = sub.add_parser("search", help="Search the local store")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    research = sub.add_parser("research", help="Search public HTML web results and store selected pages")
    research.add_argument("query")
    research.add_argument("--limit", type=int, default=3, choices=range(1, 11))
    refresh = sub.add_parser("refresh", help="Refresh already stored web pages")
    refresh.add_argument("--cycles", type=int, default=1)
    refresh.add_argument("--interval", type=int, default=3600)
    ask = sub.add_parser("ask", help="Retrieve excerpts; optionally use your experimental model")
    ask.add_argument("query")
    ask.add_argument("--run", help="Omit for reliable source excerpts without model generation")
    ask.add_argument("--web", action="store_true", help="Search web if local retrieval has no hits")
    ask.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    chat = sub.add_parser("chat", help="Interactive retrieval assistant; /exit to quit")
    chat.add_argument("--run")
    chat.add_argument("--model-only", action="store_true", help="Send every natural-language turn to learned weights")
    model_eval = sub.add_parser("model-eval", help="Evaluate held-out instruction examples")
    model_eval.add_argument("--run", default="workspace/run-instruction")
    model_eval.add_argument("--data", default="workspace/dataset-instruction")
    model_eval.add_argument("--output", default="workspace/run-instruction/evaluation.json")
    chat.add_argument("--web", action="store_true")
    chat.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    code_eval = sub.add_parser("code-eval", help="Coding benchmark: pass@1 / pass@k on seen and unseen task families")
    code_eval.add_argument("--run", required=True)
    code_eval.add_argument("--output", help="Default: <run>/code-eval.json")
    code_eval.add_argument("--samples", type=int, default=8)
    code_eval.add_argument("--per-family", type=int, default=2)
    code_eval.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    code_eval.add_argument("--agent", action="store_true",
                           help="Also score agent-assisted solving (public-test feedback + repair, hidden-test scoring)")
    code_eval.add_argument("--repair-rounds", type=int, default=1)
    code = sub.add_parser("code", help="Write a function, verified against your examples")
    code.add_argument("task", help="What the function should do")
    code.add_argument("--function", required=True, help="Function name")
    code.add_argument("--args", default="x", help="Comma-separated parameter names")
    code.add_argument("--language", choices=["python", "javascript"], default="python")
    code.add_argument("--example", action="append", default=[],
                      help='Test case as JSON, e.g. {"args": [[1, 2]], "expected": 3} (repeatable)')
    code.add_argument("--run", default="workspace/run-code-sft")
    code.add_argument("--samples", type=int, default=16)
    code.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    play = sub.add_parser("playground", help="Interactive: type code, the model continues it")
    play.add_argument("--run", default="workspace/run-code-pretrain")
    play.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    play.add_argument("--temperature", type=float, default=0.6)
    pindex = sub.add_parser("index-project", help="Index a code project's functions/classes (incremental)")
    pindex.add_argument("--root", default=".", help="Project folder to scan")
    pindex.add_argument("--db", help="Index file (default: <home>/project-index.sqlite3)")
    pindex.add_argument("--compact", action="store_true", help="Optimize and VACUUM after indexing")
    findc = sub.add_parser("find-code", help="Find functions in the indexed project for a request")
    findc.add_argument("request", help='e.g. "login fails with TypeError in check_password"')
    findc.add_argument("--db", help="Index file (default: <home>/project-index.sqlite3)")
    findc.add_argument("--limit", type=int, default=8)
    findc.add_argument("--full", action="store_true", help="Print whole function bodies")
    blib = sub.add_parser("build-codelib", help="Build the central code library from CodeSearchNet parquet")
    blib.add_argument("parquet", nargs="+", help="CodeSearchNet parquet files")
    blib.add_argument("--db", help="Library file (default: <home>/codelib.sqlite3)")
    blib.add_argument("--languages", nargs="*", help="Keep only these languages (e.g. python javascript)")
    blib.add_argument("--limit", type=int, help="Stop after this many functions")
    fex = sub.add_parser("find-example", help="Search the central code library")
    fex.add_argument("request")
    fex.add_argument("--db", help="Library file (default: <home>/codelib.sqlite3)")
    fex.add_argument("--language")
    fex.add_argument("--limit", type=int, default=5)
    aso = sub.add_parser("ask-so", help="Search Stack Overflow accepted answers (cached, CC BY-SA)")
    aso.add_argument("query")
    aso.add_argument("--tagged", help="e.g. python;flask")
    aso.add_argument("--limit", type=int, default=3)
    ap = sub.add_parser("apply-patch", help="Apply SEARCH/REPLACE edit blocks to a project (all-or-nothing)")
    ap.add_argument("patch", help="File with SEARCH/REPLACE blocks ('-' = stdin)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true", help="Show the diff only")
    rpt = sub.add_parser("run-project-tests", help="Run a project's tests on a temporary copy")
    rpt.add_argument("--root", default=".")
    rpt.add_argument("--timeout", type=int, default=120)
    agent = sub.add_parser("agent", help="Full agent: search -> summarize -> patch -> test -> repeat -> remember -> report")
    agent.add_argument("request", help='e.g. "login fails with TypeError in check_password"')
    agent.add_argument("--root", default=".")
    agent.add_argument("--run", default="workspace/run-english-sft2", help="Model checkpoint that writes patches")
    agent.add_argument("--patches", nargs="+", help="Use these SEARCH/REPLACE files instead of a model (demo/testing)")
    agent.add_argument("--rounds", type=int, default=3)
    agent.add_argument("--timeout", type=int, default=120)
    agent.add_argument("--codelib", help="Central library file (default: <home>/codelib.sqlite3 if present)")
    agent.add_argument("--web", action="store_true", help="Also search Stack Overflow for the error")
    agent.add_argument("--allow-unverified", action="store_true", help="Keep changes in projects without tests")
    agent.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    tool = sub.add_parser("tool", help="Explicit tool invocation, never auto-executes model text")
    tool.add_argument("name")
    tool.add_argument("--args", default="{}", help="JSON arguments")
    tool.add_argument("--args-file", help="UTF-8 JSON file; easier for multiline code")
    return root


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args()
    try:
        if args.command == "doctor":
            from .hardware import inspect
            emit(inspect())
        elif args.command == "agent":
            from .agent import ModelBrain, ScriptedBrain, format_report, run_agent
            brain = (ScriptedBrain([Path(p).read_text(encoding="utf-8") for p in args.patches]) if args.patches
                     else ModelBrain(args.run, device=args.device))
            home = Path(args.home)
            report = run_agent(args.root, args.request, brain, index_db=home / "project-index.sqlite3",
                               experience_db=home / "experience.sqlite3",
                               codelib_db=args.codelib or home / "codelib.sqlite3", web=args.web,
                               rounds=args.rounds, timeout=args.timeout, allow_unverified=args.allow_unverified,
                               log=lambda msg: print(msg, file=sys.stderr))
            print(format_report(report))
        elif args.command == "apply-patch":
            from .patch import apply_patch
            text = sys.stdin.read() if args.patch == "-" else Path(args.patch).read_text(encoding="utf-8")
            result = apply_patch(args.root, text, dry_run=args.dry_run)
            print(result["diff"] or "(no changes)")
            print(("would change: " if args.dry_run else "changed: ") + ", ".join(result["files"]))
        elif args.command == "run-project-tests":
            from .projtest import run_project_tests
            result = run_project_tests(args.root, timeout=args.timeout)
            tail = result.pop("output_tail", "")
            emit(result)
            if result["status"] != "pass" and tail and not result.get("first_failure"):
                print(tail)
        elif args.command in {"build-codelib", "find-example"}:
            from .codelib import CodeLibrary
            lib = CodeLibrary(args.db or Path(args.home) / "codelib.sqlite3")
            if args.command == "build-codelib":
                result = lib.build_from_parquet(args.parquet, languages=args.languages, limit=args.limit)
                lib.compact()
                emit({**result, "library": str(lib.path), "stats": lib.stats()})
            else:
                hits = lib.search(args.request, language=args.language, limit=args.limit)
                if not hits:
                    print("no match")
                for h in hits:
                    print(f"[{h['language']}] {h['name']}  (score {h['score']:.2f})\n    {h['url']}")
                    print("    " + "\n    ".join(h["code"].splitlines()[:12]) + "\n")
                if hits:
                    print("license: each example keeps its source repository's license (see url)")
        elif args.command == "ask-so":
            from .stackexchange import StackExchange
            se = StackExchange(Path(args.home) / "stackexchange-cache.sqlite3")
            for r in se.search(args.query, tagged=args.tagged, limit=args.limit):
                print(f"■ {r['title']}  (score {r['score']}, by {r['author']})\n  {r['link']}  [{r['license']}]")
                print("  " + "\n  ".join(r["answer"].splitlines()[:15]) + "\n")
            if se.quota_remaining is not None:
                print(f"quota remaining today: {se.quota_remaining}")
        elif args.command in {"index-project", "find-code"}:
            from .projindex import ProjectIndex
            index = ProjectIndex(args.db or Path(args.home) / "project-index.sqlite3")
            if args.command == "index-project":
                result = index.index(args.root)
                if args.compact:
                    index.compact()
                emit({**result, **{"index": str(index.path)}, "stats": index.stats()})
            else:
                hits = index.search(args.request, limit=args.limit)
                if not hits:
                    print("no match (run index-project first, or use other words)")
                for h in hits:
                    print(f"{h['path']}:{h['start']}-{h['end']}  [{h['kind']}] {h['name']}  (score {h['score']:.2f})")
                    body = h["body"] if args.full else "\n".join(h["body"].splitlines()[:6])
                    print("    " + body.replace("\n", "\n    ") + "\n")
                if hits:
                    print("keywords:", ", ".join(hits[0]["keywords"]))
        elif args.command == "prepare":
            from .config import Config
            from .data import prepare
            emit(prepare(args.source, args.output, Config.load(args.config)))
        elif args.command == "build-corpus":
            from .corpus import build_corpus
            from .hardware import be_polite
            be_polite()
            emit(build_corpus(args.code, args.english, args.output, args.vocab_size, args.max_tokens,
                              tokenizer=args.tokenizer))
        elif args.command == "build-anneal":
            from .corpus import build_anneal
            from .hardware import be_polite
            be_polite()
            extras = []
            for item in args.extra:
                name, path = item.split("=", 1)
                extras.append((name, path, args.extra_share, 1))
            emit(build_anneal(args.output, args.tokenizer, args.code, args.pyedu, args.english, args.max_tokens,
                              personal_file=args.personal, extras=extras))
        elif args.command == "collect-personal":
            from .personal import collect
            emit(collect(args.folders, args.output))
        elif args.command == "collect-offensive":
            from .extra import offensive_collect
            emit(offensive_collect(args.root, args.output))
        elif args.command == "prepare-instructions":
            from .config import Config
            from .instruction import prepare_instructions
            emit(prepare_instructions(args.source, args.output, Config.load(args.config), args.tokenizer))
        elif args.command == "make-code-tasks":
            from .tasks import sft_examples
            out = Path(args.output)
            if out.exists():
                raise ValueError("Output exists; choose a new file")
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as stream:
                for row in sft_examples(args.count, args.seed, tuple(args.languages)):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            emit({"output": str(out), "examples": args.count})
        elif args.command == "train":
            from .config import Config
            from .train import train
            cfg = Config.load(args.config)
            if args.steps is not None:
                cfg.steps = args.steps
            if args.device:
                cfg.device = args.device
            cfg.__post_init__()
            emit(train(args.data, args.run, cfg, args.resume, args.init))
        elif args.command == "model-eval":
            from .inference import evaluate_instruction
            emit(evaluate_instruction(args.run, args.data, args.output))
        elif args.command == "code-eval":
            from .coder import evaluate_code
            output = args.output or str(Path(args.run) / "code-eval.json")
            emit(evaluate_code(args.run, output, args.samples, args.per_family, args.device,
                               agent=args.agent, repair_rounds=args.repair_rounds))
        elif args.command == "code":
            from .coder import CodeModel, Library, solve
            from .tasks import LANGUAGES
            tests = [json.loads(e) for e in args.example]
            params = ", ".join(a.strip() for a in args.args.split(",") if a.strip())
            prompt = f"Write a {LANGUAGES[args.language]} function `{args.function}({params})` that {args.task.rstrip('.')}."
            if tests:
                shown = ", ".join(json.dumps(a) for a in tests[0]["args"])
                prompt += f"\nExample: {args.function}({shown}) returns {json.dumps(tests[0]['expected'])}"
            model = CodeModel(args.run, args.device)
            if tests:
                result = solve(model, {"prompt": prompt, "language": args.language,
                                       "function": args.function, "tests": tests},
                               samples=args.samples, library=Library(Path(args.home) / "code-library.sqlite3"))
                result["verified_against"] = len(tests)
            else:
                result = {"code": model.sample(prompt, 1, greedy=True)[0], "status": "unverified",
                          "note": "No --example given, so the code was not tested. Add examples to verify it."}
            print(result["code"])
            emit({k: v for k, v in result.items() if k != "code"})
        elif args.command == "playground":
            from .playground import main as playground
            playground(args.run, args.device, temperature=args.temperature)
        elif args.command == "generate":
            from .train import generate
            emit({"text": generate(args.run, args.prompt, args.max_tokens,
                                   args.temperature, args.device)})
        else:
            from .tools import Tools
            tools = Tools(args.home)
            knowledge = tools.knowledge
            if args.command == "tools":
                emit(tools.describe())
            elif args.command == "tool":
                raw = Path(args.args_file).read_text(encoding="utf-8") if args.args_file else args.args
                emit(tools.call(args.name, json.loads(raw)))
            elif args.command == "stats":
                emit(knowledge.stats())
            elif args.command == "ingest":
                from .assistant import ingest
                emit(ingest(knowledge, args.source))
            elif args.command == "search":
                emit(knowledge.search(args.query, args.limit))
            elif args.command == "research":
                from .assistant import research
                emit(research(knowledge, args.query, args.limit))
            elif args.command == "refresh":
                from .assistant import refresh
                if args.cycles < 1 or args.interval < 1:
                    raise ValueError("cycles and interval must be positive")
                for cycle in range(args.cycles):
                    emit(refresh(knowledge))
                    if cycle + 1 < args.cycles:
                        time.sleep(args.interval)
            elif args.command in {"ask", "chat"}:
                from .assistant import answer
                if args.command == "ask":
                    emit(answer(knowledge, args.query, args.run, args.web, args.device))
                else:
                    from .chat import Chat
                    session = Chat(tools, args.run, args.web, args.device, args.model_only)
                    print("LocalBrain: พิมพ์ /help เพื่อดูความสามารถ; /exit เพื่อออก")
                    if args.model_only:
                        print("โหมดโมเดลฝึกเอง: คำถามทั่วไปเข้าโมเดลโดยตรง; ไม่มีคำตอบตายตัว")
                        print("โมเดลทดลอง ยังอาจตอบผิดหรือไม่รู้เรื่อง")
                    else:
                        print("โหมดค้นเอกสารและเครื่องมือ; โมเดลฝึกเองยังเป็นรุ่นทดลอง")
                    while True:
                        try:
                            query = input("คุณ> ").strip()
                        except EOFError:
                            break
                        if query in {"/exit", "/quit"}:
                            break
                        if query:
                            try:
                                emit_chat(session.handle(query))
                            except Exception as error:
                                print(f"\nLocalBrain> ทำรายการไม่สำเร็จ: {error}\n")
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        emit({"error": type(error).__name__, "message": str(error)})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
