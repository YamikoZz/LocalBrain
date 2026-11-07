"""The LocalBrain agent: all 9 steps in one loop.

    1. request      -> keywords                                   (projindex.keywords)
    2. check code   -> index the project, find relevant functions (projindex)
    3. research     -> experience library, central library, Stack Overflow (optional)
    4. summarize    -> short plan the user can read; only this goes to the model, not raw results
    5. fix          -> the "brain" writes SEARCH/REPLACE blocks   (patch.apply_patch)
    6. verify       -> run the project's tests on a copy          (projtest)
    7. repeat       -> show the new failure, try again (bounded rounds); worse results are reverted
    8. remember     -> store the patch that made tests pass       (experience)
    9. report       -> files changed, diff, tests before/after, sources, warnings

The tests decide. A change is kept only when the project's tests pass afterwards; without
tests nothing is written unless the caller explicitly allows an unverified change.

The brain is pluggable: ModelBrain uses a LocalBrain checkpoint; ScriptedBrain replays
given patches (tests and demos). The 20M model is far too small to write useful patches;
the loop is built so a bigger model can be dropped in without changing it.
"""
import re
from pathlib import Path

from .experience import Experience, error_signature
from .patch import PatchError, apply_patch, revert
from .projindex import ProjectIndex, keywords
from .projtest import run_project_tests

PATCH_FORMAT = ("Reply with SEARCH/REPLACE blocks only:\n"
                "path/to/file.py\n<<<<<<< SEARCH\nexact current lines\n=======\nnew lines\n>>>>>>> REPLACE")


class ScriptedBrain:
    """Returns pre-written replies in order (for tests and demos)."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def write_patch(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


class ModelBrain:
    """A LocalBrain checkpoint writes the patch. Prompt is trimmed to the model's context."""
    def __init__(self, run, device="auto", max_tokens=300):
        from .coder import CodeModel
        self.model = CodeModel(run, device=device)
        self.max_tokens = max_tokens

    def write_patch(self, prompt):
        tok, ctx = self.model.tokenizer, self.model.cfg.context
        budget = ctx - self.max_tokens - 16
        ids = tok.encode(prompt).ids
        if len(ids) > budget:  # keep the end: the task, the failure and the format come last
            prompt = tok.decode(ids[-budget:])
        return self.model.sample(prompt, n=1, greedy=True, max_tokens=self.max_tokens, raw=True)[0]


def _fails(result):
    if result["status"] in {"error", "timeout"}:
        return float("inf")
    return result.get("failed") or 0


def summarize(request, hits, failure, past, examples, answers):
    """Step 4: a short extractive summary (deterministic; no model needed)."""
    lines = [f"Task: {request.strip()}"]
    err = error_signature(failure)
    if err:
        lines.append(f"Error: {err}")
        where = re.findall(r'File "<project>[\\/](.+?)", line (\d+)', failure)
        if where:
            lines.append("Raised at: " + ", ".join(f"{f}:{n}" for f, n in where[-2:]))
    if hits:
        lines.append("Relevant code: " + ", ".join(f"{h['path']}:{h['start']} {h['name']}" for h in hits[:4]))
    if past:
        lines.append(f"Similar past fix: {past[0]['request'][:80]} (files: {past[0]['files']})")
    if answers:
        lines.append(f"Stack Overflow: {answers[0]['title'][:90]} ({answers[0]['link']})")
    if examples:
        lines.append(f"Reference example: {examples[0]['name']} ({examples[0]['url']})")
    return "\n".join(lines)


def build_prompt(summary, hits, failure, past, max_chars=6000):
    parts = [summary, ""]
    if past:
        parts += ["A patch that fixed a similar problem before:", past[0]["patch"].strip()[:1200], ""]
    for h in hits[:3]:
        parts += [f"{h['path']} (lines {h['start']}-{h['end']}):", h["body"][:1500], ""]
    if failure:
        parts += ["Failing test output:", failure.strip()[-1200:], ""]
    parts.append(PATCH_FORMAT)
    text = "\n".join(parts)
    return text if len(text) <= max_chars else text[-max_chars:]


def run_agent(root, request, brain, index_db, experience_db=None, codelib_db=None, web=False,
              rounds=3, timeout=120, allow_unverified=False, log=print):
    root = Path(root).resolve()
    report = {"request": request, "root": str(root), "rounds": [], "sources": [], "warnings": [],
              "changed": [], "diff": "", "status": "unchanged"}

    # 1-2. keywords, index, relevant code
    index = ProjectIndex(index_db)
    index.index(root)
    hits = index.search(request, limit=8)
    is_test = lambda h: bool(re.search(r"(^|/)tests?/|(^|/)test_[^/]*$|\.test\.", h["path"]))
    hits = sorted(hits, key=is_test)[:5]  # code under change first, its tests after (stable order)
    report["keywords"] = keywords(request)
    log(f"[1-2] keywords {report['keywords'][:8]} -> {len(hits)} relevant functions")

    # 6 (baseline). tests before any change
    before = run_project_tests(root, timeout=timeout)
    report["tests_before"] = {k: before.get(k) for k in ("status", "tests", "failed")}
    failure = before.get("first_failure") or (before.get("output_tail", "") if before["status"] == "error" else "")
    log(f"[6] tests before: {before['status']} ({before.get('failed')} failing of {before.get('tests')})")
    if before["status"] == "pass" and not allow_unverified:
        report["warnings"].append("tests already pass; nothing proves a change is needed or correct")

    # 3. research
    experience = Experience(experience_db) if experience_db else None
    err = error_signature(failure)
    past = experience.search(request, err) if experience else []
    examples, answers = [], []
    if codelib_db and Path(codelib_db).exists():
        from .codelib import CodeLibrary
        examples = CodeLibrary(codelib_db).search(request, limit=2)
        report["warnings"] += ["library examples are references; each keeps its source repo's license"] if examples else []
    if web and err:
        from .stackexchange import StackExchange
        try:
            answers = StackExchange(Path(index_db).with_name("stackexchange-cache.sqlite3")).search(err, limit=2)
        except Exception as error:  # network trouble must not stop the agent
            report["warnings"].append(f"Stack Overflow search failed: {error}")
    report["sources"] = ([f"past fix #{p['id']}" for p in past[:1]] + [e["url"] for e in examples]
                         + [a["link"] for a in answers])
    log(f"[3] research: {len(past)} past fixes, {len(examples)} examples, {len(answers)} SO answers")

    # 4. summary
    summary = summarize(request, hits, failure, past, examples, answers)
    report["summary"] = summary
    log("[4] summary:\n    " + summary.replace("\n", "\n    "))

    # 5-7. fix -> verify -> repeat
    best = before
    kept = []
    for n in range(1, rounds + 1):
        prompt = build_prompt(summary, hits, failure, past if n == 1 else [])
        reply = brain.write_patch(prompt)
        entry = {"round": n}
        try:
            applied = apply_patch(root, reply)
        except PatchError as error:
            entry.update(result="patch-rejected", detail=str(error)[:300])
            report["rounds"].append(entry)
            log(f"[5] round {n}: patch rejected: {str(error).splitlines()[0]}")
            failure = f"Your previous reply could not be applied: {error}\n" + failure
            continue
        after = run_project_tests(root, timeout=timeout)
        entry.update(files=applied["files"], tests=after["status"], failed=after.get("failed"))
        improved = _fails(after) < _fails(best) or (after["status"] == "pass" and best["status"] != "pass")
        if after["status"] == "pass" or improved or (before["status"] == "no-tests" and allow_unverified):
            applied["reply"] = reply
            kept.append(applied)
            best = after
            entry["result"] = "kept"
        else:
            revert(applied["backup"])
            entry["result"] = "reverted (tests not better)"
        report["rounds"].append(entry)
        log(f"[5-6] round {n}: {entry['result']} -> tests {after['status']} ({after.get('failed')} failing)")
        if after["status"] == "pass":
            break
        failure = after.get("first_failure") or after.get("output_tail", "")

    success = best["status"] == "pass" and kept
    unverified_ok = before["status"] == "no-tests" and allow_unverified and kept
    if not (success or unverified_ok):
        for applied in reversed(kept):  # leave the project exactly as it was
            revert(applied["backup"])
        kept = []
        if before["status"] == "no-tests":
            report["warnings"].append("project has no tests; changes were not kept (use allow_unverified)")
    report["tests_after"] = {k: best.get(k) for k in ("status", "tests", "failed")}
    report["changed"] = sorted({f for a in kept for f in a["files"]})
    report["diff"] = "".join(a["diff"] for a in kept)
    report["status"] = "fixed" if success else ("changed-unverified" if unverified_ok else "not-fixed")

    # 8. remember
    if success and experience:
        patch_text = "\n".join(a["reply"].strip() for a in kept)
        rid = experience.add(root, request, err, report["changed"], patch_text, summary)
        report["remembered"] = rid
        log(f"[8] saved to experience library (#{rid})")
    log(f"[9] {report['status']}: changed {report['changed'] or 'nothing'}")
    return report


def format_report(report):
    """Step 9: human-readable report."""
    tb, ta = report.get("tests_before", {}), report.get("tests_after", {})
    out = [f"Status: {report['status']}",
           f"Tests: {tb.get('status')} ({tb.get('failed')} failing) -> {ta.get('status')} ({ta.get('failed')} failing)",
           f"Changed: {', '.join(report['changed']) or 'nothing'}",
           f"Rounds: " + "; ".join(f"{r['round']}={r['result']}" for r in report["rounds"]) if report["rounds"] else "Rounds: none",
           "", "Summary:", report.get("summary", "")]
    if report["diff"]:
        out += ["", "Diff:", report["diff"]]
    if report["sources"]:
        out += ["", "Sources: " + ", ".join(report["sources"])]
    for w in report["warnings"]:
        out.append("Warning: " + w)
    return "\n".join(out)
