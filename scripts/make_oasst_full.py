"""Extract far more chat pairs from OpenAssistant oasst2 (already downloaded, Apache-2.0).

The first SFT mix used only root-prompt -> first-reply pairs (5,284 rows). oasst2 is a tree
of multi-turn conversations, so every later assistant turn was thrown away. Here every good
English assistant reply becomes a pair, with the earlier turns written into the question as
plain "User:/Assistant:" history so the single-turn SFT format still works. oasst2 already
contains the oasst1 messages, so oasst1 is not downloaded separately.

Quality filters match chatdata.oasst_pairs: English, not deleted, not synthetic, review not
failed, best-ranked reply (rank 0). Replies that open with a refusal are dropped (owner's
choice, same as chatdata). Over-long pairs are later dropped by prepare-instructions.
"""
import json
from pathlib import Path

import pyarrow.parquet as pq

from localbrain.chatdata import REFUSAL

FILES = ["workspace/raw-chat/oasst2-train.parquet", "workspace/raw-chat/oasst2-validation.parquet"]
OUT = Path("workspace/raw-sft/oasst-full.jsonl")
MAX_CHARS = 3000  # keep pairs that can fit a 512-token context


def ok(m):
    return m and m["lang"] == "en" and not m["deleted"] and not m["synthetic"] and m["review_result"] is not False


def main():
    rows = []
    for f in FILES:
        rows += pq.read_table(f, columns=["message_id", "parent_id", "text", "role", "lang", "deleted",
                                          "review_result", "rank", "synthetic"]).to_pylist()
    by_id = {r["message_id"]: r for r in rows}
    out, first, deeper, refused, too_long = [], 0, 0, 0, 0
    for r in rows:
        if r["role"] != "assistant" or not ok(r) or r["rank"] not in (0, None):
            continue
        chain, cur, good = [], by_id.get(r["parent_id"]), True
        while cur is not None:
            if not ok(cur):
                good = False
                break
            chain.append(cur)
            cur = by_id.get(cur["parent_id"]) if cur["parent_id"] else None
        if not good or not chain:
            continue
        chain.reverse()  # root first; last item is the user prompt being answered
        answer = r["text"].strip()
        if REFUSAL.match(answer):
            refused += 1
            continue
        if len(chain) == 1:
            question = chain[0]["text"].strip()
            first += 1
        else:
            history = "\n\n".join(("User: " if m["role"] == "prompter" else "Assistant: ") + m["text"].strip()
                                  for m in chain[:-1])
            question = history + "\n\nUser: " + chain[-1]["text"].strip()
            deeper += 1
        if len(question) + len(answer) > MAX_CHARS:
            too_long += 1
            continue
        out.append({"question": question, "answer": answer, "source": "oasst2-full"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as h:
        for row in out:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(out), "first_turn": first, "later_turns": deeper,
                      "refusals_dropped": refused, "too_long_dropped": too_long, "output": str(OUT)}))


if __name__ == "__main__":
    main()
