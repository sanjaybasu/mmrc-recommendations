"""
26_adopt_dual_labels.py — Make the two-developer classification the one the whole
pipeline reads.

Claude Opus 5.5 becomes the primary label set and GPT-6-astra the independent
second coder. They are written to the file names the downstream scripts already
read (llm_labels_pass{A,B}.jsonl, intervention_map_pass{A,B}.jsonl,
cause_label_map.json), so every analysis reruns on them without other changes and
agreement between "pass A" and "pass B" is now agreement between model families.
The earlier single-model files are archived, not deleted.

Output: the files above; results/archive_opus5/*
"""
from __future__ import annotations
import json, shutil
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
PRIMARY, SECOND = "claude-opus-5-5", "gpt-6-astra"


def main() -> None:
    arc = RES / "archive_opus5"; arc.mkdir(exist_ok=True)
    for f in ("llm_labels_passA.jsonl", "llm_labels_passB.jsonl", "intervention_map_passA.jsonl",
              "intervention_map_passB.jsonl", "cause_label_map.json"):
        if (RES / f).exists() and not (arc / f).exists():
            shutil.copy2(RES / f, arc / f)
    n = {}
    for src, dst in ((f"labels_{PRIMARY}.jsonl", "llm_labels_passA.jsonl"),
                     (f"labels_{SECOND}.jsonl", "llm_labels_passB.jsonl"),
                     (f"imap_{PRIMARY}.jsonl", "intervention_map_passA.jsonl"),
                     (f"imap_{SECOND}.jsonl", "intervention_map_passB.jsonl")):
        rows = {json.loads(l)["id"]: l for l in open(RES / src)}
        (RES / dst).write_text("".join(rows[k] for k in sorted(rows)))
        n[dst] = len(rows)
    shutil.copy2(RES / f"cause_label_map_{PRIMARY}.json", RES / "cause_label_map.json")
    json.dump({"primary": PRIMARY, "second": SECOND, "counts": n},
              open(RES / "label_provenance.json", "w"), indent=1)
    print(n)


if __name__ == "__main__":
    main()
