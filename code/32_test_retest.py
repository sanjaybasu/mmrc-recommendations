"""
32_test_retest.py — Test-retest reliability of the primary model.

Claude Opus 5.5 reclassifies a random 250 recommendations with the identical prompts
and settings used for the full run, in a new random order and batching, and
agreement with the original run is reported per field. This measures run-to-run
stability under default (nondeterministic) sampling and sensitivity to batch
composition and order.

Output: results/retest_labels.jsonl, results/retest_imap.jsonl, results/retest_scores.json
"""
from __future__ import annotations
import importlib.util, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N, SEED, MODEL = 250, 20260928, "claude-opus-5-5"


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "code" / f)
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m)
    return m


def main() -> None:
    dm = _load("dm", "25_dual_model_classify.py"); lc, im = dm.lc, dm.im
    recs = lc.load_recs()
    rng = random.Random(SEED)
    sub = rng.sample(recs, N); rng.shuffle(sub)
    fields = ["cause_domain", "policy_lever", "actor_authority", "coverage_period"]
    allowed = {"cause_domain": set(lc.CAUSE_DOMAINS), "policy_lever": set(lc.LEVERS),
               "actor_authority": set(lc.AUTHORITY), "coverage_period": set(lc.PERIODS)}
    dm.run_task(MODEL, sub, RES / "retest_labels.jsonl",
                lambda b: lc.PROMPT_A.format(schema=lc.SCHEMA, items=lc.fmt(b)),
                fields, lambda r: all(r.get(f) in allowed[f] for f in fields))
    reg = im.registry(); keys = {k for k, _ in reg} | {"none"}
    dm.run_task(MODEL, sub, RES / "retest_imap.jsonl",
                lambda b: im.PROMPT_A.format(menu=im.menu(reg), items="\n".join(
                    f"[{x['id']}] {x['text'][:1200].replace(chr(10), ' ')}" for x in b)),
                ["intervention"], lambda r: r.get("intervention") in keys)

    s = importlib.util.spec_from_file_location("sv", ROOT / "code/30_score_validation.py")
    sv = importlib.util.module_from_spec(s); sys.modules["sv"] = sv; s.loader.exec_module(sv)
    orig = {json.loads(l)["id"]: json.loads(l) for l in open(RES / f"labels_{MODEL}.jsonl")}
    oi = {json.loads(l)["id"]: json.loads(l)["intervention"] for l in open(RES / f"imap_{MODEL}.jsonl")}
    rt = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "retest_labels.jsonl")}
    ri = {json.loads(l)["id"]: json.loads(l)["intervention"] for l in open(RES / "retest_imap.jsonl")}
    out = {"model": MODEL, "n": len(rt), "fields": {}}
    for f in fields:
        ids = sorted(set(rt) & set(orig))
        k, po = sv.kappa([orig[i][f] for i in ids], [rt[i][f] for i in ids])
        out["fields"][f] = {"kappa": round(k, 3), "agreement": round(po, 3), "n": len(ids)}
    ids = sorted(set(ri) & set(oi))
    k, po = sv.kappa([oi[i] for i in ids], [ri[i] for i in ids])
    out["fields"]["intervention"] = {"kappa": round(k, 3), "agreement": round(po, 3), "n": len(ids)}
    json.dump(out, open(RES / "retest_scores.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
