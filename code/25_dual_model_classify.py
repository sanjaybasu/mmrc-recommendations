"""
25_dual_model_classify.py — Classify every recommendation independently with two
frontier models from different developers.

Each model receives the identical prompts used for the primary classification:
the recommendation-field prompt (cause, lever, actor, coverage period) and the
intervention prompt. Neither model sees the other's output. Agreement between
them measures reliability across model families, which is more informative than
agreement between two prompts to one model, because errors of one model family
are less likely to be shared by the other. Disagreements go to physician
adjudication; a separate blinded physician sample is the reference standard.

Usage:  python code/25_dual_model_classify.py --model claude-opus-5-5
        python code/25_dual_model_classify.py --model gpt-6-astra
Output: results/labels_<model>.jsonl, results/imap_<model>.jsonl
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, ROOT / "code" / fname)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m


lc = _load("lc", "07_llm_classify.py")
im = _load("im", "21_intervention_map.py")


def make_caller(model: str):
    if model.startswith("claude"):
        import anthropic
        client = anthropic.Anthropic()

        def call(prompt: str) -> str:
            m = client.messages.create(model=model, max_tokens=16000,
                                       messages=[{"role": "user", "content": prompt}])
            return "\n".join(b.text for b in m.content if getattr(b, "type", "") == "text")
    else:
        import openai
        client = openai.OpenAI()

        def call(prompt: str) -> str:
            r = client.chat.completions.create(model=model,
                                               messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
    return call


def with_retry(fn, prompt, parse, tries=5):
    for k in range(tries):
        try:
            out = parse(fn(prompt))
            if out:
                return out
            raise ValueError("unparseable")
        except Exception as e:
            if k == tries - 1:
                print(f"  FAILED: {str(e)[:150]}", file=sys.stderr)
                return None
            time.sleep(5 * (k + 1))


def parse_array(txt: str):
    s, e = txt.find("["), txt.rfind("]")
    return json.loads(txt[s:e + 1]) if s >= 0 and e > s else None


def run_task(model, recs, outp, make_prompt, keep, valid, workers=8, batch=25):
    done = {json.loads(l)["id"] for l in open(outp)} if outp.exists() else set()
    todo = [r for r in recs if r["id"] not in done]
    call = make_caller(model)
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]

    def one(b):
        res = with_retry(call, make_prompt(b), parse_array) or []
        got = {}
        for r in res:
            try:
                got[int(r["id"])] = r
            except Exception:
                continue
        rows = []
        for x in b:
            r = got.get(x["id"])
            if r and valid(r):
                rows.append({"id": x["id"], **{k: r.get(k) for k in keep}})
        return rows

    n = len(done)
    with open(outp, "a") as fh, ThreadPoolExecutor(workers) as pool:
        for rows in pool.map(one, batches):
            for r in rows:
                fh.write(json.dumps(r) + "\n"); n += 1
            fh.flush()
            print(f"  {outp.name}: {n}/{len(recs)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    a = ap.parse_args()
    recs = lc.load_recs()
    tag = a.model.replace("/", "_")

    fields = ["cause_domain", "policy_lever", "actor_authority", "coverage_period"]
    allowed = {"cause_domain": set(lc.CAUSE_DOMAINS), "policy_lever": set(lc.LEVERS),
               "actor_authority": set(lc.AUTHORITY), "coverage_period": set(lc.PERIODS)}
    run_task(a.model, recs, RES / f"labels_{tag}.jsonl",
             lambda b: lc.PROMPT_A.format(schema=lc.SCHEMA, items=lc.fmt(b)),
             fields, lambda r: all(r.get(f) in allowed[f] for f in fields))

    reg = im.registry(); keys = {k for k, _ in reg} | {"none"}
    run_task(a.model, recs, RES / f"imap_{tag}.jsonl",
             lambda b: im.PROMPT_A.format(menu=im.menu(reg), items="\n".join(
                 f"[{x['id']}] {x['text'][:1200].replace(chr(10), ' ')}" for x in b)),
             ["intervention"], lambda r: r.get("intervention") in keys)


if __name__ == "__main__":
    main()
