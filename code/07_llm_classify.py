"""
07_llm_classify.py — Classify every MMRC recommendation against a fixed schema.

Replaces the rule-based mapping used in the pilot, which left half of
recommendations unclassified by cause and a third unclassified by lever.

Four fields per recommendation:
  cause_domain     the cause of pregnancy-related death addressed, or Cross-cutting
  policy_lever     what the recommendation asks someone to do
  actor_authority  the most binding body named, if any
  coverage_period  for coverage recommendations only: antepartum, postpartum, both, unspecified

Two independent passes are run. Pass A presents the schema and asks for direct
classification. Pass B is deliberately different: it presents the fields in a
different order, withholds the pilot's category frequencies, and asks the model
to reason about the recommendation's target before labeling. Agreement between
passes is reported as a reliability statistic for the automated step. It is NOT
a substitute for human adjudication; a stratified sample is written out for
physician coding, and no human label is ever machine-filled.

Usage:  python code/07_llm_classify.py [--limit N] [--pass A|B]
Output: results/llm_labels_pass{A,B}.jsonl, results/validation_sample.csv
"""
from __future__ import annotations
import argparse, csv, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
OUT = Path(__file__).resolve().parent.parent / "results"
MODEL = "claude-opus-5"
BATCH = 25
WORKERS = 8

CAUSE_DOMAINS = [
    "Mental health conditions", "Substance use disorder", "Cardiovascular conditions",
    "Cardiomyopathy", "Hemorrhage", "Infection", "Embolism", "Hypertensive disorders",
    "Cerebrovascular accident", "Injury, homicide, and violence", "Other medical conditions",
    "Cross-cutting",
]
LEVERS = [
    "Education or training", "Screening or referral", "Coverage or payment",
    "Care coordination or navigation", "Workforce or staffing", "Clinical protocol or bundle",
    "Community or social services", "Data, surveillance, or review",
]
AUTHORITY = [
    "Legislature or statute", "Medicaid or state health agency",
    "Managed care organization or payer", "Licensing, accreditation, or perinatal collaborative",
    "Named institution or facility", "Diffuse addressee", "No actor identified",
]
PERIODS = ["Antepartum", "Postpartum", "Both", "Unspecified", "Not a coverage recommendation"]

SCHEMA = f"""Return one JSON object per input, in an array, in the same order.
Each object: {{"id": <int>, "cause_domain": <one of {CAUSE_DOMAINS}>,
"policy_lever": <one of {LEVERS}>, "actor_authority": <one of {AUTHORITY}>,
"coverage_period": <one of {PERIODS}>}}

Definitions.
cause_domain: the cause of pregnancy-related death the recommendation is meant to prevent.
  Use "Cross-cutting" only when no single cause is addressed, for example recommendations
  about committee process, data systems, or general access.
policy_lever: what the recommendation asks someone to DO, independent of the cause.
  "Clinical protocol or bundle" means a standardized clinical process, checklist, drill,
  safety bundle, or level-of-care designation. "Education or training" means teaching
  people, including implicit-bias training and continuing education.
actor_authority: the MOST BINDING body the recommendation names. A body that can compel
  action outranks one that cannot. "Diffuse addressee" means it addresses providers,
  clinicians, communities, or stakeholders in general with no specific body named.
coverage_period: only for recommendations about insurance coverage, eligibility, or payment.
  Otherwise "Not a coverage recommendation".

Output only the JSON array. No prose."""

PROMPT_A = """You are classifying prevention recommendations issued by US state maternal
mortality review committees. Apply the schema exactly.

{schema}

Recommendations:
{items}"""

PROMPT_B = """Below are prevention recommendations from US state maternal mortality review
committees. For each one, first consider: who is being asked to act, what they are being
asked to do, and which cause of maternal death it would prevent. Then assign labels.

Judge only what the text says. Do not infer a cause that is not indicated, and do not
upgrade the named actor beyond what is written.

{schema}

Recommendations:
{items}"""


def load_recs() -> list[dict]:
    rows = list(csv.DictReader(open(SRC / "mmrc_recommendations.csv")))
    return [{"id": i, "state": r["state"], "year": r.get("report_year", ""),
             "category": (r.get("category") or "").strip(),
             "text": (r.get("recommendation") or "").strip()}
            for i, r in enumerate(rows)]


def fmt(batch: list[dict]) -> str:
    return "\n".join(
        f'{r["id"]}. [category: {r["category"][:90] or "none"}] {r["text"][:600]}' for r in batch)


def parse(txt: str) -> list | None:
    """Recover a list of label objects from a model reply.

    Accepts a bare JSON array, an array wrapped in a fenced block or prose, and
    a newline-delimited sequence of objects.
    """
    txt = re.sub(r"^```(?:json)?\s*|```\s*$", "", txt.strip(), flags=re.M).strip()
    try:
        v = json.loads(txt)
        return v if isinstance(v, list) else [v]
    except Exception:
        pass
    i, j = txt.find("["), txt.rfind("]")
    if i >= 0 and j > i:
        try:
            v = json.loads(txt[i:j + 1])
            if isinstance(v, list):
                return v
        except Exception:
            pass
    objs, dec = [], json.JSONDecoder()
    k = 0
    while k < len(txt):
        if txt[k] != "{":
            k += 1
            continue
        try:
            v, end = dec.raw_decode(txt, k)
        except Exception:
            k += 1
            continue
        objs.append(v)
        k = end
    return objs or None


def call(client, prompt: str, tries: int = 5):
    for k in range(tries):
        try:
            m = client.messages.create(model=MODEL, max_tokens=8000,
                                       messages=[{"role": "user", "content": prompt}])
            txt = "\n".join(b.text for b in m.content if getattr(b, "type", "") == "text")
            out = parse(txt)
            if out is None:
                raise ValueError("unparseable reply")
            return out
        except Exception as e:
            if k == tries - 1:
                print(f"    FAILED: {str(e)[:140]}", file=sys.stderr)
                return None
            time.sleep(4 * (k + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pass", dest="which", choices=["A", "B"], default="A")
    a = ap.parse_args()

    recs = load_recs()
    if a.limit:
        recs = recs[:a.limit]
    template = PROMPT_A if a.which == "A" else PROMPT_B
    outp = OUT / f"llm_labels_pass{a.which}.jsonl"

    done = set()
    if outp.exists():
        for line in open(outp):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [r for r in recs if r["id"] not in done]
    print(f"pass {a.which}: {len(recs)} recommendations, {len(done)} already labeled, {len(todo)} to do")

    client = anthropic.Anthropic()
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]

    def work(batch):
        return batch, call(client, template.format(schema=SCHEMA, items=fmt(batch)))

    n_done = 0
    with open(outp, "a") as fh, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for batch, res in ex.map(work, batches):
            n_done += len(batch)
            if not res:
                continue
            valid = {r["id"] for r in batch}
            for obj in res:
                if isinstance(obj, dict) and obj.get("id") in valid:
                    fh.write(json.dumps(obj) + "\n")
            fh.flush()
            print(f"  {n_done}/{len(todo)}", flush=True)

    # Stratified sample for physician adjudication. Labels are left EMPTY on purpose.
    if a.which == "A" and not a.limit and not (OUT / "validation_sample.csv").exists():
        labels = {json.loads(l)["id"]: json.loads(l) for l in open(outp)}
        by_dom: dict[str, list] = {}
        for r in recs:
            d = labels.get(r["id"], {}).get("cause_domain", "unlabeled")
            by_dom.setdefault(d, []).append(r)
        rng = random.Random(20260924)
        sample = []
        for d, rows in by_dom.items():
            sample += rng.sample(rows, min(len(rows), 25))
        with open(OUT / "validation_sample.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "state", "category", "recommendation",
                        "coder_cause_domain", "coder_policy_lever",
                        "coder_actor_authority", "coder_coverage_period", "coder_initials"])
            for r in sample:
                w.writerow([r["id"], r["state"], r["category"], r["text"], "", "", "", "", ""])
        print(f"\nwrote validation_sample.csv with {len(sample)} rows, coder columns intentionally blank")


if __name__ == "__main__":
    main()
