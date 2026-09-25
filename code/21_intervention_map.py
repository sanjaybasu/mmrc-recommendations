"""
21_intervention_map.py — Which specific intervention, if any, does each
recommendation ask for?

The policy lever says what kind of thing a recommendation asks someone to do. It
does not say which intervention, and evidence attaches to interventions, not to
levers: "screening or referral" contains low-dose aspirin prophylaxis, which has
demonstrated benefit, and violence screening, which does not. Grading a lever as a
whole therefore mixes them. This pass maps every recommendation to one entry of
the model's intervention registry, or to "No specific intervention" when it asks
for something the registry does not contain or for nothing specific at all. The
evidence grade then attaches to the recommendation through the intervention it
names.

Two passes with different prompts, as for the other fields; agreement is reported
as a reliability statistic for the automated step only.

Usage:  python code/21_intervention_map.py [--pass A|B]
Output: results/intervention_map_pass{A,B}.jsonl
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
MODEL = "claude-opus-5"
BATCH, WORKERS = 25, 8


def registry() -> list[tuple[str, str]]:
    s = importlib.util.spec_from_file_location("iv", HERE / "12_interventions.py")
    m = importlib.util.module_from_spec(s); sys.modules["iv"] = m; s.loader.exec_module(m)
    P = json.load(open(RES / "model_params.json"))["params"]
    return [(i.key, i.label) for i in m._registry(P)]


def menu(reg) -> str:
    lines = [f'  "{k}": {lab}' for k, lab in reg]
    lines.append('  "none": No specific intervention in this list (process, awareness, '
                 'general quality improvement, research, or an intervention not listed)')
    return "\n".join(lines)


PROMPT_A = """You are coding state maternal mortality review committee recommendations.
For each recommendation, choose the ONE intervention from the list that the recommendation
most directly asks someone to implement. Choose "none" if it asks for none of them.

Guidance. A recommendation to train clinicians on hemorrhage recognition is
"clinician_education", not "hemorrhage_bundle", unless it asks for bundle or protocol
adoption. A recommendation to adopt or implement AIM hemorrhage protocols is
"hemorrhage_bundle". Recommendations to screen for depression and refer are
"depression_prevention" only when they ask for preventive counseling or treatment for
people at risk; screening alone with no treatment attached is "none". Recommendations
about doulas are "doula_support"; about home visiting are "nurse_home_visiting"; about
Medicaid postpartum coverage length are "postpartum_extension"; about buprenorphine,
methadone, or MOUD access or coverage are "moud_access"; about data systems, MMRC
process, or surveillance are "surveillance".

Interventions:
{menu}

Return a JSON array, one object per input in order: {{"id": <int>, "intervention": "<key>"}}.

Recommendations:
{items}"""

PROMPT_B = """Below are prevention recommendations from state maternal mortality reviews,
followed by a fixed list of interventions. First decide, for each recommendation, what
concrete action it requests and who would perform it. Then map it to the single
intervention key that matches that action, or "none" if no key matches. Be strict: a
recommendation that only raises awareness, educates, or studies a problem is not an
implementation of a clinical intervention, and education of clinicians maps to
"clinician_education".

Recommendations:
{items}

Interventions:
{menu}

Return only a JSON array: [{{"id": <int>, "intervention": "<key>"}}, ...] in input order."""


def call(client, prompt, tries=5):
    for k in range(tries):
        try:
            m = client.messages.create(model=MODEL, max_tokens=8000,
                                       messages=[{"role": "user", "content": prompt}])
            txt = "\n".join(b.text for b in m.content if getattr(b, "type", "") == "text")
            s, e = txt.find("["), txt.rfind("]")
            return json.loads(txt[s:e + 1])
        except Exception as ex:
            if k == tries - 1:
                print(f"  FAILED: {str(ex)[:120]}", file=sys.stderr)
                return None
            time.sleep(4 * (k + 1))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--pass", dest="which", default="A")
    a = ap.parse_args()
    reg = registry(); keys = {k for k, _ in reg} | {"none"}
    recs = list(csv.DictReader(open(SRC / "mmrc_recommendations.csv")))
    out = RES / f"intervention_map_pass{a.which}.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [(i, r["recommendation"].replace("\n", " ").strip()) for i, r in enumerate(recs)
            if i not in done]
    tmpl = PROMPT_A if a.which == "A" else PROMPT_B
    client = anthropic.Anthropic()
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]

    def one(b):
        items = "\n".join(f"[{i}] {t[:1200]}" for i, t in b)
        res = call(client, tmpl.format(menu=menu(reg), items=items)) or []
        got = {int(r["id"]): r.get("intervention") for r in res if "id" in r}
        return [{"id": i, "intervention": got[i] if got.get(i) in keys else None}
                for i, _ in b if i in got]

    n = len(done)
    with open(out, "a") as fh, ThreadPoolExecutor(WORKERS) as pool:
        for rows in pool.map(one, batches):
            for r in rows:
                if r["intervention"]:
                    fh.write(json.dumps(r) + "\n"); n += 1
            fh.flush()
            print(f"  {n}/{len(recs)}", flush=True)


if __name__ == "__main__":
    main()
