"""
07b_llm_causes.py — Map the corpus's free-text leading-cause labels onto the
same cause vocabulary used for recommendations.

Committees write their own cause labels, and the corpus carries 422 distinct
strings for what are roughly eleven conditions. The burden side of the
alignment analysis has to speak the same vocabulary as the recommendation side
or the comparison is between two different classifications rather than between
burden and attention. Distinct strings are classified once and cached.

Usage:  python code/07b_llm_causes.py
Output: results/cause_label_map.json
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

import anthropic

SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
OUT = Path(__file__).resolve().parent.parent / "results"
MODEL = "claude-opus-5"
CHUNK = 60

DOMAINS = ["Mental health conditions", "Substance use disorder", "Cardiovascular conditions",
           "Cardiomyopathy", "Hemorrhage", "Infection", "Embolism", "Hypertensive disorders",
           "Cerebrovascular accident", "Injury, homicide, and violence",
           "Other medical conditions", "Not a cause of death"]

PROMPT = """These are cause-of-death labels written by US state maternal mortality review
committees. Map each to one category.

Categories: {doms}

Rules. "Infection" covers sepsis, COVID-19, influenza, and pneumonia. "Embolism" covers
amniotic fluid embolism and thrombotic or pulmonary embolism. "Cardiovascular conditions"
excludes cardiomyopathy and hypertensive disorders, which have their own categories.
Overdose and poisoning go to "Substance use disorder". Suicide goes to "Mental health
conditions". Homicide, motor vehicle crash, and other injury go to "Injury, homicide, and
violence". Use "Not a cause of death" for a string that names a contributing factor, a
population, a year, or a data artifact rather than a cause.

Return a JSON array of {{"label": <the input string verbatim>, "domain": <one category>}}.
Output only the array.

Labels:
{items}"""


def main() -> None:
    import argparse, importlib.util, sys
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    model = a.model
    outname = "cause_label_map.json" if model == MODEL else f"cause_label_map_{model}.json"
    spec = importlib.util.spec_from_file_location("dm", Path(__file__).parent / "25_dual_model_classify.py")
    dm = importlib.util.module_from_spec(spec); sys.modules["dm"] = dm; spec.loader.exec_module(dm)
    call = dm.make_caller(model)
    rows = list(csv.DictReader(open(SRC / "mmrc_findings.csv")))
    labels = sorted({r["leading_cause"].strip() for r in rows if r["leading_cause"].strip()})
    print(f"{len(labels)} distinct leading-cause strings")

    cache = {}
    if (OUT / outname).exists():
        cache = json.load(open(OUT / outname)).get("map", {})
    todo = [l for l in labels if l not in cache]
    print(f"{len(cache)} cached, {len(todo)} to classify")

    for i in range(0, len(todo), CHUNK):
        part = todo[i:i + CHUNK]
        items = "\n".join(f"- {l[:200]}" for l in part)
        txt = call(PROMPT.format(doms=DOMAINS, items=items))
        txt = re.sub(r"^```(?:json)?\s*|```\s*$", "", txt.strip(), flags=re.M).strip()
        j, k = txt.find("["), txt.rfind("]")
        for o in json.loads(txt[j:k + 1]):
            lab, dom = o.get("label", "").strip(), o.get("domain")
            if dom in DOMAINS:
                # match back to the exact input string, which may have been truncated
                hit = next((l for l in part if l[:200].strip() == lab or l == lab), None)
                if hit:
                    cache[hit] = dom
        print(f"  {min(i+CHUNK, len(todo))}/{len(todo)}", flush=True)

    miss = [l for l in labels if l not in cache]
    json.dump({"model": model, "n_labels": len(labels), "n_mapped": len(cache),
               "unmapped": miss, "map": cache},
              open(OUT / outname, "w"), indent=1)
    print(f"\nmapped {len(cache)}/{len(labels)}; unmapped {len(miss)}")
    import collections
    for k, v in collections.Counter(cache.values()).most_common():
        print(f"  {k:34s} {v:4d}")


if __name__ == "__main__":
    main()
