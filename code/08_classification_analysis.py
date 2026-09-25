"""
08_classification_analysis.py — Aims 1 and 2 on the language-model labels.

Three things happen here.

Reliability. Two independent classification passes were run over every
recommendation with different prompts. Agreement between them is reported as
Cohen's kappa per field. This measures the stability of the automated step and
is not a substitute for human coding; a stratified sample is held out for
physician adjudication and its coder columns are deliberately empty.

Attention. The distribution of recommendations across cause domains, policy
levers, named actors, and coverage periods, with the rule-based pilot
classification reported beside it so the reader can see how much the
classification method moves each number.

Alignment. For each state and cause domain, a normalized concordance ratio: the
share of that state's recommendations addressing the cause divided by the share
of that state's documented leading causes that is the cause. A ratio of one
means attention tracks documented burden. Confidence intervals come from a
bootstrap over recommendations within state.

Output: results/classification.json, results/concordance.json
"""
from __future__ import annotations
import collections, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_shim import map_cause, map_recommendation, map_lever

SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
RES = Path(__file__).resolve().parent.parent / "results"
FIELDS = ["cause_domain", "policy_lever", "actor_authority", "coverage_period"]
BOOT = 2000
SEED = 20260924

BINDING = {"Legislature or statute", "Medicaid or state health agency",
           "Managed care organization or payer",
           "Licensing, accreditation, or perinatal collaborative"}


def kappa(a: list, b: list) -> tuple[float, float]:
    """Cohen's kappa and simple agreement for two aligned label vectors."""
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float((m.sum(0) / n) @ (m.sum(1) / n))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def load_labels(which: str) -> dict:
    f = RES / f"llm_labels_pass{which}.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(f)} if f.exists() else {}


def main() -> None:
    recs = list(csv.DictReader(open(SRC / "mmrc_recommendations.csv")))
    A, B = load_labels("A"), load_labels("B")          # A = adjudicated final, B = second model
    M1 = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "labels_claude-opus-5-5.jsonl")}
    M2 = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "labels_gpt-6-astra.jsonl")}
    both = sorted(set(M1) & set(M2))
    print(f"recommendations {len(recs)}; pass A {len(A)}; pass B {len(B)}; both {len(both)}")

    rel = {}
    for f in FIELDS:
        k, po = kappa([M1[i].get(f) for i in both], [M2[i].get(f) for i in both])
        rel[f] = {"kappa": round(k, 3), "agreement": round(po, 3), "n": len(both)}
        print(f"  {f:18s} kappa {k:5.3f}   agreement {po:5.3f}")

    # Pass A is the primary classification. Pass B is carried through the whole
    # analysis as a sensitivity so that no result rests on one prompt.
    cause_map = json.load(open(RES / "cause_label_map.json"))["map"]

    def tabulate(lab: dict) -> dict:
        t = {}
        for f in FIELDS:
            c = collections.Counter(lab[i].get(f) for i in lab)
            n = sum(c.values())
            t[f] = [{"label": k, "n": v, "pct": round(100 * v / n, 1)}
                    for k, v in c.most_common()]
        binding = sum(1 for i in lab if lab[i].get("actor_authority") in BINDING)
        t["pct_naming_binding_actor"] = round(100 * binding / len(lab), 1)
        return t

    tabA, tabB = tabulate(A), tabulate(M2)
    tabM1 = tabulate(M1)

    # Rule-based pilot, for comparison.
    rb_lever = collections.Counter(
        map_lever(r.get("recommendation"), r.get("category")) for r in recs)
    rb_lever.pop(None, None)
    rb_n = sum(rb_lever.values())

    # ---- burden and attention by state --------------------------------------
    findings = list(csv.DictReader(open(SRC / "mmrc_findings.csv")))
    burden = collections.defaultdict(collections.Counter)
    for f in findings:
        lab = f["leading_cause"].strip()
        d = cause_map.get(lab)
        if d and d != "Not a cause of death":
            burden[f["state"]][d] += 1

    attention = collections.defaultdict(collections.Counter)
    rec_state, rec_dom = [], []
    for i, r in enumerate(recs):
        d = A.get(i, {}).get("cause_domain")
        if d and d != "Cross-cutting":
            attention[r["state"]][d] += 1
        rec_state.append(r["state"])
        rec_dom.append(d)

    domains = sorted({d for s in burden.values() for d in s})
    rng = np.random.default_rng(SEED)
    rows, silence = [], []
    for d in domains:
        states = [s for s in burden if burden[s][d] > 0]
        ratios = []
        for s in states:
            bshare = burden[s][d] / sum(burden[s].values())
            ashare = attention[s][d] / max(sum(attention[s].values()), 1)
            ratios.append(ashare / bshare if bshare > 0 else np.nan)
        ratios = np.array([r for r in ratios if np.isfinite(r)])
        if len(ratios) < 3:
            continue
        bs = np.array([np.median(rng.choice(ratios, len(ratios), replace=True))
                       for _ in range(BOOT)])
        zero = sum(1 for s in states if attention[s][d] == 0)
        rows.append(dict(domain=d, n_states=len(states),
                         median_ratio=round(float(np.median(ratios)), 3),
                         ci=[round(float(np.percentile(bs, 2.5)), 3),
                             round(float(np.percentile(bs, 97.5)), 3)],
                         iqr=[round(float(np.percentile(ratios, 25)), 3),
                              round(float(np.percentile(ratios, 75)), 3)]))
        silence.append(dict(domain=d, states_documenting=len(states),
                            states_zero_recs=zero,
                            pct=round(100 * zero / len(states), 1)))

    print(f"\n{'cause domain':34s} {'states':>7s} {'silent':>7s} {'%':>6s} "
          f"{'concordance':>12s} {'95% CI':>18s}")
    for r, s in zip(rows, silence):
        print(f"{r['domain']:34s} {s['states_documenting']:7d} {s['states_zero_recs']:7d} "
              f"{s['pct']:6.1f} {r['median_ratio']:12.2f} "
              f"  [{r['ci'][0]:5.2f}, {r['ci'][1]:5.2f}]")

    json.dump({"reliability": rel, "primary_pass": "A", "n_recommendations": len(recs),
               "pass_A": tabA, "pass_B": tabB, "model_primary": tabM1,
               "rule_based_levers": [{"label": k, "n": v, "pct": round(100 * v / rb_n, 1)}
                                     for k, v in rb_lever.most_common()],
               "rule_based_n_classified": rb_n},
              open(RES / "classification.json", "w"), indent=1)
    json.dump({"bootstrap_draws": BOOT, "seed": SEED,
               "concordance": rows, "silence": silence,
               "n_states_with_burden": len(burden)},
              open(RES / "concordance.json", "w"), indent=1)
    print(f"\nwrote classification.json and concordance.json")


if __name__ == "__main__":
    main()
