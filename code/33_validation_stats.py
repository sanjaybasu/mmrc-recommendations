"""
33_validation_stats.py — Accuracy statistics for the classification that the
headline results rest on.

1. Diagnostic accuracy against blinded physician coding for the three yes/no
   decisions behind the main results (requested an intervention with evidence of
   benefit; addressed an obstetric or cardiovascular cause; named a binding
   actor) and for naming any registry intervention: sensitivity, specificity,
   positive and negative predictive value, each with a Wilson 95% CI, and
   bootstrap 95% CIs on every Cohen kappa.
2. Bounds on the final labels. For each decision, recommendations fall into three
   sets: both models agreed, the disagreement was adjudicated, or the models
   disagreed and it was not adjudicated. The corpus-level share for the decision
   is bounded by assigning every unadjudicated disagreement to either model's label.
3. Deterministic keyword audit of the full text: mentions of tranexamic acid,
   uterotonic agents, and aspirin, independent of any model.

Output: results/validation_stats.json
"""
from __future__ import annotations
import csv, importlib.util, json, math, re, sys
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent.parent
RES, VAL = ROOT / "results", ROOT / "validation"
BINDING = {"Legislature or statute", "Medicaid or state health agency",
           "Managed care organization or payer"}
OBCV = {"Hemorrhage", "Hypertensive disorders", "Cardiomyopathy", "Cardiovascular conditions",
        "Embolism", "Infection", "Cerebrovascular accident"}
SEED, BOOT = 20260929, 2000


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "code" / f)
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m)
    return m


def jl(p, field=None):
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(p)}
    return {k: (v[field] if field else v) for k, v in rows.items()}


def main() -> None:
    sv = _load("sv", "30_score_validation.py"); iv = _load("iv", "12_interventions.py"); lc = _load("lc", "07_llm_classify.py")
    P = json.load(open(RES / "model_params.json"))["params"]
    ints = iv.build(P)
    lab2key = {i.label: i.key for i in iv._registry(P)}; lab2key["No specific intervention"] = "none"
    eff = {i.key for i in ints if i.grade.startswith(("demonstrated", "moderate"))}

    key = [r for r in csv.DictReader(open(RES / "validation_key.csv")) if r["sheet"] == "Recommendations"]
    rows = {int(r["row"]): int(r["id_or_string"]) for r in key}
    wb = load_workbook(VAL / "MMRC_blinded_coding_ADJ.xlsx", data_only=True)
    ws = wb["Recommendations"]; hdr = [c.value for c in ws[1]]
    phys = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        i = rows[int(r[0])]
        phys[i] = {"cause": r[hdr.index("Cause of death addressed")],
                   "actor": r[hdr.index("Most authoritative actor named")],
                   "int": lab2key.get(r[hdr.index("Specific intervention requested")], "none"),
                   "lever": r[hdr.index("Policy lever")]}
    FL = jl(RES / "llm_labels_passA.jsonl"); FI = jl(RES / "intervention_map_passA.jsonl", "intervention")

    decisions = {
        "evidence_of_benefit": (lambda i: phys[i]["int"] in eff, lambda i: FI[i] in eff),
        "any_registry_intervention": (lambda i: phys[i]["int"] != "none", lambda i: FI[i] != "none"),
        "obstetric_cardiovascular_cause": (lambda i: phys[i]["cause"] in OBCV, lambda i: FL[i]["cause_domain"] in OBCV),
        "binding_actor": (lambda i: phys[i]["actor"] in BINDING, lambda i: FL[i]["actor_authority"] in BINDING),
    }
    out = {"n": len(phys), "diagnostic": {}, "kappa_ci": {}, "bounds": {}, "keyword_audit": {}}
    ids = sorted(phys)
    for d, (truth, pred) in decisions.items():
        tp = sum(truth(i) and pred(i) for i in ids); fn = sum(truth(i) and not pred(i) for i in ids)
        fp = sum((not truth(i)) and pred(i) for i in ids); tn = sum((not truth(i)) and not pred(i) for i in ids)
        out["diagnostic"][d] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": [tp, tp + fn, wilson(tp, tp + fn)], "specificity": [tn, tn + fp, wilson(tn, tn + fp)],
            "ppv": [tp, tp + fp, wilson(tp, tp + fp)], "npv": [tn, tn + fn, wilson(tn, tn + fn)]}

    rng = np.random.default_rng(SEED)
    fields = {"cause_domain": (lambda i: phys[i]["cause"], lambda i: FL[i]["cause_domain"]),
              "policy_lever": (lambda i: phys[i]["lever"], lambda i: FL[i]["policy_lever"]),
              "intervention": (lambda i: phys[i]["int"], lambda i: FI[i]),
              "binding_actor": (lambda i: phys[i]["actor"] in BINDING, lambda i: FL[i]["actor_authority"] in BINDING)}
    for f, (a, b) in fields.items():
        k0, _ = sv.kappa([a(i) for i in ids], [b(i) for i in ids])
        bs = []
        for _ in range(BOOT):
            s = rng.choice(ids, len(ids), replace=True)
            try:
                k, _ = sv.kappa([a(i) for i in s], [b(i) for i in s])
                if np.isfinite(k):
                    bs.append(k)
            except ZeroDivisionError:
                pass
        out["kappa_ci"][f] = {"kappa": round(k0, 3), "ci": [round(float(np.percentile(bs, 2.5)), 3),
                                                           round(float(np.percentile(bs, 97.5)), 3)]}
    # inter-model kappa CIs on the full corpus
    M1 = jl(RES / "labels_claude-opus-5-5.jsonl"); M2 = jl(RES / "labels_gpt-6-astra.jsonl")
    I1 = jl(RES / "imap_claude-opus-5-5.jsonl", "intervention"); I2 = jl(RES / "imap_gpt-6-astra.jsonl", "intervention")
    allids = np.array(sorted(M1))
    for f in ("cause_domain", "policy_lever", "actor_authority"):
        bs = []
        for _ in range(500):
            s = rng.choice(allids, len(allids), replace=True)
            bs.append(sv.kappa([M1[i][f] for i in s], [M2[i][f] for i in s])[0])
        out["kappa_ci"][f"intermodel_{f}"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    bs = []
    for _ in range(500):
        s = rng.choice(allids, len(allids), replace=True)
        bs.append(sv.kappa([I1[i] for i in s], [I2[i] for i in s])[0])
    out["kappa_ci"]["intermodel_intervention"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

    # bounds on corpus-level shares
    adj_ids = {int(r["id_or_string"]) for r in csv.DictReader(open(RES / "adjudication_key.csv"))
               if r["sheet"] == "1 Priority"}
    N = len(FL)
    for d, f1, f2, ff in (
            ("evidence_of_benefit", lambda i: I1[i] in eff, lambda i: I2[i] in eff, lambda i: FI[i] in eff),
            ("binding_actor", lambda i: M1[i]["actor_authority"] in BINDING, lambda i: M2[i]["actor_authority"] in BINDING,
             lambda i: FL[i]["actor_authority"] in BINDING)):
        agree_pos = sum(f1(i) and f2(i) for i in FL)
        dis = [i for i in FL if f1(i) != f2(i)]
        dis_adj = [i for i in dis if i in adj_ids]
        dis_un = [i for i in dis if i not in adj_ids]
        adj_pos = sum(ff(i) for i in dis_adj)
        lo, hi = agree_pos + adj_pos, agree_pos + adj_pos + len(dis_un)
        out["bounds"][d] = {"final": sum(ff(i) for i in FL), "lower": lo, "upper": hi, "n": N,
                            "lower_pct": round(100 * lo / N, 1), "upper_pct": round(100 * hi / N, 1),
                            "disagreements": len(dis), "adjudicated": len(dis_adj), "unadjudicated": len(dis_un)}

    # keyword audit
    recs = lc.load_recs()
    pats = {"tranexamic_acid": r"tranexamic|\bTXA\b",
            "uterotonic": r"uterotonic|oxytocin|pitocin|methylergonovine|methergine|ergometrine|carboprost|hemabate|misoprostol",
            "aspirin": r"aspirin", "naloxone": r"naloxone|narcan", "firearm": r"firearm|gun\b|guns\b|lethal means"}
    for k, pat in pats.items():
        hits = [r["id"] for r in recs if re.search(pat, r["text"], re.I)]
        out["keyword_audit"][k] = {"mentions": len(hits),
                                   "coded_as": {x: sum(FI[i] == x for i in hits) for x in sorted({FI[i] for i in hits})}}
    json.dump(out, open(RES / "validation_stats.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("diagnostic", "bounds", "keyword_audit")}, indent=0)[:3000])
    print(out["kappa_ci"])


if __name__ == "__main__":
    main()
