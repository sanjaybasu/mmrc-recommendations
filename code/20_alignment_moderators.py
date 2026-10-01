"""
20_alignment_moderators.py — Aim 2, second part. Does alignment between what a
committee documents and what it recommends vary with the committee's resourcing,
its state's mortality, its membership, or the kind of cause?

Unit. A state and cause-of-death domain pair in which the committee documented
the cause among its leading causes; the same pairs as the concordance analysis.

Outcomes. Silence, whether the committee issued no recommendation on the cause,
modeled by logistic regression; and the concordance ratio, modeled by linear
regression on its logarithm after adding one half recommendation to every count so
that silent pairs stay in the model. Both use generalized estimating equations
with an exchangeable working correlation within state, so inference respects the
fact that pairs from one committee are not independent.

Predictors. Cause group (obstetric and cardiovascular, behavioral health, injury
and violence, other), report length in pages as the protocol's proxy for
committee resourcing, and the state's pregnancy-related mortality ratio. Medicaid
agency membership is read from the report roster and is available only where the
report prints one, so it enters a separate model on that subset.

Output: results/alignment_moderators.json
"""
from __future__ import annotations
import collections, csv, json, statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

RES = Path(__file__).resolve().parent.parent / "results"
SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"

GROUP = {"Hemorrhage": "Obstetric and cardiovascular",
         "Hypertensive disorders": "Obstetric and cardiovascular",
         "Cardiomyopathy": "Obstetric and cardiovascular",
         "Cardiovascular conditions": "Obstetric and cardiovascular",
         "Embolism": "Obstetric and cardiovascular",
         "Cerebrovascular accident": "Obstetric and cardiovascular",
         "Infection": "Obstetric and cardiovascular",
         "Mental health conditions": "Behavioral health",
         "Substance use disorder": "Behavioral health",
         "Injury, homicide, and violence": "Injury and violence",
         "Other medical conditions": "Other medical conditions"}


def main() -> None:
    recs = list(csv.DictReader(open(SRC / "mmrc_recommendations.csv")))
    labels = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "llm_labels_passA.jsonl")}
    cmap = json.load(open(RES / "cause_label_map.json"))["map"]
    F = list(csv.DictReader(open(SRC / "mmrc_findings.csv")))
    docs = list(csv.DictReader(open(SRC / "documents.csv")))
    comp = json.load(open(RES / "committee_composition.json"))["reports"]

    burden = collections.defaultdict(collections.Counter)
    for f in F:
        d = cmap.get(f["leading_cause"].strip())
        if d and d != "Not a cause of death":
            burden[f["state"]][d] += 1
    import re, importlib.util, sys
    spec = importlib.util.spec_from_file_location("chain", Path(__file__).parent / "04_accountability_chain.py")
    chain = importlib.util.module_from_spec(spec); sys.modules["chain"] = chain; spec.loader.exec_module(chain)
    term = {d: re.compile(p_, re.I) for d, p_ in chain.CAUSE_TERMS.items()}
    named = collections.defaultdict(set)   # causes named in the text of cross-cutting recommendations
    att = collections.defaultdict(collections.Counter)
    n_recs = collections.Counter()
    xclin = collections.defaultdict(int)   # cross-cutting clinical protocol recommendations
    for i, r in enumerate(recs):
        lab = labels.get(i, {})
        d = lab.get("cause_domain")
        n_recs[r["state"]] += 1
        if d and d != "Cross-cutting":
            att[r["state"]][d] += 1
        else:
            if lab.get("policy_lever") == "Clinical protocol or bundle":
                xclin[r["state"]] += 1
            for dd, pat in term.items():
                if pat.search(r.get("recommendation") or ""):
                    named[r["state"]].add(dd)

    # pregnancy-related mortality ratio per state: median over overall rows,
    # the same rule the model calibration uses
    prmr = collections.defaultdict(list)
    for f in F:
        if f["population_group"].strip().lower() in ("overall", "total", "all", "all women",
                                                     "statewide"):
            try:
                v = float(f["pregnancy_related_mortality_ratio"])
            except ValueError:
                continue
            if 0 < v < 200:
                prmr[f["state"]].append(v)
    prmr = {s: st.median(v) for s, v in prmr.items()}

    pages = collections.defaultdict(int)
    for d in docs:
        pages[d["jurisdiction"]] = max(pages[d["jurisdiction"]], int(d["n_pages"] or 0))
    medicaid = {}
    for r in comp:
        if r.get("roster_found"):
            v = r.get("medicaid_agency_member")
            medicaid[r["state"]] = bool(medicaid.get(r["state"])) or bool(v)

    rows = []
    for s, b in burden.items():
        if s == "US":
            continue
        tb, ta = sum(b.values()), sum(att[s].values())
        for d, n in b.items():
            if d not in GROUP:
                continue
            bshare = n / tb
            ashare = (att[s][d] + 0.5) / (ta + 0.5 * len(GROUP))
            obcv = GROUP[d] == "Obstetric and cardiovascular"
            rows.append({"state": s, "domain": d, "group": GROUP[d],
                         "silent": int(att[s][d] == 0),
                         # sensitivity: a cross-cutting clinical protocol
                         # recommendation counts toward every obstetric and
                         # cardiovascular cause the state documented
                         "silent_sens": int(att[s][d] == 0 and not (obcv and xclin[s] > 0)),
                         # intermediate rule: credit a cross-cutting recommendation to a
                         # cause only when its text names that condition
                         "silent_mid": int(att[s][d] == 0 and d not in named[s]),
                         "log_recs": float(np.log(n_recs[s])) if n_recs[s] else np.nan,
                         "n_causes": len(b),
                         "few_specific": int(ta < 5),
                         "log_concordance": float(np.log(ashare / bshare)),
                         "pages": pages.get(s) or np.nan,
                         "prmr": prmr.get(s, np.nan),
                         "medicaid_member": medicaid.get(s)})
    df = pd.DataFrame(rows)
    df["log_pages"] = np.log(df["pages"])
    df["prmr10"] = df["prmr"] / 10.0
    df["group"] = pd.Categorical(df["group"], categories=[
        "Behavioral health", "Obstetric and cardiovascular", "Injury and violence",
        "Other medical conditions"])

    out = {"n_pairs": int(len(df)), "n_states": int(df["state"].nunique()),
           "n_pairs_with_prmr": int(df["prmr"].notna().sum()),
           "n_states_with_prmr": int(df.loc[df["prmr"].notna(), "state"].nunique()),
           "n_states_with_roster": len(medicaid),
           "n_states_medicaid_member": int(sum(medicaid.values())),
           "models": {}}

    def fit(name, formula, data, family):
        data = data.dropna(subset=[c for c in ("log_pages", "prmr10") if c in formula])
        m = smf.gee(formula, groups="state", data=data, family=family,
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = m.conf_int()
        terms = {}
        for t in m.params.index:
            est, lo, hi = m.params[t], ci.loc[t, 0], ci.loc[t, 1]
            if isinstance(family, sm.families.Binomial):
                est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
            terms[t] = {"estimate": round(float(est), 3), "ci": [round(float(lo), 3),
                        round(float(hi), 3)], "p": round(float(m.pvalues[t]), 4)}
        out["models"][name] = {"formula": formula, "n_pairs": int(len(data)),
                               "n_states": int(data["state"].nunique()),
                               "scale": "odds ratio" if isinstance(family, sm.families.Binomial)
                               else "difference in log concordance", "terms": terms}
        print(f"\n{name}  (n = {len(data)} pairs, {data['state'].nunique()} states)")
        for t, v in terms.items():
            print(f"  {t:48s} {v['estimate']:7.3f} ({v['ci'][0]:.3f} to {v['ci'][1]:.3f})  "
                  f"p = {v['p']:.3f}")

    fam_b, fam_g = sm.families.Binomial(), sm.families.Gaussian()
    fit("silence_by_group", "silent ~ group", df, fam_b)
    fit("silence_recs_causes", "silent ~ group + log_recs + n_causes", df, fam_b)
    fit("silence_sensitivity_crosscutting", "silent_sens ~ group + log_recs + n_causes", df, fam_b)
    fit("silence_intermediate_rule", "silent_mid ~ group + log_recs + n_causes", df, fam_b)
    fit("silence_pages_recs", "silent ~ group + log_pages + log_recs + n_causes", df, fam_b)
    fit("silence_pages", "silent ~ group + log_pages", df, fam_b)
    fit("concordance_pages", "log_concordance ~ group + log_pages", df, fam_g)
    fit("silence_pages_prmr", "silent ~ group + log_pages + prmr10", df, fam_b)
    fit("concordance_pages_prmr", "log_concordance ~ group + log_pages + prmr10", df, fam_g)
    sub = df[df["medicaid_member"].notna()].copy()
    sub["medicaid_member"] = sub["medicaid_member"].astype(int)
    if sub["state"].nunique() >= 8 and sub["medicaid_member"].nunique() == 2:
        fit("silence_medicaid_member_subset", "silent ~ group + medicaid_member", sub, fam_b)

    out["silence_sens_by_group_raw"] = {
        g: {"pairs": int(len(x)), "silent": int(x["silent_sens"].sum()),
            "pct": round(100 * x["silent_sens"].mean(), 1)}
        for g, x in df.groupby("group", observed=True)}
    out["silence_mid_by_group_raw"] = {
        g: {"pairs": int(len(x)), "silent": int(x["silent_mid"].sum()),
            "pct": round(100 * x["silent_mid"].mean(), 1)}
        for g, x in df.groupby("group", observed=True)}
    out["pairs_with_fewer_than_5_specific_recs"] = int(df["few_specific"].sum())
    out["states_with_fewer_than_5_specific_recs"] = int(df.loc[df["few_specific"] == 1, "state"].nunique())
    # crude odds ratio, obstetric-cardiovascular vs behavioral health
    t = df[df["group"].isin(["Obstetric and cardiovascular", "Behavioral health"])]
    a_ = ((t["group"] == "Obstetric and cardiovascular") & (t["silent"] == 1)).sum()
    b_ = ((t["group"] == "Obstetric and cardiovascular") & (t["silent"] == 0)).sum()
    c_ = ((t["group"] == "Behavioral health") & (t["silent"] == 1)).sum()
    d_ = ((t["group"] == "Behavioral health") & (t["silent"] == 0)).sum()
    out["crude_or_obcv_vs_bh"] = round(float(a_ * d_ / (b_ * c_)), 2) if b_ * c_ else None
    # raw silence by group, for the reader
    out["silence_by_group_raw"] = {
        g: {"pairs": int(len(x)), "silent": int(x["silent"].sum()),
            "pct": round(100 * x["silent"].mean(), 1)}
        for g, x in df.groupby("group", observed=True)}
    json.dump(out, open(RES / "alignment_moderators.json", "w"), indent=1)
    print(f"\nwrote {RES/'alignment_moderators.json'}")


if __name__ == "__main__":
    main()
