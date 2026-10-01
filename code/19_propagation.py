"""
19_propagation.py — Aim 4. Does a cause travel from the committee's finding, to a
recommendation, to a vehicle that can bind?

Unit of analysis. A state and cause-of-death domain pair, restricted to pairs in
which the committee documented the cause among its leading causes. Burden and
recommendation content use the same language-model classification as Aims 1 and
2, so the funnel and the concordance analysis speak one vocabulary.

Channels. Four vehicles are coded where public records allow.
  contract      Medicaid managed care procurement and contract documents, a cause
                counted when it appears within 300 characters of a maternal term;
                tied to accountability when withhold or incentive language sits
                within 1,000 characters of that mention
  pqc           perinatal quality collaborative maternal initiatives
  legislation   enacted state maternal health legislation, 2018 to 2026
  consortium    statewide maternal health consortium or task force priorities
A channel with no record for a state is missing for that state, not negative.

Timing. A recommendation can only propagate forward. The primary contract
linkage therefore uses only documents dated in or after the year of the state's
review report; undated documents are excluded from it. All documents are used in
a sensitivity analysis.

Baseline. A funnel alone cannot distinguish propagation from background: if every
Medicaid contract names behavioral health, a cause will "reach" contract language
whether or not a committee recommended it. For each channel the share of
documented pairs present in that channel is therefore compared between pairs the
committee recommended on and pairs it did not, with a state-clustered bootstrap
interval on the difference.

Output: results/propagation.json, results/contract_doc_scan.csv
"""
from __future__ import annotations
import collections, csv, importlib.util, json, os, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
TAG = os.environ.get("MMRC_WINDOW_TAG", "")  # e.g. "_ctx150"; empty = primary windows
SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
SEED, BOOT = 20260924, 2000


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, HERE / path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


chain = _load("chain", "04_accountability_chain.py")
tight = _load("tight", "06_withhold_proximity.py")

MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
BINDING = {"Legislature or statute", "Medicaid or state health agency",
           "Managed care organization or payer",
           "Licensing, accreditation, or perinatal collaborative"}


def doc_year(name: str) -> int | None:
    """Year a procurement document was issued, read from its file name."""
    s = name.lower()
    m = re.findall(r"(?<!\d)(20[12]\d)(?!\d)", s)
    if m:
        return max(int(x) for x in m)
    m = re.search(rf"(?:{MONTHS})[a-z]*[-_ ]?(\d{{2}})(?!\d)", s)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r"(?<!\d)\d{1,2}[-_.]\d{1,2}[-_.](\d{2})(?!\d)", s)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r"\bfy[-_ ]?(\d{2})(?!\d)", s)
    if m:
        return 2000 + int(m.group(1))
    return None


def scan_contracts() -> list[dict]:
    """Per-document, per-domain presence and same-provision withhold, cached."""
    out = RES / f"contract_doc_scan{TAG}.csv"
    if out.exists():
        return list(csv.DictReader(open(out)))
    compiled = {d: re.compile(p, re.I) for d, p in chain.CAUSE_TERMS.items()}
    rows = []
    for name, abbr in chain.STATE_ABBR.items():
        sd = chain.RFP / name
        if not sd.exists():
            continue
        for f in sd.rglob("*.txt"):
            try:
                text = f.read_text(errors="ignore")
            except Exception:
                continue
            spans = [m.start() for m in chain.MATERNAL_CONTEXT.finditer(text)]
            has_w = bool(chain.WITHHOLD.search(text))
            yr = doc_year(f.name)
            for d, pat in compiled.items():
                present = chain.context_hits(text, pat, spans) > 0 if spans else False
                w = present and has_w and tight.tight_withhold(text, pat, spans)
                if present:
                    rows.append({"state": abbr, "file": f.name, "year": yr or "",
                                 "domain": d, "present": True, "withhold": bool(w)})
            rows.append({"state": abbr, "file": f.name, "year": yr or "", "domain": "_doc",
                         "present": True, "withhold": False})
        print(f"  scanned {name}", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def load_json(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def main() -> None:
    recs = list(csv.DictReader(open(SRC / "mmrc_recommendations.csv")))
    labels = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "llm_labels_passA.jsonl")}
    cause_map = json.load(open(RES / "cause_label_map.json"))["map"]
    findings = list(csv.DictReader(open(SRC / "mmrc_findings.csv")))

    burden = collections.defaultdict(set)
    report_year = collections.defaultdict(list)
    for f in findings:
        d = cause_map.get(f["leading_cause"].strip())
        if d and d != "Not a cause of death":
            burden[f["state"]].add(d)
    rec_dom = collections.defaultdict(set)
    rec_binding = collections.defaultdict(set)
    for i, r in enumerate(recs):
        lab = labels.get(i, {})
        d = lab.get("cause_domain")
        try:
            report_year[r["state"]].append(int(r["report_year"]))
        except ValueError:
            pass
        if d and d != "Cross-cutting":
            rec_dom[r["state"]].add(d)
            if lab.get("actor_authority") in BINDING:
                rec_binding[r["state"]].add(d)
    ryear = {s: min(v) for s, v in report_year.items()}

    # ---- contract channel -----------------------------------------------------
    scan = scan_contracts()
    docs = collections.defaultdict(set)
    for r in scan:
        if r["domain"] == "_doc":
            docs[r["state"]].add((r["file"], r["year"]))
    contract_states = set(docs)

    def contract_sets(postdating: bool):
        pres, wh = collections.defaultdict(set), collections.defaultdict(set)
        for r in scan:
            if r["domain"] == "_doc":
                continue
            st = r["state"]
            if postdating:
                if not r["year"] or st not in ryear or int(r["year"]) < ryear[st]:
                    continue
            if str(r["present"]) == "True":
                pres[st].add(r["domain"])
            if str(r["withhold"]) == "True":
                wh[st].add(r["domain"])
        return pres, wh

    dated = {s: sum(1 for _, y in v if y) for s, v in docs.items()}
    post = {s: sum(1 for _, y in v if y and s in ryear and int(y) >= ryear[s])
            for s, v in docs.items()}

    # ---- other channels, where the files exist -------------------------------
    pqc_aim = load_json("channel_pqc_aim.json")
    legis = load_json("channel_legislation_consortium.json")

    def channel_from(obj, getter):
        if not obj:
            return None, set()
        present, covered = collections.defaultdict(set), set()
        for st, v in obj.get("states", {}).items():
            got = getter(v)
            if got is None:
                continue
            covered.add(st)
            present[st] |= set(got)
        return present, covered

    def pqc_get(v):
        p = v.get("pqc") or {}
        if p.get("status") != "found":
            return None
        return {d for ini in p.get("initiatives", []) for d in ini.get("domains", [])}

    def legis_get(v):
        l = v.get("legislation") or {}
        if l.get("status") not in ("found", "none found"):
            return None
        return {d for b in l.get("bills", []) for d in b.get("domains", [])}

    def cons_get(v):
        c = v.get("consortium") or {}
        if c.get("status") != "found":
            return None
        return set(c.get("priority_domains", []))

    def legis_post_get_factory():
        def g(v, st=None):
            return None
        return g
    pqc_p, pqc_cov = channel_from(pqc_aim, pqc_get)
    # legislation enacted in or after the year of the state's report
    leg_post, leg_post_cov = collections.defaultdict(set), set()
    if legis:
        for st, v in legis.get("states", {}).items():
            l = v.get("legislation") or {}
            if l.get("status") not in ("found", "none found") or st not in ryear:
                continue
            leg_post_cov.add(st)
            for b in l.get("bills", []):
                if isinstance(b.get("year"), int) and b["year"] >= ryear[st]:
                    leg_post[st] |= set(b.get("domains", []))
    leg_p, leg_cov = channel_from(legis, legis_get)
    con_p, con_cov = channel_from(legis, cons_get)
    aim = {}
    if pqc_aim:
        for st, v in pqc_aim.get("states", {}).items():
            a = v.get("aim") or {}
            if a.get("status") == "found":
                aim[st] = a.get("enrolled")
    den = json.load(open(RES / "denominators.json"))["states"]

    # ---- pairs ---------------------------------------------------------------
    domains = [d for d in chain.CAUSE_TERMS]
    pairs = []
    for st in sorted(burden):
        if st == "US":
            continue
        for d in sorted(burden[st]):
            if d not in domains:
                continue
            pairs.append({"state": st, "domain": d, "recommended": d in rec_dom[st],
                          "binding_actor": d in rec_binding[st]})

    channels = {}
    for tag, postdating in (("contract_postdating", True), ("contract_all_documents", False)):
        pres, wh = contract_sets(postdating)
        cov = {s for s in contract_states if (post[s] if postdating else len(docs[s])) > 0}
        channels[tag] = (pres, cov)
        channels[tag.replace("contract", "withhold")] = (wh, cov)
    if pqc_p is not None:
        channels["pqc"] = (pqc_p, pqc_cov)
    if leg_p is not None:
        channels["legislation"] = (leg_p, leg_cov)
        channels["legislation_postdating"] = (leg_post, leg_post_cov)
    if con_p is not None:
        channels["consortium"] = (con_p, con_cov)

    rng = np.random.default_rng(SEED)

    def contrast(sub: list[dict]) -> dict:
        """Share present among recommended vs not recommended pairs, with a
        bootstrap over states."""
        by_state = collections.defaultdict(list)
        for p in sub:
            by_state[p["state"]].append(p)
        states = list(by_state)

        def diff(ps):
            a = [p["present"] for p in ps if p["recommended"]]
            b = [p["present"] for p in ps if not p["recommended"]]
            if not a or not b:
                return None, (np.mean(a) if a else None), (np.mean(b) if b else None)
            return float(np.mean(a) - np.mean(b)), float(np.mean(a)), float(np.mean(b))

        d0, ra, rb = diff(sub)
        bs = []
        for _ in range(BOOT):
            pick = rng.choice(len(states), len(states), replace=True)
            ps = [p for k in pick for p in by_state[states[k]]]
            d_, _, _ = diff(ps)
            if d_ is not None:
                bs.append(d_)
        n_rec = sum(p["recommended"] for p in sub)
        return {"n_pairs": len(sub), "n_states": len(states),
                "n_recommended": n_rec, "n_not_recommended": len(sub) - n_rec,
                "present_if_recommended": ra, "present_if_not_recommended": rb,
                "difference": d0,
                "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                if bs else None}

    GROUP = {"Mental health conditions": "Behavioral health",
             "Substance use disorder": "Behavioral health",
             "Injury, homicide, and violence": "Injury and violence",
             "Other medical conditions": "Other medical conditions"}

    def adjusted(sub: list[dict]) -> dict | None:
        """Risk difference for presence given a recommendation, stratified by cause
        group with Mantel-Haenszel weights, and a bootstrap over states. Cause mix
        is the obvious confounder: behavioral health is both the most recommended
        cause and the most common topic of Medicaid contract language. The
        Mantel-Haenszel estimator is used rather than a regression because several
        strata have no events among pairs without a recommendation, which leaves a
        logistic model without a finite estimate."""
        if len(sub) < 30:
            return None
        grp = lambda p: GROUP.get(p["domain"], "Obstetric and cardiovascular")

        def mh(ps):
            num = den = 0.0
            strata = collections.defaultdict(lambda: [[], []])
            for p in ps:
                strata[grp(p)][0 if p["recommended"] else 1].append(p["present"])
            for a_, b_ in strata.values():
                n1, n0 = len(a_), len(b_)
                if n1 and n0:
                    w = n1 * n0 / (n1 + n0)
                    num += w * (np.mean(a_) - np.mean(b_)); den += w
            return num / den if den else None

        est = mh(sub)
        by_state = collections.defaultdict(list)
        for p in sub:
            by_state[p["state"]].append(p)
        states = list(by_state); bs = []
        for _ in range(BOOT):
            pick = rng.choice(len(states), len(states), replace=True)
            v = mh([p for k in pick for p in by_state[states[k]]])
            if v is not None:
                bs.append(v)
        return {"mh_risk_difference": est,
                "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None,
                "n_pairs": len(sub)}

    def fixed_effects(sub: list[dict]) -> dict | None:
        """Linear probability model of presence on recommendation with state and
        cause fixed effects, standard errors clustered by state. State fixed
        effects absorb anything about a state that raises both its committee's
        attention and its policy activity; cause fixed effects absorb the fact
        that some causes are named in contracts everywhere."""
        import pandas as pd, statsmodels.formula.api as smf
        if len(sub) < 30:
            return None
        df = pd.DataFrame([{"state": p["state"], "domain": p["domain"],
                            "present": int(p["present"]), "recommended": int(p["recommended"])}
                           for p in sub])
        if df["recommended"].nunique() < 2 or df["present"].nunique() < 2:
            return None
        try:
            m = smf.ols("present ~ recommended + C(state) + C(domain)", data=df).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(df["state"])[0]})
            ci = m.conf_int().loc["recommended"]
            varying = df.groupby("state")["recommended"].nunique()
            ident = varying[varying > 1].index
            return {"mdd": float(2.80 * (ci[1] - ci[0]) / 3.92),
                    "identifying_states": int(len(ident)),
                    "identifying_pairs": int(df["state"].isin(ident).sum()),
                    "difference": float(m.params["recommended"]),
                    "ci": [float(ci[0]), float(ci[1])], "p": float(m.pvalues["recommended"]),
                    "n_pairs": int(len(df)), "n_states": int(df["state"].nunique())}
        except Exception as e:
            return {"error": str(e)[:200]}

    def by_group(sub):
        out = {}
        for g in ("Behavioral health", "Obstetric and cardiovascular", "Injury and violence",
                  "Other medical conditions"):
            ps = [p for p in sub if GROUP.get(p["domain"], "Obstetric and cardiovascular") == g]
            a = [p["present"] for p in ps if p["recommended"]]
            b = [p["present"] for p in ps if not p["recommended"]]
            out[g] = {"n_rec": len(a), "n_not": len(b),
                      "present_if_recommended": float(np.mean(a)) if a else None,
                      "present_if_not_recommended": float(np.mean(b)) if b else None}
        return out

    results = {}
    for tag, (pres, cov) in channels.items():
        sub = [dict(p, present=p["domain"] in pres.get(p["state"], set()))
               for p in pairs if p["state"] in cov]
        funnel = []
        for d in domains:
            ps = [p for p in sub if p["domain"] == d]
            if len(ps) < 3:
                continue
            funnel.append({"domain": d, "documented": len(ps),
                           "recommended": sum(p["recommended"] for p in ps),
                           "recommended_and_present": sum(p["recommended"] and p["present"]
                                                          for p in ps),
                           "present_not_recommended": sum((not p["recommended"]) and p["present"]
                                                          for p in ps)})
        results[tag] = {"states_covered": sorted(cov), "n_states_covered": len(cov),
                        "funnel": funnel, "overall": contrast(sub),
                        "adjusted_for_cause_group": adjusted(sub), "by_cause_group": by_group(sub),
                        "fixed_effects": fixed_effects(sub),
                        "binding_actor": _binding_contrast(sub)}

    # ---- moderators of propagation to contract language ---------------------
    mods = {}
    pres, cov = channels["contract_postdating"]
    sub = [dict(p, present=p["domain"] in pres.get(p["state"], set()))
           for p in pairs if p["state"] in cov and p["recommended"]]
    for name, key in (("aim_enrolled", lambda s: aim.get(s)),
                      ("postpartum_12mo_extension",
                       lambda s: den.get(s, {}).get("postpartum_12mo_extension"))):
        grp = collections.defaultdict(list)
        for p in sub:
            v = key(p["state"])
            if v is not None:
                grp[str(bool(v))].append(p["present"])
        mods[name] = {k: {"n_pairs": len(v), "share_present": float(np.mean(v))}
                      for k, v in grp.items()}

    json.dump({"n_pairs_documented": len(pairs),
               "n_states_with_burden": len({p['state'] for p in pairs}),
               "report_year_by_state": ryear,
               "contract_documents": {s: {"all": len(docs[s]), "dated": dated[s],
                                          "postdating_report": post[s]} for s in docs},
               "channels": results, "moderators_contract_postdating": mods,
               "channel_files_present": {"pqc_aim": pqc_aim is not None,
                                         "legislation_consortium": legis is not None}},
              open(RES / f"propagation{TAG}.json", "w"), indent=1)

    print(f"documented state-cause pairs: {len(pairs)} across "
          f"{len({p['state'] for p in pairs})} states")
    for tag, r in results.items():
        o = r["overall"]
        f = lambda v: "  n/a" if v is None else f"{100*v:5.1f}"
        ci = o["ci"] or [float('nan')] * 2
        print(f"{tag:28s} states {r['n_states_covered']:2d}  pairs {o['n_pairs']:3d}  "
              f"present | rec {f(o['present_if_recommended'])}%  | not rec "
              f"{f(o['present_if_not_recommended'])}%  diff "
              f"{f(o['difference'])} ({100*ci[0]:5.1f} to {100*ci[1]:5.1f})")
        fe = r.get("fixed_effects") or {}
        if "difference" in fe:
            print(f"{'':28s} state and cause fixed effects {100*fe['difference']:5.1f} "
                  f"({100*fe['ci'][0]:5.1f} to {100*fe['ci'][1]:5.1f})")
        a = r.get("adjusted_for_cause_group") or {}
        if a.get("mh_risk_difference") is not None:
            print(f"{'':28s} cause-group-stratified difference {100*a['mh_risk_difference']:5.1f} "
                  f"({100*a['ci'][0]:5.1f} to {100*a['ci'][1]:5.1f})")


def _binding_contrast(sub: list[dict]) -> dict:
    """Among recommended pairs, presence by whether a binding actor was named."""
    rec = [p for p in sub if p["recommended"]]
    a = [p["present"] for p in rec if p["binding_actor"]]
    b = [p["present"] for p in rec if not p["binding_actor"]]
    return {"n_binding": len(a), "n_nonbinding": len(b),
            "share_present_binding": float(np.mean(a)) if a else None,
            "share_present_nonbinding": float(np.mean(b)) if b else None}


if __name__ == "__main__":
    main()
