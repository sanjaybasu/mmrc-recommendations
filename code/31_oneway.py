"""
31_oneway.py — One-way sensitivity analysis of the evidence-weighted and recommended
portfolios, each against no implementation.

Allocations are held at their base-case coverage, as in the probabilistic analysis.
Each parameter moves to its low and high bound with everything else at base case:
every intervention's relative risk, unit cost, and maximum reach, and the
calibration, valuation, and cost parameters. Reported for each portfolio are
pregnancy-related deaths averted and the cost per QALY gained relative to no
implementation.

Output: results/oneway.json
"""
from __future__ import annotations
import copy, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
N, SEED = 60_000, 20260924


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m)
    return m


pc = _load("pc", "13_portfolio_cea.py")
ms = pc.ms

P_KEYS = ["prmr_for_model", "smm_rate", "cost_smm_increment", "utility_healthy_reproductive_age_female",
          "life_expectancy_female_30", "discount_rate", "inc_postpartum_hemorrhage",
          "inc_preeclampsia_eclampsia", "inc_perinatal_depression", "inc_oud_pregnancy",
          "prev_oud_pregnancy", "prev_chronic_hypertension"]
LABEL = {"prmr_for_model": "Pregnancy-related mortality ratio (calibration)",
         "smm_rate": "Severe maternal morbidity rate (calibration)",
         "cost_smm_increment": "Hospital cost of severe maternal morbidity",
         "utility_healthy_reproductive_age_female": "Health utility, women of reproductive age",
         "life_expectancy_female_30": "Life expectancy at age 30",
         "discount_rate": "Discount rate",
         "inc_postpartum_hemorrhage": "Incidence of postpartum hemorrhage",
         "inc_preeclampsia_eclampsia": "Incidence of preeclampsia or eclampsia",
         "inc_perinatal_depression": "Incidence of perinatal depression",
         "inc_oud_pregnancy": "Incidence of opioid use disorder complications",
         "prev_oud_pregnancy": "Prevalence of opioid use disorder",
         "prev_chronic_hypertension": "Prevalence of chronic hypertension"}


def summarize(m, cov):
    o = m.outcomes({k: min(v, m.ints[k].max_reach) for k, v in cov.items()})
    return {"deaths": o["deaths_averted"], "icer": o["cost_per_qaly"]}


def main() -> None:
    P = ms.load_params()
    ishares = pc.intervention_shares_from(RES / "intervention_map_passA.jsonl")
    m0 = pc.Model(P, N, seed=SEED)
    cov = {"recommended": m0.requested_full(ishares)}
    cov["evidence"] = m0.greedy(m0.outcomes(cov["recommended"])["program_cost"], steps=120)
    base = {k: summarize(m0, c) for k, c in cov.items()}
    rows = []

    # intervention attributes: no recalibration needed
    for key, i in m0.ints.items():
        if cov["evidence"].get(key, 0) == 0 and cov["recommended"].get(key, 0) == 0:
            continue
        for attr, lo_a, hi_a, lab in (("rr", "rr_lo", "rr_hi", "relative risk"),
                                      ("unit_cost", "cost_lo", "cost_hi", "unit cost"),
                                      ("max_reach", "reach_lo", "reach_hi", "maximum reach")):
            lo, hi = getattr(i, lo_a), getattr(i, hi_a)
            if lo is None or hi is None or lo == hi:
                continue
            res = {}
            for tag, v in (("low", lo), ("high", hi)):
                old = getattr(i, attr); setattr(i, attr, v)
                if attr == "unit_cost":
                    pass
                res[tag] = {k: summarize(m0, c) for k, c in cov.items()}
                setattr(i, attr, old)
            rows.append({"parameter": f"{i.label}: {lab}", "low": lo, "high": hi, **res})

    # model parameters: rebuild and recalibrate
    for key in P_KEYS:
        e = P.get(key)
        if not isinstance(e, dict) or e.get("low") is None or e.get("high") is None or e["low"] == e["high"]:
            continue
        res = {}
        for tag in ("low", "high"):
            Q = copy.deepcopy(P); Q[key]["value"] = e[tag]
            try:
                m = pc.Model(Q, N, seed=SEED)
                res[tag] = {k: summarize(m, c) for k, c in cov.items()}
            except Exception as ex:
                res[tag] = None
        if res["low"] and res["high"]:
            rows.append({"parameter": LABEL.get(key, key), "low": e["low"], "high": e["high"], **res})

    for r in rows:
        for p in ("evidence", "recommended"):
            r[f"swing_deaths_{p}"] = abs(r["high"][p]["deaths"] - r["low"][p]["deaths"])
            r[f"swing_icer_{p}"] = abs(r["high"][p]["icer"] - r["low"][p]["icer"])
    json.dump({"base": base, "rows": rows, "n_cohort": N}, open(RES / "oneway.json", "w"), indent=1)
    top = sorted(rows, key=lambda r: -r["swing_deaths_evidence"])[:6]
    for r in top:
        print(f"{r['parameter'][:60]:60s} deaths {r['low']['evidence']['deaths']:.1f} to "
              f"{r['high']['evidence']['deaths']:.1f}")
    print("base", base)


if __name__ == "__main__":
    main()
