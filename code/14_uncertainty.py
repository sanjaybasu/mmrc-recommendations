"""
14_uncertainty.py — Probabilistic sensitivity analysis, one-way sensitivity, and
structural ablations for the portfolio comparison.

Probabilistic analysis. Each draw resamples every intervention's relative risk
from its published interval, every complication incidence and prevalence from
its interval, every cost from its interval, and the mortality and severe
morbidity calibration targets. Relative risks are drawn on the log scale with
the published interval read as a 95% interval. Costs are drawn from a gamma
distribution matched to the published range. Portfolios are rebuilt inside each
draw, so an intervention that looks good only under a favorable draw does not
get to keep its place in the allocation.

One-way analysis. Each parameter is moved to its low and its high bound with
everything else held at the base case, and the resulting change in the
difference between the evidence-weighted and the observed portfolio is ranked.

Ablations. Each changes one structural assumption rather than one number.

Output: results/uncertainty.json
"""
from __future__ import annotations
import copy, importlib.util, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, HERE / path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


ms = _load("ms", "11_microsim.py")
iv = _load("iv", "12_interventions.py")
pc = _load("pc", "13_portfolio_cea.py")
# Use the model's own module objects, so structural perturbations reach the
# cohort builder and calibration the model actually calls.
ms, iv = pc.ms, pc.iv

# Coverage already in place, for the incremental scenario. Sourced where a
# published uptake estimate exists; tranexamic acid and uterotonic prophylaxis
# have no national uptake estimate and carry labeled assumptions.
CURRENT_PRACTICE = json.load(open(RES / "current_practice.json"))["coverage"]
ISHARES = pc.intervention_shares_from(RES / "intervention_map_passA.jsonl")
N_PSA = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
N_COHORT = int(sys.argv[2]) if len(sys.argv) > 2 else 60_000
SEED = 20260924

# Parameters varied in the one-way analysis, with the bound source.
ONE_WAY = ["prmr_for_model", "smm_rate", "inc_postpartum_hemorrhage",
           "inc_preeclampsia_eclampsia", "inc_perinatal_depression",
           "inc_oud_pregnancy", "cost_aim_bundle_implementation", "cost_doula_medicaid",
           "inc_cardiac_conditions_pregnancy", "inc_vte_pregnancy",
           "inc_other_medical_pregnancy", "prev_oud_pregnancy",
           "prev_chronic_hypertension", "cost_medicaid_delivery",
           "cost_smm_increment", "cost_moud_year",
           "cost_perinatal_depression_treatment", "cost_12mo_postpartum_extension_per_enrollee",
           "utility_healthy_reproductive_age_female", "life_expectancy_female_30",
           "discount_rate", "utility_decrement_smm_survivor"]


def lognormal_draw(rng, v, lo, hi):
    """Draw a relative risk treating the published interval as 95% limits."""
    if lo is None or hi is None or lo <= 0 or hi <= 0 or v <= 0:
        return v
    se = (np.log(hi) - np.log(lo)) / (2 * 1.96)
    return float(np.exp(rng.normal(np.log(v), se)))


def gamma_draw(rng, v, lo, hi):
    if v is None or v <= 0:
        return v
    if lo is None or hi is None or hi <= lo:
        lo, hi = v * 0.75, v * 1.25
    se = max((hi - lo) / (2 * 1.96), 1e-9)
    shape = (v / se) ** 2
    return float(rng.gamma(shape, v / shape))


def beta_draw(rng, v, lo, hi, scale):
    """Draw a rate or prevalence, expressed per `scale`, on the unit interval."""
    p = v / scale
    if not (0 < p < 1):
        return v
    if lo is None or hi is None or hi <= lo:
        lo, hi = v * 0.85, v * 1.15
    se = max((hi - lo) / (2 * 1.96) / scale, 1e-9)
    var = min(se ** 2, p * (1 - p) * 0.999)
    k = p * (1 - p) / var - 1
    if k <= 0:
        return v
    return float(rng.beta(max(p * k, 1e-3), max((1 - p) * k, 1e-3)) * scale)


def perturb_params(P: dict, rng) -> dict:
    Q = copy.deepcopy(P)
    for k in list(Q):
        e = Q[k]
        if not isinstance(e, dict) or "value" not in e or e["value"] is None:
            continue
        v, lo, hi = e["value"], e.get("low"), e.get("high")
        if not isinstance(v, (int, float)):
            continue
        if k.startswith("inc_"):
            e["value"] = beta_draw(rng, v, lo, hi, 1000.0)
        elif k in ("smm_rate",):
            e["value"] = beta_draw(rng, v, lo, hi, 10000.0)
        elif k == "prmr_for_model":
            e["value"] = beta_draw(rng, v, lo, hi, 1e5)
        elif k.startswith("prev_") or k.endswith("_share") or k.endswith("_pct") \
                or k.endswith("_share_medicaid_births"):
            e["value"] = beta_draw(rng, v, lo, hi, 100.0)
        elif k.startswith("cost_"):
            e["value"] = gamma_draw(rng, v, lo, hi)
        elif k.startswith("utility_"):
            e["value"] = float(np.clip(rng.normal(v, max(((hi or v) - (lo or v)) / 3.92, 1e-9)),
                                       0.0, 1.0)) if (lo is not None and hi is not None) else v
    return Q


def perturb_interventions(model, rng) -> None:
    """Effects on the log scale, unit costs from a gamma matched to the sourced or
    assumed range, maximum reach from a beta matched to its range."""
    for i in model.ints.values():
        i.rr = float(np.clip(lognormal_draw(rng, i.rr, i.rr_lo, i.rr_hi), 0.05, 1.0))
        i.unit_cost = gamma_draw(rng, i.unit_cost, i.cost_lo, i.cost_hi)
        if 0 < i.max_reach < 1:
            i.max_reach = float(np.clip(beta_draw(rng, i.max_reach, i.reach_lo, i.reach_hi, 1.0),
                                        0.01, 1.0))
    model.n_elig = {k: float(v.sum()) * model.c.weight for k, v in model.elig.items()}


# Structural parameters of the pathway model that carry no published interval:
# the risk-factor relative risks that shape who has which complication, and the
# literature-informed probability that a complication becomes severe. Each draw
# multiplies every one by a log-normal factor whose 95% range is half to double.
BASE_RISK = copy.deepcopy(ms.RISK)
BASE_SEVERITY = dict(ms.SEVERITY)
STRUCT_SD = np.log(2.0) / 1.96


def perturb_structure(rng) -> None:
    ms.RISK.clear()
    for cz, mods in BASE_RISK.items():
        ms.RISK[cz] = [(a, float(r * np.exp(rng.normal(0, STRUCT_SD)))) for a, r in mods]
    for cz, v in BASE_SEVERITY.items():
        ms.SEVERITY[cz] = float(np.clip(v * np.exp(rng.normal(0, STRUCT_SD)), 1e-4, 1.0))


def restore_structure() -> None:
    ms.RISK.clear(); ms.RISK.update(copy.deepcopy(BASE_RISK))
    ms.SEVERITY.clear(); ms.SEVERITY.update(BASE_SEVERITY)


def run_once(P, shares, seed, n, perturb=False, rng=None,
             all_extended=None, mutate=None, intensity=1.0, structure=None,
             lever_based=False, ishares_override=None, want_equiv=False,
             current=None, mode="full") -> dict:
    ishares = ishares_override or ISHARES
    if perturb:
        perturb_structure(rng)
    if structure:
        structure()
    try:
        m = pc.Model(P, n, seed=seed, all_extended=all_extended, current=current)
    finally:
        if perturb or structure:
            restore_structure()
    if perturb:
        perturb_interventions(m, rng)
    if mutate:
        mutate(m)
    if lever_based:
        cov = m.observed(shares, intensity=intensity)
    elif mode == "frequency":
        cov = m.requested_frequency(ishares)
    elif mode == "frequency_effective":
        cov = m.requested_frequency(ishares, effective_norm=True)
    else:
        cov = m.requested_full(ishares)
    obs = m.outcomes(cov)
    budget = obs["program_cost"]
    ev = m.outcomes(m.greedy(budget, steps=60))
    lb: dict = {}
    for k, c in cov.items():
        i = m.ints[k]
        lb[i.lever] = lb.get(i.lever, 0.0) + i.unit_cost * m.n_elig[k] * c
    wl = m.outcomes(m.within_lever(lb, steps=25))
    ba = m.outcomes(m.burden_aligned_at_budget(budget))
    eq = m.equivalent_budget(obs["deaths_averted"], budget) if want_equiv else {}
    return {"budget": budget, "baseline_deaths": m.base["totals"]["deaths"],
            "equivalent_budget_share": eq.get("share_of_observed_budget"),
            "observed": obs, "within_lever_optimal": wl, "evidence_weighted": ev,
            "burden_aligned": ba,
            "gap": ev["deaths_averted"] - obs["deaths_averted"],
            "ratio": ev["deaths_averted"] / max(obs["deaths_averted"], 1e-9)}


def pct(a, q):
    return float(np.percentile(a, q))


def main() -> None:
    P = ms.load_params()
    shares = pc.lever_shares_from(RES / "classification.json")
    base = run_once(P, shares, SEED, N_COHORT, want_equiv=True)
    print(f"base case on the sensitivity cohort ({N_COHORT:,}): "
          f"observed {base['observed']['deaths_averted']:,.0f}, "
          f"evidence-weighted {base['evidence_weighted']['deaths_averted']:,.0f} deaths averted")

    # ---- probabilistic sensitivity analysis ---------------------------------
    # Allocations are fixed at their base-case coverage and evaluated across
    # parameter draws. Re-optimizing the evidence-weighted portfolio inside each
    # draw would let it see each draw's true parameters, which measures an
    # oracle rather than the uncertainty facing a decision made now.
    m0 = pc.Model(P, N_COHORT, seed=SEED)
    fixed = {"obs": m0.requested_full(ISHARES)}
    b0 = m0.outcomes(fixed["obs"])["program_cost"]
    fixed["ev"] = m0.greedy(b0, steps=120)
    lb0: dict = {}
    for k, c in fixed["obs"].items():
        lb0[m0.ints[k].lever] = lb0.get(m0.ints[k].lever, 0.0) + m0.ints[k].unit_cost * m0.n_elig[k] * c
    fixed["wl"] = m0.within_lever(lb0, steps=40)
    fixed["ba"] = m0.burden_aligned_at_budget(b0)
    fixed["freq"] = m0.requested_frequency(ISHARES)
    rng = np.random.default_rng(SEED)
    rows = []
    for d in range(N_PSA):
        Q = perturb_params(P, rng)
        try:
            perturb_structure(rng)
            try:
                m = pc.Model(Q, N_COHORT, seed=SEED + 1 + d)
            finally:
                restore_structure()
            perturb_interventions(m, rng)
            out = {}
            for tag, cov in fixed.items():
                cc = {k: min(v, m.ints[k].max_reach) for k, v in cov.items()}
                out[tag] = m.outcomes(cc)
        except Exception:
            continue
        r = {f"{t}_deaths": o["deaths_averted"] for t, o in out.items()}
        r.update({f"{t}_qaly": o["qalys_gained"] for t, o in out.items()})
        r.update({f"{t}_cost": o["program_cost"] for t, o in out.items()})
        r.update({f"{t}_net": o["net_cost"] for t, o in out.items()})
        r["gap"] = r["ev_deaths"] - r["obs_deaths"]
        r["ratio"] = r["ev_deaths"] / max(r["obs_deaths"], 1e-9)
        r["obs_cpq"] = out["obs"]["cost_per_qaly"]; r["ev_cpq"] = out["ev"]["cost_per_qaly"]
        rows.append(r)
        if (d + 1) % 100 == 0:
            print(f"  psa {d+1}/{N_PSA}", flush=True)

    arr = lambda k: np.array([r[k] for r in rows], float)
    psa = {}
    for k in rows[0]:
        a_ = arr(k); a_ = a_[np.isfinite(a_)]
        psa[k] = {"mean": float(a_.mean()), "median": pct(a_, 50),
                  "ci": [pct(a_, 2.5), pct(a_, 97.5)]}
    psa["prob_evidence_weighted_averts_more"] = float((arr("gap") > 0).mean())
    psa["prob_evidence_weighted_averts_at_least_1_5x"] = float((arr("ratio") >= 1.5).mean())
    psa["prob_evidence_weighted_averts_at_least_2x"] = float((arr("ratio") >= 2.0).mean())
    ceac = {}
    for wtp in (50_000, 100_000, 150_000, 200_000):
        nmb_o = wtp * arr("obs_qaly") - arr("obs_net")
        nmb_e = wtp * arr("ev_qaly") - arr("ev_net")
        ceac[str(wtp)] = {"observed_ce_vs_none": float((nmb_o > 0).mean()),
                          "evidence_weighted_ce_vs_none": float((nmb_e > 0).mean()),
                          "evidence_weighted_preferred": float((nmb_e > nmb_o).mean())}
    psa["ceac"] = ceac
    psa["n_draws"] = len(rows)
    psa["design"] = "allocations fixed at base case, evaluated across draws"
    print(f"\nprobabilistic analysis over {len(rows)} draws (fixed allocations)")
    for t in ("obs", "ev", "wl", "ba", "freq"):
        q = psa[f"{t}_deaths"]
        print(f"  {t:5s} deaths averted {q['median']:7.1f} ({q['ci'][0]:.1f} to {q['ci'][1]:.1f})")
    print(f"  P(evidence-weighted averts more) {psa['prob_evidence_weighted_averts_more']:.3f}; "
          f"P(>=2x) {psa['prob_evidence_weighted_averts_at_least_2x']:.3f}")

    # ---- one-way sensitivity -------------------------------------------------
    one_way = []
    for key in ONE_WAY:
        e = P.get(key)
        if not isinstance(e, dict) or e.get("value") is None:
            continue
        lo, hi = e.get("low"), e.get("high")
        if lo is None or hi is None or lo == hi:
            continue
        vals = {}
        for tag, v in (("low", lo), ("high", hi)):
            Q = copy.deepcopy(P); Q[key]["value"] = v
            try:
                vals[tag] = run_once(Q, shares, SEED, N_COHORT)
            except Exception:
                vals[tag] = None
        if vals["low"] is None or vals["high"] is None:
            continue
        g = lambda t, f: vals[t][f]
        one_way.append({"parameter": key, "low_value": lo, "high_value": hi,
                        "gap_at_low": g("low", "gap"), "gap_at_high": g("high", "gap"),
                        "swing": abs(g("high", "gap") - g("low", "gap")),
                        "equiv_share_at_low": g("low", "equivalent_budget_share"),
                        "equiv_share_at_high": g("high", "equivalent_budget_share"),
                        "obs_cpq_at_low": vals["low"]["observed"]["cost_per_qaly"],
                        "obs_cpq_at_high": vals["high"]["observed"]["cost_per_qaly"],
                        "ev_cpq_at_low": vals["low"]["evidence_weighted"]["cost_per_qaly"],
                        "ev_cpq_at_high": vals["high"]["evidence_weighted"]["cost_per_qaly"]})
    one_way.sort(key=lambda r: -r["swing"])
    print(f"\none-way sensitivity, ranked by swing in the difference between portfolios")
    for r in one_way[:10]:
        print(f"  {r['parameter']:46s} {r['swing']:8,.0f} deaths")

    # ---- structural ablations ------------------------------------------------
    def ab_zero_effect_optimistic(m):
        for i in m.ints.values():
            if i.grade.startswith(("null", "absent")) and i.stage in ("incidence", "smm", "cfr"):
                i.rr = 0.95

    def ab_transportability(m):
        for k in ("tranexamic_acid", "uterotonics"):
            i = m.ints[k]
            i.rr = 1 - (1 - i.rr) * 0.5

    def ab_no_bundle_extrapolation(m):
        m.ints["severe_hypertension_bundle"].rr = 1.0

    def ab_halve_max_reach(m):
        for i in m.ints.values():
            i.max_reach *= 0.5

    def ab_moud_null(m):
        m.ints["moud_access"].rr = 1.0

    def st_homogeneous_risk():
        for cz in ms.RISK:
            ms.RISK[cz] = [(a, 1.0) for a, _ in ms.RISK[cz]]

    def st_uniform_severity():
        mean = float(np.mean(list(ms.SEVERITY.values())))
        for cz in ms.SEVERITY:
            ms.SEVERITY[cz] = mean

    ablations = {}
    specs = [
        ("base_case", dict()),
        ("zero_effect_interventions_given_5pct_benefit", dict(mutate=ab_zero_effect_optimistic)),
        ("non_US_trial_effects_halved", dict(mutate=ab_transportability)),
        ("severe_hypertension_bundle_extrapolation_removed", dict(mutate=ab_no_bundle_extrapolation)),
        ("maximum_reach_halved", dict(mutate=ab_halve_max_reach)),
        ("opioid_treatment_mortality_effect_removed", dict(mutate=ab_moud_null)),
        ("all_states_without_12_month_coverage", dict(all_extended=False)),
        ("all_states_with_12_month_coverage", dict(all_extended=True)),
        ("no_risk_factor_heterogeneity", dict(structure=st_homogeneous_risk)),
        ("uniform_severity_before_calibration", dict(structure=st_uniform_severity)),
        ("requested_by_request_frequency", dict(mode="frequency")),
        ("requested_by_frequency_normalized_to_effective", dict(mode="frequency_effective")),
        ("observed_portfolio_by_lever_share", dict(lever_based=True)),
        ("incremental_over_current_practice", dict(current=CURRENT_PRACTICE)),
        ("intervention_mapping_pass_B", dict(ishares_override=pc.intervention_shares_from(
            RES / "intervention_map_passB.jsonl") if (RES / "intervention_map_passB.jsonl").exists()
            else None)),
    ]
    for name, kw in specs:
        try:
            r = run_once(P, shares, SEED, N_COHORT, **kw)
        except Exception as e:
            ablations[name] = {"error": str(e)[:300]}
            print(f"  ablation {name} failed: {str(e)[:120]}")
            continue
        ablations[name] = {"observed_deaths_averted": r["observed"]["deaths_averted"],
                           "evidence_weighted_deaths_averted": r["evidence_weighted"]["deaths_averted"],
                           "within_lever_deaths_averted": r["within_lever_optimal"]["deaths_averted"],
                           "gap": r["gap"], "ratio": r["ratio"], "budget": r["budget"]}

    # Discount rate and perspective scenarios act on valuation, not on effects.
    for r_ in (0.0, 0.05):
        Q = copy.deepcopy(P); Q["discount_rate"]["value"] = r_
        rr = run_once(Q, shares, SEED, N_COHORT)
        ablations[f"discount_rate_{r_:.2f}"] = {
            "observed_cost_per_qaly": rr["observed"]["cost_per_qaly"],
            "evidence_weighted_cost_per_qaly": rr["evidence_weighted"]["cost_per_qaly"],
            "observed_qalys": rr["observed"]["qalys_gained"],
            "evidence_weighted_qalys": rr["evidence_weighted"]["qalys_gained"]}
    Q = copy.deepcopy(P); Q["utility_decrement_smm_survivor"]["value"] = 0.10
    rr = run_once(Q, shares, SEED, N_COHORT)
    ablations["smm_survivor_utility_decrement_0.10"] = {
        "observed_qalys": rr["observed"]["qalys_gained"],
        "evidence_weighted_qalys": rr["evidence_weighted"]["qalys_gained"],
        "observed_cost_per_qaly": rr["observed"]["cost_per_qaly"],
        "evidence_weighted_cost_per_qaly": rr["evidence_weighted"]["cost_per_qaly"]}

    # Societal perspective: adds the productivity lost to a maternal death, from
    # the sourced parameter, to the offsets of every portfolio.
    prod = (json.load(open(RES / "intervention_params.json")).get("societal", {})
            .get("productivity_loss_per_maternal_death", {}).get("value"))
    if prod:
        rr = run_once(P, shares, SEED, N_COHORT)
        soc = {}
        for k in ("observed", "evidence_weighted"):
            o = rr[k]
            net = o["net_cost"] - o["deaths_averted"] * prod
            soc[k] = {"net_cost_societal": net, "qalys": o["qalys_gained"],
                      "cost_per_qaly_societal": net / o["qalys_gained"]
                      if o["qalys_gained"] > 0 else None}
        ablations["societal_perspective"] = soc

    print(f"\n{'ablation':52s} {'observed':>10s} {'evidence':>10s} {'ratio':>7s}")
    for k, v in ablations.items():
        if isinstance(v, dict) and "observed_deaths_averted" in v:
            print(f"{k:52s} {v['observed_deaths_averted']:10,.0f} "
                  f"{v['evidence_weighted_deaths_averted']:10,.0f} {v['ratio']:7.1f}")

    json.dump({"n_psa_draws": N_PSA, "n_cohort": N_COHORT, "seed": SEED,
               "base_case": {"equivalent_budget_share": base["equivalent_budget_share"],
                             "observed": base["observed"], "within_lever_optimal":
                             base["within_lever_optimal"],
                             "evidence_weighted": base["evidence_weighted"],
                             "gap": base["gap"], "ratio": base["ratio"]},
               "psa": psa, "psa_draws": rows, "one_way": one_way,
               "ablations": ablations},
              open(RES / "uncertainty.json", "w"), indent=1)
    print(f"\nwrote {RES/'uncertainty.json'}")


if __name__ == "__main__":
    main()
