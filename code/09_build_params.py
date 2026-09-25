"""
09_build_params.py — Assemble the flat parameter set the microsimulation reads.

Four inputs, each with citations attached to every value:
  results/denominators.json      state births, Medicaid share, coverage status
  results/epi_cost_params.json   rates, costs, utilities
  results/cohort_params.json     cohort composition and complication incidence
  results/deflators.json         medical care price index

Three things happen here and nowhere else, so that every derived quantity has
one place a reader can check it.

First, the cause distribution is translated from the categories the national
review-committee data uses into the eleven the model runs. Committees report
mental health conditions as a single category that contains suicide, overdose
related to substance use disorder, and other deaths a committee attributed to a
mental health condition. The model needs those separated, because committees
recommend on them separately, so the category is split on the published share of
mental health deaths that were suicides.

Second, the calibration target for pregnancy-related mortality is taken from the
corpus, as the protocol specifies, rather than from a national surveillance
figure. The national figure is retained as an external check.

Third, every cost is inflated to a common dollar year.

Output: results/model_params.json
"""
from __future__ import annotations
import csv, json, sys, statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
MMRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"

# Suicide share of pregnancy-related deaths attributed to mental health
# conditions. The remainder is overdose related to substance use disorder and
# other deaths a committee attributed to a mental health condition; the model
# assigns that remainder to the substance use pathway and says so.
SUICIDE_SHARE_OF_MH = 0.63
SUICIDE_SHARE_SOURCE = (
    "Trost SL, Beauregard JL, Smoots AN, et al. Preventing pregnancy-related mental health "
    "deaths: insights from 14 US maternal mortality review committees, 2008-17. "
    "Health Aff (Millwood). 2021;40(10):1551-1559. PMID 34606354. doi:10.1377/hlthaff.2021.00615")

# National review-committee cause category -> model cause pathway.
CAUSE_XWALK = {
    "Hemorrhage": "Hemorrhage",
    "Cardiac and coronary conditions": "Cardiovascular conditions",
    "Infection": "Infection",
    "Thrombotic embolism": "Embolism",
    "Amniotic fluid embolism": "Embolism",
    "Cardiomyopathy": "Cardiomyopathy",
    "Hypertensive disorders of pregnancy": "Hypertensive disorders",
    "Cerebrovascular accident": "Cerebrovascular accident",
    "Injury/homicide": "Injury, homicide, and violence",
    "Other": "Other medical conditions",
}

# Cost parameters and the dollar year each is expressed in.
COST_YEARS = {
    "cost_medicaid_delivery": 2022, "cost_smm_increment": 2022,
    "cost_postpartum_year_medicaid": 2020, "cost_aim_bundle_implementation": 2021,
    "cost_low_dose_aspirin": 2026, "cost_moud_year": 2023,
    "cost_perinatal_depression_treatment": 2014,
    "cost_12mo_postpartum_extension_per_enrollee": 2023,
    "cost_doula_medicaid": 2024,
}


def corpus_prmr() -> dict:
    """Pregnancy-related mortality ratio from the corpus, the protocol's target.

    One value per state, taken as the median across that state's overall
    observations, then summarized across states two ways: unweighted median and
    a mean weighted by the state's Medicaid-financed births.
    """
    rows = list(csv.DictReader(open(MMRC / "mmrc_findings.csv")))
    overall = [r for r in rows
               if r["population_group"].strip().lower() in
               ("overall", "total", "all", "all women", "statewide")
               and r["pregnancy_related_mortality_ratio"].strip()]
    by_state: dict[str, list[float]] = {}
    for r in overall:
        try:
            v = float(r["pregnancy_related_mortality_ratio"])
        except ValueError:
            continue
        if 0 < v < 300:
            by_state.setdefault(r["state"], []).append(v)
    med = {s: st.median(v) for s, v in by_state.items()}

    den = json.load(open(RES / "denominators.json"))["states"]
    num = dn = 0.0
    for s, v in med.items():
        d = den.get(s)
        if d and d.get("births_2024") and d.get("medicaid_share_pct"):
            w = d["births_2024"] * d["medicaid_share_pct"] / 100.0
            num += v * w
            dn += w
    vals = sorted(med.values())
    return {"n_states": len(med),
            "median_of_state_medians": round(st.median(vals), 2),
            "iqr": [round(vals[len(vals)//4], 2), round(vals[3*len(vals)//4], 2)],
            "medicaid_birth_weighted_mean": round(num / dn, 2) if dn else None,
            "by_state": {k: round(v, 2) for k, v in sorted(med.items())}}


def main() -> None:
    epi = json.load(open(RES / "epi_cost_params.json"))
    coh = json.load(open(RES / "cohort_params.json"))
    defl = json.load(open(RES / "deflators.json"))
    fac = {int(k): v for k, v in defl["factor_to_2025"].items() if v is not None}

    out: dict = {}
    prov: dict = {}

    def put(key, value, unit, source, note="", low=None, high=None):
        out[key] = {"value": value, "low": low, "high": high, "unit": unit,
                    "source": source, "note": note}

    # ---- cohort composition -------------------------------------------------
    for k in ["age_distribution_medicaid_births", "black_share_medicaid_births",
              "nulliparous_share", "cesarean_rate_pct", "prev_obesity_pregnancy",
              "prev_chronic_hypertension", "prev_pregestational_diabetes",
              "prev_depression_history", "prev_oud_pregnancy"]:
        if k not in coh:
            raise KeyError(f"{k} missing from cohort_params.json")
        out[k] = coh[k]

    # ---- complication incidence, per 1,000 ----------------------------------
    inc_map = {
        "inc_postpartum_hemorrhage": "inc_postpartum_hemorrhage",
        "inc_preeclampsia_eclampsia": "inc_preeclampsia_eclampsia",
        "inc_peripartum_cardiomyopathy": "inc_peripartum_cardiomyopathy",
        "inc_cardiac_conditions_pregnancy": "inc_cardiac_conditions_pregnancy",
        "inc_vte_pregnancy": "inc_vte_pregnancy",
        "inc_maternal_sepsis": "inc_maternal_sepsis",
        "inc_perinatal_depression": "inc_perinatal_depression",
        "inc_oud_pregnancy": "inc_oud_pregnancy",
        "inc_ipv_pregnancy": "inc_ipv_pregnancy",
        "inc_stroke_pregnancy": "inc_stroke_pregnancy",
        "inc_other_medical_pregnancy": "inc_other_medical_pregnancy",
    }
    for dest, src in inc_map.items():
        if src not in coh:
            raise KeyError(f"{src} missing from cohort_params.json")
        out[dest] = coh[src]

    # ---- cause distribution, translated and split ---------------------------
    raw = epi["cause_distribution_prmr"]["value"]
    dist: dict[str, float] = {}
    for k, v in raw.items():
        if k == "Mental health conditions":
            dist["Mental health conditions"] = v * SUICIDE_SHARE_OF_MH
            dist["Substance use disorder"] = v * (1 - SUICIDE_SHARE_OF_MH)
        else:
            dist[CAUSE_XWALK[k]] = dist.get(CAUSE_XWALK[k], 0.0) + v
    tot = sum(dist.values())
    dist = {k: round(v * 100 / tot, 3) for k, v in dist.items()}
    put("cause_distribution_prmr", dist, "percent of pregnancy-related deaths",
        epi["cause_distribution_prmr"]["source"],
        "Mental health conditions as reported nationally contains suicide, overdose related to "
        "substance use disorder, and other deaths attributed to a mental health condition. It is "
        f"split here at a suicide share of {SUICIDE_SHARE_OF_MH:.0%} ({SUICIDE_SHARE_SOURCE}); the "
        "remainder is assigned to the substance use pathway. Thrombotic and amniotic fluid embolism "
        "are combined into one embolism pathway.")

    # ---- mortality calibration target ---------------------------------------
    cp = corpus_prmr()
    smm_all = epi["smm_rate"]["value"]
    smm_mcd = epi["smm_rate_medicaid"]["value"]
    ratio = smm_mcd / smm_all
    target = cp["medicaid_birth_weighted_mean"] * ratio
    put("prmr_for_model", round(target, 2),
        "pregnancy-related deaths per 100,000 Medicaid-financed live births",
        "Derived: corpus state ratios weighted by Medicaid-financed births, scaled by the "
        "Medicaid to all-payer severe maternal morbidity ratio from "
        + epi["smm_rate_medicaid"]["source"],
        "No payer-stratified national pregnancy-related mortality ratio is published. The corpus "
        f"gives a Medicaid-birth-weighted state ratio of {cp['medicaid_birth_weighted_mean']} across "
        f"{cp['n_states']} states. Severe maternal morbidity is {ratio:.3f} times as common among "
        "Medicaid-financed as among all deliveries, and that ratio is applied to the mortality "
        "target. The scaling is an assumption and is varied in sensitivity analysis.",
        low=round(cp["medicaid_birth_weighted_mean"], 2),
        high=round(cp["medicaid_birth_weighted_mean"] * ratio * 1.25, 2))
    out["corpus_prmr_detail"] = cp
    out["external_prmr_check"] = epi["us_prmr_overall"]

    put("smm_rate", smm_mcd, epi["smm_rate_medicaid"]["unit"],
        epi["smm_rate_medicaid"]["source"], epi["smm_rate_medicaid"].get("note", ""),
        low=smm_mcd * 0.9, high=smm_mcd * 1.1)

    # ---- timing, utilities, horizon -----------------------------------------
    out["timing_of_pregnancy_related_deaths"] = epi["timing_of_pregnancy_related_deaths"]
    for k in ["utility_healthy_reproductive_age_female",
              "utility_decrement_postpartum_depression",
              "life_expectancy_female_30", "discount_rate"]:
        out[k] = epi[k]

    # No published lasting decrement after severe maternal morbidity exists. The
    # only United States model that states an assumption sets it to zero. That is
    # the base case here, and a decrement is applied in sensitivity analysis.
    put("utility_decrement_smm_survivor", 0.0, "utility decrement, first year after the event",
        epi["utility_decrement_smm"]["source"],
        "Base case follows the only United States model that states an assumption, which sets a "
        "lasting decrement to zero. A first-year decrement of 0.05 to 0.15 is applied in "
        "sensitivity analysis. Treat as an assumption, not an estimate.",
        low=0.0, high=0.15)

    # ---- costs, inflated ----------------------------------------------------
    for key, yr in COST_YEARS.items():
        src = epi.get(key)
        if src is None:
            continue
        f = fac.get(yr)
        if f is None:
            raise KeyError(f"no deflator factor for {yr} (needed by {key})")
        scale = lambda v: None if v is None else round(v * f, 2)
        out[key] = {"value": scale(src["value"]), "low": scale(src.get("low")),
                    "high": scale(src.get("high")),
                    "unit": f"2025 USD; source year {yr}; {src['unit']}",
                    "source": src["source"],
                    "note": (src.get("note", "") +
                             f" Inflated from {yr} to 2025 by a factor of {f:.4f} using "
                             f"{defl['index_name']}.").strip()}

    # No national doula reimbursement figure is published. State Medicaid rates
    # span an order of magnitude, so the midpoint of the published state range
    # is used and the whole range is carried into sensitivity analysis.
    d = epi["cost_doula_medicaid"]
    f = fac[COST_YEARS["cost_doula_medicaid"]]
    lo, hi = d["low"] * f, d["high"] * f
    put("cost_doula_medicaid", round((lo + hi) / 2, 2),
        f"2025 USD per birth; source year {COST_YEARS['cost_doula_medicaid']}",
        d["source"],
        (d.get("note", "") + " No national point estimate exists. The midpoint of the published "
         f"state range is used and the full range of ${lo:,.0f} to ${hi:,.0f} is carried into "
         "sensitivity analysis.").strip(),
        low=round(lo, 2), high=round(hi, 2))

    # Cost of clinician training has no published per-clinician figure. It is
    # constructed from released time at a published wage and labeled as such.
    put("cost_clinician_training", None, "2025 USD per clinician trained",
        "No published per-clinician cost identified",
        "Constructed in the model as released hours multiplied by a published clinician wage; "
        "the constructed value and its range are reported with the intervention, not here.")

    # ---- residual pathway ---------------------------------------------------
    # No national incidence estimate exists for the residual category of other
    # medical conditions. Rather than choose a number, the incidence is set to
    # whatever makes that category exactly as lethal, once a complication has
    # become severe, as the average of the eleven pathways. The residual is then
    # neither more nor less deadly than the conditions the model can measure,
    # and the assumption is visible instead of buried in a rate.
    if out["inc_other_medical_pregnancy"]["value"] is None:
        from importlib.util import spec_from_file_location, module_from_spec
        sp = spec_from_file_location("ms", Path(__file__).parent / "11_microsim.py")
        ms = module_from_spec(sp); sys.modules['ms'] = ms; sp.loader.exec_module(ms)
        SEV, OTH = ms.SEVERITY, "Other medical conditions"
        inc = {cz: out[ms.INC_KEY[cz]]["value"] / 1000.0
               for cz in ms.CAUSES if cz != OTH}
        S = sum(inc[cz] * SEV[cz] for cz in inc)
        tgt_smm = out["smm_rate"]["value"] / 1e4
        prmr = out["prmr_for_model"]["value"] / 1e5
        sh = out["cause_distribution_prmr"]["value"]
        x = 0.005 * SEV[OTH]
        for _ in range(200):
            kappa = tgt_smm / (S + x)
            cfrs = [(prmr * sh[cz] / 100.0) / (inc[cz] * kappa * SEV[cz]) for cz in inc]
            m = sum(cfrs) / len(cfrs)
            D = prmr * sh[OTH] / 100.0 / m
            x_new = D * S / (tgt_smm - D)
            if abs(x_new - x) < 1e-12:
                x = x_new
                break
            x = 0.5 * x + 0.5 * x_new
        val = x / SEV[OTH] * 1000.0
        out["inc_other_medical_pregnancy"] = {
            "value": round(val, 3), "low": round(val * 0.5, 3), "high": round(val * 2.0, 3),
            "unit": "per 1,000 deliveries, derived",
            "source": "Derived, not estimated. No national incidence estimate exists for this "
                      "residual category.",
            "note": "Set so that case fatality given a severe event in this residual category "
                    "equals the mean case fatality across the other ten pathways. Varied by a "
                    "factor of two in each direction in sensitivity analysis."}
        print(f"derived residual incidence: {val:.3f} per 1,000")

    json.dump({"generated_for": "MMRC recommendation portfolio microsimulation",
               "dollar_year": 2025, "deflator": defl["index_name"],
               "params": out}, open(RES / "model_params.json", "w"), indent=1)

    nulls = [k for k, v in out.items()
             if isinstance(v, dict) and "value" in v and v["value"] is None]
    print(f"wrote {RES/'model_params.json'} with {len(out)} entries")
    print(f"corpus PRMR: median of state medians {cp['median_of_state_medians']}, "
          f"Medicaid-birth-weighted {cp['medicaid_birth_weighted_mean']} "
          f"across {cp['n_states']} states")
    print(f"model calibration target PRMR: {out['prmr_for_model']['value']} per 100,000")
    print(f"national external check: {epi['us_prmr_overall']['value']} per 100,000")
    print(f"\ncause distribution after translation and split:")
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {k:34s} {v:6.2f}%")
    print(f"\nnull-valued entries: {nulls}")


if __name__ == "__main__":
    main()
