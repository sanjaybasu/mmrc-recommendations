"""
11_microsim.py — Individual-level model of a Medicaid-financed US birth cohort.

Structure. Each simulated pregnancy carries a state, an age, a parity, and a
clinical history. Eleven cause-specific pathways run in parallel. Within a
pathway a complication occurs or does not, a complication may progress to severe
maternal morbidity, and a severe event may end in death. Interventions act at
whichever of those three stages the evidence for them measured its effect, which
is what stops the model crediting an intervention with a mortality effect no
study estimated.

Monte Carlo is used for who exists, not for what happens to them. Individuals
and their risk factors are sampled; the pathway probabilities are then combined
in closed form over that sampled cohort. Expectations are available exactly, so
two portfolios evaluated on the same cohort differ only by their effects and not
by simulation noise. Sampling uncertainty in the cohort is handled by repeating
the draw, and parameter uncertainty by probabilistic sensitivity analysis.

Calibration. Complication incidence comes from the literature. One scalar is
then solved so that severe maternal morbidity reaches the published rate among
Medicaid-financed deliveries, and one case-fatality rate per cause is solved so
that cause-specific deaths reproduce the corpus mortality ratio multiplied by
that cause's national share. Nothing else is tuned.

Mortality is a modeled projection. Only two interventions in this literature
carry a direct estimate of an effect on maternal death and both were tested
largely outside the United States.
"""
from __future__ import annotations
import json, sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"

CAUSES = ["Hemorrhage", "Hypertensive disorders", "Cardiomyopathy",
          "Cardiovascular conditions", "Embolism", "Infection",
          "Mental health conditions", "Substance use disorder",
          "Injury, homicide, and violence", "Cerebrovascular accident",
          "Other medical conditions"]

# Probability that a complication, once it occurs, meets the severe maternal
# morbidity definition. These are set from what each condition is, not fitted.
# Sepsis and stroke are severe morbidity by definition and sit at one. Postpartum
# hemorrhage and preeclampsia are common and mostly not severe, so they sit low.
# Perinatal depression and intimate partner violence reach severe morbidity only
# through a rare severe event, so they sit lowest. A single multiplier is then
# solved across all of them so the cohort reproduces the published rate; that
# multiplier is reported, and a value far from one would mean the eleven
# pathways do not span what the published rate counts.
SEVERITY = {"Hemorrhage": 0.08, "Hypertensive disorders": 0.04, "Cardiomyopathy": 0.30,
            "Cardiovascular conditions": 0.30, "Embolism": 0.30, "Infection": 1.00,
            "Mental health conditions": 0.005, "Substance use disorder": 0.02,
            "Injury, homicide, and violence": 0.005, "Cerebrovascular accident": 1.00,
            "Other medical conditions": 0.05}

# Published deaths per complication, where an estimate exists. These check the
# calibration; they do not drive it. The comparison is made on the published
# denominator, which is everyone with the complication, not only those whose
# complication became severe.
PUBLISHED_DEATHS_PER_COMPLICATION = {
    "Infection": (0.032, "Hensley MK, et al. JAMA. 2019;322(9):890-892. Deaths within 42 days "
                         "of delivery discharge among maternal sepsis cases."),
    "Cardiomyopathy": (0.018, "Krishnamoorthy P, et al. J Cardiovasc Med. 2016;17(10):756-761. "
                              "In-hospital mortality during the index admission."),
    "Hypertensive disorders": (0.00064, "MacKay AP, et al. Obstet Gynecol. 2001;97(4):533-538. "
                                        "Deaths per preeclampsia or eclampsia delivery."),
}

RISK = {
    "Hypertensive disorders": [("chronic_htn", 5.1), ("pregest_dm", 2.0), ("obesity", 2.1),
                               ("nullip", 2.4), ("age35", 1.5), ("black", 1.2)],
    "Hemorrhage": [("cesarean", 1.7), ("obesity", 1.2), ("age35", 1.3)],
    "Cardiomyopathy": [("black", 3.4), ("age35", 2.0), ("chronic_htn", 2.6), ("obesity", 1.6)],
    "Cardiovascular conditions": [("chronic_htn", 3.0), ("obesity", 1.8), ("age35", 2.1),
                                  ("pregest_dm", 2.0), ("black", 1.5)],
    "Embolism": [("cesarean", 2.1), ("obesity", 2.6), ("age35", 1.6)],
    "Infection": [("cesarean", 3.0), ("pregest_dm", 1.7), ("obesity", 1.4)],
    "Mental health conditions": [("depression_hx", 5.0), ("oud", 2.5)],
    "Substance use disorder": [("oud", 30.0), ("depression_hx", 2.0)],
    "Injury, homicide, and violence": [("oud", 3.0), ("age35", 0.6)],
    "Cerebrovascular accident": [("chronic_htn", 4.0), ("age35", 1.8), ("black", 1.5)],
    "Other medical conditions": [("pregest_dm", 2.0), ("obesity", 1.5), ("age35", 1.4)],
}

INC_KEY = {
    "Hemorrhage": "inc_postpartum_hemorrhage",
    "Hypertensive disorders": "inc_preeclampsia_eclampsia",
    "Cardiomyopathy": "inc_peripartum_cardiomyopathy",
    "Cardiovascular conditions": "inc_cardiac_conditions_pregnancy",
    "Embolism": "inc_vte_pregnancy",
    "Infection": "inc_maternal_sepsis",
    "Mental health conditions": "inc_perinatal_depression",
    "Substance use disorder": "inc_oud_pregnancy",
    "Injury, homicide, and violence": "inc_ipv_pregnancy",
    "Cerebrovascular accident": "inc_stroke_pregnancy",
    "Other medical conditions": "inc_other_medical_pregnancy",
}


def load_params() -> dict:
    return json.load(open(RES / "model_params.json"))["params"]


@dataclass
class Cohort:
    n: int
    weight: float
    state: np.ndarray
    age: np.ndarray
    nullip: np.ndarray
    black: np.ndarray
    chronic_htn: np.ndarray
    pregest_dm: np.ndarray
    obesity: np.ndarray
    depression_hx: np.ndarray
    oud: np.ndarray
    cesarean: np.ndarray
    extension: np.ndarray
    p_comp: dict = field(default_factory=dict)


def _bern(rng, p, n):
    return rng.random(n) < np.clip(p, 0.0, 0.999)


def build_cohort(P: dict, n: int, rng: np.random.Generator,
                 all_extended: bool | None = None) -> Cohort:
    den = json.load(open(RES / "denominators.json"))["states"]
    st = [(k, v) for k, v in den.items()
          if v.get("births_2024") and v.get("medicaid_share_pct")]
    codes = np.array([k for k, _ in st])
    mb = np.array([v["births_2024"] * v["medicaid_share_pct"] / 100.0 for _, v in st])
    ext_by_state = np.array([bool(v.get("postpartum_12mo_extension")) for _, v in st])
    idx = rng.choice(len(st), size=n, p=mb / mb.sum())

    share = np.asarray(P["age_distribution_medicaid_births"]["value"], float)
    share = share / share.sum()
    edges = np.array([15, 20, 25, 30, 35, 40])
    g = rng.choice(len(share), size=n, p=share)
    age = edges[g] + rng.integers(0, 5, size=n)

    black = _bern(rng, P["black_share_medicaid_births"]["value"] / 100.0, n)
    nullip = _bern(rng, P["nulliparous_share"]["value"] / 100.0, n)
    cesarean = _bern(rng, P["cesarean_rate_pct"]["value"] / 100.0, n)

    def correlated(key, mods):
        """Prevalence with age and adiposity gradients, rescaled to the
        published national prevalence so the gradient redistributes rather than
        adds cases."""
        base = P[key]["value"] / 100.0
        r = np.ones(n)
        for cond, rr in mods:
            r = r * np.where(cond, rr, 1.0)
        pr = base * r / r.mean()
        return _bern(rng, pr, n)

    obesity = correlated("prev_obesity_pregnancy", [(age >= 35, 1.1)])
    chtn = correlated("prev_chronic_hypertension",
                      [(age >= 35, 2.2), (obesity, 2.5), (black, 1.8)])
    dm = correlated("prev_pregestational_diabetes", [(age >= 35, 1.9), (obesity, 3.0)])
    dep = correlated("prev_depression_history", [(age < 25, 1.2)])
    oud = correlated("prev_oud_pregnancy", [(dep, 3.0)])

    ext = np.ones(n, bool) if all_extended is True else (
        np.zeros(n, bool) if all_extended is False else ext_by_state[idx])

    c = Cohort(n=n, weight=float(mb.sum()) / n, state=codes[idx], age=age, nullip=nullip,
               black=black, chronic_htn=chtn, pregest_dm=dm, obesity=obesity,
               depression_hx=dep, oud=oud, cesarean=cesarean, extension=ext)

    for cz in CAUSES:
        r = np.ones(n)
        for attr, rr in RISK[cz]:
            v = (age >= 35) if attr == "age35" else getattr(c, attr)
            r = r * np.where(v, rr, 1.0)
        base = P[INC_KEY[cz]]["value"] / 1000.0
        c.p_comp[cz] = np.clip(base * r / r.mean(), 0.0, 1.0)
    return c


class CalibrationError(RuntimeError):
    """The calibration cannot satisfy its targets with the parameters given."""


def calibrate(c: Cohort, P: dict) -> dict:
    """Solve the progression multiplier and the per-cause case-fatality rates.

    The product of progression and case fatality is pinned for every cause,
    because incidence is fixed by the literature and cause-specific deaths are
    fixed by the calibration target. Only the split between the two stages is
    free, and it matters because interventions act at one stage or the other.
    Progression is set to its literature-informed value scaled by a single
    multiplier, bounded below by the pinned product so that no case-fatality
    rate can exceed one, and bounded above at one. The multiplier is found by
    bisection so the cohort reproduces the published severe morbidity rate.
    """
    inc = {cz: float(c.p_comp[cz].mean()) for cz in CAUSES}
    prmr = P["prmr_for_model"]["value"] / 1e5
    shares = P["cause_distribution_prmr"]["value"]
    target_smm = P["smm_rate"]["value"] / 1e4

    # pinned product of progression and case fatality
    pin = {cz: (prmr * shares[cz] / 100.0) / inc[cz] if inc[cz] > 0 else 0.0
           for cz in CAUSES}
    for cz, v in pin.items():
        if v > 1.0:
            raise CalibrationError(
                f"{cz}: deaths exceed complications. Incidence of {inc[cz]*1e3:.3f} per 1,000 "
                f"cannot support {shares[cz]:.2f}% of a ratio of {prmr*1e5:.1f} per 100,000. "
                "The incidence parameter measures a narrower condition than the cause of death.")

    sev_at = lambda k: {cz: float(np.clip(SEVERITY[cz] * k, pin[cz], 1.0)) for cz in CAUSES}
    smm_at = lambda k: sum(inc[cz] * sev_at(k)[cz] for cz in CAUSES)

    lo, hi = 1e-6, 1e6
    if smm_at(lo) > target_smm:
        raise CalibrationError(
            f"minimum attainable severe morbidity {smm_at(lo)*1e4:.1f} per 10,000 already exceeds "
            f"the target of {target_smm*1e4:.1f}; the mortality target and the incidence "
            "parameters are mutually inconsistent.")
    for _ in range(200):
        mid = (lo + hi) / 2
        if smm_at(mid) < target_smm:
            lo = mid
        else:
            hi = mid
    kappa = (lo + hi) / 2
    sev = sev_at(kappa)
    cfr = {cz: (pin[cz] / sev[cz] if sev[cz] > 0 else 0.0) for cz in CAUSES}
    for cz, v in cfr.items():
        if v > 1.0 + 1e-9:
            raise CalibrationError(f"{cz}: case fatality solved to {v:.3f}")

    check = {}
    for cz, (pub, src) in PUBLISHED_DEATHS_PER_COMPLICATION.items():
        check[cz] = {"modeled_deaths_per_complication": round(pin[cz], 6),
                     "published": pub, "ratio": round(pin[cz] / pub, 2), "source": src}
    return {"kappa": kappa, "cfr": cfr, "p_smm_given_comp": sev,
            "achieved_smm_per_10k": smm_at(kappa) * 1e4,
            "case_fatality_check": check,
            "clipped_at_one": [cz for cz in CAUSES if sev[cz] >= 1.0 - 1e-9]}


def evaluate(c: Cohort, cal: dict, mult: dict | None = None) -> dict:
    """Expected complications, severe morbidity, and deaths over the cohort.

    mult maps a cause to a dict of three per-person multiplier arrays, one for
    each stage. Absent entries are one.
    """
    mult = mult or {}
    out = {"comp": {}, "smm": {}, "death": {}}
    for cz in CAUSES:
        m = mult.get(cz, {})
        p_c = c.p_comp[cz] * m.get("incidence", 1.0)
        p_s = np.clip(cal["p_smm_given_comp"][cz] * m.get("smm", 1.0), 0, 1)
        p_d = np.clip(cal["cfr"][cz] * m.get("cfr", 1.0), 0, 1)
        out["comp"][cz] = p_c
        out["smm"][cz] = p_c * p_s
        out["death"][cz] = p_c * p_s * p_d
    w = c.weight
    out["totals"] = {
        "births": c.n * w,
        "complications": {cz: float(out["comp"][cz].sum() * w) for cz in CAUSES},
        "smm_events": float(sum(out["smm"][cz].sum() for cz in CAUSES) * w),
        "smm_by_cause": {cz: float(out["smm"][cz].sum() * w) for cz in CAUSES},
        "deaths": float(sum(out["death"][cz].sum() for cz in CAUSES) * w),
        "deaths_by_cause": {cz: float(out["death"][cz].sum() * w) for cz in CAUSES},
    }
    out["totals"]["smm_per_10k"] = out["totals"]["smm_events"] / out["totals"]["births"] * 1e4
    out["totals"]["prmr"] = out["totals"]["deaths"] / out["totals"]["births"] * 1e5
    return out


def main() -> None:
    P = load_params()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    rng = np.random.default_rng(20260924)
    c = build_cohort(P, n, rng)
    cal = calibrate(c, P)
    r = evaluate(c, cal)
    t = r["totals"]

    print(f"simulated pregnancies       {c.n:,} (each represents {c.weight:.2f} births)")
    print(f"Medicaid-financed births    {t['births']:,.0f}")
    print(f"severe maternal morbidity   {t['smm_per_10k']:.1f} per 10,000  "
          f"(target {P['smm_rate']['value']:.1f})")
    print(f"pregnancy-related deaths    {t['prmr']:.1f} per 100,000  "
          f"(target {P['prmr_for_model']['value']:.1f}),  n = {t['deaths']:,.0f}")
    print(f"states with 12-month coverage in cohort: "
          f"{100*c.extension.mean():.1f}% of simulated births")
    print(f"\n{'cause':34s} {'compl/1,000':>12s} {'SMM':>9s} {'deaths':>8s} {'share':>7s} "
          f"{'case fatality':>14s}")
    for cz in CAUSES:
        print(f"{cz:34s} {t['complications'][cz]/t['births']*1e3:12.2f} "
              f"{t['smm_by_cause'][cz]:9,.0f} {t['deaths_by_cause'][cz]:8,.0f} "
              f"{100*t['deaths_by_cause'][cz]/t['deaths']:6.1f}% {100*cal['cfr'][cz]:13.2f}%")

    print(f"\nprogression multiplier {cal['kappa']:.3f}; achieved severe morbidity "
          f"{cal['achieved_smm_per_10k']:.1f} per 10,000")
    print("deaths per complication against published estimates, where one exists:")
    for cz, v in cal["case_fatality_check"].items():
        print(f"  {cz:32s} modeled {100*v['modeled_deaths_per_complication']:6.3f}%   "
              f"published {100*v['published']:6.3f}%   ratio {v['ratio']:.2f}")
    json.dump({"n_simulated": n, "seed": 20260924, "totals": t,
               "calibration": {"kappa": cal["kappa"], "cfr": cal["cfr"],
                               "p_smm_given_comp": cal["p_smm_given_comp"],
                               "achieved_smm_per_10k": cal["achieved_smm_per_10k"],
                               "case_fatality_check": cal["case_fatality_check"],
                               "progression_clipped_at_one": cal["clipped_at_one"]}},
              open(RES / "microsim_baseline.json", "w"), indent=1)
    print(f"\nwrote {RES/'microsim_baseline.json'}")


if __name__ == "__main__":
    main()
