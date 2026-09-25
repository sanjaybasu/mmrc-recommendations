"""
12_interventions.py — The interventions a recommendation can ask for, what each
one is known to do, and to whom it applies.

Every intervention carries four things the analysis needs. The policy lever is
what the recommendation asks someone to do, and it is the field that connects an
intervention to the recommendation counts. The stage is where in the pathway the
effect was measured: on whether a complication occurs, on whether a complication
becomes severe, or on whether a severe event ends in death. Putting each effect
at the stage its trial measured is what keeps the model from crediting an
intervention with a mortality effect no study has estimated.

Interventions whose evidence for patient outcomes is null or absent are present
with a relative risk of exactly 1.0. They are not omitted, because they account
for a large share of what committees recommend, and omitting them would flatter
the current portfolio by removing its cost without removing its ineffectiveness.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

SOURCED = Path(__file__).resolve().parent.parent / "results" / "intervention_params.json"


@dataclass
class Intervention:
    key: str
    label: str
    lever: str
    causes: tuple[str, ...]
    stage: str                 # "incidence", "smm", or "cfr"
    rr: float
    rr_lo: float
    rr_hi: float
    grade: str
    unit_cost: float           # 2025 USD per person reached, per pregnancy
    max_reach: float
    postpartum: bool           # effect delivered after discharge
    eligible: Callable
    source: str
    cost_note: str = ""
    cost_lo: float | None = None
    cost_hi: float | None = None
    reach_lo: float | None = None
    reach_hi: float | None = None
    provenance: dict = field(default_factory=dict)


ALL_ELIGIBLE = lambda c: np.ones(c.n, bool)

# Preeclampsia prophylaxis eligibility follows the task force rule: one high-risk
# factor, or two moderate-risk factors.
def _aspirin_eligible(c):
    high = c.chronic_htn | c.pregest_dm
    mod = (c.nullip.astype(int) + c.obesity.astype(int) +
           (c.age >= 35).astype(int) + c.black.astype(int))
    return high | (mod >= 2)


def _cardiac_eligible(c):
    return c.chronic_htn | c.obesity | (c.age >= 35) | c.black


def _vte_eligible(c):
    return c.cesarean & (c.obesity | (c.age >= 35))


def _depression_eligible(c):
    return c.depression_hx | c.oud


def _registry(P: dict) -> list[Intervention]:
    """The registry with its default values, before sourced values are applied."""
    v = lambda k: P[k]["value"]

    # Cost of clinician training is not published. It is constructed as four
    # released hours at the national mean hourly wage for obstetricians and
    # gynecologists plus registered nurses in a two-to-one nurse-to-physician
    # mix, amortized over the clinician's annual delivery volume. The
    # construction, not a literature value, is what enters the model.
    train_cost_per_birth = 4.0 * (0.33 * 128.0 + 0.67 * 43.0) / 120.0

    return [
        # ---- clinical protocol or bundle ------------------------------------
        Intervention(
            "hemorrhage_bundle", "Obstetric hemorrhage safety bundle",
            "Clinical protocol or bundle", ("Hemorrhage",), "smm",
            0.792, 0.70, 0.90, "moderate",
            v("cost_aim_bundle_implementation"), 0.95, False, ALL_ELIGIBLE,
            "Main EK, et al. Am J Obstet Gynecol. 2017;216(3):298.e1-298.e11. Severe morbidity "
            "among women with hemorrhage fell 20.8% in collaborative hospitals versus 1.2% in "
            "comparison hospitals; the relative risk is the ratio of those changes.",
            "AIM-aligned bundle implementation and maintenance, per birth."),
        Intervention(
            "uterotonics", "Preferred uterotonic regimen for hemorrhage prevention",
            "Clinical protocol or bundle", ("Hemorrhage",), "incidence",
            0.76, 0.64, 0.90, "demonstrated benefit",
            1.50, 0.95, False, ALL_ELIGIBLE,
            "Gallos I, et al. Cochrane Database Syst Rev. 2025;CD011689.pub4. Network "
            "meta-analysis, 122 trials, 121,931 women; ergometrine plus oxytocin versus "
            "oxytocin for blood loss of 500 mL or more.",
            "Incremental drug cost over oxytocin alone."),
        Intervention(
            "tranexamic_acid", "Early tranexamic acid for hemorrhage treatment",
            "Clinical protocol or bundle", ("Hemorrhage",), "cfr",
            0.69, 0.52, 0.91, "demonstrated benefit",
            12.0, 0.90, False, ALL_ELIGIBLE,
            "WOMAN Trial Collaborators. Lancet. 2017;389(10084):2105-2116. Randomized trial, "
            "n = 20,060; death due to bleeding when given within three hours.",
            "Drug cost per treated hemorrhage, spread across deliveries at the hemorrhage rate."),
        Intervention(
            "severe_hypertension_bundle", "Severe hypertension treatment bundle",
            "Clinical protocol or bundle", ("Hypertensive disorders",), "smm",
            0.81, 0.68, 0.96, "weak (extrapolated from blood pressure control)",
            v("cost_aim_bundle_implementation"), 0.95, False, ALL_ELIGIBLE,
            "Alavifard S, et al. Pregnancy Hypertens. 2019;18:179-187, network meta-analysis of "
            "17 trials for blood pressure control, applied at the severe-morbidity stage at the "
            "magnitude observed for the hemorrhage bundle in Main EK, et al. 2017. The effect on "
            "severe morbidity is an extrapolation from blood pressure control and is labeled as "
            "such throughout.",
            "AIM-aligned bundle implementation and maintenance, per birth."),
        Intervention(
            "levels_of_care", "Risk-appropriate care and maternal levels of care",
            "Clinical protocol or bundle",
            ("Hemorrhage", "Hypertensive disorders", "Cardiomyopathy",
             "Cardiovascular conditions"), "smm",
            0.92, 0.80, 1.05, "weak",
            22.0, 0.70, False, ALL_ELIGIBLE,
            "Osei-Poku GK, et al. Am J Obstet Gynecol. 2024;231(5):546.e1. Delivery at an inadequate level of care, "
            "severe morbidity adjusted odds ratio 3.34 (2.24 to 4.96). Confounding by indication "
            "is unresolved, so the modeled effect is set well below what the odds ratio implies.",
            "Designation, verification, and transfer infrastructure, per birth."),
        Intervention(
            "cardiac_pathway", "Cardiac conditions assessment and referral pathway",
            "Clinical protocol or bundle", ("Cardiomyopathy", "Cardiovascular conditions"),
            "smm", 1.00, 0.85, 1.00, "absent (untested for maternal outcomes)",
            18.0, 0.80, False, _cardiac_eligible,
            "No controlled evaluation of a maternal cardiac pathway reporting severe morbidity or "
            "mortality was identified. Entered at no effect, with an upward band in sensitivity "
            "analysis.",
            "Screening algorithm, echocardiography, and cardiology referral, per eligible person."),
        Intervention(
            "vte_prophylaxis", "Postcesarean thromboprophylaxis protocol",
            "Clinical protocol or bundle", ("Embolism",), "incidence",
            1.00, 0.60, 1.00, "absent (untested for maternal outcomes)",
            45.0, 0.85, False, _vte_eligible,
            "No randomized evidence demonstrating that universal postcesarean pharmacologic "
            "prophylaxis reduces venous thromboembolism in this population was identified. "
            "Entered at no effect, with an upward band in sensitivity analysis.",
            "Low molecular weight heparin course, per eligible person."),
        Intervention(
            "sepsis_bundle", "Maternal sepsis recognition and treatment bundle",
            "Clinical protocol or bundle", ("Infection",), "cfr",
            1.00, 0.70, 1.00, "absent (untested for maternal outcomes)",
            v("cost_aim_bundle_implementation"), 0.90, False, ALL_ELIGIBLE,
            "No maternal-specific controlled evaluation of a sepsis bundle reporting case fatality "
            "was identified. Entered at no effect, with an upward band in sensitivity analysis.",
            "Bundle implementation and maintenance, per birth."),

        # ---- screening or referral -------------------------------------------
        Intervention(
            "aspirin_prophylaxis", "Low-dose aspirin for preeclampsia prevention",
            "Screening or referral", ("Hypertensive disorders",), "incidence",
            0.82, 0.77, 0.88, "demonstrated benefit",
            v("cost_low_dose_aspirin") + 30.0, 0.80, False, _aspirin_eligible,
            "Duley L, et al. Cochrane Database Syst Rev. 2019;CD004659.pub3. Individual "
            "participant data, 77 trials, 40,249 women; proteinuric preeclampsia.",
            "Drug cost plus one risk-assessment visit increment, per eligible person."),
        Intervention(
            "depression_prevention", "Counseling to prevent perinatal depression in at-risk people",
            "Screening or referral", ("Mental health conditions",), "incidence",
            0.61, 0.47, 0.78, "demonstrated benefit",
            v("cost_perinatal_depression_treatment"), 0.60, True, _depression_eligible,
            "O'Connor E, et al. JAMA. 2019;321(6):588-601, for the US Preventive Services Task "
            "Force. Systematic review, 17 trials; onset of perinatal depression.",
            "Course of interpersonal psychotherapy or cognitive behavioral therapy."),
        Intervention(
            "ipv_screening", "Intimate partner violence screening in pregnancy",
            "Screening or referral", ("Injury, homicide, and violence",), "incidence",
            1.00, 0.85, 1.00, "null (tested; no effect on outcomes)",
            26.0, 0.85, False, ALL_ELIGIBLE,
            "O'Doherty L, et al. Cochrane Database Syst Rev. 2015;CD007007.pub3. Screening raises "
            "identification without reducing abuse or health outcomes. Entered at no effect.",
            "Screening time and referral coordination, per person screened."),

        # ---- coverage or payment ---------------------------------------------
        Intervention(
            "moud_access", "Medication for opioid use disorder in pregnancy and postpartum",
            "Coverage or payment", ("Substance use disorder",), "cfr",
            0.76, 0.55, 1.00, "moderate",
            v("cost_moud_year"), 0.60, True, lambda c: c.oud,
            "Santo T Jr, et al. JAMA Psychiatry. 2021;78(9):979-993. Meta-analysis of 36 cohorts, "
            "drug-related mortality during versus outside opioid agonist treatment. Estimated in the "
            "general population with opioid use disorder; no pregnancy or postpartum cohort reports "
            "an adjusted estimate, so transportability is tested by ablation.",
            "Annual medication and visit cost per treated person."),
        Intervention(
            "postpartum_extension", "Twelve-month postpartum Medicaid coverage",
            "Coverage or payment", (), "exposure",
            1.00, 1.00, 1.00, "moderate for coverage, absent for mortality",
            v("cost_12mo_postpartum_extension_per_enrollee"), 1.00, True,
            lambda c: ~c.extension,
            "No outcome evaluation of the twelve-month extension itself reporting mortality or severe "
            "morbidity was identified. The nearest evidence concerns the pandemic continuous "
            "coverage requirement (Meille G, et al. JAMA Health Forum. 2026; Daw JR, et al. JAMA "
            "Health Forum. 2024). In the model the extension carries no direct effect and acts only "
            "by lengthening the window over which postpartum interventions can reach someone.",
            "Incremental Medicaid cost of months three through twelve."),

        # ---- education or training --------------------------------------------
        Intervention(
            "clinician_education", "Clinician education, training, and implicit bias training",
            "Education or training", (), "none",
            1.00, 0.90, 1.00, "absent (no study measured patient outcomes)",
            train_cost_per_birth, 0.90, False, ALL_ELIGIBLE,
            "Hagiwara N, et al. Sci Adv. 2024;10(33):eado5957. Systematic review of 77 implicit bias "
            "training studies; none measured patient outcomes. Entered at no effect on any pathway, "
            "with an upward band in sensitivity analysis.",
            "Constructed as four released hours at published clinician wages, amortized over "
            "annual delivery volume. Not a literature value."),

        # ---- care coordination or navigation -----------------------------------
        Intervention(
            "nurse_home_visiting", "Nurse home visiting",
            "Care coordination or navigation", (), "none",
            1.00, 0.90, 1.00, "absent for maternal outcomes (null for infant composite)",
            7400.0, 0.30, True, ALL_ELIGIBLE,
            "McConnell MA, et al. JAMA. 2022;328(1):27-37. Randomized trial, n = 5,670, South "
            "Carolina Medicaid; composite adverse birth outcome adjusted difference 0.5% "
            "(-2.1 to 3.1). A null result from a well-powered trial, not an evidence gap.",
            "Program cost per enrolled family, from the trial's own program."),
        Intervention(
            "remote_bp_monitoring", "Remote postpartum blood pressure monitoring",
            "Care coordination or navigation", ("Hypertensive disorders",), "smm",
            1.00, 0.85, 1.00, "null (tested; no effect on readmission)",
            180.0, 0.70, True, lambda c: c.chronic_htn | c.obesity,
            "Zullo F, et al. J Perinat Med. 2025;53(4):439-448. Meta-analysis, 4 trials, n = 714; no "
            "reduction in hypertension-related readmission and more emergency visits. Entered at "
            "no effect.",
            "Device, data plan, and monitoring staff time per enrolled person."),

        # ---- workforce or staffing ----------------------------------------------
        Intervention(
            "doula_support", "Doula and continuous labor support",
            "Workforce or staffing", (), "none",
            1.00, 0.90, 1.00, "absent for maternal outcomes (reduces cesarean)",
            v("cost_doula_medicaid"), 0.50, False, ALL_ELIGIBLE,
            "Bohren MA, et al. Cochrane Database Syst Rev. 2017;CD003766.pub6. Continuous support "
            "lowers cesarean (relative risk 0.75) with no effect on maternal complications. "
            "Entered at no effect on the severe-morbidity and mortality pathways.",
            "Midpoint of state Medicaid doula benefit rates."),

        # ---- community or social services -----------------------------------------
        Intervention(
            "group_prenatal_care", "Group prenatal care and community-based support",
            "Community or social services", (), "none",
            1.00, 0.90, 1.00, "absent for maternal outcomes (null for preterm birth)",
            420.0, 0.40, False, ALL_ELIGIBLE,
            "Carter EB, et al. Obstet Gynecol. 2016;128(3):551-561. Null for preterm birth and low "
            "birth weight in randomized trials; no maternal morbidity outcome. Entered at no effect.",
            "Incremental facilitation and space cost per participant."),

        # ---- data, surveillance, or review -----------------------------------------
        Intervention(
            "surveillance", "Data systems, surveillance, and review capacity",
            "Data, surveillance, or review", (), "none",
            1.00, 1.00, 1.00, "not directly effect-bearing",
            9.0, 1.00, False, ALL_ELIGIBLE,
            "Surveillance enables action and carries no independent effect estimate. Entered at no "
            "direct effect; its value is that it produces the reports this study analyzes.",
            "State review committee and data infrastructure cost, per birth."),
    ]


def build(P: dict) -> list[Intervention]:
    """Instantiate the registry, then overwrite every effect, unit cost, and
    maximum reach for which a verified source exists.

    The sourced file carries a value, a low and high bound, and a basis for each
    number. Where it supplies a number the default is discarded. Where it supplies
    none the default is kept and flagged as an assumption, and it is given a range
    of half to one and a half times its value so that the probabilistic analysis
    carries the uncertainty the missing source implies. The provenance of every
    number is recorded on the intervention and written into the appendix table.
    """
    ints = _registry(P)
    src = json.load(open(SOURCED)).get("interventions", {}) if SOURCED.exists() else {}
    # A sourced relative risk replaces the default only when its outcome is the
    # model stage the intervention acts on. Sourced effects on cesarean delivery,
    # low birth weight, postpartum hospitalization, or a crude nonsignificant
    # readmission ratio describe outcomes that are not stages of the pathway and
    # are recorded in the appendix but not applied; a single-state inverse odds
    # ratio for levels of care is likewise not applied, and the conservative
    # default stands as a labeled assumption.
    RR_APPLIES = {"hemorrhage_bundle", "uterotonics", "tranexamic_acid",
                  "aspirin_prophylaxis", "depression_prevention", "moud_access"}
    pph = P["inc_postpartum_hemorrhage"]["value"] / 1000.0
    for i in ints:
        e = src.get(i.key, {})
        prov = {}
        for fld, lo_attr, hi_attr in (("rr", "rr_lo", "rr_hi"),
                                      ("unit_cost", "cost_lo", "cost_hi"),
                                      ("max_reach", "reach_lo", "reach_hi")):
            rec = dict(e.get(fld) or {})
            v = rec.get("value")
            if fld == "rr" and i.key not in RR_APPLIES:
                if isinstance(v, (int, float)):
                    prov["rr_sourced_not_applied"] = {"value": v, "basis": rec.get("basis")
                                                      or rec.get("outcome")}
                v = None
            if fld == "unit_cost" and i.key == "tranexamic_acid" and isinstance(v, (int, float)):
                # priced per treated hemorrhage; the model prices per delivery
                for kk in ("value", "low", "high"):
                    if isinstance(rec.get(kk), (int, float)):
                        rec[kk] = rec[kk] * pph
                v = rec["value"]
                rec["basis"] = (rec.get("basis") or "") + " Converted to per delivery at the hemorrhage rate."
            if isinstance(v, (int, float)):
                setattr(i, fld, float(v))
                lo, hi = rec.get("low"), rec.get("high")
                if isinstance(lo, (int, float)):
                    setattr(i, lo_attr, float(lo))
                if isinstance(hi, (int, float)):
                    setattr(i, hi_attr, float(hi))
                prov[fld] = {"status": "sourced", "basis": rec.get("basis") or rec.get("outcome"),
                             "source_key": rec.get("source_key"),
                             "dollar_year": rec.get("dollar_year")}
            else:
                prov[fld] = {"status": "assumption"}
            cur = getattr(i, fld)
            if getattr(i, lo_attr) is None:
                setattr(i, lo_attr, cur * 0.5 if fld != "max_reach" else max(cur * 0.5, 0.0))
            if getattr(i, hi_attr) is None:
                setattr(i, hi_attr, cur * 1.5 if fld != "max_reach" else min(cur * 1.5, 1.0))
        i.provenance = prov
    # Final evidence grades: unanimous across the investigators and 2 models, or
    # adjudicated by a physician where they disagreed (30_score_validation.py).
    fg = SOURCED.parent / "evidence_grades_final.json"
    if fg.exists():
        final = json.load(open(fg))["final"]
        word = {"Demonstrated benefit": "demonstrated benefit", "Moderate": "moderate",
                "Weak": "weak", "Null": "null", "Absent": "absent",
                "Not applicable": "not directly effect-bearing"}
        for i in ints:
            if i.key in final:
                i.grade = word[final[i.key]]
    return ints


LEVERS = ["Education or training", "Screening or referral", "Coverage or payment",
          "Care coordination or navigation", "Workforce or staffing",
          "Clinical protocol or bundle", "Community or social services",
          "Data, surveillance, or review"]
