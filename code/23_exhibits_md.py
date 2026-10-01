"""
23_exhibits_md.py — Write the main tables, figure legends, and appendix tables as
Markdown from the result files.

No number is typed here. Main exhibits go to manuscript/exhibits.md, which the
renderer appends to the manuscript; appendix tables go to manuscript/etables.md,
which the supplement template includes. Legends carry {{key}} tokens that the
renderer resolves from canonical_numbers.json.
"""
from __future__ import annotations
import collections, csv, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES, MAN = ROOT / "results", ROOT / "manuscript"
load = lambda n: json.load(open(RES / n))
f1 = lambda x: f"{x:.1f}"
money = lambda x: f"{x:,.0f}"


def registry():
    s = importlib.util.spec_from_file_location("iv", ROOT / "code/12_interventions.py")
    m = importlib.util.module_from_spec(s); sys.modules["iv"] = m; s.loader.exec_module(m)
    return m.build(load("model_params.json")["params"])


def grade_of(g: str) -> str:
    g = g.lower()
    if "absent for mortality" in g:
        return "Null or absent"
    for pre, lab in (("demonstrated", "Demonstrated benefit"), ("moderate", "Moderate"),
                     ("weak", "Weak"), ("null", "Null or absent"), ("absent", "Null or absent")):
        if g.startswith(pre):
            return lab
    return "Not effect-bearing"


def table1(ints) -> str:
    lab = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "llm_labels_passA.jsonl")}
    im = {json.loads(l)["id"]: json.loads(l)["intervention"]
          for l in open(RES / "intervention_map_passA.jsonl")}
    g = {i.key: grade_of(i.grade) for i in ints}
    name = {i.key: i.label for i in ints}
    n = len(lab)
    eff = lambda i: im.get(i) in g and g[im[i]] in ("Demonstrated benefit", "Moderate")
    out = ["## Table 1. Content of {{corpus.recommendations}} State Maternal Mortality Review "
           "Committee Recommendations", "",
           "| Characteristic | No. (%) | Naming an intervention with evidence of benefit, No. (%)^a^ |",
           "|---|---|---|"]

    def block(title, field, order=None):
        out.append(f"| *{title}* | | |")
        c = collections.Counter(lab[i][field] for i in lab)
        keys = order or [k for k, _ in c.most_common()]
        disp = {"Diffuse addressee": "General audience (eg, clinicians, communities)",
                "No actor identified": "No entity specified",
                "Licensing, accreditation, or perinatal collaborative": "Licensing, accreditation, or quality collaborative"}
        for k in keys:
            ids = [i for i in lab if lab[i][field] == k]
            e = sum(eff(i) for i in ids)
            out.append(f"| {disp.get(k, k)} | {len(ids):,} ({f1(100*len(ids)/n)}) | {e:,} ({f1(100*e/max(len(ids),1))}) |")

    block("Type of action", "policy_lever")
    block("Entity addressed", "actor_authority")
    out.append("| *Intervention named* | | |")
    c = collections.Counter(im.values())
    for k, v in c.most_common():
        nm = "None of the interventions in the table" if k == "none" else name.get(k, k)
        gr = "" if k == "none" else f" ({g.get(k, '').lower()})"
        out.append(f"| {nm}{gr} | {v:,} ({f1(100*v/n)}) | |")
    for k in ("tranexamic_acid", "uterotonics"):
        if k not in c:
            out.append(f"| {name[k]} ({g[k].lower()}) | 0 | |")
    out += ["", "^a^Demonstrated or moderate evidence of benefit on a stage of the pathway to "
            "pregnancy-related death, as graded independently by the investigators and 2 models with "
            "physician adjudication (eTable 2). Percentages in the final column are of the row. "
            "Primary classification by Claude Opus 5.5; agreement with GPT-6-astra, classifying "
            "independently, is reported in the text and eTable 4."]
    return "\n".join(out)


def table2() -> str:
    pc = load("portfolio_cea.json"); P = pc["portfolios"]
    u = load("uncertainty.json")["psa"]
    uc = load("uncertainty_current.json")["psa"] if (RES / "uncertainty_current.json").exists() else {}
    z0 = lambda x: 0.0 if abs(x) < 0.5 else x
    ui = lambda ps, k: f' ({z0(ps[k]["ci"][0]):,.0f} to {z0(ps[k]["ci"][1]):,.0f})' if k in ps else ""
    def row(lab, k, ps=None, pk=None):
        r = P[k]
        cpq = r["cost_per_qaly"]
        return (f"| {lab} | {r['deaths_averted']:,.0f}{ui(ps, pk) if ps is not None and pk else ''} | "
                f"{r['smm_averted']:,.0f} | {r['qalys_gained']:,.0f} | {r['program_cost']/1e6:,.0f} | "
                f"{cpq:,.0f} |")
    out = ["## Table 2. Projected Annual Outcomes of Implementation Scenarios Among "
           "{{pf.baseline.births}} Medicaid-Financed Births", "",
           "| Scenario | Pregnancy-related deaths averted (95% UI) | Severe maternal morbidity "
           "events averted | QALYs gained | Program cost, $ millions | Incremental cost per QALY gained, $ |",
           "|---|---|---|---|---|---|",
           "| *Beyond current practice (primary)* | | | | | |",
           row("Named interventions", "observed_incremental_current_practice", uc, "obs_deaths"),
           row("Named interventions, hemorrhage bundle including tranexamic acid",
               "requested_with_bundle_components_incremental_current_practice", uc, "comp_deaths"),
           row("Interventions with evidence of benefit", "evidence_weighted_incremental_current_practice", uc, "ev_deaths"),
           "| *Without crediting current uptake* | | | | | |",
           row("Named interventions", "observed", u, "obs_deaths"),
           row("Named interventions, hemorrhage bundle including tranexamic acid",
               "requested_with_bundle_components", u, "comp_deaths"),
           row("Interventions with evidence of benefit", "evidence_weighted", u, "ev_deaths"),
           row("Coverage proportional to how often each intervention was named", "requested_frequency", u, "freq_deaths"),
           ""]
    out += ["Abbreviations: QALY, quality-adjusted life-year; UI, uncertainty interval from "
            "{{psa.draws}} probabilistic draws with each scenario's allocation held fixed. Baseline: "
            "{{pf.baseline.deaths}} pregnancy-related deaths and {{pf.baseline.smm}} severe maternal "
            "morbidity events per year. Costs in 2025 US dollars, health care sector perspective, lifetime "
            "horizon; costs and QALYs discounted at 3%. Incremental cost per QALY gained is relative to no "
            "additional implementation. The named-intervention scenario implements every intervention named "
            "by at least 1 recommendation at full feasible coverage; the evidence-based scenario funds every "
            "intervention with evidence of benefit at full feasible coverage and costs less while gaining at "
            "least as many QALYs, so it dominates the named-intervention scenario. Current practice credits "
            "existing uptake of each intervention to baseline (eTable 3)."]
    return "\n".join(out)


LEGENDS = """## Figures

![](figures/Figure1_silence_and_concordance.png)

**Figure 1. Documented Causes of Death Without a Cause-Specific Recommendation**

A, Share of states listing each cause among their leading causes of pregnancy-related death whose published recommendations included none addressing that cause, counting recommendations coded to the cause; labels give states without a recommendation over states listing the cause. B, Median alignment ratio across states, the cause's share of the state's cause-specific recommendations divided by its share of the state's leading-cause entries, with bootstrap 95% CIs; a ratio of 1 indicates recommendations in proportion to documented causes. Hollow markers indicate a median of 0 in every bootstrap sample. Daggers mark causes listed by fewer than 10 states. {{mod.n_pairs}} documented causes in {{mod.n_states}} states. Results under 2 broader counting rules are in the text and eTable 5.

![](figures/Figure2_levers_by_evidence.png)

**Figure 2. Type of Action Requested and Evidence for the Intervention Named**

Each bar is a type of action as a share of {{corpus.recommendations}} recommendations, divided by the evidence grade of the intervention each recommendation named, graded for effect on a stage of the pathway to pregnancy-related death (eTable 2). Recommendations naming none of the {{registry.n}} interventions in the table are shown as no intervention in the table.

![](figures/Figure3_propagation.png)

**Figure 3. Documented Causes Appearing in State Policy, by Whether a Recommendation Addressed Them**

A, Share of documented causes appearing in each state policy, among causes with and without a recommendation; counts give documented causes and states with records. B, Difference associated with a recommendation, unadjusted (bootstrap 95% CI over states) and from a linear probability model with state and cause fixed effects (state-clustered 95% CI). Medicaid contracts and state laws are enforceable; quality collaborative projects and task force priorities are voluntary. Contract language requires a cause term within 300 characters of a maternal term; a withhold or incentive provision requires such language within 1,000 characters of that mention. On investigator review, the positive predictive value was {{cval.cause.ppv.pct}}% for contract language and {{cval.pay.ppv.pct}}% for withhold or incentive provisions (eTable 14). After report restricts to documents or laws dated in or after the year of the state's report.
"""


POL = {"contract_postdating": "Medicaid contract language, after report", "withhold_postdating": "Medicaid withhold or incentive, after report",
       "contract_all_documents": "Medicaid contract language", "withhold_all_documents": "Medicaid withhold or incentive",
       "pqc": "Quality collaborative project", "legislation": "State law", "legislation_postdating": "State law, after report",
       "consortium": "Task force priority"}


def etables(ints) -> str:
    P = load("model_params.json")["params"]
    out = []
    # eTable 2 intervention registry
    out += ["## eTable 2. Evidence Table: Interventions, Effects, Evidence Grades, Costs, and Coverage", "",
            "| Intervention | Policy lever | Causes | Stage | Relative risk (range) | Evidence grade (investigators / Claude Opus 5.5 / GPT-6-astra; final) | "
            "Unit cost, 2025 $ (range) | Full feasible coverage (range) | Provenance of cost and coverage | Effect source |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    eg = load("evidence_grades.json"); fin = load("evidence_grades_final.json")
    for i in ints:
        pv = i.provenance
        prov = "; ".join(f"{k.replace('_', ' ')} {v.get('status')}" for k, v in pv.items()
                         if isinstance(v, dict) and "status" in v and k != "rr")
        out.append(f"| {i.label} | {i.lever} | {', '.join(i.causes) or 'None'} | {i.stage} | "
                   f"{i.rr:.2f} ({i.rr_lo:.2f}-{i.rr_hi:.2f}) | "
                   f"{eg['investigator'][i.key]} / {eg['claude-opus-5-5'][i.key]} / {eg['gpt-6-astra'][i.key]}; "
                   f"**{fin['final'][i.key]}**{' (physician adjudicated)' if fin['source'][i.key] != 'unanimous' else ''} | "
                   f"{i.unit_cost:,.2f} ({(i.cost_lo or 0):,.2f}-{(i.cost_hi or 0):,.2f}) | "
                   f"{i.max_reach:.2f} ({(i.reach_lo or 0):.2f}-{(i.reach_hi or 0):.2f}) | {prov} | "
                   f"{i.source} |")
    out += ["", "Stage indicates where the effect acts: incidence of the complication, progression "
            "to severe maternal morbidity (smm), case fatality (cfr), none, or coverage exposure. "
            "Assumption indicates a value with no published source, varied from half to 1.5 times "
            "its value in sensitivity analysis.", ""]
    # eTable 3 model parameters
    out += ["## eTable 3. Model Parameters", "",
            "| Parameter | Value | Range | Unit | Source |", "|---|---|---|---|---|"]
    for k, v in P.items():
        if not isinstance(v, dict):
            continue
        val = v.get("value")
        if isinstance(val, dict):
            val = "; ".join(f"{kk}: {vv}" for kk, vv in val.items())
        elif isinstance(val, list):
            val = ", ".join(str(x) for x in val)
        rng = "" if v.get("low") is None else f"{v.get('low')} to {v.get('high')}"
        out.append(f"| {k.replace('_', ' ')} | {val} | {rng} | {v.get('unit', '')} | "
                   f"{(v.get('source') or '').replace('|', '/')} |")
    out += ["", "Unit costs, relative risks, and coverage for each intervention are those in eTable 2, which "
            "supersede the corresponding cost rows above where they differ.", ""]
    cp = load("current_practice.json")
    out += ["", "Current practice coverage used in the incremental scenario: " +
            "; ".join(f"{k.replace('_', ' ')} {v:.2f} ({cp['sources'][k]})"
                      for k, v in cp["coverage"].items()), ""]
    # eTable 4 classification distributions
    cls = load("classification.json")
    out += ["## eTable 4. Classification by Model and Agreement", "",
            "| Field | Category | Claude Opus 5.5, No. (%) | GPT-6-astra, No. (%) |", "|---|---|---|---|"]
    for fld in ("cause_domain", "policy_lever", "actor_authority", "coverage_period"):
        a = {r["label"]: r for r in cls.get("model_primary", cls["pass_A"])[fld]}
        b = {r["label"]: r for r in cls["pass_B"][fld]}
        for k in a:
            bb = b.get(k, {"n": 0, "pct": 0})
            out.append(f"| {fld.replace('_', ' ')} | {k} | {a[k]['n']:,} ({a[k]['pct']}) | "
                       f"{bb['n']:,} ({bb['pct']}) |")
    rel = cls["reliability"]
    out += ["", "Cohen κ between models: " + "; ".join(
        f"{k.replace('_', ' ')} {v['kappa']:.2f} (agreement {100*v['agreement']:.1f}%)"
        for k, v in rel.items()) + "; specific intervention {{kappa.intervention}}. The rule-based "
        "pilot classified " + f"{cls['rule_based_n_classified']:,}" + " recommendations by lever.", ""]
    # eTable 5 moderators
    mo = load("alignment_moderators.json")
    out += ["## eTable 5. Correlates of Documented Causes Without a Recommendation", "",
            "| Model | Term | Estimate (95% CI) | P value | Pairs | States |", "|---|---|---|---|---|---|"]
    for name, m in mo["models"].items():
        for t, v in m["terms"].items():
            if t == "Intercept":
                continue
            out.append(f"| {name.replace('_', ' ')} ({m['scale']}) | {t} | {v['estimate']:.2f} "
                       f"({v['ci'][0]:.2f} to {v['ci'][1]:.2f}) | {v['p']:.3f} | {m['n_pairs']} | {m['n_states']} |")
    out += ["", "Generalized estimating equations with exchangeable correlation within state. "
            "Reference cause group: behavioral health. Pages enter as natural log; mortality ratio "
            "per 10 deaths per 100,000 live births.", ""]
    # eTable 6 ablations
    u = load("uncertainty.json")
    out += ["## eTable 6. Structural Sensitivity Analyses", "",
            "| Analysis | Named interventions, deaths averted | Interventions with evidence of benefit, deaths averted | Ratio | Named-intervention scenario cost, $ millions |",
            "|---|---|---|---|---|"]
    ABL = {"base_case": "Base case (no current uptake)",
           "zero_effect_interventions_given_5pct_benefit": "Untested and null interventions credited with a 5% benefit",
           "non_US_trial_effects_halved": "Effects from trials outside the US halved",
           "severe_hypertension_bundle_extrapolation_removed": "Severe hypertension bundle effect removed",
           "maximum_reach_halved": "Full feasible coverage halved",
           "opioid_treatment_mortality_effect_removed": "Opioid use disorder treatment effect removed",
           "all_states_without_12_month_coverage": "No state with 12-month postpartum coverage",
           "all_states_with_12_month_coverage": "Every state with 12-month postpartum coverage",
           "no_risk_factor_heterogeneity": "No risk-factor heterogeneity",
           "uniform_severity_before_calibration": "Uniform severity before calibration",
           "requested_by_request_frequency": "Coverage proportional to how often each intervention was named",
           "requested_by_frequency_normalized_to_effective": "Coverage by frequency, scaled to the most-named effective intervention",
           "observed_portfolio_by_lever_share": "Coverage by type-of-action share",
           "incremental_over_current_practice": "Beyond current practice",
           "intervention_mapping_pass_B": "GPT-6-astra labels throughout",
           "discount_rate_0.00": "Discount rate 0%", "discount_rate_0.05": "Discount rate 5%",
           "smm_survivor_utility_decrement_0.10": "Utility decrement 0.10 for survivors of severe morbidity",
           "societal_perspective": "Societal perspective"}
    for k, v in u["ablations"].items():
        if isinstance(v, dict) and "observed_deaths_averted" in v:
            out.append(f"| {ABL.get(k, k.replace('_', ' '))} | {v['observed_deaths_averted']:,.1f} | "
                       f"{v['evidence_weighted_deaths_averted']:,.1f} | {v['ratio']:.1f} | "
                       f"{v['budget']/1e6:,.0f} |")
    for k, v in u["ablations"].items():
        if isinstance(v, dict) and "observed_deaths_averted" not in v and "error" not in v:
            desc = "; ".join(f"{kk.replace('_', ' ')}: " + (f"{vv:,.0f}" if isinstance(vv, (int, float)) else
                             "; ".join(f"{a2.replace('_', ' ')} {b2:,.0f}" for a2, b2 in vv.items()
                                       if isinstance(b2, (int, float))))
                             for kk, vv in v.items())
            desc = (desc.replace("observed", "named interventions").replace("evidence weighted", "evidence of benefit")
                    .replace("cost per qaly", "cost per QALY gained").replace("qalys", "QALYs"))
            out.append(f"| {ABL.get(k, k.replace('_', ' '))} | {desc} | | | |")
    out += ["", f"Without crediting current uptake, on a cohort of {u['n_cohort']:,} simulated pregnancies; "
            "each analysis changes 1 assumption and rebuilds both scenarios.", ""]
    # eTable 7 one-way (allocations fixed, against no implementation)
    ow = load("oneway.json")
    out += ["## eTable 7. One-Way Sensitivity Analysis", "",
            "| Parameter | Low | High | Evidence of benefit: deaths averted, low to high | Evidence of benefit: $ per QALY gained, low to high | Named interventions: deaths averted, low to high |",
            "|---|---|---|---|---|---|"]
    for r in sorted(ow["rows"], key=lambda r: -r["swing_deaths_evidence"]):
        fmtv = lambda v: f"{v:,.3g}" if isinstance(v, float) else str(v)
        out.append(f"| {r['parameter']} | {fmtv(r['low'])} | {fmtv(r['high'])} | "
                   f"{r['low']['evidence']['deaths']:.1f} to {r['high']['evidence']['deaths']:.1f} | "
                   f"{r['low']['evidence']['icer']:,.0f} to {r['high']['evidence']['icer']:,.0f} | "
                   f"{r['low']['recommended']['deaths']:.1f} to {r['high']['recommended']['deaths']:.1f} |")
    out += ["", f"Base case: evidence of benefit {ow['base']['evidence']['deaths']:.1f} deaths averted, "
            f"${ow['base']['evidence']['icer']:,.0f} per QALY gained; named interventions "
            f"{ow['base']['recommended']['deaths']:.1f} deaths averted. Without crediting current uptake.", ""]
    # eTable 8 propagation by vehicle and cause group
    pr = load("propagation.json")["channels"]
    out += ["## eTable 8. State Policy by Cause Group and Recommendation Status", "",
            "| State policy | Cause group | With a recommendation, No. | Appearing, % | Without a recommendation, No. | Appearing, % |",
            "|---|---|---|---|---|---|"]
    for k, v in pr.items():
        for gname, gv in v["by_cause_group"].items():
            fmt = lambda x: "—" if x is None else f1(100 * x)
            out.append(f"| {POL.get(k, k)} | {gname} | {gv['n_rec']} | "
                       f"{fmt(gv['present_if_recommended'])} | {gv['n_not']} | "
                       f"{fmt(gv['present_if_not_recommended'])} |")
    out += ["", "| State policy | States | Documented causes | Fixed-effects difference, pp (95% CI) | Minimum detectable difference, pp | Mantel-Haenszel difference, pp (95% CI) |",
            "|---|---|---|---|---|---|"]
    for k, v in pr.items():
        fe, mh = v.get("fixed_effects") or {}, v.get("adjusted_for_cause_group") or {}
        fes = (f"{100*fe['difference']:.1f} ({100*fe['ci'][0]:.1f} to {100*fe['ci'][1]:.1f})"
               if "difference" in fe else "—")
        mhs = (f"{100*mh['mh_risk_difference']:.1f} ({100*mh['ci'][0]:.1f} to {100*mh['ci'][1]:.1f})"
               if mh.get("mh_risk_difference") is not None else "—")
        mdd = f"{100*fe['mdd']:.0f}" if "mdd" in fe else "—"
        out.append(f"| {POL.get(k, k)} | {v['n_states_covered']} | {v['overall']['n_pairs']} | {fes} | {mdd} | {mhs} |")
    out.append("")
    # eTable 9 committee composition
    cc = load("committee_composition.json")["reports"]
    out += ["## eTable 9. Committee Rosters Read From Reports", "",
            "| State | Report year | Pages | Roster printed | Medicaid agency member | Managed care or payer member | Obstetrician or MFM member |",
            "|---|---|---|---|---|---|---|"]
    yn = lambda v: "Unknown" if v is None else ("Yes" if v else "No")
    for r in sorted(cc, key=lambda r: (r["state"], r["report_year"])):
        out.append(f"| {r['state']} | {r['report_year']} | {r['n_pages']} | {yn(r.get('roster_found'))} | "
                   f"{yn(r.get('medicaid_agency_member'))} | {yn(r.get('mco_or_payer_member'))} | "
                   f"{yn(r.get('obstetrician_or_mfm_member'))} |")
    out.append("")
    # eTable 10 calibration
    pc = load("portfolio_cea.json")
    cal = pc["calibration"]
    out += ["## eTable 10. Calibration and External Validation", "",
            "| Cause | Probability complication is severe | Case fatality among severe | Modeled deaths per complication | Published | Ratio |",
            "|---|---|---|---|---|---|"]
    chk = cal["case_fatality_check"]
    for cz, cf in cal["cfr"].items():
        c = chk.get(cz, {})
        out.append(f"| {cz} | {cal['p_smm_given_comp'][cz]:.3f} | {cf:.4f} | "
                   f"{c.get('modeled_deaths_per_complication', '')} | {c.get('published', '')} | "
                   f"{c.get('ratio', '')} |")
    out += ["", f"Progression multiplier {cal['kappa']:.3f}; achieved severe maternal morbidity "
            f"{cal['achieved_smm_per_10k']:.1f} per 10,000 against a target of {{{{model.smm_target}}}}. "
            "Published comparisons are external checks and did not drive calibration.", ""]
    out.append("<!-- SPLIT -->")
    # eTable 12 accuracy
    vst = load("validation_stats.json")
    names = {"evidence_of_benefit": "Named an intervention with evidence of benefit",
             "any_registry_intervention": "Named any intervention in the table",
             "obstetric_cardiovascular_cause": "Addressed an obstetric or cardiovascular cause",
             "binding_actor": "Named an entity with authority to act"}
    ci = lambda x: f"{x[0]} of {x[1]} ({100*x[2][0]:.1f}-{100*x[2][1]:.1f})" if x[1] else "—"
    out += ["## eTable 12. Accuracy of the Classification Against Blinded Physician Coding", "",
            "| Decision | Sensitivity (95% CI) | Specificity (95% CI) | Positive predictive value (95% CI) | Negative predictive value (95% CI) |",
            "|---|---|---|---|---|"]
    for d, v in vst["diagnostic"].items():
        out.append(f"| {names[d]} | {ci(v['sensitivity'])} | {ci(v['specificity'])} | {ci(v['ppv'])} | {ci(v['npv'])} |")
    kc = vst["kappa_ci"]
    out += ["", "| Field | κ vs blinded physician (95% CI) |", "|---|---|"]
    for f, lab_ in (("cause_domain", "Cause of death addressed"), ("policy_lever", "Type of action"),
                    ("intervention", "Intervention named"), ("binding_actor", "Entity with authority to act")):
        out.append(f"| {lab_} | {kc[f]['kappa']:.2f} ({kc[f]['ci'][0]:.2f}-{kc[f]['ci'][1]:.2f}) |")
    b = vst["bounds"]
    out += ["", f"Sample of {vst['n']} recommendations drawn from those on which the 2 models agreed; percentages "
            "with Wilson 95% CIs. Bounds on corpus shares, assigning every unadjudicated disagreement to either "
            f"model: intervention with evidence of benefit {b['evidence_of_benefit']['lower_pct']}%-"
            f"{b['evidence_of_benefit']['upper_pct']}%; entity with authority to act "
            f"{b['binding_actor']['lower_pct']}%-{b['binding_actor']['upper_pct']}%. Keyword audit of full text: "
            + "; ".join(f"{k.replace('_', ' ')} {v['mentions']} mentions" for k, v in vst["keyword_audit"].items()) + ".", ""]
    # eTable 13 outcomes by race
    P2 = load("portfolio_cea.json")["portfolios"]
    out += ["## eTable 13. Projected Outcomes by Race", "",
            "| Scenario | Deaths averted per 100,000 births, Black | Deaths averted per 100,000 births, other |", "|---|---|---|"]
    for k, lab_ in (("observed_incremental_current_practice", "Named interventions, beyond current practice"),
                    ("requested_with_bundle_components_incremental_current_practice", "Named, bundle including tranexamic acid, beyond current practice"),
                    ("evidence_weighted_incremental_current_practice", "Evidence of benefit, beyond current practice"),
                    ("observed", "Named interventions, no current uptake"),
                    ("evidence_weighted", "Evidence of benefit, no current uptake")):
        br = P2[k].get("by_race")
        if br:
            out.append(f"| {lab_} | {br['black_per_100k']:.1f} | {br['other_per_100k']:.1f} |")
    br0 = P2["observed"]["by_race"]
    out += ["", f"Baseline pregnancy-related mortality ratio in the modeled cohort: {br0['baseline_black_prmr']:.1f} "
            f"per 100,000 births among Black and {br0['baseline_other_prmr']:.1f} among other births.", ""]
    # eTable 14 contract text rules: accuracy and window width
    cv = load("contract_validation_scores.json")["rules"]
    w = lambda x: f"{x['k']} of {x['n']} ({100*x['value']:.1f}; {100*x['ci'][0]:.1f}-{100*x['ci'][1]:.1f})"
    out += ["## eTable 14. Accuracy of the Contract Text Rules and Sensitivity to Window Width", "",
            "| Rule | Positive predictive value, No. (%; 95% CI) | Negative predictive value, No. (%; 95% CI) | Positive predictive value, narrower window, No. (%; 95% CI) |",
            "|---|---|---|---|"]
    for sheet, lab_ in (("Cause mentions", "Contract language: cause term within 300 characters of a maternal term"),
                        ("Payment provisions", "Withhold or incentive provision: payment term within 1,000 characters")):
        r = cv[sheet]
        out.append(f"| {lab_} | {w(r['ppv'])} | {w(r['npv'])} | {w(r['ppv_narrow_window'])} at {r['narrow_window_chars']:,} characters |")
    out += ["", "| Window widths | Contract language, fixed-effects difference, percentage points (95% CI) | Withhold or incentive provision, fixed-effects difference, percentage points (95% CI) |",
            "|---|---|---|"]
    fe = lambda d, ch: d["channels"][ch]["fixed_effects"]
    fmt = lambda f: f"{100*f['difference']:.1f} ({100*f['ci'][0]:.1f} to {100*f['ci'][1]:.1f})"
    for tag, lab_ in (("", "300 and 1,000 characters (primary)"), ("_ctx150", "150 and 1,000"), ("_ctx600", "600 and 1,000"),
                      ("_wh500", "300 and 500"), ("_wh2000", "300 and 2,000")):
        d = load(f"propagation{tag}.json")
        out.append(f"| {lab_} | {fmt(fe(d, 'contract_all_documents'))} | {fmt(fe(d, 'withhold_all_documents'))} |")
    out += ["", "An investigator blinded to the rule's decision reviewed 80 passages per rule, 40 flagged and 40 not "
            "flagged, sampled across states and causes. Positive predictive value is the share of flagged passages "
            "that referred to the condition in pregnant or postpartum people (contract language) or tied its care to "
            "a payment consequence (withhold or incentive); negative predictive value is the share of unflagged "
            "passages that did not. No passage was judged unclear. The narrower-window value is computed on flagged "
            "passages that also met the narrower window, measured within each passage. Window widths are given as "
            "contract-language window and payment window; fixed-effects models include state and cause fixed effects "
            "with state-clustered standard errors, using all documents.", ""]
    return "\n".join(out)


def main() -> None:
    ints = registry()
    (MAN / "exhibits.md").write_text(table1(ints) + "\n\n" + table2() + "\n\n" + LEGENDS)
    pre, post = etables(ints).split("<!-- SPLIT -->")
    (MAN / "etables.md").write_text(pre)
    (MAN / "etables_post.md").write_text(post)
    print("wrote exhibits.md and etables.md")


if __name__ == "__main__":
    main()
