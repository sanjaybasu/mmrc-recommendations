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
           "| Characteristic | No. (%) | Requesting an intervention with evidence of benefit, No. (%)^a^ |",
           "|---|---|---|"]

    def block(title, field, order=None):
        out.append(f"| *{title}* | | |")
        c = collections.Counter(lab[i][field] for i in lab)
        keys = order or [k for k, _ in c.most_common()]
        for k in keys:
            ids = [i for i in lab if lab[i][field] == k]
            e = sum(eff(i) for i in ids)
            out.append(f"| {k} | {len(ids):,} ({f1(100*len(ids)/n)}) | {e:,} ({f1(100*e/max(len(ids),1))}) |")

    block("Policy lever", "policy_lever")
    block("Most binding actor named", "actor_authority")
    out.append("| *Specific intervention requested* | | |")
    c = collections.Counter(im.values())
    for k, v in c.most_common():
        nm = "No specific intervention" if k == "none" else name.get(k, k)
        gr = "" if k == "none" else f" ({g.get(k, '').lower()})"
        out.append(f"| {nm}{gr} | {v:,} ({f1(100*v/n)}) | |")
    for k in ("tranexamic_acid", "uterotonics"):
        if k not in c:
            out.append(f"| {name[k]} ({g[k].lower()}) | 0 | |")
    out += ["", "^a^Demonstrated or moderate evidence of benefit on a stage of the pathway to "
            "pregnancy-related death (eTable 2). Percentages in the final column are of the row. "
            "Primary classification by Claude Opus 5.5; agreement with GPT-6-astra, classifying "
            "independently, is reported in the text and eTable 4."]
    return "\n".join(out)


def table2() -> str:
    pc, u = load("portfolio_cea.json"), load("uncertainty.json")
    P, ps = pc["portfolios"], u["psa"]
    ui = lambda k: f'{ps[k]["ci"][0]:,.0f} to {ps[k]["ci"][1]:,.0f}' if k in ps else "—"
    rows = [("Requested portfolio (every requested intervention at full reach)", "observed", "obs_deaths"),
            ("Within-lever portfolio", "within_lever_optimal", "wl_deaths"),
            ("Burden-aligned portfolio", "burden_aligned", "ba_deaths"),
            ("Evidence-weighted portfolio", "evidence_weighted", "ev_deaths")]
    out = ["## Table 2. Projected Annual Outcomes of Recommendation Portfolios Among "
           "{{pf.baseline.births}} Medicaid-Financed Births", "",
           "| Portfolio | Pregnancy-related deaths averted (95% UI) | Severe maternal morbidity "
           "events averted | QALYs gained | Program cost, $ millions | Cost per QALY, $ |",
           "|---|---|---|---|---|---|",
           "| *Compared with no implementation* | | | | | |"]
    for lab, k, pk in rows:
        r = P[k]
        out.append(f"| {lab} | {r['deaths_averted']:,.0f} ({ui(pk)}) | {r['smm_averted']:,.0f} | "
                   f"{r['qalys_gained']:,.0f} | {r['program_cost']/1e6:,.0f} | "
                   f"{r['cost_per_qaly']:,.0f} |")
    out.append("| *Alternative constructions of the requested portfolio* | | | | | |")
    for lab, k, pk in (("Coverage proportional to request frequency", "requested_frequency", "freq_deaths"),
                       ("Every requested intervention except nurse home visiting",
                        "requested_excluding_home_visiting", None)):
        if k in P:
            r = P[k]
            out.append(f"| {lab} | {r['deaths_averted']:,.0f}{(' (' + ui(pk) + ')') if pk else ''} | "
                       f"{r['smm_averted']:,.0f} | {r['qalys_gained']:,.0f} | "
                       f"{r['program_cost']/1e6:,.0f} | {r['cost_per_qaly']:,.0f} |")
    out.append("| *Incremental over current practice* | | | | | |")
    for lab, k in (("Requested portfolio", "observed_incremental_current_practice"),
                   ("Evidence-weighted portfolio", "evidence_weighted_incremental_current_practice")):
        r = P[k]
        out.append(f"| {lab} | {r['deaths_averted']:,.0f} | {r['smm_averted']:,.0f} | "
                   f"{r['qalys_gained']:,.0f} | {r['program_cost']/1e6:,.0f} | "
                   f"{r['cost_per_qaly']:,.0f} |")
    out += ["", "Abbreviations: QALY, quality-adjusted life-year; UI, uncertainty interval from "
            "{{psa.draws}} probabilistic draws. Baseline: {{pf.baseline.deaths}} pregnancy-related "
            "deaths and {{pf.baseline.smm}} severe maternal morbidity events per year. Costs in 2025 "
            "US dollars, health care sector perspective; costs and QALYs discounted at 3%. The "
            "requested portfolio implements every intervention requested by at least 1 "
            "recommendation at its maximum attainable reach. The evidence-weighted portfolio allocates the requested "
            "portfolio's budget to maximize deaths averted and exhausts all interventions with "
            "evidence of benefit before spending it. The within-lever portfolio holds each policy "
            "lever's budget fixed; the burden-aligned portfolio sets coverage by cause-specific "
            "share of deaths. Current practice credits existing uptake of each intervention to the "
            "baseline (eTable 3). Uncertainty intervals hold each portfolio's base-case allocation "
            "fixed across draws."]
    return "\n".join(out)


LEGENDS = """## Figures

![](figures/Figure1_silence_and_concordance.png)

**Figure 1. Silence and Alignment of Recommendations With Documented Causes of Death**

A, Share of states documenting each cause among their leading causes of pregnancy-related death that issued no recommendation addressing it; labels give states silent over states documenting. B, Median concordance ratio across states, the share of the state's cause-specific recommendations addressing the cause divided by the cause's share of the state's documented leading causes, with bootstrap 95% CIs; a ratio of 1 indicates recommendations in proportion to documented burden. Hollow markers indicate a median of 0 in every bootstrap sample. Daggers mark causes documented by fewer than 10 states. {{mod.n_pairs}} state-cause pairs in {{mod.n_states}} states.

![](figures/Figure2_levers_by_evidence.png)

**Figure 2. Policy Levers and the Evidence for the Intervention Requested**

Each bar is a policy lever as a share of {{corpus.recommendations}} recommendations, divided by the evidence grade of the specific intervention each recommendation requested, graded for effect on a stage of the pathway to pregnancy-related death (eTable 2). Recommendations requesting no intervention in the registry are shown as no specific intervention.

![](figures/Figure3_propagation.png)

**Figure 3. Presence of Documented Causes in Downstream Policy Vehicles by Committee Recommendation Status**

A, Share of documented state-cause pairs present in each vehicle, among pairs on which the committee did and did not issue a recommendation; counts give pairs and states with records for the vehicle. B, Difference in presence associated with a recommendation, unadjusted and from a linear probability model with state and cause fixed effects, with 95% CIs (bootstrap over states for unadjusted; state-clustered for fixed effects). Binding vehicles can require action; voluntary vehicles organize it. Contract language requires a cause term within 300 characters of a maternal term; a withhold or incentive provision requires such language within 1,000 characters of that mention. After report restricts to documents or bills dated in or after the year of the state's review report.
"""


def etables(ints) -> str:
    P = load("model_params.json")["params"]
    out = []
    # eTable 2 intervention registry
    out += ["## eTable 2. Intervention Registry: Effects, Evidence Grades, Costs, and Reach", "",
            "| Intervention | Policy lever | Causes | Stage | Relative risk (range) | Evidence grade (investigators / Claude Opus 5.5 / GPT-6-astra; final) | "
            "Unit cost, 2025 $ (range) | Maximum reach (range) | Provenance of cost and reach | Effect source |",
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
    out += ["## eTable 5. Correlates of Silence and Concordance", "",
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
    out += ["## eTable 6. Structural Ablations", "",
            "| Scenario | Requested portfolio, deaths averted | Evidence-weighted portfolio, deaths averted | Ratio | Requested portfolio cost, $ millions |",
            "|---|---|---|---|---|"]
    for k, v in u["ablations"].items():
        if isinstance(v, dict) and "observed_deaths_averted" in v:
            out.append(f"| {k.replace('_', ' ')} | {v['observed_deaths_averted']:,.1f} | "
                       f"{v['evidence_weighted_deaths_averted']:,.1f} | {v['ratio']:.1f} | "
                       f"{v['budget']/1e6:,.0f} |")
    for k, v in u["ablations"].items():
        if isinstance(v, dict) and "observed_deaths_averted" not in v and "error" not in v:
            desc = "; ".join(f"{kk.replace('_', ' ')}: " + (f"{vv:,.0f}" if isinstance(vv, (int, float)) else
                             "; ".join(f"{a2.replace('_', ' ')} {b2:,.0f}" for a2, b2 in vv.items()
                                       if isinstance(b2, (int, float))))
                             for kk, vv in v.items())
            out.append(f"| {k.replace('_', ' ')} | {desc} | | | |")
    out += ["", f"Sensitivity cohort of {u['n_cohort']:,} simulated pregnancies.", ""]
    # eTable 7 one-way
    out += ["## eTable 7. One-Way Sensitivity Analysis", "",
            "| Parameter | Low | High | Difference in deaths averted at low | at high | Requested cost per QALY at low | at high | Evidence-weighted cost per QALY at low | at high |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in u["one_way"]:
        g = lambda k: f"{r[k]:,.0f}" if isinstance(r.get(k), (int, float)) and r[k] < 1e12 else "—"
        out.append(f"| {r['parameter'].replace('_', ' ')} | {r['low_value']} | {r['high_value']} | "
                   f"{g('gap_at_low')} | {g('gap_at_high')} | {g('obs_cpq_at_low')} | "
                   f"{g('obs_cpq_at_high')} | {g('ev_cpq_at_low')} | {g('ev_cpq_at_high')} |")
    out.append("")
    # eTable 8 propagation by vehicle and cause group
    pr = load("propagation.json")["channels"]
    out += ["## eTable 8. Presence in Binding Vehicles by Cause Group and Recommendation Status", "",
            "| Vehicle | Cause group | Recommended, No. | Present if recommended, % | Not recommended, No. | Present if not recommended, % |",
            "|---|---|---|---|---|---|"]
    for k, v in pr.items():
        for gname, gv in v["by_cause_group"].items():
            fmt = lambda x: "—" if x is None else f1(100 * x)
            out.append(f"| {k.replace('_', ' ')} | {gname} | {gv['n_rec']} | "
                       f"{fmt(gv['present_if_recommended'])} | {gv['n_not']} | "
                       f"{fmt(gv['present_if_not_recommended'])} |")
    out += ["", "| Vehicle | States | Pairs | Fixed-effects difference, pp (95% CI) | Mantel-Haenszel difference, pp (95% CI) |",
            "|---|---|---|---|---|"]
    for k, v in pr.items():
        fe, mh = v.get("fixed_effects") or {}, v.get("adjusted_for_cause_group") or {}
        fes = (f"{100*fe['difference']:.1f} ({100*fe['ci'][0]:.1f} to {100*fe['ci'][1]:.1f})"
               if "difference" in fe else "—")
        mhs = (f"{100*mh['mh_risk_difference']:.1f} ({100*mh['ci'][0]:.1f} to {100*mh['ci'][1]:.1f})"
               if mh.get("mh_risk_difference") is not None else "—")
        out.append(f"| {k.replace('_', ' ')} | {v['n_states_covered']} | {v['overall']['n_pairs']} | {fes} | {mhs} |")
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
    return "\n".join(out)


def main() -> None:
    ints = registry()
    (MAN / "exhibits.md").write_text(table1(ints) + "\n\n" + table2() + "\n\n" + LEGENDS)
    (MAN / "etables.md").write_text(etables(ints))
    print("wrote exhibits.md and etables.md")


if __name__ == "__main__":
    main()
