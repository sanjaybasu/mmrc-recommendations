"""
17_canonical.py — Emit every number the manuscript is allowed to state.

CANONICAL_NUMBERS.md is the single source of truth. Each entry is a key, the
value, and the result file the value came from. The consistency audit reads this
file and checks the manuscript against it, so a number that changes in an
analysis cannot survive in the prose.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
load = lambda n: json.load(open(RES / n))

rows: list[tuple[str, str, str]] = []
add = lambda k, v, src: rows.append((k, v, src))


def main() -> None:
    pil = load("pilot_numbers.json")
    c = pil["corpus"]
    add("corpus.documents", f'{c["documents"]}', "pilot_numbers.json")
    add("corpus.jurisdictions", f'{c["jurisdictions"]}', "pilot_numbers.json")
    add("corpus.year_first", c["report_years"][0], "pilot_numbers.json")
    add("corpus.year_last", c["report_years"][1], "pilot_numbers.json")
    add("corpus.recommendations", f'{c["n_recommendations"]:,}', "pilot_numbers.json")
    add("corpus.findings", f'{c["n_findings"]:,}', "pilot_numbers.json")
    add("corpus.states_with_recommendations", f'{c["states_with_recommendations"]}',
        "pilot_numbers.json")

    cls = load("classification.json")
    for f, v in cls["reliability"].items():
        add(f"kappa.{f}", f'{v["kappa"]:.2f}', "classification.json")
        add(f"agreement.{f}", f'{v["agreement"]:.3f}', "classification.json")
    for r in cls["pass_A"]["policy_lever"]:
        key = r["label"].split(" or ")[0].split(",")[0].lower().replace(" ", "_")
        add(f"lever.{key}.n", f'{r["n"]:,}', "classification.json")
        add(f"lever.{key}.pct", f'{r["pct"]:.1f}', "classification.json")
    for r in cls["pass_A"]["cause_domain"]:
        key = r["label"].split(",")[0].lower().replace(" ", "_")
        add(f"cause.{key}.n", f'{r["n"]:,}', "classification.json")
        add(f"cause.{key}.pct", f'{r["pct"]:.1f}', "classification.json")
    for r in cls["pass_A"]["actor_authority"]:
        key = r["label"].split(",")[0].split(" or ")[0].lower().replace(" ", "_")
        add(f"actor.{key}.n", f'{r["n"]:,}', "classification.json")
        add(f"actor.{key}.pct", f'{r["pct"]:.1f}', "classification.json")
    add("actor.binding.pct", f'{cls["pass_A"]["pct_naming_binding_actor"]:.1f}',
        "classification.json")
    for r in cls["pass_A"]["coverage_period"]:
        key = r["label"].split(" ")[0].lower()
        add(f"coverage_period.{key}.n", f'{r["n"]:,}', "classification.json")
        add(f"coverage_period.{key}.pct", f'{r["pct"]:.1f}', "classification.json")

    co = load("concordance.json")
    for r in co["concordance"]:
        k = r["domain"].split(",")[0].lower().replace(" ", "_")
        add(f"concordance.{k}.median", f'{r["median_ratio"]:.2f}', "concordance.json")
        add(f"concordance.{k}.ci", f'{r["ci"][0]:.2f} to {r["ci"][1]:.2f}', "concordance.json")
        add(f"concordance.{k}.n_states", f'{r["n_states"]}', "concordance.json")
    for r in co["silence"]:
        k = r["domain"].split(",")[0].lower().replace(" ", "_")
        add(f"silence.{k}.documenting", f'{r["states_documenting"]}', "concordance.json")
        add(f"silence.{k}.zero", f'{r["states_zero_recs"]}', "concordance.json")
        add(f"silence.{k}.pct", f'{r["pct"]:.1f}', "concordance.json")

    ac = {"funnel": [], "n_states": 0, "window_chars": 0}  # superseded by propagation.json
    # rule-based funnel superseded by propagation.json; not emitted

    P = load("model_params.json")["params"]
    add("model.prmr_target", f'{P["prmr_for_model"]["value"]:.1f}', "model_params.json")
    add("model.smm_target", f'{P["smm_rate"]["value"]:.1f}', "model_params.json")
    add("model.corpus_prmr_weighted",
        f'{P["corpus_prmr_detail"]["medicaid_birth_weighted_mean"]:.1f}', "model_params.json")
    add("model.corpus_prmr_median",
        f'{P["corpus_prmr_detail"]["median_of_state_medians"]:.1f}', "model_params.json")
    add("model.corpus_prmr_states", f'{P["corpus_prmr_detail"]["n_states"]}', "model_params.json")
    add("model.external_prmr", f'{P["external_prmr_check"]["value"]:.1f}', "model_params.json")
    add("model.discount_rate", f'{P["discount_rate"]["value"]:.2f}', "model_params.json")
    add("model.life_expectancy_30", f'{P["life_expectancy_female_30"]["value"]:.1f}',
        "model_params.json")

    den = load("denominators.json")
    add("denominator.us_births", f'{den["national"]["total_births_2024"]:,}',
        "denominators.json")
    add("denominator.medicaid_share",
        f'{den["national"]["medicaid_share_of_births_2024_pct"]:.1f}', "denominators.json")
    add("denominator.medicaid_births", f'{den["national"]["medicaid_financed_births_2024"]:,}',
        "denominators.json")

    pc = load("portfolio_cea.json")
    b = pc["baseline"]
    add("sim.births", f'{b["births"]:,.0f}', "portfolio_cea.json")
    add("sim.deaths", f'{b["deaths"]:,.0f}', "portfolio_cea.json")
    add("sim.smm", f'{b["smm_events"]:,.0f}', "portfolio_cea.json")
    add("sim.prmr", f'{b["prmr"]:.1f}', "portfolio_cea.json")
    add("sim.smm_per_10k", f'{b["smm_per_10k"]:.1f}', "portfolio_cea.json")
    add("sim.n_simulated", f'{pc["n_simulated"]:,}', "portfolio_cea.json")
    cal = pc["calibration"]
    add("sim.progression_multiplier", f'{cal["kappa"]:.2f}', "portfolio_cea.json")
    for cz, v in cal["case_fatality_check"].items():
        k = cz.split(",")[0].lower().replace(" ", "_")
        add(f"validation.{k}.modeled", f'{100*v["modeled_deaths_per_complication"]:.3f}',
            "portfolio_cea.json")
        add(f"validation.{k}.published", f'{100*v["published"]:.3f}', "portfolio_cea.json")
        add(f"validation.{k}.ratio", f'{v["ratio"]:.2f}', "portfolio_cea.json")

    for name, r in pc["portfolios"].items():
        add(f"portfolio.{name}.deaths_averted", f'{r["deaths_averted"]:,.0f}',
            "portfolio_cea.json")
        add(f"portfolio.{name}.smm_averted", f'{r["smm_averted"]:,.0f}', "portfolio_cea.json")
        add(f"portfolio.{name}.qalys", f'{r["qalys_gained"]:,.0f}', "portfolio_cea.json")
        add(f"portfolio.{name}.program_cost_m", f'{r["program_cost"]/1e6:,.0f}',
            "portfolio_cea.json")
        add(f"portfolio.{name}.net_cost_m", f'{r["net_cost"]/1e6:,.0f}', "portfolio_cea.json")
        add(f"portfolio.{name}.cost_per_qaly", f'{r["cost_per_qaly"]:,.0f}',
            "portfolio_cea.json")
    ev, ob = pc["portfolios"]["evidence_weighted"], pc["portfolios"]["observed"]
    add("headline.ratio_deaths", f'{ev["deaths_averted"]/ob["deaths_averted"]:.1f}',
        "portfolio_cea.json")
    add("headline.cost_share_pct", f'{100*ev["program_cost"]/ob["program_cost"]:.0f}',
        "portfolio_cea.json")
    add("headline.extra_deaths", f'{ev["deaths_averted"]-ob["deaths_averted"]:,.0f}',
        "portfolio_cea.json")
    add("budget.observed_total_b", f'{pc["observed_budget"]/1e9:.2f}', "portfolio_cea.json")
    add("budget.per_birth", f'{pc["budget_per_birth"]:,.0f}', "portfolio_cea.json")
    for lv, v in pc["lever_budget"].items():
        k = lv.split(" or ")[0].split(",")[0].lower().replace(" ", "_")
        add(f"spend.{k}.m", f'{v/1e6:,.0f}', "portfolio_cea.json")
        add(f"spend.{k}.pct", f'{100*v/pc["observed_budget"]:.1f}', "portfolio_cea.json")

    if (RES / "uncertainty.json").exists():
        u = load("uncertainty.json")
        add("psa.draws", f'{u["psa"]["n_draws"]:,}', "uncertainty.json")
        for k in ["obs_deaths", "ev_deaths", "wl_deaths", "ba_deaths", "freq_deaths", "comp_deaths", "gap",
                  "ratio", "obs_cost", "ev_cost", "obs_qaly", "ev_qaly"]:
            if k not in u["psa"]:
                continue
            s = u["psa"][k]
            d = 1 if k == "ratio" else 0
            add(f"psa.{k}.median", f'{s["median"]:,.{d}f}', "uncertainty.json")
            lo_, hi_ = [0.0 if abs(x) < 0.5 * 10 ** (-d) else x for x in s["ci"]]
            add(f"psa.{k}.ci", f'{lo_:,.{d}f} to {hi_:,.{d}f}', "uncertainty.json")
        if "prob_evidence_weighted_averts_at_least_2x" in u["psa"]:
            add("psa.prob_2x", f'{100*u["psa"]["prob_evidence_weighted_averts_at_least_2x"]:.1f}',
                "uncertainty.json")
        if "prob_evidence_weighted_averts_at_least_1_5x" in u["psa"]:
            add("psa.prob_1_5x", f'{100*u["psa"]["prob_evidence_weighted_averts_at_least_1_5x"]:.1f}',
                "uncertainty.json")
        for w, v in (u["psa"].get("ceac") or {}).items():
            for kk, vv in v.items():
                add(f"psa.ceac.{int(w)//1000}k.{kk}", f"{100*vv:.1f}", "uncertainty.json")
        add("psa.prob_better", f'{100*u["psa"]["prob_evidence_weighted_averts_more"]:.1f}',
            "uncertainty.json")
        if u["one_way"]:
            add("oneway.top_parameter", u["one_way"][0]["parameter"], "uncertainty.json")
            lab = {"prmr_for_model": "calibration pregnancy-related mortality ratio",
                   "smm_rate": "calibration severe maternal morbidity rate"}
            top = u["one_way"][0]["parameter"]
            add("oneway.top_label", lab.get(top, top.replace("_", " ")), "uncertainty.json")
            add("oneway.top_swing", f'{u["one_way"][0]["swing"]:,.0f}', "uncertainty.json")
        add("ablation.count", str(sum(1 for k, v in u["ablations"].items()
                                      if isinstance(v, dict) and "observed_deaths_averted" in v
                                      and k != "base_case")), "uncertainty.json")
        for k, v in u["ablations"].items():
            if isinstance(v, dict) and "observed_deaths_averted" in v:
                add(f"ablation.{k}.observed", f'{v["observed_deaths_averted"]:,.0f}',
                    "uncertainty.json")
                add(f"ablation.{k}.evidence", f'{v["evidence_weighted_deaths_averted"]:,.0f}',
                    "uncertainty.json")
                add(f"ablation.{k}.ratio", f'{v["ratio"]:.1f}', "uncertainty.json")

    # ---- additions: intervention mapping, moderators, propagation, model ----
    import csv, collections, importlib.util, sys
    f1 = lambda x: f"{(0.0 if abs(x) < 0.05 else x):.1f}"
    z = lambda v: 0.0 if abs(v) < 0.05 else v
    ci = lambda a, b, d=1, sc=1.0: f"{z(a*sc):,.{d}f} to {z(b*sc):,.{d}f}"
    n_recs = sum(1 for _ in open(RES / "intervention_map_passA.jsonl"))
    imA = [json.loads(l) for l in open(RES / "intervention_map_passA.jsonl")]
    imB = {json.loads(l)["id"]: json.loads(l)["intervention"]
           for l in open(RES / "imap_gpt-6-astra.jsonl")}
    imM1 = [json.loads(l) for l in open(RES / "imap_claude-opus-5-5.jsonl")]
    cnt = collections.Counter(r["intervention"] for r in imA)
    for k, v in cnt.items():
        add(f"imap.{k}.n", f"{v:,}", "intervention_map_passA.jsonl")
        add(f"imap.{k}.pct", f1(100 * v / n_recs), "intervention_map_passA.jsonl")
    for k in ("tranexamic_acid", "uterotonics", "aspirin_prophylaxis"):
        if k not in cnt:
            add(f"imap.{k}.n", "0", "intervention_map_passA.jsonl")
    add("imap.aspirin.n", str(cnt.get("aspirin_prophylaxis", 0)), "intervention_map_passA.jsonl")
    # kappa between passes
    import numpy as np
    ids = [r["id"] for r in imA if r["id"] in imB]
    cats = sorted({r["intervention"] for r in imA} | set(imB.values())); ix = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    cats = sorted({r["intervention"] for r in imM1} | set(imB.values())); ix = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    for r in imM1:
        if r["id"] in imB:
            M[ix[r["intervention"]], ix[imB[r["id"]]]] += 1
    po = np.trace(M) / M.sum(); pe = (M.sum(0) / M.sum()) @ (M.sum(1) / M.sum())
    add("kappa.intervention", f"{(po - pe) / (1 - pe):.2f}", "intervention_map_pass{A,B}.jsonl")
    # evidence grade of requested intervention
    spec = importlib.util.spec_from_file_location("iv", ROOT / "code/12_interventions.py")
    ivm = importlib.util.module_from_spec(spec); sys.modules["iv"] = ivm; spec.loader.exec_module(ivm)
    ints = ivm.build(P)
    add("registry.n", str(len(ints)), "12_interventions.py")
    gmap = {}
    for i in ints:
        g = i.grade.lower()
        if "absent for mortality" in g:
            gmap[i.key] = "absent"; continue
        gmap[i.key] = ("demonstrated" if g.startswith("demonstrated") else "moderate" if
                       g.startswith("moderate") else "weak" if g.startswith("weak") else
                       "null" if g.startswith("null") else "absent" if g.startswith("absent")
                       else "not_effect_bearing")
    gc = collections.Counter(gmap.get(r["intervention"], "none") if r["intervention"] != "none"
                             else "none" for r in imA)
    for g, v in gc.items():
        add(f"grade.{g}.n", f"{v:,}", "intervention_map_passA.jsonl + 12_interventions.py")
        add(f"grade.{g}.pct", f1(100 * v / n_recs), "intervention_map_passA.jsonl + 12_interventions.py")
    eff = gc.get("demonstrated", 0) + gc.get("moderate", 0)
    add("imap.effective.n", f"{eff:,}", "intervention_map_passA.jsonl + 12_interventions.py")
    add("imap.effective.pct", f1(100 * eff / n_recs), "intervention_map_passA.jsonl + 12_interventions.py")
    add("imap.none.pct", f1(100 * cnt.get("none", 0) / n_recs), "intervention_map_passA.jsonl")
    add("imap.none.n", f"{cnt.get('none', 0):,}", "intervention_map_passA.jsonl")
    # among recommendations naming a specific intervention
    named = n_recs - cnt.get("none", 0)
    add("imap.named.n", f"{named:,}", "intervention_map_passA.jsonl")
    add("imap.effective_of_named.pct", f1(100 * eff / named), "intervention_map_passA.jsonl")
    add("imap.null_of_named.pct", f1(100 * (gc.get("null", 0) + gc.get("absent", 0)) / named),
        "intervention_map_passA.jsonl")
    add("imap.nullonly_of_named.pct", f1(100 * gc.get("null", 0) / named), "intervention_map_passA.jsonl")
    add("imap.absent_of_named.pct", f1(100 * gc.get("absent", 0) / named), "intervention_map_passA.jsonl")
    vk = list(csv.DictReader(open(RES / "validation_key.csv")))
    add("validation.n", str(sum(1 for r in vk if r["sheet"] == "Recommendations")), "validation_key.csv")
    add("validation.n_cause_labels", str(sum(1 for r in vk if r["sheet"] == "Cause labels")), "validation_key.csv")

    # adjudication
    adj = load("adjudication_summary.json")
    add("adj.items", str(adj["adjudicated_items"]), "adjudication_summary.json")
    add("adj.primary", str(adj["chose_primary"]), "adjudication_summary.json")
    add("adj.second", str(adj["chose_second"]), "adjudication_summary.json")
    add("adj.neither", str(adj["chose_neither"]), "adjudication_summary.json")
    add("adj.changed", str(adj["labels_changed_from_primary"]), "adjudication_summary.json")
    rc = [x for x in adj["changes"] if x[0] == "1 Priority"]
    add("adj.rec_changes", str(len(rc)), "adjudication_summary.json")
    add("adj.to_none", str(sum(1 for x in rc if x[2] == "Specific intervention requested" and x[4] == "none")),
        "adjudication_summary.json")
    tiers = load("adjudication_tiers.json")
    add("adj.priority_recs", str(tiers["priority_recommendations"]), "adjudication_tiers.json")
    add("adj.cause_labels", str(tiers["cause_label_items"]), "adjudication_tiers.json")
    add("adj.remaining_items", f'{tiers["remaining_items"]:,}', "adjudication_tiers.json")

    # blinded physician validation and evidence grading
    vs = load("validation_scores.json")
    add("val.n", str(vs["n_recommendations"]), "validation_scores.json")
    add("val.unsure", str(vs["n_unsure"]), "validation_scores.json")
    for src, short in (("claude-opus-5-5", "opus"), ("gpt-6-astra", "gpt"), ("final_adjudicated", "final")):
        for f, v in vs["fields"][src].items():
            add(f"val.{short}.{f}.kappa", f"{v['kappa']:.2f}", "validation_scores.json")
            add(f"val.{short}.{f}.agree", f1(100 * v["agreement"]), "validation_scores.json")
    eg = load("evidence_grades_final.json")
    add("grading.n", str(eg["n_interventions"]), "evidence_grades_final.json")
    add("grading.unanimous", str(eg["n_unanimous"]), "evidence_grades_final.json")
    add("grading.adjudicated", str(eg["n_adjudicated"]), "evidence_grades_final.json")
    add("grading.changed", str(eg["adjudicated_changed_investigator_grade"]), "evidence_grades_final.json")
    vk = list(csv.DictReader(open(RES / "validation_key.csv")))
    n_agree_pool = n_recs - len({r["id_or_string"] for r in csv.DictReader(open(RES / "adjudication_key.csv"))
                               if not r["sheet"].endswith("Cause labels")})
    add("val.agree_pool.n", f"{n_agree_pool:,}", "adjudication_key.csv")
    add("val.agree_pool.pct", f1(100 * n_agree_pool / n_recs), "adjudication_key.csv")

    if (RES / "uncertainty_current.json").exists():
        uc = load("uncertainty_current.json")["psa"]
        for k in ("obs_deaths", "ev_deaths", "comp_deaths", "obs_qaly", "ev_qaly"):
            if k in uc:
                lo_, hi_ = [0.0 if abs(x) < 0.5 else x for x in uc[k]["ci"]]
                add(f"psacur.{k}.ci", f'{lo_:,.0f} to {hi_:,.0f}', "uncertainty_current.json")
                add(f"psacur.{k}.median", f'{uc[k]["median"]:,.0f}', "uncertainty_current.json")
        add("psacur.prob_better", f'{100*uc["prob_evidence_weighted_averts_more"]:.1f}', "uncertainty_current.json")
        for w, v in (uc.get("ceac") or {}).items():
            for kk, vv in v.items():
                add(f"psacur.ceac.{int(w)//1000}k.{kk}", f"{100*vv:.1f}", "uncertainty_current.json")

    # accuracy statistics, test-retest, keyword audit
    vst = load("validation_stats.json")
    pct3 = lambda x: f"{100*x:.1f}"
    for d, v in vst["diagnostic"].items():
        for m_ in ("sensitivity", "specificity", "ppv", "npv"):
            k_, n_, ci_ = v[m_]
            if n_:
                add(f"acc.{d}.{m_}", f"{k_} of {n_}", "validation_stats.json")
                add(f"acc.{d}.{m_}.pct", pct3(k_ / n_), "validation_stats.json")
                add(f"acc.{d}.{m_}.ci", f"{pct3(ci_[0])} to {pct3(ci_[1])}", "validation_stats.json")
    for f, v in vst["kappa_ci"].items():
        if isinstance(v, dict):
            add(f"kci.{f}", f"{v['ci'][0]:.2f} to {v['ci'][1]:.2f}", "validation_stats.json")
        else:
            add(f"kci.{f}", f"{v[0]:.2f} to {v[1]:.2f}", "validation_stats.json")
    for d, v in vst["bounds"].items():
        add(f"bound.{d}.lower_pct", f"{v['lower_pct']:.1f}", "validation_stats.json")
        add(f"bound.{d}.upper_pct", f"{v['upper_pct']:.1f}", "validation_stats.json")
        add(f"bound.{d}.unadjudicated", str(v["unadjudicated"]), "validation_stats.json")
    for k_, v in vst["keyword_audit"].items():
        add(f"kw.{k_}.mentions", str(v["mentions"]), "validation_stats.json")
    rt = load("retest_scores.json")
    add("retest.n", str(rt["n"]), "retest_scores.json")
    for f, v in rt["fields"].items():
        add(f"retest.{f}.kappa", f"{v['kappa']:.2f}", "retest_scores.json")

    # coverage recommendations
    lab = [json.loads(l) for l in open(RES / "llm_labels_passA.jsonl")]
    cov = [r for r in lab if r.get("coverage_period") != "Not a coverage recommendation"]
    add("coverage.n", f"{len(cov):,}", "llm_labels_passA.jsonl")
    for per in ("Unspecified", "Postpartum", "Antepartum", "Both"):
        k = per.lower()
        add(f"coverage.{k}.of_cov_pct", f1(100 * sum(r["coverage_period"] == per for r in cov) / len(cov)),
            "llm_labels_passA.jsonl")

    # moderators
    mo = load("alignment_moderators.json")
    for g, v in mo["silence_by_group_raw"].items():
        key = {"Obstetric and cardiovascular": "obcv", "Behavioral health": "bh",
               "Injury and violence": "injury", "Other medical conditions": "other"}[g]
        add(f"silence.{key}.pct", f1(v["pct"]), "alignment_moderators.json")
        add(f"silence.{key}.n", f"{v['silent']} of {v['pairs']}", "alignment_moderators.json")
    add("mod.n_pairs", str(mo["n_pairs"]), "alignment_moderators.json")
    add("mod.n_states", str(mo["n_states"]), "alignment_moderators.json")
    add("mod.n_states_roster", str(mo["n_states_with_roster"]), "alignment_moderators.json")
    add("mod.n_states_medicaid", str(mo["n_states_medicaid_member"]), "alignment_moderators.json")
    add("mod.n_states_prmr", str(mo["n_states_with_prmr"]), "alignment_moderators.json")
    tname = {"group[T.Obstetric and cardiovascular]": "obcv", "group[T.Injury and violence]": "injury",
             "group[T.Other medical conditions]": "other", "log_pages": "pages", "prmr10": "prmr",
             "medicaid_member": "medicaid", "log_recs": "recs", "n_causes": "ncauses"}
    for mname, m in mo["models"].items():
        short = {"silence_by_group": "silence0", "silence_pages": "silence",
                 "concordance_pages": "conc", "silence_pages_prmr": "silence_prmr",
                 "concordance_pages_prmr": "conc_prmr",
                 "silence_medicaid_member_subset": "silence_medicaid",
                 "silence_recs_causes": "silence_rc",
                 "silence_sensitivity_crosscutting": "silence_sens",
                 "silence_pages_recs": "silence_pr",
                 "silence_intermediate_rule": "silence_mid"}.get(mname, mname)
        for t, v in m["terms"].items():
            if t in tname:
                d = 2
                add(f"mod.{short}.{tname[t]}.or", f"{v['estimate']:.{d}f}", "alignment_moderators.json")
                add(f"mod.{short}.{tname[t]}.ci", f"{v['ci'][0]:.{d}f} to {v['ci'][1]:.{d}f}",
                    "alignment_moderators.json")
        add(f"mod.{short}.n_pairs", str(m["n_pairs"]), "alignment_moderators.json")
        add(f"mod.{short}.n_states", str(m["n_states"]), "alignment_moderators.json")

    import math
    for mname, short in (("silence_recs_causes", "silence_rc"), ("silence_pages_recs", "silence_pr")):
        t = mo["models"].get(mname, {}).get("terms", {}).get("log_recs")
        if t:
            dbl = lambda v: math.exp(math.log(v) * math.log(2))
            add(f"mod.{short}.recs.or_doubling", f"{dbl(t['estimate']):.2f}", "alignment_moderators.json")
            add(f"mod.{short}.recs.ci_doubling", f"{dbl(t['ci'][0]):.2f} to {dbl(t['ci'][1]):.2f}",
                "alignment_moderators.json")
    t = mo["models"].get("silence_pages_recs", {}).get("terms", {}).get("log_pages")
    if t:
        dbl = lambda v: math.exp(math.log(v) * math.log(2))
        add("mod.silence_pr.pages.or_doubling", f"{dbl(t['estimate']):.2f}", "alignment_moderators.json")
        add("mod.silence_pr.pages.ci_doubling", f"{dbl(t['ci'][0]):.2f} to {dbl(t['ci'][1]):.2f}",
            "alignment_moderators.json")
    add("mod.crude_or", f"{mo['crude_or_obcv_vs_bh']:.2f}", "alignment_moderators.json")
    for g, v in mo["silence_sens_by_group_raw"].items():
        key = {"Obstetric and cardiovascular": "obcv", "Behavioral health": "bh",
               "Injury and violence": "injury", "Other medical conditions": "other"}[g]
        add(f"silence_sens.{key}.pct", f1(v["pct"]), "alignment_moderators.json")
        add(f"silence_sens.{key}.n", f"{v['silent']} of {v['pairs']}", "alignment_moderators.json")
    for g, v in mo.get("silence_mid_by_group_raw", {}).items():
        key = {"Obstetric and cardiovascular": "obcv", "Behavioral health": "bh",
               "Injury and violence": "injury", "Other medical conditions": "other"}[g]
        add(f"silence_mid.{key}.pct", f1(v["pct"]), "alignment_moderators.json")
    add("mod.few_specific_pairs", str(mo["pairs_with_fewer_than_5_specific_recs"]), "alignment_moderators.json")
    add("mod.few_specific_states", str(mo["states_with_fewer_than_5_specific_recs"]), "alignment_moderators.json")
    strict = {"Legislature or statute", "Medicaid or state health agency", "Managed care organization or payer"}
    nb = sum(1 for r in lab if r.get("actor_authority") in strict)
    add("actor.binding_strict.pct", f1(100 * nb / len(lab)), "llm_labels_passA.jsonl")
    add("actor.binding_strict.n", f"{nb:,}", "llm_labels_passA.jsonl")

    # committee composition
    cc = load("committee_composition.json")["reports"]
    add("roster.reports", str(sum(1 for r in cc if r.get("roster_found"))), "committee_composition.json")
    add("roster.medicaid", str(sum(1 for r in cc if r.get("roster_found") and r.get("medicaid_agency_member"))),
        "committee_composition.json")

    # propagation
    pr = load("propagation.json")
    add("prop.n_pairs", str(pr["n_pairs_documented"]), "propagation.json")
    add("prop.n_states", str(pr["n_states_with_burden"]), "propagation.json")
    cd = pr["contract_documents"]
    add("contract.n_docs", f'{sum(v["all"] for v in cd.values()):,}', "propagation.json")
    add("contract.n_states", str(len(cd)), "propagation.json")
    add("contract.n_dated", f'{sum(v["dated"] for v in cd.values()):,}', "propagation.json")
    add("contract.n_postdating", f'{sum(v["postdating_report"] for v in cd.values()):,}', "propagation.json")
    short = {"contract_postdating": "contract_post", "contract_all_documents": "contract_all",
             "withhold_postdating": "withhold_post", "withhold_all_documents": "withhold_all",
             "pqc": "pqc", "legislation": "legis", "consortium": "consortium",
             "legislation_postdating": "legis_post"}
    for k, v in pr["channels"].items():
        o, a, sk = v["overall"], v.get("adjusted_for_cause_group") or {}, short[k]
        add(f"prop.{sk}.states", str(v["n_states_covered"]), "propagation.json")
        add(f"prop.{sk}.pairs", str(o["n_pairs"]), "propagation.json")
        if o["present_if_recommended"] is not None:
            add(f"prop.{sk}.rec", f1(100 * o["present_if_recommended"]), "propagation.json")
            add(f"prop.{sk}.notrec", f1(100 * o["present_if_not_recommended"]), "propagation.json")
            add(f"prop.{sk}.diff", f1(100 * o["difference"]), "propagation.json")
            add(f"prop.{sk}.diff_ci", ci(o["ci"][0], o["ci"][1], 1, 100), "propagation.json")
        fe = v.get("fixed_effects") or {}
        if "difference" in fe:
            add(f"prop.{sk}.mdd", f"{100 * fe['mdd']:.0f}", "propagation.json")
            add(f"prop.{sk}.ident_states", str(fe["identifying_states"]), "propagation.json")
            add(f"prop.{sk}.fe", f1(100 * fe["difference"]), "propagation.json")
            add(f"prop.{sk}.fe_ci", ci(fe["ci"][0], fe["ci"][1], 1, 100), "propagation.json")
        if a.get("mh_risk_difference") is not None:
            add(f"prop.{sk}.mh", f1(100 * a["mh_risk_difference"]), "propagation.json")
            add(f"prop.{sk}.mh_ci", ci(a["ci"][0], a["ci"][1], 1, 100), "propagation.json")
        for g, gv in v["by_cause_group"].items():
            gk = {"Obstetric and cardiovascular": "obcv", "Behavioral health": "bh",
                  "Injury and violence": "injury", "Other medical conditions": "other"}[g]
            for fld, lab in (("present_if_recommended", "rec"), ("present_if_not_recommended", "notrec")):
                if gv[fld] is not None:
                    add(f"prop.{sk}.{gk}.{lab}", f1(100 * gv[fld]), "propagation.json")
            add(f"prop.{sk}.{gk}.n_rec", str(gv["n_rec"]), "propagation.json")
            add(f"prop.{sk}.{gk}.n_not", str(gv["n_not"]), "propagation.json")
            add(f"prop.{sk}.{gk}.n_total", str(gv["n_rec"] + gv["n_not"]), "propagation.json")
    lc = json.load(open(RES / "channel_legislation_consortium.json"))
    yrs = [b.get("year") for v in lc["states"].values()
           for b in (v.get("legislation") or {}).get("bills", []) if isinstance(b.get("year"), int)]
    add("legis.year_first", "2018", "channel_legislation_consortium.json")
    add("legis.year_last", "2026", "channel_legislation_consortium.json")
    add("legis.n_bills", f"{len(yrs):,}", "channel_legislation_consortium.json")
    pq = json.load(open(RES / "channel_pqc_aim.json"))["states"]
    ini = [i for v in pq.values() for i in ((v.get("pqc") or {}).get("initiatives") or [])]
    add("pqc.n_initiatives", f"{len(ini):,}", "channel_pqc_aim.json")

    # accuracy of the contract text rules (investigator review, script 38)
    cv = load("contract_validation_scores.json")
    for sheet, tag in (("Cause mentions", "cause"), ("Payment provisions", "pay")):
        r = cv["rules"][sheet]
        for stat in ("ppv", "npv", "ppv_narrow_window"):
            x = r[stat]; short = {"ppv": "ppv", "npv": "npv", "ppv_narrow_window": "ppv_narrow"}[stat]
            add(f"cval.{tag}.{short}.k", str(x["k"]), "contract_validation_scores.json")
            add(f"cval.{tag}.{short}.n", str(x["n"]), "contract_validation_scores.json")
            add(f"cval.{tag}.{short}.pct", f1(100 * x["value"]), "contract_validation_scores.json")
            add(f"cval.{tag}.{short}.ci", ci(x["ci"][0], x["ci"][1], 1, 100), "contract_validation_scores.json")
        add(f"cval.{tag}.narrow_chars", f'{r["narrow_window_chars"]:,}', "contract_validation_scores.json")
        add(f"cval.{tag}.n_states", str(r["n_states"]), "contract_validation_scores.json")
    add("cval.n_passages", str(sum(v["n_flagged"] + v["n_not_flagged"] for v in cv["rules"].values())),
        "contract_validation_scores.json")

    # window-width sensitivity of the contract channels (script 19 rerun with other windows)
    fes = {}
    for tag in ("_ctx150", "_ctx600", "_wh500", "_wh2000"):
        w = load(f"propagation{tag}.json")
        for ch, sk in (("contract_all_documents", "contract_all"), ("withhold_all_documents", "withhold_all")):
            fe = w["channels"][ch].get("fixed_effects") or {}
            if "difference" in fe:
                t = tag.strip("_")
                add(f"win.{t}.{sk}.fe", f1(100 * fe["difference"]), f"propagation{tag}.json")
                add(f"win.{t}.{sk}.fe_ci", ci(fe["ci"][0], fe["ci"][1], 1, 100), f"propagation{tag}.json")
                fes.setdefault(sk, []).append(100 * fe["difference"])
    for sk, v in fes.items():
        add(f"win.{sk}.fe_min", f1(min(v)), "propagation_*.json")
        add(f"win.{sk}.fe_max", f1(max(v)), "propagation_*.json")
    for dname, dk in (("Hemorrhage", "hem"), ("Hypertensive disorders", "htn"),
                      ("Substance use disorder", "sud"), ("Mental health conditions", "mh"),
                      ("Cardiomyopathy", "cm"), ("Cardiovascular conditions", "cv"),
                      ("Infection", "inf"), ("Embolism", "emb")):
        add(f"pqc.states_with.{dk}", str(sum(1 for v in pq.values() if any(
            dname in i.get("domains", []) for i in ((v.get("pqc") or {}).get("initiatives") or [])))),
            "channel_pqc_aim.json")

    # model: base case at full cohort
    pcj = load("portfolio_cea.json")
    b = pcj["baseline"]
    add("pf.baseline.births", f'{b["births"]:,.0f}', "portfolio_cea.json")
    add("pf.baseline.births_m", f'{b["births"]/1e6:.2f}', "portfolio_cea.json")
    add("pf.baseline.deaths", f'{b["deaths"]:,.0f}', "portfolio_cea.json")
    add("pf.baseline.smm", f'{b["smm_events"]:,.0f}', "portfolio_cea.json")
    add("model.smm_target", f'{P["smm_rate"]["value"]:.1f}', "model_params.json")
    add("model.prmr_target", f'{P["prmr_for_model"]["value"]:.1f}', "model_params.json")
    add("den.medicaid_share", f'{den["national"]["medicaid_share_of_births_2024_pct"]:.1f}',
        "denominators.json")
    pnames = {"observed": "observed", "evidence_weighted": "evidence",
              "within_lever_optimal": "withinlever", "burden_aligned": "burden",
              "full_effective_menu": "fullmenu", "observed_lever_based": "obslever",
              "evidence_weighted_at_lever_based_budget": "evlever",
              "observed_incremental_current_practice": "obscur",
              "evidence_weighted_incremental_current_practice": "evcur",
              "requested_frequency": "reqfreq",
              "requested_frequency_effective_norm": "reqfreqeff",
              "requested_excluding_home_visiting": "reqnohv",
              "burden_aligned_unconstrained": "burdenunc",
              "requested_with_bundle_components": "reqcomp",
              "requested_with_bundle_components_incremental_current_practice": "reqcompcur"}
    for k, r in pcj["portfolios"].items():
        if k not in pnames:
            continue
        n = pnames[k]
        add(f"pf.{n}.deaths", f'{r["deaths_averted"]:,.0f}', "portfolio_cea.json")
        add(f"pf.{n}.deaths_pct", f1(100 * r["deaths_averted"] / b["deaths"]), "portfolio_cea.json")
        add(f"pf.{n}.smm", f'{r["smm_averted"]:,.0f}', "portfolio_cea.json")
        add(f"pf.{n}.qaly", f'{r["qalys_gained"]:,.0f}', "portfolio_cea.json")
        add(f"pf.{n}.cost_m", f'{r["program_cost"]/1e6:,.0f}', "portfolio_cea.json")
        add(f"pf.{n}.net_m", f'{r["net_cost"]/1e6:,.0f}', "portfolio_cea.json")
        cpq = r["cost_per_qaly"]
        add(f"pf.{n}.cpq", f"{cpq:,.0f}" if cpq < 1e12 else "not estimable", "portfolio_cea.json")
        br = r.get("by_race")
        if br:
            add(f"pf.{n}.black_per100k", f"{br['black_per_100k']:.1f}", "portfolio_cea.json")
            add(f"pf.{n}.other_per100k", f"{br['other_per_100k']:.1f}", "portfolio_cea.json")
            if not any(r_[0] == "pf.baseline.black_prmr" for r_ in rows):
                add("pf.baseline.black_prmr", f"{br['baseline_black_prmr']:.1f}", "portfolio_cea.json")
                add("pf.baseline.other_prmr", f"{br['baseline_other_prmr']:.1f}", "portfolio_cea.json")
        cpd = r["cost_per_death_averted"]
        add(f"pf.{n}.cpd_m", f"{cpd/1e6:,.1f}" if cpd < 1e15 else "not estimable", "portfolio_cea.json")
    obs, ev = pcj["portfolios"]["observed"], pcj["portfolios"]["evidence_weighted"]
    oc_, ec_ = pcj["portfolios"].get("observed_incremental_current_practice"), pcj["portfolios"].get("evidence_weighted_incremental_current_practice")
    if oc_ and ec_:
        add("pf.evcur_cost_share", f1(100 * ec_["program_cost"] / oc_["program_cost"]), "portfolio_cea.json")
    add("pf.ratio", f'{ev["deaths_averted"]/obs["deaths_averted"]:.1f}', "portfolio_cea.json")
    add("pf.ev_cost_share", f1(100 * ev["program_cost"] / obs["program_cost"]), "portfolio_cea.json")
    eq = pcj.get("equivalent_budget") or {}
    if eq.get("budget") is not None:
        add("pf.equiv_budget_k", f'{eq["budget"]/1e3:,.0f}', "portfolio_cea.json")
        add("pf.equiv_share", f'{100*eq["share_of_observed_budget"]:.2f}', "portfolio_cea.json")
    lb = pcj["lever_budget"]; tot = sum(lb.values())
    for lv, v in lb.items():
        key = lv.split(" or ")[0].split(",")[0].lower().replace(" ", "_")
        add(f"spend.{key}.pct", f1(100 * v / tot), "portfolio_cea.json")
    add("pf.pp_reach", f1(100 * pcj["postpartum_reachable_without_extension"]), "portfolio_cea.json")

    body = ["# Canonical numbers", "",
            "Generated by `code/17_canonical.py` from the result files named in the last column.",
            "Every number in the manuscript, tables, figure legends, and appendix must match an",
            "entry here. Regenerate after any analysis rerun, then run `code/18_audit.py`.", "",
            "| Key | Value | Source |", "|---|---|---|"]
    body += [f"| `{k}` | {v} | `{s}` |" for k, v, s in rows]
    (ROOT / "CANONICAL_NUMBERS.md").write_text("\n".join(body) + "\n")
    json.dump({k: v for k, v, _ in rows}, open(ROOT / "canonical_numbers.json", "w"), indent=1)
    print(f"wrote CANONICAL_NUMBERS.md with {len(rows)} entries")


if __name__ == "__main__":
    main()
