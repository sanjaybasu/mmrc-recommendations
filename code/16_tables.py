"""
16_tables.py — Build every table from the canonical result files.

No number is typed here. Main-text tables go to manuscript/tables.md and
appendix tables to manuscript/etables.md, both in the journal's plain format.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES, MAN = ROOT / "results", ROOT / "manuscript"
MAN.mkdir(exist_ok=True)
MMRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"

load = lambda n: json.load(open(RES / n))
fmt = lambda v, d=1: f"{v:,.{d}f}"


def md(rows, header):
    w = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    w += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(w)


def table1() -> str:
    cls = load("classification.json")
    pil = load("pilot_numbers.json")
    docs = list(csv.DictReader(open(MMRC / "documents.csv")))
    rows = []
    c = pil["corpus"]
    rows += [["Published review reports", c["documents"]],
             ["Jurisdictions represented", c["jurisdictions"]],
             ["Report years", f'{c["report_years"][0]} to {c["report_years"][1]}'],
             ["Quantitative findings extracted", f'{c["n_findings"]:,}'],
             ["Prevention recommendations extracted", f'{c["n_recommendations"]:,}'],
             ["States issuing at least one recommendation", c["states_with_recommendations"]]]
    out = ["**Corpus**", md(rows, ["Characteristic", "Value"]), ""]

    for field, title in [("policy_lever", "Policy lever"),
                         ("cause_domain", "Cause domain addressed"),
                         ("actor_authority", "Most binding actor named"),
                         ("coverage_period", "Coverage period named")]:
        r = [[x["label"], f'{x["n"]:,}', f'{x["pct"]:.1f}'] for x in cls["pass_A"][field]]
        out += [f"**{title}**", md(r, [title, "No.", "%"]), ""]
    out += [f'Recommendations naming a body able to bind anyone: '
            f'{cls["pass_A"]["pct_naming_binding_actor"]:.1f}%', ""]
    return "\n".join(out)


def table2() -> str:
    t = load("accountability_chain_tight.json")
    rows = []
    for f in t["funnel"]:
        d = f["documented"]
        if d < 3:
            continue
        rows.append([f["domain"], d, f["recommended"], f["in_contract"],
                     f["withhold_same_provision"],
                     f'{100*f["recommended"]/d:.0f}',
                     f'{100*f["withhold_same_provision"]/d:.0f}'])
    return md(rows, ["Cause domain", "States documenting the cause",
                     "and issuing a recommendation", "and naming it in contract language",
                     "and tying it to a payment withhold", "% reaching a recommendation",
                     "% reaching a withhold"])


def table3() -> str:
    p = load("portfolio_cea.json")
    u = load("uncertainty.json")
    names = [("observed", "Observed"), ("within_lever_optimal", "Within-lever optimal"),
             ("burden_aligned", "Burden-aligned"), ("evidence_weighted", "Evidence-weighted")]
    rows = []
    for k, lab in names:
        r = p["portfolios"][k]
        rows.append([lab, fmt(r["deaths_averted"], 0), fmt(r["smm_averted"], 0),
                     fmt(r["qalys_gained"], 0), f'{r["net_cost"]/1e6:,.0f}',
                     f'{r["cost_per_qaly"]:,.0f}' if r["cost_per_qaly"] < 1e7 else "dominated",
                     f'{r["cost_per_death_averted"]/1e6:,.2f}'])
    head = ["Portfolio", "Deaths averted", "Severe morbidity events averted",
            "QALYs gained", "Net cost, $ millions", "Cost per QALY, $",
            "Cost per death averted, $ millions"]
    psa = u["psa"]
    note = (f'\nProbabilistic analysis over {psa["n_draws"]:,} draws: the evidence-weighted '
            f'portfolio averted {psa["gap"]["median"]:,.0f} more deaths than the observed '
            f'portfolio (95% uncertainty interval {psa["gap"]["ci"][0]:,.0f} to '
            f'{psa["gap"]["ci"][1]:,.0f}); it averted more in '
            f'{100*psa["prob_evidence_weighted_averts_more"]:.1f}% of draws.')
    return md(rows, head) + "\n" + note


def etables() -> str:
    out = []
    cls = load("classification.json")
    r = [[k.replace("_", " "), f'{v["kappa"]:.3f}', f'{v["agreement"]:.3f}', f'{v["n"]:,}']
         for k, v in cls["reliability"].items()]
    out += ["### eTable 1. Agreement between two independent classification passes", "",
            md(r, ["Field", "Cohen kappa", "Simple agreement", "Recommendations"]), ""]

    r = [[x["label"], f'{x["n"]:,}', f'{x["pct"]:.1f}'] for x in cls["rule_based_levers"]]
    out += ["### eTable 2. Policy lever under the rule-based pilot classification", "",
            md(r, ["Policy lever", "No.", "%"]),
            f'\nClassified by the rule-based method: {cls["rule_based_n_classified"]:,} of '
            f'{cls["n_recommendations"]:,} recommendations. The language-model classification '
            "assigned a lever to every recommendation.", ""]

    co = load("concordance.json")
    r = [[x["domain"], x["n_states"], f'{x["median_ratio"]:.2f}',
          f'{x["ci"][0]:.2f} to {x["ci"][1]:.2f}', f'{x["iqr"][0]:.2f} to {x["iqr"][1]:.2f}']
         for x in co["concordance"]]
    out += ["### eTable 3. Concordance between recommendation content and documented burden", "",
            md(r, ["Cause domain", "States", "Median concordance ratio",
                   "95% CI", "Interquartile range"]),
            f'\nBootstrap over {co["bootstrap_draws"]:,} resamples of states within cause domain.',
            ""]

    if (RES / "model_params.json").exists():
        P = load("model_params.json")["params"]
        r = []
        for k, v in P.items():
            if not isinstance(v, dict) or "value" not in v:
                continue
            val = v["value"]
            if isinstance(val, (dict, list)):
                val = json.dumps(val)[:90]
            rng = (f'{v.get("low")} to {v.get("high")}'
                   if v.get("low") is not None and v.get("high") is not None else "—")
            r.append([k.replace("_", " "), val, rng, str(v.get("unit", ""))[:60],
                      str(v.get("source", ""))[:190]])
        out += ["### eTable 4. Model parameters, ranges, and sources", "",
                md(r, ["Parameter", "Value", "Range", "Unit", "Source"]), ""]

    if (RES / "portfolio_cea.json").exists():
        import importlib.util
        s = importlib.util.spec_from_file_location("iv", Path(__file__).parent / "12_interventions.py")
        iv = importlib.util.module_from_spec(s); sys.modules['iv'] = iv
        s.loader.exec_module(iv)
        P = load("model_params.json")["params"]
        r = [[i.label, i.lever, ", ".join(i.causes) or "none", i.stage,
              f'{i.rr:.2f}', f'{i.rr_lo:.2f} to {i.rr_hi:.2f}', i.grade,
              f'{i.unit_cost:,.2f}', f'{i.max_reach:.2f}', i.source[:230]]
             for i in iv.build(P)]
        out += ["### eTable 5. Interventions, effect estimates, costs, and evidence sources", "",
                md(r, ["Intervention", "Policy lever", "Cause pathway", "Stage of effect",
                       "Relative risk", "Interval", "Evidence grade",
                       "Unit cost, 2025 $", "Maximum reach", "Source"]), ""]

        p = load("portfolio_cea.json")
        r = []
        for name, pf in p["portfolios"].items():
            for k, v in sorted(pf["coverage"].items(), key=lambda kv: -kv[1]):
                r.append([name.replace("_", " "), k.replace("_", " "), f"{v:.3f}"])
        out += ["### eTable 6. Coverage achieved by each intervention under each portfolio", "",
                md(r, ["Portfolio", "Intervention", "Coverage of eligible population"]), ""]

    if (RES / "uncertainty.json").exists():
        u = load("uncertainty.json")
        r = [[x["parameter"].replace("_", " "), f'{x["low_value"]:,.4g}',
              f'{x["high_value"]:,.4g}', f'{x["gap_at_low"]:,.0f}',
              f'{x["gap_at_high"]:,.0f}', f'{x["swing"]:,.0f}'] for x in u["one_way"]]
        out += ["### eTable 7. One-way sensitivity analysis", "",
                md(r, ["Parameter", "Low", "High", "Difference at low", "Difference at high",
                       "Swing"]),
                "\nThe difference is additional pregnancy-related deaths averted by the "
                "evidence-weighted portfolio relative to the observed portfolio.", ""]

        r = []
        for k, v in u["ablations"].items():
            if "observed_deaths_averted" in v:
                r.append([k.replace("_", " "), f'{v["observed_deaths_averted"]:,.0f}',
                          f'{v["evidence_weighted_deaths_averted"]:,.0f}',
                          f'{v["gap"]:,.0f}', f'{v["ratio"]:.1f}'])
        out += ["### eTable 8. Structural ablations", "",
                md(r, ["Ablation", "Observed portfolio, deaths averted",
                       "Evidence-weighted, deaths averted", "Difference", "Ratio"]), ""]

    t = load("accountability_chain_tight.json")
    rows = list(csv.DictReader(open(RES / "accountability_chain_long.csv")))
    r = [[x["state"], x["domain"], x["documented"], x["recommended"], x["in_contract"],
          x.get("withhold_same_provision", ""), x["contract_mentions"]]
         for x in rows if x["documented"] == "True"]
    out += ["### eTable 9. Accountability propagation by state and cause domain", "",
            md(r, ["State", "Cause domain", "Documented", "Recommendation issued",
                   "Named in contract language", "Tied to a payment withhold",
                   "Contract mentions"]),
            f'\nStates with both a review report and procurement documents: {t["n_states"]}. '
            f'A cause counts as named in contract language only when the cause term appears '
            f'within 300 characters of a maternal or perinatal term, and as tied to a withhold '
            f'only when payment-withhold language also appears within {t["window_chars"]:,} '
            f'characters of that mention.', ""]
    return "\n".join(out)


def main():
    t = [f"## Table 1. Corpus and classification of 3,351 prevention recommendations\n",
         table1(),
         "\n## Table 2. Propagation from documented cause to recommendation to "
         "Medicaid contract accountability\n", table2()]
    if (RES / "portfolio_cea.json").exists() and (RES / "uncertainty.json").exists():
        t += ["\n## Table 3. Modeled outcomes of four allocations of the same budget\n", table3()]
    (MAN / "tables.md").write_text("\n".join(t) + "\n")
    (MAN / "etables.md").write_text(etables() + "\n")
    print(f"wrote {MAN/'tables.md'} and {MAN/'etables.md'}")


if __name__ == "__main__":
    main()
