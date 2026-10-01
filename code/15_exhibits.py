"""
15_exhibits.py — Build every table and figure from the canonical result files.

No number is typed here. Each exhibit reads the JSON its analysis wrote, so a
rerun of any analysis propagates to the manuscript without hand editing.

Output: figures/*.pdf, figures/*.png, manuscript/tables.md, manuscript/etables.md
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, MAN = ROOT / "results", ROOT / "figures", ROOT / "manuscript"
FIG.mkdir(exist_ok=True); MAN.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
INK = "#1a1a1a"
GREY = "#8c8c8c"
ACCENT = "#7d1128"
BLUE = "#1b3a5c"


def load(name):
    return json.load(open(RES / name))


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=400)
    plt.close(fig)
    print(f"  {stem}")


SHORT = {"Injury, homicide, and violence": "Injury, homicide, and violence",
         "Mental health conditions": "Mental health conditions",
         "Substance use disorder": "Substance use disorder"}
short = lambda s: SHORT.get(s, s)

OBSTETRIC_CV = {"Hemorrhage", "Hypertensive disorders", "Cardiomyopathy",
                "Cardiovascular conditions", "Embolism", "Cerebrovascular accident", "Infection"}
FEW_STATES = 10   # causes documented by fewer states than this are marked


# ---------------------------------------------------------------- Figure 1
def figure1():
    """Silence and concordance by cause, grouped by cause type, sorted within
    group, labeled on both panels so either panel reads alone."""
    co = load("concordance.json")
    sil = {r["domain"]: r for r in co["silence"]}
    conc = {r["domain"]: r for r in co["concordance"]}
    doms = [d for d in conc if d in sil]
    grp = lambda d: 0 if d in OBSTETRIC_CV else 1
    doms.sort(key=lambda d: (grp(d), -sil[d]["pct"]))
    doms = doms[::-1]                          # top of the plot = first in list
    y = np.arange(len(doms))
    lab = [short(d) + ("†" if sil[d]["states_documenting"] < FEW_STATES else "")
           for d in doms]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"wspace": 0.95})
    pct = [sil[d]["pct"] for d in doms]
    col = [ACCENT if d in OBSTETRIC_CV else GREY for d in doms]
    ax[0].barh(y, pct, color=col, height=0.62)
    for i, d in enumerate(doms):
        ax[0].text(pct[i] + 2, i, f"{sil[d]['states_zero_recs']}/{sil[d]['states_documenting']}",
                   va="center", fontsize=6.8, color=INK)
    ax[0].set_yticks(y); ax[0].set_yticklabels(lab)
    ax[0].set_xlim(0, 118); ax[0].set_xticks([0, 25, 50, 75, 100])
    ax[0].set_xlabel("States listing the cause with no\nrecommendation addressing it, %")
    ax[0].set_title("A  Causes without a recommendation", loc="left", fontweight="bold")
    from matplotlib.lines import Line2D
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=ACCENT), plt.Rectangle((0, 0), 1, 1, color=GREY),
                Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK, mew=1.0),
                Line2D([], [], ls="", marker="")],
               ["Obstetric and cardiovascular causes", "Other causes",
                "Median 0 in every bootstrap sample (no interval)",
                f"\u2020 Listed by fewer than {FEW_STATES} states"],
               frameon=False, loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.16), handlelength=1.6, columnspacing=2.0)

    ax[1].axvline(1.0, color=GREY, lw=0.8, ls="--", zorder=0)
    for i, d in enumerate(doms):
        r = conc[d]
        m, lo, hi = r["median_ratio"], r["ci"][0], r["ci"][1]
        c = ACCENT if d in OBSTETRIC_CV else GREY
        if hi > lo:
            ax[1].plot([lo, hi], [i, i], color=c, lw=1.1)
            ax[1].plot(m, i, "o", ms=4, color=c)
        else:
            # every bootstrap median was the same value, which here is zero:
            # most states documenting the cause issued nothing on it
            ax[1].plot(m, i, "o", ms=4, mfc="white", mec=c, mew=1.1)
    ax[1].set_yticks(y); ax[1].set_yticklabels(lab)
    ax[1].set_xlabel("Alignment ratio, median across states\n"
                     "(share of recommendations / share of documented causes)")
    ax[1].set_title("B  Alignment with documented causes", loc="left", fontweight="bold")
    ax[1].set_xlim(left=-0.08)
    save(fig, "Figure1_silence_and_concordance")


# ---------------------------------------------------------------- Figure 2
GRADE_ORDER = ["Demonstrated benefit", "Moderate", "Weak", "Null or absent",
               "Not effect-bearing", "No intervention in the table"]
GRADE_COLOR = {"Demonstrated benefit": BLUE, "Moderate": "#5b8db8", "Weak": "#b7c9da",
               "Null or absent": ACCENT, "Not effect-bearing": "#f0e2c8",
               "No intervention in the table": "#e6e6e6"}


def grade_of(g: str) -> str:
    g = (g or "").lower()
    if "absent for mortality" in g:
        return "Null or absent"      # evidence for coverage outcomes only, none on the pathway
    if g.startswith("demonstrated"):
        return "Demonstrated benefit"
    if g.startswith("moderate"):
        return "Moderate"
    if g.startswith("weak"):
        return "Weak"
    if g.startswith(("null", "absent")):
        return "Null or absent"
    return "Not effect-bearing"


def recommendation_grades() -> dict:
    """Evidence grade of each recommendation through the intervention it asks for."""
    import importlib.util, sys
    s = importlib.util.spec_from_file_location("iv", ROOT / "code/12_interventions.py")
    m = importlib.util.module_from_spec(s); sys.modules["iv"] = m; s.loader.exec_module(m)
    P = load("model_params.json")["params"]
    g = {i.key: grade_of(i.grade) for i in m.build(P)}
    imap = {json.loads(l)["id"]: json.loads(l)["intervention"]
            for l in open(RES / "intervention_map_passA.jsonl")}
    return {i: (g.get(k, "No intervention in the table") if k != "none"
                else "No intervention in the table") for i, k in imap.items()}


def figure2():
    """Each lever split by the evidence grade of the specific intervention the
    recommendation asks for."""
    grades = recommendation_grades()
    lev = {json.loads(l)["id"]: json.loads(l)["policy_lever"]
           for l in open(RES / "llm_labels_passA.jsonl")}
    n = len(lev)
    tab = {}
    for i, L in lev.items():
        tab.setdefault(L, {k: 0 for k in GRADE_ORDER})[grades.get(i, "No intervention in the table")] += 1
    levers = sorted(tab, key=lambda L: sum(tab[L].values()))
    y = np.arange(len(levers))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    left = np.zeros(len(levers))
    for g in GRADE_ORDER:
        w = np.array([100 * tab[L][g] / n for L in levers])
        ax.barh(y, w, left=left, color=GRADE_COLOR[g], height=0.64, label=g,
                edgecolor="white", linewidth=0.4)
        left += w
    for i, L in enumerate(levers):
        ax.text(left[i] + 0.4, i, f"{left[i]:.1f}%", va="center", fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(levers)
    ax.set_xlabel(f"Share of {n:,} recommendations, %")
    ax.set_xlim(0, left.max() * 1.15)
    ax.legend(title="Evidence for the intervention named", loc="lower right",
              frameon=False, title_fontsize=7.2, fontsize=6.8)
    save(fig, "Figure2_levers_by_evidence")


# ---------------------------------------------------------------- Figure 3
def figure4():
    u = load("uncertainty.json")
    pf = load("portfolio_cea.json")["portfolios"]
    names = [("observed", "Named\ninterventions"), ("burden_aligned", "Cause-\nproportional"),
             ("evidence_weighted", "Evidence of\nbenefit")]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"wspace": 0.32})

    x = np.arange(len(names))
    vals = [pf[k]["deaths_averted"] for k, _ in names]
    ax[0].bar(x, vals, color=[ACCENT, "#5b8db8", BLUE], width=0.6)
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.02, f"{v:,.0f}", ha="center", fontsize=7.2, color=INK)
    ax[0].set_xticks(x); ax[0].set_xticklabels([n for _, n in names], rotation=18, ha="right")
    ax[0].set_ylabel("Pregnancy-related deaths averted per year")
    ax[0].set_ylim(0, max(vals) * 1.16)
    ax[0].set_title("A  Deaths averted", loc="left", fontweight="bold")

    d = u["psa_draws"]
    ax[1].scatter([r["obs_qaly"] for r in d], [r["obs_cpq"] for r in d],
                  s=2.4, alpha=0.20, color=ACCENT, label="Named interventions", rasterized=True)
    ax[1].scatter([r["ev_qaly"] for r in d], [r["ev_cpq"] for r in d],
                  s=2.4, alpha=0.20, color=BLUE, label="Evidence of benefit", rasterized=True)
    ax[1].set_xlabel("Quality-adjusted life years gained")
    ax[1].set_ylabel("Incremental cost per QALY gained (2025 US$)")
    ax[1].set_yscale("symlog", linthresh=1e4)
    lg = ax[1].legend(frameon=False, markerscale=4, loc="upper right")
    for h in lg.legend_handles:
        h.set_alpha(1.0)
    ax[1].set_title("B  Probabilistic analysis", loc="left", fontweight="bold")
    save(fig, "eFigure_portfolio_outcomes")


# ---------------------------------------------------------------- Figure 3
CHANNELS = [("withhold_all_documents", "Medicaid contract: withhold or incentive"),
            ("contract_all_documents", "Medicaid contract: language"),
            ("contract_postdating", "Medicaid contract: language, after report"),
            ("legislation", "State law"),
            ("legislation_postdating", "State law, after report"),
            ("pqc", "Quality collaborative project"),
            ("consortium", "Task force priority")]


def figure3():
    """Share of documented state-cause pairs present in each vehicle, with and
    without a committee recommendation, and the difference before and after
    stratifying by cause group."""
    pr = load("propagation.json")["channels"]
    rows = [(k, lab) for k, lab in CHANNELS if k in pr and pr[k]["overall"]["n_pairs"] > 0]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"wspace": 0.08,
                                                                "width_ratios": [1.05, 1]})
    y = np.arange(len(rows))[::-1]
    for i, (k, lab) in zip(y, rows):
        o = pr[k]["overall"]
        a, b = 100 * o["present_if_recommended"], 100 * o["present_if_not_recommended"]
        ax[0].plot([b, a], [i, i], color="#bbbbbb", lw=1.2, zorder=1)
        ax[0].plot(b, i, "o", ms=4.5, mfc="white", mec=INK, zorder=2)
        ax[0].plot(a, i, "o", ms=4.5, color=BLUE, zorder=3)
        ax[0].text(101, i, f"{o['n_pairs']} causes, {pr[k]['n_states_covered']} states",
                   va="center", fontsize=6.3, color=GREY)
    ax[0].set_yticks(y); ax[0].set_yticklabels([lab for _, lab in rows])
    ax[0].set_xlim(0, 135); ax[0].set_xticks([0, 25, 50, 75, 100])
    ax[0].set_xlabel("Documented causes appearing in the policy, %")
    ax[0].set_title("A  By recommendation status", loc="left", fontweight="bold")
    from matplotlib.lines import Line2D
    ax[0].legend([Line2D([], [], ls="", marker="o", color=BLUE),
                  Line2D([], [], ls="", marker="o", mfc="white", mec=INK)],
                 ["Recommendation addressed the cause", "No recommendation"],
                 frameon=False, loc="lower center", bbox_to_anchor=(0.45, -0.34), fontsize=6.8)

    ax[1].axvline(0, color=GREY, lw=0.8, ls="--")
    for i, (k, lab) in zip(y, rows):
        o, a = pr[k]["overall"], pr[k].get("adjusted_for_cause_group") or {}
        if o["difference"] is not None and o["ci"]:
            ax[1].plot([100 * v for v in o["ci"]], [i + 0.14] * 2, color=GREY, lw=1.1)
            ax[1].plot(100 * o["difference"], i + 0.14, "o", ms=4, mfc="white", mec=GREY)
        fe = pr[k].get("fixed_effects") or {}
        if "difference" in fe:
            ax[1].plot([100 * v for v in fe["ci"]], [i - 0.14] * 2, color=BLUE, lw=1.3)
            ax[1].plot(100 * fe["difference"], i - 0.14, "o", ms=4, color=BLUE)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel("Difference, percentage points (95% CI)")
    ax[1].set_title("B  Difference associated with a recommendation", loc="left",
                    fontweight="bold")
    ax[1].legend([Line2D([], [], ls="-", marker="o", mfc="white", mec=GREY, color=GREY),
                  Line2D([], [], ls="-", marker="o", color=BLUE)],
                 ["Unadjusted", "State and cause fixed effects"], frameon=False,
                 loc="lower center", bbox_to_anchor=(0.5, -0.34), fontsize=6.8)
    save(fig, "Figure3_propagation")


# ---------------------------------------------------------------- eFigures
SHORT_INT = {"Obstetric hemorrhage safety bundle": "Hemorrhage bundle",
             "Preferred uterotonic regimen for hemorrhage prevention": "Uterotonic prophylaxis",
             "Early tranexamic acid for hemorrhage treatment": "Tranexamic acid",
             "Severe hypertension treatment bundle": "Severe hypertension bundle",
             "Risk-appropriate care and maternal levels of care": "Levels of maternal care",
             "Low-dose aspirin for preeclampsia prevention": "Low-dose aspirin",
             "Counseling to prevent perinatal depression in at-risk people": "Depression prevention counseling",
             "Medication for opioid use disorder in pregnancy and postpartum": "MOUD"}


def short_param(p: str) -> str:
    if ": " in p:
        a, b = p.split(": ", 1)
        return f"{SHORT_INT.get(a, a)}, {b}"
    return p


def efigure_tornado():
    """One-way sensitivity of the evidence-weighted portfolio against no
    implementation: deaths averted (A) and cost per QALY gained (B)."""
    ow = load("oneway.json")
    rows = sorted(ow["rows"], key=lambda r: -r["swing_deaths_evidence"])[:12][::-1]
    rows_b = sorted(ow["rows"], key=lambda r: -r["swing_icer_evidence"])[:12][::-1]
    base = ow["base"]["evidence"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 4.2), gridspec_kw={"wspace": 1.05})
    for axx, rr, key, b0, xl in ((ax[0], rows, "deaths", base["deaths"], "Pregnancy-related deaths averted per year"),
                                 (ax[1], rows_b, "icer", base["icer"] / 1000,
                                  "Cost per QALY gained, thousands of 2025 US$")):
        for j, r in enumerate(rr):
            lo = r["low"]["evidence"][key]; hi = r["high"]["evidence"][key]
            if key == "icer":
                lo, hi = lo / 1000, hi / 1000
            a_, b_ = sorted([lo, hi])
            axx.barh(j, b_ - a_, left=a_, height=0.6, color=BLUE, alpha=0.8)
        axx.axvline(b0, color=ACCENT, lw=1.1)
        axx.set_yticks(range(len(rr)))
        axx.set_yticklabels([short_param(r["parameter"]) for r in rr], fontsize=6.4)
        axx.set_xlabel(xl)
    ax[0].set_title("A  Deaths averted", loc="left", fontweight="bold")
    ax[1].set_title("B  Incremental cost per QALY gained", loc="left", fontweight="bold")
    save(fig, "eFigure_tornado")


def efigure_ceac():
    """Probability that each portfolio is cost-effective against no
    implementation, across willingness-to-pay thresholds."""
    u = load("uncertainty.json")
    d = u["psa_draws"]
    wtp = np.geomspace(1e4, 1e7, 200)
    series = [("ev", "Interventions with evidence of benefit", BLUE, "-"), ("ba", "Cause-proportional", "#5b8db8", "--"),
              ("obs", "Named interventions", ACCENT, "-"), ("comp", "Named, bundle including tranexamic acid", "#c46a7c", "-."),
              ("freq", "Named, by frequency", GREY, ":")]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for key, lab, col, ls in series:
        if f"{key}_qaly" not in d[0]:
            continue
        q = np.array([r[f"{key}_qaly"] for r in d]); c = np.array([r[f"{key}_net"] for r in d])
        ax.plot(wtp / 1000, [float(np.mean(w * q - c > 0)) for w in wtp], color=col, ls=ls, lw=1.6, label=lab)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}K" if v < 1000 else f"${v/1000:,.0f}M"))
    ax.set_xlabel("Willingness to pay per QALY gained, 2025 US$ (log scale)")
    ax.set_ylabel("Probability cost-effective\nvs no implementation")
    ax.set_ylim(-0.02, 1.02)
    for x in (100, 150):
        ax.axvline(x, color="#dddddd", lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(fig, "eFigure_acceptability_curve")


def efigure_deaths_by_cause():
    pf = load("portfolio_cea.json")
    base = pf["baseline"]["deaths_by_cause"]
    obs = pf["portfolios"]["observed"]["deaths_by_cause"]
    ev = pf["portfolios"]["evidence_weighted"]["deaths_by_cause"]
    causes = sorted(base, key=lambda c: -base[c])
    y = np.arange(len(causes))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(y + 0.22, [base[c] - obs[c] for c in causes], height=0.38,
            color=ACCENT, label="Named interventions")
    ax.barh(y - 0.22, [base[c] - ev[c] for c in causes], height=0.38,
            color=BLUE, label="Interventions with evidence of benefit")
    ax.set_yticks(y); ax.set_yticklabels([short(c) for c in causes])
    ax.invert_yaxis()
    ax.set_xlabel("Pregnancy-related deaths averted per year")
    ax.legend(frameon=False)
    save(fig, "eFigure_deaths_averted_by_cause")


def efigure_pass_agreement():
    cls = load("classification.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.55})
    for k, (field, title) in enumerate([("policy_lever", "Policy lever"),
                                        ("cause_domain", "Cause domain")]):
        a = {r["label"]: r["pct"] for r in cls["pass_A"][field]}
        b = {r["label"]: r["pct"] for r in cls["pass_B"][field]}
        labs = sorted(set(a) | set(b), key=lambda l: -a.get(l, 0))
        y = np.arange(len(labs))
        ax[k].barh(y + 0.2, [a.get(l, 0) for l in labs], height=0.36, color=BLUE, label="Claude Opus 5.5")
        ax[k].barh(y - 0.2, [b.get(l, 0) for l in labs], height=0.36, color=GREY, label="GPT-6-astra")
        ax[k].set_yticks(y); ax[k].set_yticklabels([short(l) for l in labs])
        ax[k].invert_yaxis(); ax[k].set_xlabel("Share of recommendations (%)")
        ax[k].set_title(title, loc="left", fontweight="bold")
        if k == 0:
            ax[k].legend(frameon=False, loc="lower right")
    save(fig, "eFigure_pass_agreement")


def main():
    print("figures:")
    figure1(); figure2(); figure3()
    if (RES / "portfolio_cea.json").exists() and (RES / "uncertainty.json").exists():
        figure4()
    if (RES / "portfolio_cea.json").exists():
        efigure_deaths_by_cause()
    if (RES / "uncertainty.json").exists():
        efigure_ceac()
    if (RES / "oneway.json").exists():
        efigure_tornado()
    efigure_pass_agreement()
    print("done")


if __name__ == "__main__":
    main()
