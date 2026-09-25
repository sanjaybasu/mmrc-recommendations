"""
00_pilot.py — Pilot analysis for the MMRC recommendations paper.

Question: what do state Maternal Mortality Review Committees recommend, and who
is asked to act? Novelty check (2026-09-20): Qian et al. Am J Obstet Gynecol
2025;232:394.e1 covered preventability and contributing factors across 42
jurisdictions and explicitly did not analyze recommendations. A Georgetown CCF
policy brief (Oct 2023) scanned 12 states for managed-care involvement but was
not peer reviewed. No peer-reviewed multi-state catalogue of MMRC-issued
recommendations exists.

Input: dark-health-data v0.4.0 MMRC release (public-record state MMRC PDFs).
Output: results/pilot_numbers.json  — every number quoted in the memo.
"""
from __future__ import annotations
import csv, json, re, collections, statistics
from pathlib import Path

SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

load = lambda n: list(csv.DictReader(open(SRC / n)))
num = lambda x: float(x) if x not in (None, "", "NA") else None

# Actor-naming patterns. Deliberately generous: any mention counts, so the
# payer-naming share is an upper bound on how often payers are addressed.
PATTERNS = {
    "education_training": r"\beducat|\btrain|\bawareness",
    "coverage_reimbursement": r"\bcoverage\b|\breimburs|\bbenefit\b",
    "names_medicaid": r"\bmedicaid\b",
    "names_plan_payer": r"\bmanaged care\b|\bMCO\b|\bhealth plan\b|\bpayer\b|\bpayor\b|\binsurer\b",
    "mental_health_substance_use": r"\bmental health\b|\bsubstance\b|\bopioid\b|\bsuicid|\bdepress|\bbehavioral health\b|\boverdose\b",
    "doula_chw": r"\bdoula\b|\bcommunity health worker\b|\bCHW\b",
}


def race_group(g: str | None) -> str | None:
    g = (g or "").lower()
    if "black" in g: return "Black"
    if "white" in g: return "White"
    return None


def main() -> None:
    F, R, D = load("mmrc_findings.csv"), load("mmrc_recommendations.csv"), load("documents.csv")
    text = lambda r: f"{r['recommendation'] or ''} {r['category'] or ''}"
    n = len(R)

    res: dict = {
        "corpus": {
            "documents": len(D),
            "jurisdictions": len(set(d["jurisdiction"] for d in D)),
            "report_years": [min(d["report_year"] for d in D), max(d["report_year"] for d in D)],
            "n_recommendations": n,
            "states_with_recommendations": len(set(r["state"] for r in R)),
            "n_findings": len(F),
        },
        "recommendation_themes": {},
        "target_level": dict(collections.Counter(r["target_level"] for r in R).most_common()),
    }
    for k, p in PATTERNS.items():
        c = sum(1 for r in R if re.search(p, text(r), re.I))
        res["recommendation_themes"][k] = {"n": c, "pct": round(100 * c / n, 1)}

    # Preventability: reported for context only. Qian et al. own this endpoint.
    pv = collections.defaultdict(list)
    for f in F:
        v = num(f["pct_preventable"])
        if v is not None and f["qa_status"] == "pass":
            pv[f["state"]].append(v)
    med = {s: statistics.median(v) for s, v in pv.items()}
    res["preventability_context_only"] = {
        "states": len(med),
        "median_of_state_medians": round(statistics.median(med.values()), 1),
        "range": [round(min(med.values()), 1), round(max(med.values()), 1)],
    }

    # Black:White pregnancy-related mortality ratio, within state-period.
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in F:
        v = num(f["pregnancy_related_mortality_ratio"])
        g = race_group(f["population_group"])
        if v is not None and g and f["qa_status"] == "pass":
            cells[(f["state"], f["report_period"])][g].append(v)
    ratios = [
        statistics.median(d["Black"]) / statistics.median(d["White"])
        for d in cells.values()
        if "Black" in d and "White" in d and statistics.median(d["White"]) > 0
    ]
    rs = sorted(ratios)
    res["black_white_prmr_ratio"] = {
        "state_periods": len(rs),
        "distinct_states": len({k[0] for k, d in cells.items() if "Black" in d and "White" in d}),
        "median": round(statistics.median(rs), 2),
        "iqr": [round(rs[len(rs) // 4], 2), round(rs[3 * len(rs) // 4], 2)],
    }

    causes = collections.Counter(
        (f["leading_cause"] or "").strip().lower() for f in F if (f["leading_cause"] or "").strip()
    )
    res["leading_causes_top10"] = causes.most_common(10)

    res["external_benchmark"] = {
        "medicaid_share_of_us_births_2024_pct": 40.2,
        "source": "Osterman MJK et al. Births in the United States, 2024. NCHS Data Brief 535. July 2025.",
    }

    (OUT / "pilot_numbers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
