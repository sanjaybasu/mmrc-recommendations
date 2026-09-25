"""
03_evidence_crosstab.py — Cross the policy lever a recommendation pulls against
the strength of evidence that the lever changes maternal outcomes.

Evidence grades are assigned at the lever level from the strongest available
synthesis for each lever's representative interventions (see
results/evidence_table.md for citations). Grades are provisional and must be
re-assigned by two independent reviewers with a kappa before publication.
"""
from __future__ import annotations
import csv, collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_shim import map_lever

SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
OUT = Path(__file__).resolve().parent.parent / "results"

# lever -> (grade, anchor trial/synthesis, direction on maternal outcomes)
EVIDENCE = {
    "Clinical protocol or bundle": ("Moderate to strong", "AIM/CMQCC hemorrhage collaborative: SMM among hemorrhage -20.8% vs -1.2% comparison (Main 2017); uterotonics network meta-analysis RR 0.76 for PPH (Gallos 2025); tranexamic acid RR 0.69 for bleeding death within 3h (WOMAN 2017)", "benefit"),
    "Screening or referral": ("Weak", "Perinatal depression prevention RR 0.61 only when treatment is attached (USPSTF 2019); IPV screening raises detection without reducing abuse or morbidity (Cochrane 2015)", "conditional"),
    "Coverage or payment": ("Moderate for coverage, absent for mortality", "12-month postpartum extension raises coverage +2.8pp; no credible published effect on maternal death", "indirect"),
    "Workforce or staffing": ("Moderate for process, absent for mortality", "Continuous labor support lowers cesarean RR 0.75 with no effect on maternal complications (Cochrane 2017)", "indirect"),
    "Care coordination or navigation": ("Null", "Nurse-Family Partnership RCT n=5,670 South Carolina Medicaid: adjusted difference 0.5% (-2.1 to 3.1), null across all maternal outcomes (McConnell 2022)", "null"),
    "Community or social services": ("Weak", "Group prenatal care null in pooled RCTs; social-need interventions not powered for maternal mortality", "unknown"),
    "Education or training": ("Absent for patient outcomes", "Systematic review of 77 implicit-bias training studies found no trial demonstrating improvement in patient clinical outcomes (Science Advances 2024)", "none"),
    "Data, surveillance, or review": ("Not directly effect-bearing", "Surveillance enables action but carries no independent effect estimate", "n/a"),
}

R = list(csv.DictReader(open(SRC / "mmrc_recommendations.csv")))
counts = collections.Counter()
for r in R:
    lv = map_lever(r.get("recommendation"), r.get("category"))
    if lv:
        counts[lv] += 1
tot = sum(counts.values())

rows = []
print(f"{'policy lever':34s} {'n':>5s} {'% of mapped':>12s}  evidence grade")
print("-" * 96)
for lv, n in counts.most_common():
    grade, anchor, direction = EVIDENCE[lv]
    print(f"{lv:34s} {n:5d} {100*n/tot:11.1f}%  {grade}")
    rows.append(dict(lever=lv, n=n, pct_of_mapped=round(100 * n / tot, 1),
                     evidence_grade=grade, direction=direction, anchor=anchor))

strong = sum(n for lv, n in counts.items() if EVIDENCE[lv][2] == "benefit")
none_null = sum(n for lv, n in counts.items() if EVIDENCE[lv][2] in ("none", "null"))
print("-" * 96)
print(f"levers with a demonstrated benefit on maternal outcomes: {strong} of {tot} ({100*strong/tot:.1f}%)")
print(f"levers with null or absent evidence on patient outcomes: {none_null} of {tot} ({100*none_null/tot:.1f}%)")

json.dump({"by_lever": rows, "mapped_total": tot,
           "pct_benefit_evidence": round(100 * strong / tot, 1),
           "pct_null_or_absent_evidence": round(100 * none_null / tot, 1)},
          open(OUT / "evidence_crosstab.json", "w"), indent=1)
