"""
02_alignment_pilot.py — Do state MMRC recommendations match that state's own
documented burden of death, and what lever do they pull?

Concordance ratio follows the normalized measure used in our Inquiry Medicaid
procurement paper: the share of a state's recommendations addressing a cause
domain divided by that domain's share of the same committee's documented
leading causes. A ratio of 1.0 means recommendations track burden; above 1.0
means over-attention; below means under-attention.
"""
from __future__ import annotations
import csv, json, collections, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_shim import map_cause, map_recommendation, map_lever, DOMAINS

SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
OUT = Path(__file__).resolve().parent.parent / "results"
load = lambda n: list(csv.DictReader(open(SRC / n)))

F, R = load("mmrc_findings.csv"), load("mmrc_recommendations.csv")

# Burden: each documented leading-cause mention is one unit of attributed burden.
burden = collections.defaultdict(collections.Counter)
for f in F:
    d = map_cause(f.get("leading_cause"))
    if d:
        burden[f["state"]][d] += 1

recs = collections.defaultdict(collections.Counter)
levers = collections.Counter()
lever_by_domain = collections.defaultdict(collections.Counter)
unmapped_rec = 0
for r in R:
    d = map_recommendation(r.get("recommendation"), r.get("category"))
    if d:
        recs[r["state"]][d] += 1
    else:
        unmapped_rec += 1
    lv = map_lever(r.get("recommendation"), r.get("category"))
    if lv:
        levers[lv] += 1
        if d:
            lever_by_domain[d][lv] += 1

mapped_causes = sum(sum(c.values()) for c in burden.values())
mapped_recs = sum(sum(c.values()) for c in recs.values())
print(f"cause mentions mapped: {mapped_causes} / {sum(1 for f in F if (f.get('leading_cause') or '').strip())}")
print(f"recommendations mapped to a cause domain: {mapped_recs} / {len(R)}  (unmapped {unmapped_rec})")

# States with enough of both to compare
states = [s for s in burden if sum(burden[s].values()) >= 5 and sum(recs.get(s, {}).values()) >= 10]
print(f"\nstates with >=5 mapped causes and >=10 mapped recommendations: {len(states)}")

ratios = collections.defaultdict(list)
for s in states:
    bt, rt = sum(burden[s].values()), sum(recs[s].values())
    for d in DOMAINS:
        bs, rs = burden[s][d] / bt, recs[s][d] / rt
        if bs > 0:
            ratios[d].append(rs / bs)

print(f"\n{'cause domain':34s} {'states':>6s} {'median ratio':>13s} {'IQR':>18s}")
rows = []
for d in DOMAINS:
    v = sorted(ratios[d])
    if len(v) >= 5:
        med = statistics.median(v)
        q1, q3 = v[len(v)//4], v[3*len(v)//4]
        print(f"{d:34s} {len(v):6d} {med:13.2f}   {q1:6.2f} to {q3:6.2f}")
        rows.append(dict(domain=d, n_states=len(v), median_ratio=round(med,2), iqr=[round(q1,2), round(q3,2)]))

print(f"\n{'policy lever':36s} {'n':>6s} {'%':>6s}")
tot = sum(levers.values())
lv_rows = []
for k, v in levers.most_common():
    print(f"{k:36s} {v:6d} {100*v/tot:6.1f}")
    lv_rows.append(dict(lever=k, n=v, pct=round(100*v/tot,1)))

print("\n--- lever mix within the two largest cause domains ---")
for d in ("Mental health conditions", "Substance use disorder", "Hemorrhage", "Hypertensive disorders"):
    c = lever_by_domain[d]; t = sum(c.values())
    if t:
        top = ", ".join(f"{k} {100*v/t:.0f}%" for k, v in c.most_common(3))
        print(f"  {d:30s} n={t:4d}  {top}")

json.dump({"concordance_by_domain": rows, "levers": lv_rows,
           "n_states_compared": len(states),
           "mapped_cause_mentions": mapped_causes, "mapped_recommendations": mapped_recs,
           "total_recommendations": len(R)},
          open(OUT / "alignment_pilot.json", "w"), indent=1)
