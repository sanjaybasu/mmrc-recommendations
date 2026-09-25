"""
05_authority_and_coverage.py — Two refinements requested by the payer co-author.

1. Coverage recommendations split by period. "Coverage or payment" was a single
   lever at 13.4%. Kara asked whether antepartum and postpartum coverage behave
   differently, which matters because 12-month postpartum extension is now
   near-universal and antepartum coverage is not.

2. Whether a recommendation names an actor with the authority to act. Her
   account of serving at the state level was that committees and consortia
   discussed themes but frequently lacked authority to change practice. The
   testable version is whether the recommendation identifies a body that can
   bind anyone: a legislature, a Medicaid agency, a managed care organization,
   a licensing or accrediting body, or a named accountable institution, as
   against a diffuse addressee such as "providers" or "the community".

Output: results/authority_and_coverage.json
"""
from __future__ import annotations
import csv, json, re, collections, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_shim import map_lever

SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
OUT = Path(__file__).resolve().parent.parent / "results"

R = list(csv.DictReader(open(SRC / "mmrc_recommendations.csv")))
text = lambda r: f"{r.get('recommendation') or ''} {r.get('category') or ''}"

# ---- 1. coverage period split -------------------------------------------------
POSTPARTUM = re.compile(r"postpartum|post[- ]partum|fourth trimester|after (?:the )?(?:birth|delivery)|year after", re.I)
ANTEPARTUM = re.compile(r"prenatal|antepartum|during pregnan|presumptive eligibility|first trimester|early pregnan", re.I)
COVERAGE = re.compile(r"\bcoverage\b|reimburs|\bbenefit\b|medicaid|eligib|\bfund|\bpay(?:ment|er|or)?\b|insur", re.I)

cov = [r for r in R if COVERAGE.search(text(r))]
split = collections.Counter()
for r in cov:
    t = text(r)
    pp, ap = bool(POSTPARTUM.search(t)), bool(ANTEPARTUM.search(t))
    split["Postpartum only" if pp and not ap else
          "Antepartum only" if ap and not pp else
          "Both periods named" if pp and ap else "Period not specified"] += 1

# 12-month extension specifically
TWELVE = re.compile(r"(?:12|twelve)[- ]month.{0,40}postpartum|postpartum.{0,40}(?:12|twelve)[- ]month|extend.{0,40}postpartum", re.I)
twelve_n = sum(1 for r in R if TWELVE.search(text(r)))

print(f"recommendations touching coverage, payment, or eligibility: {len(cov)} of {len(R)} ({100*len(cov)/len(R):.1f}%)")
for k, v in split.most_common():
    print(f"  {k:24s} {v:4d} ({100*v/len(cov):.1f}% of coverage recs)")
print(f"  naming 12-month postpartum extension explicitly: {twelve_n} ({100*twelve_n/len(R):.1f}% of all recs)")

# ---- 2. authority of the named actor -----------------------------------------
# Ordered: the first tier matched wins, most binding first.
AUTHORITY = [
    ("Legislature or statute", r"legislat|statut|\bbill\b|general assembly|appropriat|state law|enact"),
    ("Medicaid agency or state agency", r"medicaid agency|state medicaid|department of health|\bDHHS\b|\bDOH\b|state agency|health department"),
    ("Managed care organization or payer", r"managed care organization|\bMCO\b|health plan|payer|payor|insurer|contract"),
    ("Licensing, accreditation, or perinatal collaborative", r"licens|accredit|certif|perinatal quality collaborative|\bPQC\b|regulat"),
    ("Named institution or facility", r"\bhospital\b|\bfacility\b|health system|birthing center|\bclinic\b"),
]
DIFFUSE = re.compile(r"\bproviders?\b|\bclinicians?\b|\bcommunit|\bstakeholder|\beveryone\b|\bthe public\b|\bfamilies\b", re.I)

auth = collections.Counter()
for r in R:
    t = text(r)
    tier = next((name for name, pat in AUTHORITY if re.search(pat, t, re.I)), None)
    auth[tier or ("Diffuse addressee" if DIFFUSE.search(t) else "No actor identified")] += 1

n = len(R)
print(f"\n{'actor named':46s} {'n':>5s} {'%':>6s}")
for k, v in auth.most_common():
    print(f"{k:46s} {v:5d} {100*v/n:6.1f}")
binding = sum(v for k, v in auth.items() if k in
              ("Legislature or statute", "Medicaid agency or state agency",
               "Managed care organization or payer", "Licensing, accreditation, or perinatal collaborative"))
print(f"\nnaming a body that can bind anyone: {binding} of {n} ({100*binding/n:.1f}%)")

# ---- 3. lever by authority ----------------------------------------------------
lev_auth = collections.defaultdict(collections.Counter)
for r in R:
    t = text(r)
    lv = map_lever(r.get("recommendation"), r.get("category"))
    if not lv:
        continue
    tier = next((name for name, pat in AUTHORITY if re.search(pat, t, re.I)), None)
    lev_auth[lv]["binding" if tier in ("Legislature or statute", "Medicaid agency or state agency",
                                       "Managed care organization or payer",
                                       "Licensing, accreditation, or perinatal collaborative") else "non-binding"] += 1
print(f"\n{'policy lever':36s} {'binding':>8s} {'total':>7s} {'% binding':>10s}")
lev_rows = []
for lv, c in sorted(lev_auth.items(), key=lambda kv: -sum(kv[1].values())):
    tot = sum(c.values())
    print(f"{lv:36s} {c['binding']:8d} {tot:7d} {100*c['binding']/tot:9.1f}%")
    lev_rows.append(dict(lever=lv, binding=c["binding"], total=tot, pct_binding=round(100*c["binding"]/tot, 1)))

json.dump({
    "coverage_recs": len(cov), "coverage_pct_of_all": round(100*len(cov)/len(R), 1),
    "coverage_period_split": dict(split),
    "twelve_month_postpartum_named": twelve_n,
    "authority": dict(auth), "pct_naming_binding_actor": round(100*binding/n, 1),
    "lever_by_authority": lev_rows, "total_recommendations": n,
}, open(OUT / "authority_and_coverage.json", "w"), indent=1)
