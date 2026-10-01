"""
04_accountability_chain.py — Does a maternal death cause travel from the review
committee's finding, to a recommendation, to state Medicaid contract language?

Kara Odom Walker's framing: committees identify causes and issue recommendations,
but the bodies that receive them often lack authority to change practice. The
testable version is a three-stage funnel, per state and cause domain:

    Stage 1  the committee documents the cause among its leading causes
    Stage 2  the committee issues at least one recommendation addressing it
    Stage 3  the topic appears in that state's Medicaid procurement or contract
             documents, and separately, appears tied to a quality withhold or
             incentive

Stage 3 uses the Medicaid managed care procurement corpus assembled for our
Inquiry analysis: 2,834 documents across 32 states.

Matching for stage 3 requires an obstetric context window, because terms such as
"hemorrhage" or "infection" appear throughout a Medicaid contract for reasons
unrelated to pregnancy. A cause term counts only when it occurs within
CONTEXT_WINDOW characters of a maternal or perinatal term.

Output: results/accountability_chain.json and a per-state long table.
"""
from __future__ import annotations
import bisect, csv, json, os, re, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_shim import map_cause, map_recommendation, DOMAINS

MMRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
RFP = Path.home() / "waymark-local/notebooks/rfp_analysis/processed_text"
OUT = Path(__file__).resolve().parent.parent / "results"
CONTEXT_WINDOW = int(os.environ.get("MMRC_CTX_WINDOW", 300))  # window-width sensitivity via env

STATE_ABBR = {
    "Arizona": "AZ", "California": "CA", "Colorado": "CO", "Delaware": "DE", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Mexico": "NM", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Rhode Island": "RI", "Tennessee": "TN", "Texas": "TX", "Virginia": "VA", "Washington": "WA",
    "Washington DC": "DC", "West Virginia": "WV",
}

MATERNAL_CONTEXT = re.compile(
    r"matern|obstetric|pregnan|postpartum|perinatal|prenatal|antepartum|birth|deliver|OB/?GYN|doula|midwif",
    re.I)

CAUSE_TERMS = {
    "Mental health conditions": r"mental health|behavioral health|depress|suicid|perinatal mood",
    "Substance use disorder": r"substance use|opioid|overdose|neonatal abstinence|buprenorphine|methadone|\bMOUD\b|\bSUD\b",
    "Cardiovascular conditions": r"cardiovascular|cardiac|heart disease",
    "Cardiomyopathy": r"cardiomyopath",
    "Hemorrhage": r"h[ae]morrhag|blood loss|transfusion",
    "Infection": r"infection|sepsis|septic",
    "Embolism": r"embolism|thromboembol|\bVTE\b|thromboprophylax",
    "Hypertensive disorders": r"hypertens|preeclamp|pre-eclamp|eclampsia",
    "Cerebrovascular accident": r"cerebrovascular|stroke",
    "Injury, homicide, and violence": r"intimate partner violence|\bIPV\b|domestic violence|homicide",
    "Other medical conditions": r"gestational diabet|diabetes in pregnan",
}

WITHHOLD = re.compile(
    r"withhold|incentive payment|pay[- ]for[- ]performance|\bP4P\b|performance bonus|"
    r"quality withhold|liquidated damages|penalt|capitation adjust|value[- ]based payment",
    re.I)


def context_hits(text: str, pattern: re.Pattern, spans: list[int]) -> int:
    """Count pattern matches falling within CONTEXT_WINDOW of a maternal term.

    spans is the sorted list of maternal-term offsets, computed once per
    document; a binary search finds the nearest one in log time.
    """
    if not spans:
        return 0
    n = 0
    for m in pattern.finditer(text):
        i = m.start()
        j = bisect.bisect_left(spans, i)
        near = False
        if j < len(spans) and spans[j] - i <= CONTEXT_WINDOW:
            near = True
        elif j > 0 and i - spans[j - 1] <= CONTEXT_WINDOW:
            near = True
        if near:
            n += 1
    return n


def scan_state(state_dir: Path) -> dict:
    """Return per-domain match counts and withhold co-occurrence for one state."""
    compiled = {d: re.compile(p, re.I) for d, p in CAUSE_TERMS.items()}
    counts = collections.Counter()
    withhold_counts = collections.Counter()
    n_docs = 0
    for f in state_dir.rglob("*.txt"):
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        n_docs += 1
        spans = [m.start() for m in MATERNAL_CONTEXT.finditer(text)]
        if not spans:
            continue
        has_withhold = bool(WITHHOLD.search(text))
        for d, pat in compiled.items():
            h = context_hits(text, pat, spans)
            if h:
                counts[d] += h
                # withhold language anywhere in a document that also ties the
                # cause to a maternal context; deliberately permissive, and
                # therefore an upper bound
                if has_withhold:
                    withhold_counts[d] += 1
    return {"n_docs": n_docs, "counts": dict(counts), "withhold_docs": dict(withhold_counts)}


def main() -> None:
    F = list(csv.DictReader(open(MMRC / "mmrc_findings.csv")))
    R = list(csv.DictReader(open(MMRC / "mmrc_recommendations.csv")))

    burden = collections.defaultdict(collections.Counter)
    for f in F:
        d = map_cause(f.get("leading_cause"))
        if d:
            burden[f["state"]][d] += 1
    recs = collections.defaultdict(collections.Counter)
    for r in R:
        d = map_recommendation(r.get("recommendation"), r.get("category"))
        if d:
            recs[r["state"]][d] += 1

    rfp = {}
    for name, abbr in STATE_ABBR.items():
        p = RFP / name
        if p.exists():
            print(f"scanning {name} ...", flush=True)
            rfp[abbr] = scan_state(p)

    states = sorted(set(rfp) & set(burden))
    print(f"\nstates with both a review report and procurement documents: {len(states)}")
    print(", ".join(states))

    rows = []
    funnel = collections.defaultdict(lambda: collections.Counter())
    for abbr in states:
        for d in DOMAINS:
            documented = burden[abbr][d] > 0
            recommended = recs.get(abbr, {}).get(d, 0) > 0
            in_contract = rfp[abbr]["counts"].get(d, 0) > 0
            withheld = rfp[abbr]["withhold_docs"].get(d, 0) > 0
            rows.append(dict(state=abbr, domain=d, documented=documented,
                             recommended=recommended, in_contract=in_contract,
                             contract_mentions=rfp[abbr]["counts"].get(d, 0),
                             withhold_docs=rfp[abbr]["withhold_docs"].get(d, 0)))
            if documented:
                funnel[d]["documented"] += 1
                if recommended:
                    funnel[d]["recommended"] += 1
                    if in_contract:
                        funnel[d]["in_contract"] += 1
                        if withheld:
                            funnel[d]["withhold"] += 1

    with open(OUT / "accountability_chain_long.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'cause domain':34s} {'documented':>11s} {'+rec':>6s} {'+contract':>10s} {'+withhold':>10s}")
    out = []
    for d in DOMAINS:
        f_ = funnel[d]
        if f_["documented"] >= 3:
            print(f"{d:34s} {f_['documented']:11d} {f_['recommended']:6d} {f_['in_contract']:10d} {f_['withhold']:10d}")
            out.append(dict(domain=d, documented=f_["documented"], recommended=f_["recommended"],
                            in_contract=f_["in_contract"], withhold=f_["withhold"]))

    json.dump({"states": states, "n_states": len(states), "funnel": out,
               "context_window_chars": CONTEXT_WINDOW},
              open(OUT / "accountability_chain.json", "w"), indent=1)


if __name__ == "__main__":
    main()
