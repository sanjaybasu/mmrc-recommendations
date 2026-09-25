"""
06_withhold_proximity.py — Tighten the accountability test's final stage.

In the first pass a cause counted as "tied to a quality withhold" if the
document mentioned the cause in a maternal context anywhere and contained
withhold or incentive language anywhere. In a 400-page contract those two can be
hundreds of pages apart, and the withhold column duplicated the contract column
almost exactly, which is the signature of a measure that is not measuring
anything.

This version requires the withhold term to fall within WINDOW characters of a
maternal-context cause mention, so the two must appear in the same provision.
Reported alongside the loose measure so the difference is visible.
"""
from __future__ import annotations
import bisect, csv, json, re, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("chain", Path(__file__).parent / "04_accountability_chain.py")
chain = importlib.util.module_from_spec(spec); spec.loader.exec_module(chain)

WINDOW = 1000   # same provision, roughly a page
OUT = Path(__file__).resolve().parent.parent / "results"


def tight_withhold(text: str, cause_pat: re.Pattern, mat_spans: list[int]) -> bool:
    """True if a withhold term sits within WINDOW of a maternal-context cause mention."""
    w = [m.start() for m in chain.WITHHOLD.finditer(text)]
    if not w:
        return False
    for m in cause_pat.finditer(text):
        i = m.start()
        j = bisect.bisect_left(mat_spans, i)
        near_mat = (j < len(mat_spans) and mat_spans[j] - i <= chain.CONTEXT_WINDOW) or \
                   (j > 0 and i - mat_spans[j - 1] <= chain.CONTEXT_WINDOW)
        if not near_mat:
            continue
        k = bisect.bisect_left(w, i)
        if (k < len(w) and w[k] - i <= WINDOW) or (k > 0 and i - w[k - 1] <= WINDOW):
            return True
    return False


def main() -> None:
    compiled = {d: re.compile(p, re.I) for d, p in chain.CAUSE_TERMS.items()}
    tight = collections.defaultdict(set)
    for name, abbr in chain.STATE_ABBR.items():
        sd = chain.RFP / name
        if not sd.exists():
            continue
        for f in sd.rglob("*.txt"):
            try:
                text = f.read_text(errors="ignore")
            except Exception:
                continue
            if not chain.WITHHOLD.search(text):
                continue
            spans = [m.start() for m in chain.MATERNAL_CONTEXT.finditer(text)]
            if not spans:
                continue
            for d, pat in compiled.items():
                if abbr not in tight[d] and tight_withhold(text, pat, spans):
                    tight[d].add(abbr)
        print(f"  {name} done", flush=True)

    # Join onto the per-state funnel so the withhold stage is conditional on the
    # earlier stages rather than a free-standing count of states.
    long = list(csv.DictReader(open(OUT / "accountability_chain_long.csv")))
    loose = json.load(open(OUT / "accountability_chain.json"))
    states = set(loose["states"])
    tobool = lambda v: str(v).strip().lower() == "true"

    rows, out_long = [], []
    for r in long:
        d, st = r["domain"], r["state"]
        r["withhold_same_provision"] = st in tight[d]
        out_long.append(r)
    with open(OUT / "accountability_chain_long.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_long[0].keys()))
        w.writeheader(); w.writerows(out_long)

    print(f"\n{'cause domain':34s} {'doc':>4s} {'+rec':>5s} {'+contract':>10s} "
          f"{'+loose':>7s} {'+same-prov':>11s} {'any same-prov':>14s}")
    for f_ in loose["funnel"]:
        d = f_["domain"]
        conditional = sum(
            1 for r in long
            if r["domain"] == d and r["state"] in states
            and tobool(r["documented"]) and tobool(r["recommended"]) and tobool(r["in_contract"])
            and r["state"] in tight[d])
        unconditional = len(tight[d] & states)
        print(f"{d:34s} {f_['documented']:4d} {f_['recommended']:5d} {f_['in_contract']:10d} "
              f"{f_['withhold']:7d} {conditional:11d} {unconditional:14d}")
        rows.append(dict(domain=d, documented=f_["documented"], recommended=f_["recommended"],
                         in_contract=f_["in_contract"], withhold_loose=f_["withhold"],
                         withhold_same_provision=conditional,
                         withhold_same_provision_unconditional=unconditional))
    json.dump({"window_chars": WINDOW, "n_states": loose["n_states"], "funnel": rows,
               "note": ("withhold_same_provision is conditional on the state having documented "
                        "the cause, recommended on it, and naming it in a maternal context in its "
                        "contract corpus; withhold_same_provision_unconditional counts all states "
                        "with such language regardless of the earlier stages")},
              open(OUT / "accountability_chain_tight.json", "w"), indent=1)


if __name__ == "__main__":
    main()
