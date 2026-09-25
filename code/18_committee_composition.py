"""
18_committee_composition.py — Who sits on each review committee, read from the
report itself.

The protocol asks whether alignment between recommendations and documented
burden varies with whether the state Medicaid agency sits on the committee. That
fact is not in the extracted corpus, so it is read here from each report's
membership roster. Report length, the protocol's proxy for committee resourcing,
is the page count already stored with each document.

Each report's text is extracted with pdftotext. Pages that look like a roster
(member, committee, representing, affiliation) are sent to a language model with
a fixed schema; the model must quote the line that supports each answer and give
its page, so every coded value can be checked against the source. A report with
no roster in its text is coded unknown, never absent.

Usage:  python code/18_committee_composition.py
Output: results/committee_composition.json
"""
from __future__ import annotations
import csv, json, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DHD = Path.home() / "waymark-local/notebooks/dark-health-data"
DOCS = DHD / "data/processed/mmrc/documents.csv"
CACHE = RES / "report_text"
MODEL = "claude-opus-5"
WORKERS = 6

ROSTER = re.compile(r"\bmembers?\b|membership|roster|representing|affiliation|committee member",
                    re.I)

SCHEMA = """Return one JSON object with these keys:
{"roster_found": true|false,
 "n_members": <integer or null>,
 "medicaid_agency_member": true|false|null,
 "medicaid_evidence": "<exact quoted line naming the Medicaid agency member, or null>",
 "medicaid_page": <page number or null>,
 "mco_or_payer_member": true|false|null,
 "mco_evidence": "<exact quoted line or null>",
 "obstetrician_or_mfm_member": true|false|null,
 "community_or_lived_experience_member": true|false|null}

Rules. A Medicaid agency member is a person listed as representing the state Medicaid
program or the state agency that administers Medicaid (for example "Department of Medical
Assistance Services", "Health Care Authority", "AHCCCS", "MassHealth", "Division of Medicaid").
A managed care organization or payer member is a person listed with a health plan or insurer
affiliation. If there is no membership list in the text, set roster_found false and every
member field null. Never infer membership from text that is not a roster. Quote exactly."""


def text_pages(pdf: Path) -> list[str]:
    out = CACHE / (pdf.stem + ".txt")
    if not out.exists():
        CACHE.mkdir(exist_ok=True)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(out)],
                       check=False, capture_output=True)
    if not out.exists():
        return []
    return out.read_text(errors="ignore").split("\f")


def roster_excerpt(pages: list[str], limit: int = 60000) -> str:
    """Pages that look like a roster, in page order, with page labels."""
    scored = [(i, len(ROSTER.findall(p))) for i, p in enumerate(pages)]
    keep = sorted(i for i, s in sorted(scored, key=lambda x: -x[1])[:10] if s >= 2)
    parts, n = [], 0
    for i in keep:
        chunk = f"\n[page {i + 1}]\n{pages[i]}"
        if n + len(chunk) > limit:
            break
        parts.append(chunk); n += len(chunk)
    return "".join(parts)


def call(client, prompt: str, tries: int = 4):
    for k in range(tries):
        try:
            m = client.messages.create(model=MODEL, max_tokens=4000,
                                       messages=[{"role": "user", "content": prompt}])
            txt = "\n".join(b.text for b in m.content if getattr(b, "type", "") == "text")
            s, e = txt.find("{"), txt.rfind("}")
            return json.loads(txt[s:e + 1])
        except Exception as ex:
            if k == tries - 1:
                return {"error": str(ex)[:200]}
            time.sleep(4 * (k + 1))


def main() -> None:
    docs = list(csv.DictReader(open(DOCS)))
    client = anthropic.Anthropic()

    def one(d):
        lp = d["local_path"]
        pdf = DHD / lp[lp.index("data/raw"):] if "data/raw" in lp else Path(lp)
        pages = text_pages(pdf)
        rec = {"document_id": d["document_id"], "state": d["jurisdiction"],
               "report_year": d["report_year"], "n_pages": int(d["n_pages"] or 0),
               "text_pages": len(pages)}
        ex = roster_excerpt(pages)
        if not ex.strip():
            rec.update({"roster_found": False, "medicaid_agency_member": None,
                        "mco_or_payer_member": None, "note": "no roster-like page in text"})
            return rec
        prompt = (f"This is text from a {d['jurisdiction']} maternal mortality review committee "
                  f"report ({d['report_year']}). {SCHEMA}\n\nTEXT:\n{ex}")
        rec.update(call(client, prompt))
        return rec

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(one, docs))

    found = [r for r in rows if r.get("roster_found")]
    med = [r for r in found if r.get("medicaid_agency_member") is True]
    print(f"reports {len(rows)}; roster found {len(found)}; Medicaid agency member {len(med)}; "
          f"errors {sum(1 for r in rows if 'error' in r)}")
    json.dump({"model": MODEL, "reports": rows}, open(RES / "committee_composition.json", "w"),
              indent=1)
    print(f"wrote {RES/'committee_composition.json'}")


if __name__ == "__main__":
    main()
