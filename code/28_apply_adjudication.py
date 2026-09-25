"""
28_apply_adjudication.py — Apply the physician's adjudication decisions to produce
the final label set the analyses use.

Inputs: validation/MMRC_adjudication_ADJ.xlsx (the physician's decisions) and
results/adjudication_key.csv (which option came from which model, never shown to
the physician). For each adjudicated item the chosen option's source label is
kept; "Neither" takes the physician's corrected label. Items not adjudicated keep
the primary model's (Claude Opus 5.5) label.

Output: results/llm_labels_passA.jsonl, results/intervention_map_passA.jsonl, and
results/cause_label_map.json overwritten with the adjudicated final labels (the
pure model outputs remain in labels_<model>.jsonl, imap_<model>.jsonl, and
cause_label_map_<model>.json), plus results/adjudication_summary.json.
"""
from __future__ import annotations
import collections, csv, importlib.util, json, sys
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PRIMARY, SECOND = "claude-opus-5-5", "gpt-6-astra"
BINDING = {"Legislature or statute", "Medicaid or state health agency",
           "Managed care organization or payer"}


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "code" / f)
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m)
    return m


def main() -> None:
    iv = _load("iv", "12_interventions.py")
    P = json.load(open(RES / "model_params.json"))["params"]
    lab2key = {i.label: i.key for i in iv._registry(P)}; lab2key["No specific intervention"] = "none"

    src = {"primary": {"lab": {json.loads(l)["id"]: json.loads(l) for l in open(RES / f"labels_{PRIMARY}.jsonl")},
                       "im": {json.loads(l)["id"]: json.loads(l)["intervention"] for l in open(RES / f"imap_{PRIMARY}.jsonl")},
                       "cm": json.load(open(RES / f"cause_label_map_{PRIMARY}.json"))["map"]},
           "second": {"lab": {json.loads(l)["id"]: json.loads(l) for l in open(RES / f"labels_{SECOND}.jsonl")},
                      "im": {json.loads(l)["id"]: json.loads(l)["intervention"] for l in open(RES / f"imap_{SECOND}.jsonl")},
                      "cm": json.load(open(RES / f"cause_label_map_{SECOND}.json"))["map"]}}
    final_lab = {k: dict(v) for k, v in src["primary"]["lab"].items()}
    final_im = dict(src["primary"]["im"])
    final_cm = dict(src["primary"]["cm"])

    key = {(r["sheet"], int(r["excel_row"])): r for r in csv.DictReader(open(RES / "adjudication_key.csv"))}
    wb = load_workbook(ROOT / "validation/MMRC_adjudication_ADJ.xlsx", data_only=True)
    tally = collections.Counter(); by_sheet = collections.defaultdict(collections.Counter)
    changes = []
    for sheet in ("1 Priority", "2 Cause labels", "3 Remaining (optional)"):
        if sheet not in wb.sheetnames:
            continue
        ws = wb[sheet]
        hdr = [c.value for c in ws[1]]
        d = hdr.index("Your decision")
        nb = next(i for i, h in enumerate(hdr) if h and str(h).startswith("If neither"))
        for row in ws.iter_rows(min_row=2):
            dec = row[d].value
            if not dec:
                continue
            k = key[(sheet, row[0].row)]
            chosen = (k["option1_source"] if dec == "Option 1" else
                      k["option2_source"] if dec == "Option 2" else "neither")
            tally[chosen] += 1; by_sheet[sheet][chosen] += 1
            fld = k["field"]
            if sheet == "2 Cause labels":
                s = k["id_or_string"]
                new = row[nb].value if chosen == "neither" else src[chosen]["cm"][s]
                if new and new != final_cm.get(s):
                    changes.append((sheet, s, fld, final_cm.get(s), new))
                final_cm[s] = new or final_cm[s]
                continue
            i = int(k["id_or_string"])
            if fld == "Specific intervention requested":
                new = (lab2key.get(str(row[nb].value).strip()) if chosen == "neither"
                       else src[chosen]["im"][i])
                if new and new != final_im[i]:
                    changes.append((sheet, i, fld, final_im[i], new))
                final_im[i] = new or final_im[i]
            elif fld == "Cause of death addressed":
                new = row[nb].value if chosen == "neither" else src[chosen]["lab"][i]["cause_domain"]
                if new and new != final_lab[i]["cause_domain"]:
                    changes.append((sheet, i, fld, final_lab[i]["cause_domain"], new))
                final_lab[i]["cause_domain"] = new or final_lab[i]["cause_domain"]
            elif fld == "Policy lever":
                new = row[nb].value if chosen == "neither" else src[chosen]["lab"][i]["policy_lever"]
                final_lab[i]["policy_lever"] = new or final_lab[i]["policy_lever"]
            elif fld == "Binding actor named" and chosen == "second":
                final_lab[i]["actor_authority"] = src["second"]["lab"][i]["actor_authority"]

    with open(RES / "llm_labels_passA.jsonl", "w") as f:
        for i in sorted(final_lab):
            f.write(json.dumps(final_lab[i]) + "\n")
    with open(RES / "intervention_map_passA.jsonl", "w") as f:
        for i in sorted(final_im):
            f.write(json.dumps({"id": i, "intervention": final_im[i]}) + "\n")
    cm = json.load(open(RES / f"cause_label_map_{PRIMARY}.json"))
    cm["map"] = final_cm; cm["model"] = f"{PRIMARY}, physician-adjudicated"
    json.dump(cm, open(RES / "cause_label_map.json", "w"), indent=1)

    summ = {"adjudicated_items": sum(tally.values()), "chose_primary": tally["primary"],
            "chose_second": tally["second"], "chose_neither": tally["neither"],
            "by_sheet": {k: dict(v) for k, v in by_sheet.items()},
            "labels_changed_from_primary": len(changes),
            "changes": [list(map(str, c)) for c in changes]}
    json.dump(summ, open(RES / "adjudication_summary.json", "w"), indent=1)
    print({k: v for k, v in summ.items() if k != "changes"})


if __name__ == "__main__":
    main()
