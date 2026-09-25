"""
24_validation_packet.py — Build the physician coding packet for validating the
automated classification.

Sample. 120 recommendations drawn with a fixed seed so that every value of the two
fields the headline results rest on is represented: up to 5 per requested
intervention (the evidence result) and at least 6 per cause domain (the silence
result), topped up at random from the rest. Rows are shuffled so that no ordering
reveals an automated label. A second sheet samples 40 of the committee-written
leading-cause strings, which define the burden side of the silence analysis.

Blinding. Coders never see the automated labels. The key linking sample rows to
automated labels is written to results/validation_key.csv, which must not be sent
to coders. Coder columns are left empty and must be filled only by the physicians.

Output: validation/MMRC_physician_coding_coder1.xlsx, ..._coder2.xlsx,
        validation/CODING_INSTRUCTIONS.md, results/validation_key.csv
"""
from __future__ import annotations
import csv, importlib.util, json, random, sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", ROOT / "validation"
SRC = Path.home() / "waymark-local/notebooks/dark-health-data/data/processed/mmrc"
SEED, N, PER_INT, PER_CAUSE, N_CAUSE_STRINGS = 20260927, 45, 2, 2, 15

spec = importlib.util.spec_from_file_location("lc", ROOT / "code/07_llm_classify.py")
lc = importlib.util.module_from_spec(spec); sys.modules["lc"] = lc; spec.loader.exec_module(lc)
spec = importlib.util.spec_from_file_location("iv", ROOT / "code/12_interventions.py")
iv = importlib.util.module_from_spec(spec); sys.modules["iv"] = iv; spec.loader.exec_module(iv)

CAUSE_STRING_OPTIONS = [c for c in lc.CAUSE_DOMAINS if c != "Cross-cutting"] + ["Not a cause of death"]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    recs = lc.load_recs()
    A = {json.loads(l)["id"]: json.loads(l) for l in open(RES / "llm_labels_passA.jsonl")}
    I = {json.loads(l)["id"]: json.loads(l)["intervention"]
         for l in open(RES / "intervention_map_passA.jsonl")}
    P = json.load(open(RES / "model_params.json"))["params"]
    reg = [(i.key, i.label) for i in iv._registry(P)]
    int_label = dict(reg); int_label["none"] = "No specific intervention"
    int_options = [lab for _, lab in reg] + ["No specific intervention"]

    # Exclude everything the adjudicator has already seen with candidate labels,
    # so the blinded sample is not anchored on them.
    seen_ids, seen_strings = set(), set()
    adj_key = RES / "adjudication_key.csv"
    if adj_key.exists():
        for r in csv.DictReader(open(adj_key)):
            if r["sheet"].endswith("Cause labels"):
                seen_strings.add(r["id_or_string"])
            else:
                seen_ids.add(int(r["id_or_string"]))
    recs = [r for r in recs if r["id"] not in seen_ids]

    rng = random.Random(SEED)
    chosen: list[int] = []
    by_int: dict = {}
    for r in recs:
        by_int.setdefault(I.get(r["id"], "none"), []).append(r["id"])
    for k, ids in sorted(by_int.items()):
        chosen += rng.sample(ids, min(len(ids), PER_INT))
    by_cause: dict = {}
    for r in recs:
        by_cause.setdefault(A[r["id"]]["cause_domain"], []).append(r["id"])
    for c, ids in sorted(by_cause.items()):
        have = [i for i in chosen if A[i]["cause_domain"] == c]
        pool = [i for i in ids if i not in chosen]
        chosen += rng.sample(pool, max(0, min(len(pool), PER_CAUSE - len(have))))
    rest = [r["id"] for r in recs if r["id"] not in set(chosen)]
    chosen += rng.sample(rest, max(0, N - len(chosen)))
    rng.shuffle(chosen)
    byid = {r["id"]: r for r in recs}
    print(f"excluded {len(seen_ids)} recommendations and {len(seen_strings)} cause labels "
          f"already shown in the adjudication workbook")

    # committee-written leading-cause strings
    cmap = json.load(open(RES / "cause_label_map.json"))["map"]
    strings = sorted(k for k in cmap if k not in seen_strings)
    cs = rng.sample(strings, min(N_CAUSE_STRINGS, len(strings)))

    # key file (never sent to coders)
    with open(RES / "validation_key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "row", "id_or_string", "llm_cause_domain", "llm_policy_lever",
                    "llm_actor_authority", "llm_coverage_period", "llm_intervention",
                    "sampling_weight_note"])
        for n, i in enumerate(chosen, 1):
            a = A[i]
            w.writerow(["Recommendations", n, i, a["cause_domain"], a["policy_lever"],
                        a["actor_authority"], a["coverage_period"], int_label[I.get(i, "none")],
                        "stratified; weight to population by intervention and cause stratum"])
        for n, s in enumerate(cs, 1):
            w.writerow(["Cause labels", n, s, cmap[s], "", "", "", "", "simple random sample"])

    cols = [("Row", 6), ("State", 7), ("Report section or category", 28), ("Recommendation text", 70),
            ("Cause of death addressed", 26), ("Policy lever", 26), ("Most authoritative actor named", 30),
            ("Coverage period", 22), ("Specific intervention requested", 38), ("Unsure? (Y)", 10),
            ("Comment", 30)]
    options = {5: lc.CAUSE_DOMAINS, 6: lc.LEVERS, 7: lc.AUTHORITY, 8: lc.PERIODS, 9: int_options}

    def hidden_list(wb, name, values):
        ws = wb[name] if name in wb.sheetnames else wb.create_sheet(name)
        for j, v in enumerate(values, 1):
            ws.cell(row=j, column=1, value=v)
        ws.sheet_state = "hidden"
        return f"'{name}'!$A$1:$A${len(values)}"

    for coder in (1,):
        wb = Workbook()
        ws = wb.active; ws.title = "Recommendations"
        hdr = PatternFill("solid", fgColor="DDE6EE")
        for j, (h, wdt) in enumerate(cols, 1):
            c = ws.cell(row=1, column=j, value=h); c.font = Font(bold=True); c.fill = hdr
            c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.column_dimensions[c.column_letter].width = wdt
        for n, i in enumerate(chosen, 2):
            r = byid[i]
            for j, v in enumerate([n - 1, r["state"], r["category"], r["text"]], 1):
                ws.cell(row=n, column=j, value=v).alignment = Alignment(wrap_text=True, vertical="top")
        for col, vals in options.items():
            ref = hidden_list(wb, f"list{col}", vals)
            dv = DataValidation(type="list", formula1=ref, allow_blank=True, showDropDown=False)
            ws.add_data_validation(dv)
            letter = ws.cell(row=1, column=col).column_letter
            dv.add(f"{letter}2:{letter}{len(chosen) + 1}")
        ws.freeze_panes = "E2"

        ws2 = wb.create_sheet("Cause labels", 1)
        for j, (h, wdt) in enumerate([("Row", 6), ("Committee-written cause label", 60),
                                      ("Cause category", 30), ("Comment", 30)], 1):
            c = ws2.cell(row=1, column=j, value=h); c.font = Font(bold=True); c.fill = hdr
            ws2.column_dimensions[c.column_letter].width = wdt
        for n, s in enumerate(cs, 2):
            ws2.cell(row=n, column=1, value=n - 1); ws2.cell(row=n, column=2, value=s)
        ref = hidden_list(wb, "listcause", CAUSE_STRING_OPTIONS)
        dv = DataValidation(type="list", formula1=ref, allow_blank=True)
        ws2.add_data_validation(dv); dv.add(f"C2:C{len(cs) + 1}")
        ws2.freeze_panes = "C2"

        ws3 = wb.create_sheet("Coder", 2)
        ws3["A1"] = "Coder initials"; ws3["A2"] = "Date completed"
        ws3["A1"].font = ws3["A2"].font = Font(bold=True); ws3.column_dimensions["A"].width = 18
        wb.save(OUT / "MMRC_blinded_coding.xlsx")

    for old in ("MMRC_physician_coding_coder1.xlsx", "MMRC_physician_coding_coder2.xlsx"):
        if (OUT / old).exists():
            (OUT / old).rename(OUT / old.replace(".xlsx", "_superseded_uncoded.xlsx"))
    # retire the earlier cause-only sample, which was never coded
    old = RES / "validation_sample.csv"
    if old.exists():
        old.rename(RES / "validation_sample_superseded_uncoded.csv")
    print(f"packet: {len(chosen)} recommendations, {len(cs)} cause labels; "
          f"interventions represented {len({I.get(i, 'none') for i in chosen})}; "
          f"causes represented {len({A[i]['cause_domain'] for i in chosen})}")


if __name__ == "__main__":
    main()
