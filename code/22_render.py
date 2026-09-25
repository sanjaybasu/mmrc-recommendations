"""
22_render.py — Render the manuscript and supplement from their templates.

Every number in the prose is a {{key}} token resolved from canonical_numbers.json,
and rendering stops if any token has no canonical value, so a number cannot
appear in the manuscript without a result file behind it. Citations are
[@key; @key] markers numbered in order of first appearance, main text and
supplement separately, and resolved against manuscript/references.json; an
unverified reference is rendered with its flag visible.

Output: manuscript/MMRC_Recommendations_JAMAHF_main.{md,docx}
        manuscript/MMRC_Recommendations_JAMAHF_supplement.{md,docx}
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAN, FIG = ROOT / "manuscript", ROOT / "figures"
REFDOC = Path.home() / ".claude/templates/sanjay_paper_reference.docx"
TOKEN = re.compile(r"\{\{([A-Za-z0-9_.\-]+)\}\}")
CITE = re.compile(r"\[(@[a-z0-9_]+(?:;\s*@[a-z0-9_]+)*)\]")


DRAFT = "--draft" in sys.argv


def fill(text: str, canon: dict) -> str:
    missing = sorted({k for k in TOKEN.findall(text) if k not in canon})
    if missing and DRAFT:
        return TOKEN.sub(lambda m: str(canon.get(m.group(1), "[pending]")), text)
    if missing:
        sys.exit("missing canonical values:\n  " + "\n  ".join(missing))
    return TOKEN.sub(lambda m: str(canon[m.group(1)]), text)


def compress(nums: list[int]) -> str:
    nums = sorted(set(nums)); out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}-{nums[j]}" if j - i >= 2 else
                   ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


def cite(text: str, refs: dict, prefix: str = "") -> tuple[str, list[str]]:
    order: list[str] = []

    def num(m):
        keys = [k.strip()[1:] for k in m.group(1).split(";")]
        for k in keys:
            if k not in order:
                order.append(k)
        return f"<sup>{prefix}{compress([order.index(k) + 1 for k in keys])}</sup>"

    body = CITE.sub(num, text)
    lines = []
    for i, k in enumerate(order, 1):
        r = refs.get(k)
        if r is None:
            lines.append(f"{prefix}{i}. [MISSING REFERENCE: {k}]")
        else:
            flag = "" if r.get("verified") else " [UNVERIFIED — check manually]"
            ama = r["ama"].replace(" [UNVERIFIED — check manually]", "")
            lines.append(f"{prefix}{i}. {ama}{flag}")
    return body, lines


def words(md: str, start: str, end: str) -> int:
    a, b = md.find(start), md.find(end)
    seg = md[a:b]
    seg = re.sub(r"<sup>.*?</sup>", "", seg)
    seg = "\n".join(l for l in seg.splitlines() if not l.startswith("#"))
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-.,%]*", seg))


def to_docx(md_path: Path) -> None:
    out = md_path.with_suffix(".docx")
    subprocess.run(["pandoc", str(md_path), "-o", str(out), f"--reference-doc={REFDOC}",
                    f"--resource-path={ROOT}"], check=True)
    print(f"  {out.name}")


def prompts() -> str:
    """The classification prompts, read from the code that sent them."""
    import importlib.util
    out = []
    for fname, names, title in (
            ("07_llm_classify.py", ["SCHEMA", "PROMPT_A"], "Recommendation fields"),
            ("21_intervention_map.py", ["PROMPT_A"], "Specific intervention requested"),
            ("07b_llm_causes.py", ["PROMPT"], "Committee-written cause labels"),
            ("29_evidence_grading.py", ["PROMPT"], "Evidence grading"),
            ("18_committee_composition.py", ["SCHEMA"], "Committee roster")):
        spec = importlib.util.spec_from_file_location(fname[:-3], ROOT / "code" / fname)
        m = importlib.util.module_from_spec(spec); sys.modules[fname[:-3]] = m
        spec.loader.exec_module(m)
        for n in names:
            txt = getattr(m, n, None)
            if isinstance(txt, str):
                txt = txt.replace("{{", "{").replace("}}", "}")
                out.append(f"*{title}, {n.replace('_', ' ').capitalize().replace('prompt a', 'prompt').replace('schema', 'schema inserted into the prompt')}.*\n\n```\n{txt.strip()}\n```")
    return "\n\n".join(out)


def main() -> None:
    canon = json.load(open(ROOT / "canonical_numbers.json"))
    refs = json.load(open(MAN / "references.json")) if (MAN / "references.json").exists() else {}

    parts = {n: (MAN / f"{n}.md").read_text() for n in
             ("template_main", "results", "discussion", "exhibits")
             if (MAN / f"{n}.md").exists()}
    main_md = parts["template_main"]
    main_md = main_md.replace("<!-- RESULTS -->", parts.get("results", "<!-- RESULTS -->"))
    main_md = main_md.replace("<!-- DISCUSSION -->", parts.get("discussion", "<!-- DISCUSSION -->"))

    canon.setdefault("wordcount.text", "0"); canon.setdefault("refcount.main", "0")
    body = fill(main_md, canon)
    body, reflines = cite(body, refs)
    n_words = words(body, "## Introduction", "## Article Information")
    canon["wordcount.text"] = f"{n_words:,}"; canon["refcount.main"] = str(len(reflines))
    body = fill(main_md, canon)
    body, reflines = cite(body, refs)
    body = body.replace("<!-- REFERENCES -->", "\n\n".join(reflines))
    if "exhibits" in parts:
        body += "\n\n" + fill(parts["exhibits"], canon)
    out = MAN / "MMRC_Recommendations_JAMAHF_main.md"
    out.write_text(body)
    print(f"main text {n_words:,} words; {len(reflines)} references")

    sup = MAN / "template_supplement.md"
    if sup.exists():
        raw = sup.read_text()
        if (MAN / "etables.md").exists():
            raw = raw.replace("<!-- ETABLES -->", (MAN / "etables.md").read_text())
        raw = raw.replace("<!-- PROMPTS -->", prompts())
        s = fill(raw, canon)
        s, elines = cite(s, refs, prefix="e")
        s = s.replace("<!-- EREFERENCES -->", "\n\n".join(elines))
        so = MAN / "MMRC_Recommendations_JAMAHF_supplement.md"
        so.write_text(s)
        print(f"supplement: {len(elines)} eReferences")
        to_docx(so)
    to_docx(out)
    unverified = [k for k, r in refs.items() if not r.get("verified")]
    if unverified:
        print("unverified references present:", ", ".join(unverified))


if __name__ == "__main__":
    main()
