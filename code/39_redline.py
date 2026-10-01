"""
39_redline.py — Tracked-changes copy of the manuscript: the new render compared
word by word with an earlier render, with insertions and deletions as Word
revisions.

Paragraphs are aligned by text; within each changed paragraph, words are diffed and
written as <w:ins>/<w:del> runs that keep each word's character formatting
(superscript citations, italics). Inserted words that reproduce a co-author's own
tracked insertions (passed with --coauthor-text) are attributed to that co-author;
all other changes to --author.

Usage: python 39_redline.py OLD.docx NEW.docx OUT.docx --author "Sanjay Basu"
       [--coauthor "Walker, Kara" --coauthor-text file_with_one_insertion_per_line]
"""
from __future__ import annotations
import argparse, copy, difflib, re, shutil, tempfile, zipfile
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
q = lambda t: f"{{{W}}}{t}"
TOK = re.compile(r"\s+|[\w$%.,'’\-–]+|[^\w\s]")


def runs_of(p):
    """(rPr element or None, text) for each simple text run; None if the paragraph has other content."""
    out = []
    for child in p:
        if child.tag == q("pPr"):
            continue
        if child.tag in (q("bookmarkStart"), q("bookmarkEnd"), q("proofErr")):
            continue
        if child.tag != q("r"):
            return None
        if child.find(q("drawing")) is not None or child.find(q("fldChar")) is not None:
            return None
        rpr = child.find(q("rPr"))
        txt = "".join((t.text or "") for t in child.iter(q("t")))
        if child.find(q("tab")) is not None or child.find(q("br")) is not None:
            return None
        out.append((rpr, txt))
    return out


def ptext(p):
    return "".join((t.text or "") for t in p.iter(q("t")))


def tokens(runs):
    out = []
    for rpr, txt in runs:
        for m in TOK.finditer(txt):
            out.append((rpr, m.group(0)))
    return out


def make_run(rpr, text, deleted=False):
    r = etree.Element(q("r"))
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    t = etree.SubElement(r, q("delText") if deleted else q("t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return r


class Ids:
    n = 9000

    @classmethod
    def next(cls):
        cls.n += 1
        return str(cls.n)


def wrap(kind, author, date, runs):
    e = etree.Element(q(kind))
    e.set(q("id"), Ids.next()); e.set(q("author"), author); e.set(q("date"), date)
    for r in runs:
        e.append(r)
    return e


def grouped(toks):
    """Merge consecutive tokens with identical formatting into runs."""
    out = []
    for rpr, t in toks:
        key = etree.tostring(rpr) if rpr is not None else b""
        if out and out[-1][0] == key:
            out[-1][2].append(t)
        else:
            out.append((key, rpr, [t]))
    return [(rpr, "".join(ts)) for _, rpr, ts in out]


def attribute(ins_toks, coauthor_seqs):
    """Split inserted tokens into (is_coauthor, tokens) spans."""
    texts = [t for _, t in ins_toks]
    flags = [False] * len(texts)
    for seq in coauthor_seqs:
        sm = difflib.SequenceMatcher(None, [x.lower() for x in texts], [x.lower() for x in seq], autojunk=False)
        for b in sm.get_matching_blocks():
            if sum(1 for x in texts[b.a:b.a + b.size] if x.strip()) >= 4:   # at least 4 words
                for i in range(b.a, b.a + b.size):
                    flags[i] = True
    spans = []
    for f, tk in zip(flags, ins_toks):
        if spans and spans[-1][0] == f:
            spans[-1][1].append(tk)
        else:
            spans.append((f, [tk]))
    return spans


def redline_paragraph(op, np_, author, coauthor, coseqs, date):
    o, n = runs_of(op), runs_of(np_)
    if o is None or n is None:
        return False
    ot, nt = tokens(o), tokens(n)
    sm = difflib.SequenceMatcher(None, [t for _, t in ot], [t for _, t in nt], autojunk=False)
    ppr = np_.find(q("pPr"))
    for c in list(np_):
        if c is not ppr:
            np_.remove(c)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for rpr, txt in grouped(nt[j1:j2]):
                np_.append(make_run(rpr, txt))
            continue
        if tag in ("delete", "replace"):
            np_.append(wrap("del", author, date, [make_run(r, t, True) for r, t in grouped(ot[i1:i2])]))
        if tag in ("insert", "replace"):
            for is_co, span in attribute(nt[j1:j2], coseqs):
                who = coauthor if (is_co and coauthor) else author
                np_.append(wrap("ins", who, date, [make_run(r, t) for r, t in grouped(span)]))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new"); ap.add_argument("out")
    ap.add_argument("--author", required=True)
    ap.add_argument("--coauthor"); ap.add_argument("--coauthor-text")
    a = ap.parse_args()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    coseqs = []
    if a.coauthor_text:
        coseqs = [[m.group(0) for m in TOK.finditer(l.strip())] for l in open(a.coauthor_text) if l.strip()]
    rd = lambda f: etree.fromstring(zipfile.ZipFile(f).read("word/document.xml"))
    odoc, ndoc = rd(a.old), rd(a.new)
    ops = list(odoc.iter(q("p"))); nps = list(ndoc.iter(q("p")))
    sm = difflib.SequenceMatcher(None, [ptext(p) for p in ops], [ptext(p) for p in nps], autojunk=False)
    changed = skipped = added = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        pairs = list(zip(range(i1, i2), range(j1, j2))) if tag == "replace" else []
        for i, j in pairs:
            if redline_paragraph(ops[i], nps[j], a.author, a.coauthor, coseqs, date):
                changed += 1
            else:
                skipped += 1
        for j in range(j1 + len(pairs), j2):      # wholly new paragraphs
            p = nps[j]; ppr = p.find(q("pPr"))
            kids = [c for c in p if c is not ppr]
            if kids and ptext(p).strip():
                for c in kids:
                    p.remove(c)
                p.append(wrap("ins", a.author, date, kids)); added += 1
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(a.new) as z:
        z.extractall(tmp)
    (tmp / "word/document.xml").write_bytes(etree.tostring(ndoc, xml_declaration=True, encoding="UTF-8", standalone=True))
    out = Path(a.out); out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp)
    print(f"redlined {changed} paragraphs, {added} new paragraphs, {skipped} changed paragraphs left untracked")


if __name__ == "__main__":
    main()
