#!/usr/bin/env python
"""Citation-freshness scan for the audit records and notes.

Extracts every ``path:line`` / ``path:a-b`` citation from ``audit/records/*.yaml``
and ``audit/notes/*.md``, resolves the path (repo, project venv site-packages,
audit scripts; bare names resolved by suffix, preferring pyspi/), and classifies
each citation:

``ok``                  a code-like token from the text around the citation occurs in the
                        cited lines (+/- 2 lines of slack)
``no-token``            no code-like token near the citation that also occurs in the cited
                        file (not judged)
``token-miss``          tokens that occur in the cited file occur nowhere in the cited lines
                        (+/- 2); ``found_at`` lists lines where they do occur (needs a look)
``stale-calculator``    pyspi/calculator.py citation whose tokens match the pre-a1f20c7
                        file but not HEAD; ``suggested`` gives the mapped HEAD lines
``out-of-range``        the cited line is past the end of the file
``unresolved``          no file found for the path (JIDT .java sources and the GitBook pages are
                        not in the repo or venv; their citations point at upstream tags/clones)
``out-of-range-all-candidates`` every file with that name is shorter than the cited line
``ambiguous``           several candidate files and the context does not pick one

The heuristic only screens; ``ok`` means a nearby token is present, not that the
sentence is right. Usage::

    .venv/bin/python audit/checks/citations.py [--out FILE] [--only-problems]
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "audit"
SITE = REPO / ".venv" / "lib" / "python3.12" / "site-packages"
OLD_CALC_COMMIT = "a1f20c7^"

CITE = re.compile(r"(?P<path>[./A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|pyx|java|m|R|c|cpp|h|yaml|md))"
                  r":(?P<a>\d+)(?:\s*[-–]\s*(?P<b>\d+))?")
TOKEN = re.compile(r"`([^`]{3,80})`|([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)|"
                   r"([A-Za-z_]*_[A-Za-z0-9_]+)|([A-Za-z_][A-Za-z0-9_]{3,})\(")
STOP = {"self", "data", "None", "True", "False", "return", "import", "numpy", "line", "lines",
        "pyspi", "file", "files", "audit", "records", "notes", "yaml", "the_", "and_"}


def _site_index():
    idx = {}
    for root in (REPO / "pyspi", SITE, AUDIT, REPO / "tests"):
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.suffix in (".py", ".pyx", ".m", ".R", ".c", ".cpp", ".h", ".yaml", ".md") and p.is_file():
                idx.setdefault(p.name, []).append(p)
    return idx


def _tokens(text):
    out = set()
    for m in TOKEN.finditer(text):
        tok = next(g for g in m.groups() if g)
        tok = tok.strip().strip("()")
        for part in re.split(r"[\s,;=()\[\]{}<>]+", tok):
            part = part.strip(".:'\"")
            if len(part) >= 4 and part not in STOP and not CITE.fullmatch(part) and not part.endswith((".py", ".java", ".m", ".R")):
                out.add(part)
                if "." in part:
                    out.add(part.rsplit(".", 1)[-1])
    return {t for t in out if len(t) >= 4 and t not in STOP}


def _read(p, cache):
    if p not in cache:
        try:
            cache[p] = p.read_text(errors="replace").splitlines()
        except Exception:
            cache[p] = []
    return cache[p]


def _resolve(path, a, b, idx, context_tokens, cache):
    """``(file, reason)``. Exact paths first; bare or partial names by suffix, keeping only
    files long enough for the cited line, then the most context tokens in the cited
    window, then the most in the whole file, then pyspi/ over third-party."""
    raw = path
    for prefix in (str(REPO) + "/",):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    if raw.startswith("./"):
        raw = raw[2:]
    if raw.startswith("/") and Path(raw).is_file():
        return Path(raw), None
    for c in (REPO / raw, SITE / raw, AUDIT / raw):
        if c.is_file():
            return c, None
    name = Path(raw).name
    cands = [p for p in idx.get(name, []) if str(p).endswith("/" + raw) or raw == name]
    if not cands:
        return None, "unresolved"
    fits = [p for p in cands if len(_read(p, cache)) >= a]
    if not fits:
        return None, "out-of-range-all-candidates"
    if len(fits) == 1:
        return fits[0], None

    def score(p):
        lines = _read(p, cache)
        win = "\n".join(lines[max(0, a - 3): min(len(lines), b + 2)])
        whole = "\n".join(lines)
        return (sum(t in win for t in context_tokens), sum(t in whole for t in context_tokens),
                1 if str(p).startswith(str(REPO / "pyspi")) else 0)
    ranked = sorted(fits, key=lambda p: score(p), reverse=True)
    if score(ranked[0]) > score(ranked[1]):
        return ranked[0], None
    return None, "ambiguous"


def _hit(lines, a, b, toks):
    lo, hi = max(0, a - 3), min(len(lines), b + 2)
    window = "\n".join(lines[lo:hi])
    return [t for t in toks if t in window]


def _old_calc():
    try:
        return subprocess.run(["git", "show", f"{OLD_CALC_COMMIT}:pyspi/calculator.py"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception:
        return None


def _map_lines(old, new):
    """old line number (1-based) -> new line number for lines in matching blocks."""
    m = {}
    for blk in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            m[blk.a + k + 1] = blk.b + k + 1
    return m


def scan():
    idx = _site_index()
    old_calc = _old_calc()
    new_calc = (REPO / "pyspi" / "calculator.py").read_text().splitlines()
    cmap = _map_lines(old_calc, new_calc) if old_calc else {}
    cache, out = {}, []
    sources = sorted((AUDIT / "records").glob("*.yaml")) + sorted((AUDIT / "notes").glob("*.md"))
    for src in sources:
        for ln, line in enumerate(src.read_text(errors="replace").splitlines(), 1):
            for m in CITE.finditer(line):
                a = int(m.group("a"))
                b = int(m.group("b") or a)
                if b < a:
                    b = a
                ctx = line[max(0, m.start() - 160): m.end() + 160]
                toks = _tokens(ctx.replace(m.group(0), " "))
                f, why = _resolve(m.group("path"), a, b, idx, toks, cache)
                rec = dict(source=str(src.relative_to(REPO)), source_line=ln, citation=m.group(0),
                           file=str(f.relative_to(REPO)) if f and str(f).startswith(str(REPO)) else (str(f) if f else None),
                           tokens=sorted(toks)[:12])
                if f is None:
                    rec["status"] = why
                    out.append(rec)
                    continue
                lines = _read(f, cache)
                whole = "\n".join(lines)
                file_toks = sorted(t for t in toks if t in whole)
                rec["tokens_in_file"] = file_toks[:12]
                if a > len(lines):
                    rec["status"] = "out-of-range"
                elif not file_toks:
                    rec["status"] = "no-token"
                else:
                    hits = _hit(lines, a, b, file_toks)
                    rec["hits"] = hits[:5]
                    rec["status"] = "ok" if hits else "token-miss"
                    if not hits:
                        rec["found_at"] = sorted({i + 1 for i, l in enumerate(lines)
                                                  for t in file_toks if t in l})[:8]
                    if f == REPO / "pyspi" / "calculator.py" and old_calc and not hits:
                        old_hits = _hit(old_calc, a, b, file_toks)
                        if old_hits:
                            na, nb = cmap.get(a), cmap.get(b)
                            rec["status"] = "stale-calculator"
                            rec["old_hits"] = old_hits[:5]
                            rec["suggested"] = (f"pyspi/calculator.py:{na}" + (f"-{nb}" if nb and nb != na else "")
                                                if na else "no line mapping (block changed)")
                rec["cited_text"] = lines[a - 1].strip()[:160] if a <= len(lines) else ""
                out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--only-problems", action="store_true")
    a = ap.parse_args()
    res = scan()
    counts = {}
    for r in res:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    if a.only_problems:
        res = [r for r in res if r["status"] in ("token-miss", "stale-calculator", "out-of-range", "ambiguous")]
    doc = dict(counts=counts, n=sum(counts.values()), citations=res)
    text = json.dumps(doc, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
