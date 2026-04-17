"""Verify every entry in local.bib against authoritative online sources.

For each entry we try to verify:
  - arXiv papers (eprint or arxiv id in journal): hit the arXiv API, cross-check
    title and first author.
  - @misc entries with URL: HEAD-check the URL resolves.
  - @inproceedings (ICLR-style): flagged as MANUAL since the canonical source
    is OpenReview and author lists change between submission and accept.

Output: prints a table and writes a JSON report to scripts/citation_report.json.

Usage:
    python scripts/verify_citations.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET


PAPER_DIR = Path(__file__).resolve().parent.parent.parent / "autoresearch-automl-paper"
BIB_PATH = PAPER_DIR / "local.bib"
REPORT_PATH = Path(__file__).parent / "citation_report.json"


def parse_bib(text: str) -> list[dict]:
    """Minimal BibTeX parser. Each entry returned as dict with keys:
    type, key, title, author, year, journal, url, eprint, arxiv_id.
    """
    entries = []
    # Split on @type{key, ... } — tolerant of whitespace/newlines inside key.
    pattern = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,([^@]+?)\n\}", re.DOTALL)
    for m in pattern.finditer(text):
        typ, key, body = m.group(1), m.group(2), m.group(3)
        fields = {}
        # Match field = {value} or field = "value"
        for fm in re.finditer(r"(\w+)\s*=\s*(\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\")", body):
            name = fm.group(1).lower()
            val = fm.group(3) if fm.group(3) is not None else fm.group(4)
            fields[name] = (val or "").strip()
        # Pull arxiv id from journal or eprint
        arxiv_id = None
        if "eprint" in fields:
            arxiv_id = fields["eprint"].strip()
        journal = fields.get("journal", "")
        vol = fields.get("volume", "")
        m_arxiv = re.search(r"(?:arXiv[:\s]?|abs/)(\d{4}\.\d{4,5})", journal + " " + vol)
        if m_arxiv and not arxiv_id:
            arxiv_id = m_arxiv.group(1)
        entries.append({
            "type": typ.lower(),
            "key": key,
            "title": fields.get("title", "").replace("{", "").replace("}", ""),
            "author": fields.get("author", ""),
            "year": fields.get("year", ""),
            "journal": journal,
            "booktitle": fields.get("booktitle", ""),
            "url": fields.get("url") or fields.get("howpublished", ""),
            "arxiv_id": arxiv_id,
        })
    return entries


def fetch_arxiv(arxiv_id: str) -> dict | None:
    """Query the arXiv API for a given ID. Returns dict with title + authors."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = resp.read().decode("utf-8")
    except Exception as e:
        return {"error": f"fetch failed: {e}"}
    try:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(data)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return {"error": "no entry in response (bad ID?)"}
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        authors = [a.findtext("atom:name", default="", namespaces=ns).strip()
                   for a in entry.findall("atom:author", ns)]
        return {"title": title, "authors": authors}
    except Exception as e:
        return {"error": f"parse failed: {e}"}


def head_check(url: str) -> bool:
    """Return True if URL responds with 2xx/3xx."""
    # Strip LaTeX \url{...}
    m = re.search(r"https?://[^\s}]+", url)
    if not m:
        return False
    clean_url = m.group(0).rstrip(".,)")
    try:
        req = urllib.request.Request(clean_url, method="HEAD",
                                     headers={"User-Agent": "citation-verifier/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 400
    except urllib.error.HTTPError as e:
        return 200 <= e.code < 400
    except Exception:
        return False


def normalize(s: str) -> str:
    return re.sub(r"\W+", "", s).lower()


def first_author_surname(author_field: str) -> str:
    """Extract first author's surname from a BibTeX author field."""
    if not author_field:
        return ""
    first = re.split(r"\s+and\s+", author_field)[0]
    if "," in first:
        return first.split(",")[0].strip()
    return first.strip().split()[-1]


def verify(entry: dict) -> dict:
    """Verify one entry against online sources. Returns report dict."""
    report = {"key": entry["key"], "status": "UNKNOWN", "issues": []}
    # arXiv-backed entries (highest signal)
    if entry["arxiv_id"]:
        arx = fetch_arxiv(entry["arxiv_id"])
        report["arxiv_id"] = entry["arxiv_id"]
        if arx is None or "error" in arx:
            report["status"] = "FAIL"
            report["issues"].append(f"arXiv fetch error: {arx.get('error') if arx else 'None'}")
            return report
        report["online_title"] = arx["title"]
        report["online_authors"] = arx["authors"]
        report["local_title"] = entry["title"]
        report["local_author"] = entry["author"]

        # Title match (fuzzy)
        if normalize(arx["title"][:40]) != normalize(entry["title"][:40]):
            report["issues"].append(
                f"title mismatch: bib='{entry['title'][:60]}' vs arxiv='{arx['title'][:60]}'"
            )
        # First-author surname match
        local_first = first_author_surname(entry["author"])
        online_first = arx["authors"][0].strip().split()[-1] if arx["authors"] else ""
        if local_first.lower() not in arx["authors"][0].lower() and online_first.lower() != local_first.lower():
            report["issues"].append(
                f"first-author mismatch: bib='{local_first}' vs arxiv first='{arx['authors'][0]}'"
            )
        # Full author list match check (just report count + any suspicious diffs)
        local_authors = [a.strip() for a in re.split(r"\s+and\s+", entry["author"])]
        if len(local_authors) != len(arx["authors"]):
            # "and others" is fine
            if "others" not in entry["author"].lower():
                report["issues"].append(
                    f"author count differs: bib={len(local_authors)} vs arxiv={len(arx['authors'])}"
                )
        report["status"] = "FAIL" if report["issues"] else "OK"
        return report

    # @misc with a URL
    if entry["type"] == "misc" and entry["url"]:
        ok = head_check(entry["url"])
        report["url"] = entry["url"]
        report["status"] = "OK" if ok else "FAIL"
        if not ok:
            report["issues"].append(f"URL not resolvable: {entry['url']}")
        return report

    # inproceedings with a booktitle but no arXiv — flag for manual review
    if entry["type"] == "inproceedings":
        report["status"] = "MANUAL"
        report["issues"].append(
            f"No arXiv id. Verify via OpenReview/DBLP: '{entry['title']}' "
            f"by {entry['author']} ({entry['booktitle']}, {entry['year']})"
        )
        return report

    report["status"] = "MANUAL"
    report["issues"].append("no verifiable source (arXiv id or URL)")
    return report


def main() -> int:
    if not BIB_PATH.exists():
        print(f"ERROR: {BIB_PATH} not found", file=sys.stderr)
        return 2
    text = BIB_PATH.read_text()
    entries = parse_bib(text)
    print(f"Parsed {len(entries)} entries from {BIB_PATH.name}\n")

    results = []
    for entry in entries:
        r = verify(entry)
        results.append(r)
        status = r["status"]
        symbol = {"OK": "OK  ", "FAIL": "FAIL", "MANUAL": "MAN "}.get(status, "?   ")
        print(f"[{symbol}] {r['key']}")
        for issue in r.get("issues", []):
            print(f"         {issue}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    counts = {"OK": 0, "FAIL": 0, "MANUAL": 0, "UNKNOWN": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for k, v in counts.items():
        print(f"  {k}: {v}")

    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print("\nFAILURES (fix these!):")
        for r in failures:
            print(f"  - {r['key']}: {'; '.join(r['issues'])}")

    REPORT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nFull JSON report: {REPORT_PATH}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
