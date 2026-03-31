#!/usr/bin/env python3
"""Clean a BibTeX file and keep only selected fields.

Behavior:
- Always back up input .bib file to script/backup before processing.
- Keep common fields: author, journal, title, year, volume, number, pages, doi, url.
- Keep booktitle for inproceedings.
- Keep institution for techreport.
- Normalize DOI to HTTPS URL format.
- Require every entry to have both DOI and URL.
- Require institution for techreport entries.
- Print missing items and wait for user input before continuing.
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

BASE_FIELDS = [
    "author",
    "journal",
    "title",
    "year",
    "volume",
    "number",
    "pages",
    "doi",
    "url",
]

EXTRA_FIELDS_BY_TYPE = {
    "inproceedings": ["booktitle"],
    "techreport": ["institution"],
}

DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
ACM_DOI_RE = re.compile(r"^https?://dl\.acm\.org/doi/(?:abs/)?(.+)$", re.IGNORECASE)
ARXIV_ABS_RE = re.compile(r"^https?://arxiv\.org/abs/([^/?#]+)$", re.IGNORECASE)
DOI_CORE_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


def split_entries(bib_text: str) -> list[str]:
    entries: list[str] = []
    i = 0
    n = len(bib_text)

    while i < n:
        at = bib_text.find("@", i)
        if at == -1:
            break

        brace_open = bib_text.find("{", at)
        if brace_open == -1:
            break

        depth = 0
        j = brace_open
        while j < n:
            c = bib_text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    entries.append(bib_text[at : j + 1])
                    i = j + 1
                    break
            j += 1
        else:
            break

    return entries


def parse_value(text: str, i: int) -> tuple[str, int]:
    if i >= len(text):
        return "", i

    if text[i] == "{":
        depth = 0
        start = i + 1
        i += 1
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                if depth == 0:
                    return text[start:i], i + 1
                depth -= 1
            i += 1
        return text[start:], i

    if text[i] == '"':
        i += 1
        start = i
        while i < len(text):
            if text[i] == '"' and text[i - 1] != "\\":
                return text[start:i], i + 1
            i += 1
        return text[start:], i

    start = i
    while i < len(text) and text[i] not in ",\n\r":
        i += 1
    return text[start:i].strip(), i


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(body)

    while i < n:
        while i < n and body[i] in " \t\n\r,":
            i += 1
        if i >= n:
            break

        name_start = i
        while i < n and (body[i].isalnum() or body[i] in "_-"):
            i += 1
        name = body[name_start:i].strip().lower()
        if not name:
            i += 1
            continue

        while i < n and body[i].isspace():
            i += 1
        if i >= n or body[i] != "=":
            while i < n and body[i] != ",":
                i += 1
            continue

        i += 1
        while i < n and body[i].isspace():
            i += 1

        value, i = parse_value(body, i)
        fields[name] = value.strip()

        while i < n and body[i].isspace():
            i += 1
        if i < n and body[i] == ",":
            i += 1

    return fields


def parse_entry(entry_text: str) -> tuple[str, str, dict[str, str]]:
    type_start = entry_text.find("@") + 1
    brace_open = entry_text.find("{", type_start)
    entry_type = entry_text[type_start:brace_open].strip().lower()

    inner = entry_text[brace_open + 1 : -1]

    depth = 0
    split_idx = -1
    for idx, ch in enumerate(inner):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            split_idx = idx
            break

    if split_idx == -1:
        return entry_type, inner.strip(), {}

    key = inner[:split_idx].strip()
    fields_body = inner[split_idx + 1 :]
    fields = parse_fields(fields_body)
    return entry_type, key, fields


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None

    raw = doi.strip()
    if raw.startswith("http://"):
        raw = "https://" + raw[len("http://") :]

    # Keep ACM DOI links as HTTPS URLs (only if suffix looks like DOI core).
    acm_match = ACM_DOI_RE.match(raw)
    if acm_match:
        suffix = acm_match.group(1).strip()
        if DOI_CORE_RE.match(suffix):
            return f"https://dl.acm.org/doi/{suffix}"
        return None

    # Already a doi.org link: normalize scheme and keep URL form.
    doi_org_match = re.match(r"^https?://(?:dx\.)?doi\.org/(.+)$", raw, re.IGNORECASE)
    if doi_org_match:
        suffix = doi_org_match.group(1).strip()
        if DOI_CORE_RE.match(suffix):
            return f"https://doi.org/{suffix}"
        return None

    # Other URLs are not accepted as DOI input.
    if raw.startswith("https://"):
        return None

    cleaned = DOI_PREFIX_RE.sub("", raw).strip()
    if not cleaned:
        return None
    if cleaned.startswith("https://"):
        return None
    if not DOI_CORE_RE.match(cleaned):
        return None
    return f"https://doi.org/{cleaned}"


def infer_doi_from_url(url: str | None) -> str | None:
    if not url:
        return None

    u = url.strip()
    doi = normalize_doi(u)
    if doi:
        return doi

    arxiv_match = ARXIV_ABS_RE.match(u)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        return f"https://doi.org/10.48550/arXiv.{arxiv_id}"

    return None


def ensure_backup(input_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{input_path.stem}_{ts}{input_path.suffix}"
    shutil.copy2(input_path, backup_file)
    return backup_file


def get_allowed_fields(entry_type: str) -> list[str]:
    return BASE_FIELDS + EXTRA_FIELDS_BY_TYPE.get(entry_type, [])


def prompt_non_empty(label: str, default: str | None = None) -> str:
    while True:
        if default:
            value = input(f"{label} [{default}]: ").strip()
            if not value:
                value = default
        else:
            value = input(f"{label}: ").strip()

        if value:
            return value
        print("Input cannot be empty.")


def ensure_required_fields(entry_type: str, key: str, fields: dict[str, str]) -> None:
    doi = normalize_doi(fields.get("doi"))
    if doi:
        fields["doi"] = doi

    url = (fields.get("url") or fields.get("URL") or "").strip()

    if not doi and url:
        inferred = infer_doi_from_url(url)
        if inferred:
            fields["doi"] = inferred
            doi = inferred
            print(f"[AUTO DOI] {key} -> {inferred}")

    if not doi:
        print(f"[MISSING DOI] {key}")
        doi = normalize_doi(prompt_non_empty("  Enter DOI (plain or URL, output will be HTTPS)", fields.get("doi")))
        while not doi:
            print("  DOI is still empty after normalization.")
            doi = normalize_doi(prompt_non_empty("  Enter DOI (plain or URL, output will be HTTPS)"))
        fields["doi"] = doi

    if not url:
        suggested_url = fields["doi"]
        print(f"[MISSING URL] {key}")
        url = prompt_non_empty("  Enter URL", suggested_url)
        fields["url"] = url
    else:
        fields["url"] = url

    if entry_type == "techreport":
        institution = (fields.get("institution") or "").strip()
        if not institution:
            print(f"[MISSING INSTITUTION] {key}")
            fields["institution"] = prompt_non_empty("  Enter institution")


def format_entry(entry_type: str, key: str, fields: dict[str, str], allowed_fields: list[str]) -> str:
    lines = [f"@{entry_type}{{{key},"]
    for name in allowed_fields:
        value = fields.get(name)
        if value:
            lines.append(f"  {name} = {{{value}}},")

    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}")
    return "\n".join(lines)


def process_bib(input_path: Path, output_path: Path, backup_dir: Path) -> None:
    backup_file = ensure_backup(input_path, backup_dir)

    content = input_path.read_text(encoding="utf-8")
    entries_text = split_entries(content)

    processed_entries: list[str] = []
    missing_summary: list[str] = []

    for e in entries_text:
        entry_type, key, fields = parse_entry(e)

        missing: list[str] = []
        if not normalize_doi(fields.get("doi")):
            missing.append("doi")
        if not (fields.get("url") or fields.get("URL")):
            missing.append("url")
        if entry_type == "techreport" and not fields.get("institution"):
            missing.append("institution")
        if missing:
            missing_summary.append(f"{key}: {', '.join(missing)}")

        ensure_required_fields(entry_type, key, fields)

        allowed_fields = get_allowed_fields(entry_type)
        reduced = {k: fields[k] for k in allowed_fields if k in fields and fields[k]}
        processed_entries.append(format_entry(entry_type, key, reduced, allowed_fields))

    output_path.write_text("\n\n".join(processed_entries) + "\n", encoding="utf-8")

    print(f"Backup created: {backup_file}")
    print(f"Written cleaned bib: {output_path}")
    if missing_summary:
        print("Entries that needed user input:")
        for info in missing_summary:
            print(f"- {info}")
    else:
        print("No missing DOI/URL/institution found.")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Clean BibTeX fields and normalize DOI/URL.")
    parser.add_argument(
        "input_bib",
        nargs="?",
        default=str((script_dir.parent / "reference.bib").resolve()),
        help="Path to input .bib file (default: ../reference.bib)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .bib file path (default: overwrite input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_bib).resolve()
    output_path = Path(args.output).resolve() if args.output else input_path
    backup_dir = script_dir / "backup"

    if not input_path.exists():
        raise FileNotFoundError(f"Input bib file not found: {input_path}")

    process_bib(input_path, output_path, backup_dir)


if __name__ == "__main__":
    main()
