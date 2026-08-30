#!/usr/bin/env python3
"""Quick sanity audit for MMTabReal/MMTBench split alignment.

Checks lexical consistency between question and table title/headers,
and reports potentially mismatched examples.
"""

import argparse
import json
import re
import string
from pathlib import Path

import datasets

STOP_WORDS = {
    "what", "which", "who", "when", "where", "how", "many", "much", "is", "are", "was", "were",
    "the", "a", "an", "in", "on", "of", "to", "for", "and", "or", "with", "did", "does", "do",
    "from", "at", "by", "that", "this", "these", "those", "it", "its", "as", "than", "most", "least",
    "all", "shown", "table",
}

KEYWORD_RULES = [
    ("flag", ["flag", "country", "dynasty", "era", "background color"]),
    ("railroad", ["railroad", "railway", "successor", "line"]),
    ("population", ["population", "inhabitants"]),
    ("album", ["album", "song", "track"]),
]


def normalize_words(text: str):
    words = [w.strip(string.punctuation) for w in (text or "").lower().split()]
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def audit_split(dataset_path: str):
    ds = datasets.load_from_disk(dataset_path)
    suspicious = []

    for i, item in enumerate(ds):
        q = (item.get("question") or "").lower()
        table = item.get("table") or {}
        title = (table.get("page_title") or "").lower()
        headers = " ".join((h or "").lower() for h in table.get("header", []))
        corpus = f"{title} {headers}".strip()

        content_words = normalize_words(q)
        overlap = sum(1 for w in content_words if w in corpus)

        reason = None
        if len(content_words) >= 4 and overlap == 0:
            reason = "zero-overlap"
        else:
            for tag, kws in KEYWORD_RULES:
                if tag in q and not any(kw in corpus for kw in kws):
                    reason = f"{tag}-mismatch"
                    break

        if reason is not None:
            suspicious.append(
                {
                    "idx": i,
                    "id": item.get("id"),
                    "reason": reason,
                    "question": item.get("question"),
                    "page_title": table.get("page_title"),
                    "headers": table.get("header", []),
                }
            )

    return len(ds), suspicious


def main():
    parser = argparse.ArgumentParser(description="Audit MMTBench/MMTabReal question-table alignment")
    parser.add_argument("--dataset_path", required=True, help="Path like /.../MMTBench/hf_dataset/mmtabreal_VQ")
    parser.add_argument("--out", default=None, help="Optional output JSON file for suspicious examples")
    args = parser.parse_args()

    total, suspicious = audit_split(args.dataset_path)
    pct = (len(suspicious) / total * 100.0) if total else 0.0

    print(f"total: {total}")
    print(f"suspicious: {len(suspicious)} ({pct:.2f}%)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump({"total": total, "suspicious": suspicious}, f, indent=2)
        print(f"wrote: {out_path}")

    preview = suspicious[:20]
    for ex in preview:
        print("-" * 60)
        print(f"idx: {ex['idx']} | id: {ex['id']} | reason: {ex['reason']}")
        print(f"q: {ex['question']}")
        print(f"title: {ex['page_title']}")


if __name__ == "__main__":
    main()
