# -*- coding: utf-8 -*-
# Load and parse qa_looseweight.txt into records shared by problem01-10.
#
# File format: each entry is separated by blank lines
#     [Category: <category>]
#     Q: <question>
#     A: <answer>
# Lines starting with # are file header/comments and are skipped.

import os
import re

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_looseweight.txt")

_HEADER_RE = re.compile(r"\[Category:\s*(.+?)\]")


def load_qa(path=DATA_PATH):
    """
    Returns a list of dicts: id, category, question, answer, text, line_no
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge base file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    category = "Uncategorized"
    question = None
    question_line = None

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("[Category:"):
            m = _HEADER_RE.match(line)
            category = m.group(1).strip() if m else line.strip("[]").replace("Category:", "").strip()
        elif line.startswith("Q:"):
            question = line[2:].strip()
            question_line = line_no
        elif line.startswith("A:") and question:
            answer = line[2:].strip()
            records.append({
                "id": len(records),
                "category": category,
                "question": question,
                "answer": answer,
                "text": f"{question} {answer}",
                "line_no": question_line,
            })
            question = None

    return records


def categories(entries=None):
    entries = entries if entries is not None else load_qa()
    return sorted(set(e["category"] for e in entries))


if __name__ == "__main__":
    data = load_qa()
    print("Total Q&A count:", len(data))
    cats = categories(data)
    print("Number of categories:", len(cats))
    print("Categories list:")
    for i, c in enumerate(cats, 1):
        print(f"  {i}. {c}")
    print("\nFirst entry example:")
    print("  Category:", data[0]["category"])
    print("  Q:", data[0]["question"])
    print("  A:", data[0]["answer"][:100] + "...")
