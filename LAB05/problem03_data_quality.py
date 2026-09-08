# -*- coding: utf-8 -*-
# Problem 03: Data Quality (Duplicate / Noise / Multi-line Parsing / Normalization)
# Simulates data noise, multi-line truncation bugs, and checks quality of qa_looseweight.txt
import re
from data_loader import load_qa


def make_noisy_samples(data, n=2):
    base = [d["question"] for d in data[:n]]
    raw = []
    for q in base:
        raw.append(q)                                      # Original
        raw.append(q)                                      # Exact duplicate
        raw.append("   " + q + "   \n")                     # Extra whitespace & newline
        raw.append(q.lower().replace(" ", "_") + "???")    # Symbol noise / case variation
    raw.append("")                                         # Empty broken row
    return raw


def normalize(text):
    text = text.lower()
    text = re.sub(r"[_\-?!@#]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run():
    print("=" * 65)
    print("PROBLEM 03: Data Quality (Duplicates, Noise & Multi-line Parsing)")
    print("Code Reference: src/document_loader.py (lines 25-46)")
    print("=" * 65)

    data = load_qa()
    raw = make_noisy_samples(data)

    print("--- 1. Simulated Dirty / Scraped Data ---")
    for idx, item in enumerate(raw, 1):
        print(f"  [{idx}] {repr(item)}")

    normalized = [normalize(x) for x in raw if x.strip()]
    unique = list(dict.fromkeys(normalized))

    print("\n--- 2. After Normalization & Deduplication ---")
    for idx, item in enumerate(unique, 1):
        print(f"  [{idx}] {repr(item)}")

    print(f"\nCount before: {len(raw)} items -> Count after: {len(unique)} clean items")

    print("\n--- 3. Dataset Audit on qa_looseweight.txt ---")
    all_q = [d["question"] for d in data]
    norm_q = [normalize(q) for q in all_q]
    dup_count = len(norm_q) - len(set(norm_q))
    print(f"Total entries in qa_looseweight.txt: {len(data)}")
    print(f"Duplicate questions detected: {dup_count}")

    print("\n--- 4. Multi-line Answer Vulnerability in src/document_loader.py ---")
    print("In LAB04 src/document_loader.py:")
    print("    elif line.startswith('A:') and question:")
    print("        records.append({... 'answer': line[2:].strip()})")
    print("        question = None")
    print("Vulnerability: If the source answer has multiple paragraphs/newlines,")
    print("subsequent lines are skipped because 'question' has already been reset to None!")
    print("Solution: Accumulate lines into an answer buffer until the next [Category:] or EOF.")

    print("\n[Analysis]")
    print("- Cause: Uncleaned source files cause duplicate embeddings and wasted vector space.")
    print("         Naive single-line parsers truncate multi-line answers.")
    print("- Verification: Run automated audits checking duplicate hashes and raw line counts.")
    print("- Solution in LAB04: Applied block-based parsing and string normalization.")


if __name__ == "__main__":
    run()

