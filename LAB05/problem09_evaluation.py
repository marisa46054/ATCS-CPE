# -*- coding: utf-8 -*-
# Problem 09: Quantitative Evaluation & The Golden Set Chunk Mismatch Bug
# Analyzes real metrics from LAB04 evaluation/metrics.py & outputs/eval_retrieval.json
import math
from data_loader import load_qa

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
EVAL_K_VALUES = [1, 3, 5, 10]


def calc_metrics(retrieved, relevant, k_values=EVAL_K_VALUES):
    rel_set = set(relevant)
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel_set:
            mrr = 1.0 / rank
            break

    hits = {f"hit@{k}": (1.0 if rel_set & set(retrieved[:k]) else 0.0) for k in k_values}
    return {"mrr": mrr, **hits}


def run():
    print("=" * 65)
    print("PROBLEM 09: Quantitative Evaluation Metrics & Golden Set Mapping Bug")
    print("Code Reference: evaluation/metrics.py, eval_retrieval.py & build_golden_set.py")
    print("=" * 65)

    data = load_qa()
    full_text = " ".join(d["text"] for d in data)
    print(f"Knowledge Base Size: {len(full_text)} characters across {len(data)} Q&A pairs")
    print(f"Chunk Configuration : CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")

    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunk_count = math.ceil((len(full_text) - CHUNK_OVERLAP) / step)
    print(f"Generated Index Size: ~{chunk_count} chunks")

    print("\n--- 1. The Real Discovery in LAB04 outputs/eval_retrieval.json ---")
    print("In LAB04, the initial retrieval evaluation returned abnormally low scores:")
    print("  Dense Only  : Hit@1 = 0.0128 (1.28%) | MRR = 0.0349 | Hit@10 = 0.0833")
    print("  BM25 Only   : Hit@1 = 0.0064 (0.64%) | MRR = 0.0340 | Hit@10 = 0.0833")
    print("  Hybrid (RRF): Hit@1 = 0.0128 (1.28%) | MRR = 0.0429 | Hit@10 = 0.1218")

    print("\n--- 2. Root Cause Analysis: Golden Set Index Off-by-One Mismatch ---")
    print("When building the index:")
    print("  QA 0 -> Chunk 0 (Part 0, contains Question + Answer part 1)")
    print("       -> Chunk 1 (Part 1, contains Answer part 2)")
    print("However, data/golden_set.json was generated with:")
    print("  Item 'g0001' -> relevant_chunk_ids: [1]  <-- Mapped ONLY to part 1!")
    print("When the retriever retrieved Chunk 0 (which contains the question and highest score),")
    print("the evaluator checked: is Chunk 0 in [1]? -> NO! (Result = MISS)")

    print("\n--- 3. Simulation of Before vs After Bug Fix ---")
    retrieved_chunks = [0, 2, 4, 6, 8]  # System retrieved chunk 0 at Rank 1

    buggy_relevant = [1]      # Mapped to part 1 only
    fixed_relevant = [0, 1]   # Correct: QA 0 spans both chunks 0 and 1

    score_buggy = calc_metrics(retrieved_chunks, buggy_relevant)
    score_fixed = calc_metrics(retrieved_chunks, fixed_relevant)

    print(f"Retrieved Top-5 by Model: {retrieved_chunks}")
    print(f"Buggy Ground Truth [1]  : Hit@1={score_buggy['hit@1']:.2f}, Hit@3={score_buggy['hit@3']:.2f}, MRR={score_buggy['mrr']:.4f}")
    print(f"Fixed Ground Truth [0,1]: Hit@1={score_fixed['hit@1']:.2f}, Hit@3={score_fixed['hit@3']:.2f}, MRR={score_fixed['mrr']:.4f}")

    print("\n[Analysis]")
    print("- Cause: Synthetic evaluation sets generated with flawed index mappings create false")
    print("         impressions of system failure when the retriever is actually performing well.")
    print("- Verification: Cross-reference chunk IDs in outputs/chunks.json with data/golden_set.json.")
    print("- Solution in LAB04:")
    print("  In build_golden_set.py, ensure by_qa[qa_id] maps to ALL chunks belonging to that Q&A pair,")
    print("  and evaluate both Hit@k and position-discounted nDCG@k.")


if __name__ == "__main__":
    run()

