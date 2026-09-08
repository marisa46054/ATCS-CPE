# -*- coding: utf-8 -*-
# Problem 06: First-stage Retrieval Ranking Bottlenecks & Re-ranking
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_loader import load_qa

GENERIC_TERMS = ["weight", "loss", "calorie", "diet"]
SPECIFIC_TERMS = ["yo-yo", "rebound", "restrictive", "starvation"]


def first_stage_score(doc):
    text = (doc["question"] + " " + doc["answer"]).lower()
    return sum(text.count(t) for t in GENERIC_TERMS)


def rerank_score(doc):
    text = (doc["question"] + " " + doc["answer"]).lower()
    base = first_stage_score(doc)
    # Re-ranking awards deep semantic matching on high-information specific keywords
    specific_boost = sum(5 * text.count(t) for t in SPECIFIC_TERMS)
    return base + specific_boost


def run():
    print("=" * 65)
    print("PROBLEM 06: The Most Relevant Document Ranked Low in First-Stage Search")
    print("Code Reference: src/rerankers.py (lines 21-49) & src/hybrid_retriever.py (lines 170-187)")
    print("=" * 65)

    data = load_qa()
    query = "What causes the yo-yo effect and weight rebound after restrictive dieting?"
    print(f"User Query: '{query}'\n")

    # First stage: Bi-encoder / BM25 lexical frequency
    first_stage_ranked = sorted(data, key=first_stage_score, reverse=True)[:6]

    print("--- 1. Before Re-ranking (Top 6 from First-stage Retrieval) ---")
    for rank, d in enumerate(first_stage_ranked, 1):
        score = first_stage_score(d)
        is_target = "yo-yo" in d["question"].lower()
        star = " * [TARGET]" if is_target else ""
        print(f"  Rank {rank} (score={score:2d}): {d['question'][:60]}...{star}")

    # Second stage: Cross-encoder Re-ranking
    reranked = sorted(first_stage_ranked, key=rerank_score, reverse=True)

    print("\n--- 2. After Re-ranking (Cross-Attention on Specific Terms) ---")
    for rank, d in enumerate(reranked, 1):
        score = rerank_score(d)
        is_target = "yo-yo" in d["question"].lower()
        star = " * [TARGET]" if is_target else ""
        print(f"  Rank {rank} (score={score:2d}): {d['question'][:60]}...{star}")

    print("\n[Analysis]")
    print("- Cause: First-stage retrieval (Bi-encoder/BM25) overweights ubiquitous domain keywords")
    print("         ('weight', 'loss', 'diet'), pushing precise needle-in-haystack documents down.")
    print("- Impact: If TOP_K=3, the target document at Rank 5 is completely omitted from LLM context!")
    print("- Verification: In outputs/eval_retrieval.json, compare hit@1 vs hit@10 and evaluate MRR gains.")
    print("- Solution in LAB04:")
    print("  Implemented Cross-Encoder reranker (BAAI/bge-reranker-v2-m3 in src/rerankers.py).")
    print("  First stage fetches CANDIDATE_K=20 candidates, and Cross-Encoder reorders them to Top-3.")


if __name__ == "__main__":
    run()

