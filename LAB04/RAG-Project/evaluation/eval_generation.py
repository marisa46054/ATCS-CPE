# Evaluate retrieval performance using data/golden_set.json.
#
# This script compares whether BM25 and reranking improve retrieval quality.
# It runs multiple retrieval methods on the same evaluation queries.
#
#     dense_only      Semantic retrieval only (baseline from Labs 1–7)
#     bm25_only       Keyword retrieval only
#     hybrid          BM25 + Dense retrieval with RRF
#     hybrid+rerank   Hybrid retrieval with cross-encoder reranking
#                     (only when USE_RERANK = True)
#
# How to interpret the results:
#   * Hybrid should perform best on partial queries and English abbreviations.
#   * Reranking should improve MRR and nDCG more than Hit@10 because it only
#     reorders retrieved results.
#   * Strong performance only on verbatim queries indicates exact word matching
#     rather than robust retrieval.


import json
import re

import config
from evaluation.eval_retrieval import load_golden_set
from src.hybrid_retriever import tokenize
from src.prompt_templates import format_context

# Number of items — LLM calls are slow, so keep this low
LIMIT = 20

# Question variant to use (natural = realistic colloquial speech, used for decisions)
VARIANT = "natural"


def word_overlap(text_a, text_b):
    """
    Proportion of words in text_a that also appear in text_b (0.0 - 1.0)

    If the answer is "copied / paraphrased" from the document, the value will be high.
    If it is made up (hallucinated), the value will be low.
    """
    words_a = set(tokenize(text_a))
    if not words_a:
        return 0.0

    words_b = set(tokenize(text_b))
    return len(words_a & words_b) / len(words_a)


def is_refusal(answer):
    """Does this answer indicate 'I don't know'?"""
    phrases = ["ไม่พบข้อมูล", "ไม่มีข้อมูล", "ไม่สามารถตอบ"]
    return any(phrase in answer for phrase in phrases)


def evaluate_one_item(rag, item):
    """Evaluate 1 question, returning a dict of scores"""
    query = item["variants"].get(VARIANT, item["question"])
    result = rag.ask(query)

    answer = result["answer"].replace(config.DISCLAIMER, "").strip()
    context = format_context(result["retrieved"])

    found_ids = {chunk["chunk_id"] for chunk in result["retrieved"]}
    correct_ids = set(item["relevant_chunk_ids"])

    return {
        "id": item["id"],
        "query": query,
        "answer": answer,
        "refused": is_refusal(answer),
        "context_hit": bool(found_ids & correct_ids),   # Found the correct chunk?
        "has_citation": bool(re.search(r"\[\d+\]", result["answer"])),
        "faithfulness": round(word_overlap(answer, context), 4),
        "correctness": round(word_overlap(answer, item["reference_answer"]), 4),
        "relevance": round(word_overlap(query, answer), 4),
        "seconds": result["timings"]["รวม"],
    }


def summarize(rows):
    """Average scores across all questions"""
    def mean(key):
        return round(sum(row[key] for row in rows) / len(rows), 4)

    return {
        "Number of items": len(rows),
        "Refusal rate": mean("refused"),
        "Correct chunk hit rate": mean("context_hit"),
        "Has citation [n]": mean("has_citation"),
        "faithfulness": mean("faithfulness"),
        "correctness": mean("correctness"),
        "relevance": mean("relevance"),
        "Average time (seconds)": mean("seconds"),
    }


def main():
    print("=== Evaluate Generation Quality ===")

    from src.rag_pipeline import RAGPipeline

    items = load_golden_set()["items"][:LIMIT]

    # Disable memory because each question must be independent
    original_memory = config.USE_MEMORY
    config.USE_MEMORY = False

    rag = RAGPipeline()
    rag.show_settings()

    if not config.USE_LLM:
        print("\n! USE_LLM = False — The answer is extracted text, not generated")
        print("  Set USE_LLM = True in config.py for meaningful numbers")

    rows = []
    for number, item in enumerate(items, start=1):
        print(f"  [{number}/{len(items)}] {item['id']}", end="\r", flush=True)
        rows.append(evaluate_one_item(rag, item))

    config.USE_MEMORY = original_memory     # Restore original setting

    summary = summarize(rows)

    print("\n\n=== Summary ===")
    for name, value in summary.items():
        print(f"  {name:26s} {value}")

    print("\n=== Answers least grounded in documents (suspected hallucinations) ===")
    for row in sorted(rows, key=lambda r: r["faithfulness"])[:3]:
        print(f"  {row['id']} ({row['faithfulness']:.3f}) {row['query'][:50]}")
        print(f"      {row['answer'][:100]}...")

    with open(config.EVAL_GENERATION_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved report to {config.EVAL_GENERATION_FILE}")


if __name__ == "__main__":
    main()
