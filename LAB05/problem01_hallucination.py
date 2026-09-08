# -*- coding: utf-8 -*-
# Problem 01: Hallucination / No evidence in Retrieved Context
# Uses qa_looseweight.txt as a real Knowledge Base
from data_loader import load_qa

DOCS = load_qa()


import re

STOP_WORDS = {
    "how", "what", "where", "when", "which", "who", "whom", "why",
    "does", "should", "the", "and", "for", "with", "can", "you", "are",
    "about", "is", "of", "a", "an", "do", "i"
}


def retrieve(question, top_k=3):
    q_words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()) if w not in STOP_WORDS]
    scored = []
    for d in DOCS:
        d_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", d["text"].lower()))
        score = sum(1 for w in q_words if w in d_words)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def bad_generate(question, context):
    if not context:
        # Simulated failure: the LLM fabricates an answer even without evidence in the KB
        return ("The capital city of France is Paris, located on the Seine River. "
                "(Fabricated / Hallucinated: Answered from model memory despite empty KB context)")
    return context[0]["answer"]


def grounded_generate(question, context):
    if not context:
        # Solution: Refusal with standard fallback message (from config.NO_CONTEXT_MESSAGE)
        return "Sorry, no relevant information found in the knowledge base."
    return context[0]["answer"]


def run():
    print("=" * 65)
    print("PROBLEM 01: Hallucination / Answer without Supporting Context")
    print("Code Reference: src/generator.py (lines 78-83) & src/prompt_templates.py (lines 12-20)")
    print("=" * 65)

    q_in_kb = "How does sustainable weight loss work in principle?"
    q_out_of_kb = "What is the capital city of France?"



    for label, q in [("In KB scope", q_in_kb), ("Out of KB scope", q_out_of_kb)]:
        ctx = retrieve(q)
        print(f"\n--- Query ({label}): {q}")
        print("Retrieved:", [d["question"] for d in ctx] or "Not found (Empty Context)")
        print("Bad answer   (Hallucinated) :", bad_generate(q, ctx))
        print("Fixed answer (Grounded Guard):", grounded_generate(q, ctx))

    print("\n[Analysis]")
    print("- Cause: The LLM tries to answer even when the Retrieved Context has no supporting evidence.")
    print("         In medical/nutrition domains, hallucinations can give dangerous health advice.")
    print("- Verification: Check if retrieved chunks are empty (len(chunks) == 0) or below similarity threshold.")
    print("- Solution in LAB04:")
    print("  1. In src/generator.py: if not chunks: return {'answer': config.NO_CONTEXT_MESSAGE}")
    print("  2. In src/prompt_templates.py: System prompt strictly forbids outside knowledge and enforces")
    print("     answering config.NO_CONTEXT_MESSAGE when data is insufficient.")


if __name__ == "__main__":
    run()

