# -*- coding: utf-8 -*-
# Problem 08: RAG Configuration Trade-offs & Pipeline Orchestration
# Directly reflects configuration architecture from LAB04 config.py & src/rag_pipeline.py
from data_loader import load_qa

CONFIG = {
    "KB_SOURCE": "qa_looseweight.txt",
    "USE_HYBRID": True,            # Search with BM25 along with dense
    "USE_RERANK": False,           # Cross-encoder reranker
    "USE_QUERY_TRANSFORM": False,   # Rewrite / multi-query before search
    "USE_MEMORY": True,            # Conversational multi-turn history
    "USE_LLM": True,               # LLM generator vs raw chunks
    "SHOW_SOURCES": True,          # Display source chunk references
}

PROFILES = {
    "1. High-Speed / Edge (Low Latency)": {
        "USE_HYBRID": False, "USE_RERANK": False, "USE_QUERY_TRANSFORM": False,
        "USE_MEMORY": False, "USE_LLM": False, "Latency": "~18 ms", "Cost": "$0"
    },
    "2. Standard Balanced (Current Default)": {
        "USE_HYBRID": True, "USE_RERANK": False, "USE_QUERY_TRANSFORM": False,
        "USE_MEMORY": True, "USE_LLM": True, "Latency": "~600-900 ms", "Cost": "1 LLM call"
    },
    "3. Maximum Precision (Clinical / Strict)": {
        "USE_HYBRID": True, "USE_RERANK": True, "USE_QUERY_TRANSFORM": True,
        "USE_MEMORY": True, "USE_LLM": True, "Latency": "~2500 ms", "Cost": "2 LLM calls + GPU rerank"
    }
}


def run():
    print("=" * 65)
    print("PROBLEM 08: Component Configuration & Architectural Trade-offs")
    print("Code Reference: config.py (lines 18-25) & src/rag_pipeline.py (lines 28-79)")
    print("=" * 65)

    data = load_qa()
    print(f"Loaded Knowledge Base: {CONFIG['KB_SOURCE']} ({len(data)} entries)")
    print("\n--- Current Active Pipeline Stages (from config.py) ---")
    step = 1
    if CONFIG["USE_QUERY_TRANSFORM"]:
        print(f"  Stage {step}: Query Transformation (LLM multi_query / rewrite)")
        step += 1
    else:
        print(f"  Stage {step}: Raw User Query (Query transform bypassed)")
        step += 1

    if CONFIG["USE_HYBRID"]:
        print(f"  Stage {step}: Hybrid Retrieval (BM25 + FAISS Dense with RRF fusion)")
        step += 1
    else:
        print(f"  Stage {step}: Dense Retrieval Only (FAISS IndexFlatIP)")
        step += 1

    if CONFIG["USE_RERANK"]:
        print(f"  Stage {step}: Cross-Encoder Re-ranking (BAAI/bge-reranker-v2-m3)")
        step += 1
    else:
        print(f"  Stage {step}: Re-ranking Disabled (Pass first-stage Top-K directly)")
        step += 1

    if CONFIG["USE_LLM"]:
        print(f"  Stage {step}: LLM Generation (Prompt template grounded in context)")
        step += 1
    else:
        print(f"  Stage {step}: Raw Text Extraction (No LLM generation)")
        step += 1

    if CONFIG["USE_MEMORY"]:
        print(f"  Stage {step}: Conversation Memory (Sliding window of 6 turns)")
        step += 1

    print("\n--- Architectural Profiles & Performance Trade-offs ---")
    for name, p in PROFILES.items():
        print(f"\n[{name}]")
        print(f"  Hybrid: {p['USE_HYBRID']} | Rerank: {p['USE_RERANK']} | Transform: {p['USE_QUERY_TRANSFORM']}")
        print(f"  Memory: {p['USE_MEMORY']} | LLM: {p['USE_LLM']}")
        print(f"  Estimated Latency: {p['Latency']} | Cost/Overhead: {p['Cost']}")

    print("\n[Analysis]")
    print("- Cause: Enabling all RAG features creates compounding latency and API costs.")
    print("         Disabling advanced features risks lower retrieval precision on complex queries.")
    print("- Verification: In outputs/eval_retrieval.json, compare timing (ms_per_query) vs MRR/Hit rate.")
    print("- Solution in LAB04: Centralized configuration in config.py enables modular component toggling.")


if __name__ == "__main__":
    run()

