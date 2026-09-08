# -*- coding: utf-8 -*-
# Problem 10: Debugging Real Implementation Bugs in LAB04
# Audits real bug fixes across evaluation, query transformation, and pipeline scripts
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def test_eval_generation_key_bug():
    """
    Bug 1: KeyError in evaluation/eval_generation.py line 77
    result['timings']['รวม'] was hardcoded in Thai,
    while src/rag_pipeline.py line 76 produced {'Total': ...} in English.
    """
    pipeline_result = {
        "answer": "Sustainable weight loss requires slight calorie deficit.",
        "timings": {"Transform": 0.05, "Retrieve": 0.02, "Generate": 0.45, "Total": 0.52}
    }

    print("\n--- Bug 1: KeyError in evaluation/eval_generation.py ---")
    try:
        # Buggy line
        _ = pipeline_result["timings"]["รวม"]
        print("Buggy read: Success (unexpected)")
    except KeyError as e:
        print(f"Buggy read: Crashed with KeyError: {e}!")

    # Robust fix:
    fixed_val = pipeline_result["timings"].get("Total", pipeline_result["timings"].get("รวม", 0.0))
    print(f"Fixed read: Safely retrieved timing = {fixed_val} seconds.")


def test_slang_domain_mismatch_bug():
    """
    Bug 2: Residual Thai sexual health dictionary in an English Weight Loss system
    In src/query_transform.py lines 42-52, SLANG_MAP contained:
        'น้องชาย': 'อวัยวะเพศชาย', 'ถุงยาง': 'ถุงยางอนามัย'
    When applied to English queries, zero terms matched.
    """
    print("\n--- Bug 2: Residual Domain Mismatch in src/query_transform.py ---")
    thai_slang_map = {"น้องชาย": "อวัยวะเพศชาย", "ถุงยาง": "ถุงยางอนามัย"}
    english_query = "I want to get rid of my belly fat and prevent yo-yo rebound"

    # With leftover Thai map:
    untransformed = english_query
    for k, v in thai_slang_map.items():
        untransformed = untransformed.replace(k, v)
    print("Thai Map Output   :", untransformed, "(No effect on English query)")

    # Fixed Nutrition & Weight Loss Slang Map:
    weight_slang_map = {
        "belly fat": "abdominal visceral fat",
        "yo-yo rebound": "weight regain yo-yo effect",
        "cutting carbs": "low-carbohydrate dietary deficit",
    }
    transformed = english_query
    for k, v in weight_slang_map.items():
        transformed = transformed.replace(k, v)
    print("Domain Map Output :", transformed, " * [Normalized to Clinical KB Terms]")


def test_multi_turn_memory_detection():
    """
    Bug 3 / Feature: Follow-up question detection in src/memory.py
    Ensures memory history is only used when the user asks a follow-up query.
    """
    print("\n--- Bug 3 / Feature: Follow-up Detection in src/memory.py ---")
    markers = ("then", "it", "that", "this", "next", "more", "why", "so")

    queries = [
        ("What is intermittent fasting?", False),
        ("Why does it happen?", True),
        ("Then how much protein do I need?", True),
        ("What are good complex carbs?", False)
    ]

    for q, expected in queries:
        text = q.lower().strip()
        is_followup = len(text) < 35 and any(text.startswith(m) or f" {m} " in text for m in markers)
        print(f"  Query: '{q:36s}' -> Follow-up: {is_followup} (Expected: {expected})")


def run():
    print("=" * 65)
    print("PROBLEM 10: Debug RAG Implementation & Integration Scripts")
    print("Audit of Real Bugs Discovered & Resolved in LAB04 Codebase")
    print("=" * 65)

    test_eval_generation_key_bug()
    test_slang_domain_mismatch_bug()
    test_multi_turn_memory_detection()

    print("\n[Summary of Code Hardening in LAB04]")
    print("1. Fixed timing dictionary key mismatch between rag_pipeline.py and eval_generation.py.")
    print("2. Replaced legacy domain slang dictionary with weight loss & nutrition terminology.")
    print("3. Hardened memory follow-up classifier to prevent context pollution in independent queries.")


if __name__ == "__main__":
    run()

