# -*- coding: utf-8 -*-
# Problem 02: Vocabulary Mismatch & Token Position (Transformer vs BoW)
# Uses real questions from qa_looseweight.txt
from data_loader import load_qa


def bow(text):
    result = {}
    for token in text.lower().replace("?", "").replace(",", "").split():
        result[token] = result.get(token, 0) + 1
    return result


def with_position(text):
    return [(i, token) for i, token in enumerate(text.split())]


def run():
    print("=" * 65)
    print("PROBLEM 02: Vocabulary Mismatch & Token Ordering (Position)")
    print("Code Reference: src/embedding_model.py & src/vector_store.py")
    print("=" * 65)

    data = load_qa()
    # Find plateau question
    kb_item = next(d for d in data if "plateaus" in d["question"])
    user_query = "Why did my scale stop dropping even though I am dieting hard?"

    print(f"KB Formal Question : {kb_item['question']}")
    print(f"User Query (Slang) : {user_query}")

    bow_kb = bow(kb_item["question"])
    bow_user = bow(user_query)

    common = set(bow_kb) & set(bow_user)
    print("\nBoW (KB)  :", bow_kb)
    print("BoW (User):", bow_user)
    print("Exact-token overlap:", common or "None (0 overlapping meaningful tokens)")
    print("-> Both questions have identical semantic intent (weight loss plateau),")
    print("   but Bag-of-Words / Keyword search fails due to Vocabulary Mismatch!")

    print("\n--- Example: Effect of Token Order on Meaning (Position) ---")
    s1 = "eat carbs before exercise"
    s2 = "exercise before eat carbs"
    print("Sentence 1 (Pre-workout meal) :", with_position(s1))
    print("Sentence 2 (Post-workout meal):", with_position(s2))
    print("BoW is identical?             :", bow(s1) == bow(s2))
    print("-> BoW treats both sentences as 100% identical despite opposite nutritional advice!")

    print("\n[Analysis]")
    print("- Cause: Traditional lexical/BoW models ignore vocabulary variation and discard word order.")
    print("- Verification: Compare keyword overlap vs dense semantic similarity scores.")
    print("- Solution in LAB04:")
    print("  Uses Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2 in src/embedding_model.py).")
    print("  Self-Attention + Positional Encodings project queries and documents into dense semantic space,")
    print("  allowing matches on conceptual intent rather than identical lexical tokens.")


if __name__ == "__main__":
    run()

