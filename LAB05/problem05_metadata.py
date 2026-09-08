# -*- coding: utf-8 -*-
# Problem 05: Metadata Filtering & Context Isolation
# Uses real data from qa_looseweight.txt across 11 categories
from data_loader import load_qa

QUERY = "How much protein should I eat per day?"


def score(query, text):
    words = query.lower().replace("?", "").split()
    return sum(w in text.lower() for w in words)


def search(data, query, category=None):
    docs = data if category is None else [d for d in data if d["category"] == category]
    if not docs:
        return None
    return max(docs, key=lambda d: score(query, d["question"] + " " + d["answer"]))


def run():
    print("=" * 65)
    print("PROBLEM 05: Metadata Filtering & Context Isolation")
    print("Code Reference: src/retriever.py & src/hybrid_retriever.py")
    print("=" * 65)

    data = load_qa()

    print(f"Query: '{QUERY}'\n")

    # 1. Search without Metadata filtering
    unfiltered = search(data, QUERY)
    print("1. Without Metadata Filtering (Global Search):")
    print(f"   Category : [{unfiltered['category']}]")
    print(f"   Question : {unfiltered['question']}")
    print(f"   Answer   : {unfiltered['answer'][:120]}...")

    # 2. Search filtered for 'Groups Requiring Special Caution'
    caution_cat = "Groups Requiring Special Caution"
    filtered_caution = search(data, "protein kidney medical condition", category=caution_cat)
    print(f"\n2. Filtered by Metadata Category = '{caution_cat}':")
    if filtered_caution:
        print(f"   Category : [{filtered_caution['category']}]")
        print(f"   Question : {filtered_caution['question']}")
        print(f"   Answer   : {filtered_caution['answer'][:120]}...")

    # 3. Search filtered for 'Choosing Proteins'
    protein_cat = "Choosing Proteins"
    filtered_protein = search(data, QUERY, category=protein_cat)
    print(f"\n3. Filtered by Metadata Category = '{protein_cat}':")
    if filtered_protein:
        print(f"   Category : [{filtered_protein['category']}]")
        print(f"   Question : {filtered_protein['question']}")
        print(f"   Answer   : {filtered_protein['answer'][:120]}...")

    print("\n[Analysis]")
    print("- Cause: Semantic similarity evaluates textual alignment, but is blind to audience constraints,")
    print("         user personas, or domain categories without explicit metadata.")
    print("- Risk: In health/diet applications, giving general high-protein advice to a kidney patient")
    print("        can have severe medical consequences.")
    print("- Verification: Compare retrieval accuracy and clinical safety metrics across user segments.")
    print("- Solution in LAB04:")
    print("  Attach 'category' and 'line_no' to every chunk in src/text_splitter.py and index_meta.py,")
    print("  allowing pre-retrieval or post-retrieval filtering based on user profile and metadata.")


if __name__ == "__main__":
    run()

