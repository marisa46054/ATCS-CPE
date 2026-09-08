# -*- coding: utf-8 -*-
# Problem 07: Retrieval Correct, but Generation Distorts Context (Faithfulness)
# Uses real data from qa_looseweight.txt (Plate proportions for healthy weight loss)
from data_loader import load_qa


def find_entry(data):
    return next(d for d in data if "calorie counting" in d["question"].lower())


def bad_generator(context):
    # Simulated failure: The generator modifies critical health advice and nutritional ratios
    return context.replace(
        "half a plate of vegetables, a quarter protein, and a quarter complex carbohydrates",
        "three-quarters meat and fat, completely cutting out all vegetables and carbohydrates"
    ).replace("equally effective", "much more dangerous")


def grounded_generator(context):
    # Grounded answer adhering strictly to evidence with citation
    return (
        "Calorie counting is not mandatory for everyone. An effective alternative is controlling "
        "plate proportions: half a plate of vegetables, a quarter protein, and a quarter "
        "complex carbohydrates [1]."
    )


def run():
    print("=" * 65)
    print("PROBLEM 07: Retrieval is Correct, but Generation Distorts Facts (Faithfulness)")
    print("Code Reference: src/generator.py (lines 75-101) & src/prompt_templates.py (lines 12-28)")
    print("=" * 65)

    data = load_qa()
    entry = find_entry(data)
    context = entry["answer"]

    print("Question:", entry["question"])
    print("\n--- Retrieved Ground Truth Context ---")
    print(context)

    print("\n--- Bad Generation (Distorts Nutritional Proportions) ---")
    print(bad_generator(context))

    print("\n--- Grounded Generation (Strictly Fact-Aligned with Citations) ---")
    print(grounded_generator(context))

    print("\n[Analysis]")
    print("- Cause: Retrieval succeeded in finding the right chunk, but the generator paraphrased")
    print("         or altered essential nutritional numbers/ratios due to parametric bias.")
    print("- Risk: Misleading nutritional instructions could harm users with dietary conditions.")
    print("- Verification: Measure 'faithfulness' score in evaluation/eval_generation.py (word overlap).")
    print("- Solution in LAB04:")
    print("  1. Low LLM_TEMPERATURE = 0.2 in config.py to reduce creative randomness.")
    print("  2. System prompt rules: 'Use only info from Reference Data. Do not add outside knowledge.'")
    print("  3. Require inline citations [n] so statements are traceable back to exact chunks.")


if __name__ == "__main__":
    run()

