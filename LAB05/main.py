# -*- coding: utf-8 -*-
# Main script for demonstrating 10 LLM & RAG problem scenarios
# Based on the Weight Loss & Nutrition RAG System developed in LAB04
#
# Run interactive menu:
#     python main.py
#
# Or select a problem directly via CLI argument:
#     python main.py 4
#     python main.py 0    (Run all)

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from problem01_hallucination import run as problem01
from problem02_transformer import run as problem02
from problem03_data_quality import run as problem03
from problem04_chunking import run as problem04
from problem05_metadata import run as problem05
from problem06_reranking import run as problem06
from problem07_generation import run as problem07
from problem08_config import run as problem08
from problem09_evaluation import run as problem09
from problem10_debug_scripts import run as problem10

PROBLEMS = {
    1: ("Hallucination / Context Grounding", problem01),
    2: ("Transformer / Vocabulary Mismatch / Position", problem02),
    3: ("Data Quality / Noise & Normalization", problem03),
    4: ("Chunk Size / Overlap / Mid-Word Truncation", problem04),
    5: ("Metadata Filtering & Category Isolation", problem05),
    6: ("First-Stage Ranking & Re-ranking", problem06),
    7: ("Generation Faithfulness & Numerical Distortion", problem07),
    8: ("RAG Configuration Trade-offs", problem08),
    9: ("Quantitative Evaluation & Golden Set Mismatches", problem09),
    10: ("Debug Real LAB04 Scripts & Bugs", problem10),
}


def show_menu():
    print("*" * 68)
    print("      LAB05: LLM & RAG System — 10 Problem-Based Simulations")
    print("          (Weight Loss & Nutrition Knowledge Base)")
    print("*" * 68)
    print(" 0. Run All Problems")
    for no, (name, _) in PROBLEMS.items():
        print(f"{no:2}. {name}")
    print("*" * 68)


def execute(number):
    if number == 0:
        for no, (name, func) in PROBLEMS.items():
            print("\n" + "#" * 68)
            print(f"RUNNING PROBLEM {no}: {name}")
            print("#" * 68)
            func()
        return

    if number not in PROBLEMS:
        print("Please choose a number between 0 and 10.")
        return

    name, func = PROBLEMS[number]
    print("\n" + "#" * 68)
    print(f"RUNNING PROBLEM {number}: {name}")
    print("#" * 68)
    func()


def main_loop():
    while True:
        show_menu()
        choice = input("Select a problem to simulate [0-10] or Q to exit: ").strip()

        if choice.upper() == "Q":
            print("Exiting LAB05 simulation program.")
            break

        try:
            number = int(choice)
        except ValueError:
            print("Invalid input. Please enter a number 0-10 or Q.\n")
            continue

        if number < 0 or (number not in PROBLEMS and number != 0):
            print("Please choose a number between 0 and 10.\n")
            continue

        execute(number)
        print("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg.upper() == "Q":
            sys.exit(0)
        try:
            execute(int(arg))
        except ValueError:
            print("Please enter a number 0-10 or Q")
    else:
        main_loop()
