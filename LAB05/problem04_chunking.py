# -*- coding: utf-8 -*-
# Problem 04: Chunking Issues (Chunk Size / Overlap / Mid-Word Truncation)
# Directly reproduces chunking behavior from LAB04 src/text_splitter.py
from data_loader import load_qa


def char_chunk(text, size, overlap=0):
    """Exact logic from LAB04 src/text_splitter.py"""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def word_chunk(words, size, overlap=0):
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def run():
    print("=" * 65)
    print("PROBLEM 04: Chunk Size, Overlap & Mid-Word Truncation")
    print("Code Reference: src/text_splitter.py (lines 9-52) & config.py (lines 53-54)")
    print("=" * 65)

    data = load_qa()
    sample = data[0]
    full_text = f"Question: {sample['question']} Answer: {sample['answer']}"

    print(f"Sample Q&A length: {len(full_text)} characters ({len(full_text.split())} words)")

    # 1. Real bug from LAB04: Character-level splitting cuts words in half!
    print("\n--- 1. LAB04 Character-Level Slicing (CHUNK_SIZE=400, OVERLAP=50) ---")
    chunks = char_chunk(full_text, 400, 50)
    for idx, c in enumerate(chunks):
        print(f"  Chunk [{idx}] (len={len(c)}):")
        print(f"    Start: {repr(c[:40])}")
        print(f"    End  : {repr(c[-40:])}")

    print("\n  [!] Notice how the boundary split word 'behavioral':")
    print(f"      End of Chunk 0  : {repr(chunks[0][-15:])}")
    print(f"      Start of Chunk 1: {repr(chunks[1][:15])}")
    print("      'behavioral' is literally cut into 'lo' and 'vioral'!")

    # 2. Child chunk missing Question header
    print("\n--- 2. Question Context Loss in Subsequent Chunks ---")
    print("  Chunk 0 has Question prefix:", "Question:" in chunks[0])
    print("  Chunk 1 has Question prefix:", "Question:" in chunks[1])
    print("  -> If a user searches using question keywords, Chunk 1 cannot be found")
    print("     because Chunk 1 has NO question header!")

    # 3. Chunk size comparison
    words = full_text.split()
    print("\n--- 3. Effect of Chunk Size on Granularity ---")
    print("Large Chunk (size=80 words) : Preserves full context, but embedding may be diluted.")
    print("Small Chunk (size=15 words) : Highly focused, but splits sentences mid-thought.")
    print("Word-Aware Chunk + Overlap  : Clean word boundaries + seamless continuity.")

    print("\n[Analysis]")
    print("- Cause: Blind character slicing text[start:start+size] ignores word boundaries.")
    print("         Only the first chunk retains 'Question:', while subsequent chunks lose topic metadata.")
    print("- Verification: Inspect outputs/chunks.json for severed words and missing question headers.")
    print("- Solution in LAB04:")
    print("  1. Use whitespace/sentence-aware token splitting (RecursiveCharacterTextSplitter).")
    print("  2. Prepend 'Question: <q>' to every child chunk (part_idx > 0) to maintain retrieval signals.")


if __name__ == "__main__":
    run()

