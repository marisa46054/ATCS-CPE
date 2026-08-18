

# prompt_templates.py
# Store all prompt templates in one place.
# Makes prompts easier to manage, compare, and update.
# Answers must be based only on the retrieved context.
# Inline citations are required for traceable and verifiable responses.


import config

SYSTEM_PROMPT = """You are an assistant providing weight loss and nutrition information. Answer based only on the provided "Reference Data".

Rules:
1. Use only information from "Reference Data". Do not add outside knowledge.
2. If there is not enough info, answer "{no_context}". Do not guess.
3. Cite source numbers like [1] [2] at the end of the sentence using that data.
4. Use polite, direct, non-judgmental language.
5. If it is a severe or emergency symptom, advise seeing a doctor immediately.
6. Keep answers concise, no more than 5-6 sentences."""

USER_PROMPT = """{history}Reference Data:
{context}

User Question: {question}

Answer using only the reference data above, with citations [n]"""


def format_context(chunks, max_chars=6000):
    """
    Format chunks into numbered reference blocks

    max_chars prevents prompt from exceeding context window — best chunks come first
    so truncating removes the least relevant pieces
    """
    blocks, used = [], 0
    for i, chunk in enumerate(chunks, start=1):
        block = f"[{i}] {chunk.get('answer') or chunk.get('text', '')}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_messages(question, chunks, history=""):
    """Assemble messages list to send to LLM"""
    history_block = f"Previous conversation:\n{history}\n\n" if history else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(no_context=config.NO_CONTEXT_MESSAGE)},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                history=history_block,
                context=format_context(chunks),
                question=question,
            ),
        },
    ]


# --------------------------------------------------- query transform
REWRITE_PROMPT = """Rewrite the question to be clear and suitable for searching a weight loss database
- Correct spelling mistakes, replace slang with medical or sports science terms
- If it's a follow-up question, add context from the previous conversation to make it standalone
- Respond with a single search query line, no explanation

{history}Original question: {question}

Rewritten query:"""

MULTI_QUERY_PROMPT = """Generate {n} variations of this question to improve search coverage
- Use different words, both conversational and medical terms
- The meaning must match the original question
- Respond with 1 question per line, no numbering

Original question: {question}

Generated questions:"""

HYDE_PROMPT = """Write a "hypothetical answer" for this question, in the style of a health education article
- Length 3-5 sentences, use specific terms that would likely be in real documents
- Don't worry about factual accuracy, it's only used as a search proxy

Question: {question}

Hypothetical answer:"""


# ------------------------------------------- LLM judge (ตอน evaluate)
JUDGE_PROMPT = """Evaluate the "answer" based on criteria {criteria}, score 1-5
(5 = Excellent, 3 = Fair, 1 = Poor)

{reference}
Question: {question}

Answer:
{answer}

Respond in JSON only: {{"score": <1-5>, "reason": "<short reason>"}}"""
