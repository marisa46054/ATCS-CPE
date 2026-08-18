# query_transform.py
# Improve user queries before retrieval.
#
# Problem
# Retrieval quality depends on the user's query. Real users often write
# short or ambiguous questions, or use slang.
#
#     "I have a sore on my private part"   ← slang
#     Knowledge base: "penis"
#
#     "So what's the difference?"          ← unclear without context
#
# Since the knowledge base uses medical terms, query transformation helps
# bridge the gap between user language and stored documents.
#
# Two levels are available:
#
# Level 1 — normalize_query()
#     No AI required. Replaces slang using a lookup table.
#     Fast, free, and always enabled.
#
# Level 2 — transform()
#     Uses an LLM when config.USE_QUERY_TRANSFORM is enabled.
#
#     rewrite
#         Rewrite the query into a clearer question.
#
#     multi_query
#         Generate multiple equivalent queries and search all of them.
#
#     hyde
#         Generate a hypothetical answer and use it as the search query.
#         This works because answer-like text is often closer to the
#         correct document than the original question.

import re

import config
from src.prompt_templates import HYDE_PROMPT, MULTI_QUERY_PROMPT, REWRITE_PROMPT

# Slang mapping table — add words as needed without modifying other code
SLANG_MAP = {
    "น้องชาย": "อวัยวะเพศชาย",
    "น้องสาว": "อวัยวะเพศหญิง",
    "จู๋": "อวัยวะเพศชาย",
    "จิ๋ม": "อวัยวะเพศหญิง",
    "ถุงยาง": "ถุงยางอนามัย",
    "เอดส์": "เอชไอวี",
    "มีอะไรกัน": "มีเพศสัมพันธ์",
    "โรคจากเซ็กส์": "โรคติดต่อทางเพศสัมพันธ์",
    "เมนส์": "ประจำเดือน",
}

# Ending words that don't help in searching
ENDING_WORDS = re.compile(r"\s*(ครับ|ค่ะ|คะ|จ้า|น้า|หน่อย)\s*$")


def normalize_query(query):
    """
    Adjust question without AI — fast and free

        "Got a sore on my little brother man"  →  "Got a sore on penis"
    """
    text = re.sub(r"\s+", " ", query).strip()       # Remove extra spaces

    for slang, formal in SLANG_MAP.items():
        text = text.replace(slang, formal)

    text = ENDING_WORDS.sub("", text)
    return text.strip() or query.strip()            # If everything is removed, use the original


def clean_line(line):
    """Remove item numbers, quotes, and prefixes that LLMs like to add"""
    text = line.strip()
    text = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", text)    # "1. " หรือ "- "
    text = re.sub(r"^(Question|Search|Query)\s*[:：]\s*", "", text)
    return text.strip().strip('"').strip("'")


class QueryTransformer:
    def __init__(self, llm):
        self.llm = llm

    def ask_llm(self, prompt):
        return self.llm.chat([{"role": "user", "content": prompt}])

    def rewrite(self, query, history):
        """Have LLM rewrite the question to be clear and self-contained"""
        history_block = f"Previous conversation:\n{history}\n\n" if history else ""
        prompt = REWRITE_PROMPT.format(history=history_block, question=query)
        return [clean_line(self.ask_llm(prompt))]

    def multi_query(self, query):
        """Have LLM generate multiple questions with the same meaning"""
        prompt = MULTI_QUERY_PROMPT.format(n=config.MULTI_QUERY_COUNT, question=query)
        answer = self.ask_llm(prompt)

        queries = [normalize_query(query)]           # Keep the original query as the first one
        for line in answer.splitlines():
            new_query = clean_line(line)
            if new_query and new_query not in queries:
                queries.append(new_query)

        return queries[: config.MULTI_QUERY_COUNT + 1]

    def hyde(self, query):
        """Have LLM generate a hypothetical answer and use it as a search query"""
        fake_answer = self.ask_llm(HYDE_PROMPT.format(question=query)).strip()

        # Keep the original query in case the hypothetical answer misses the point
        return [normalize_query(query), fake_answer]

    def transform(self, query, history=""):
        """
        Return a list of queries to search — always at least 1 item

        If LLM is unavailable, return the original query so search doesn't break here
        """
        if not config.USE_QUERY_TRANSFORM:
            return [normalize_query(query)]

        try:
            if config.QUERY_TRANSFORM_MODE == "rewrite":
                return self.rewrite(query, history)
            if config.QUERY_TRANSFORM_MODE == "hyde":
                return self.hyde(query)
            return self.multi_query(query)

        except Exception as error:
            print(f"[query_transform] Failed ({error}) — using original query")
            return [normalize_query(query)]

