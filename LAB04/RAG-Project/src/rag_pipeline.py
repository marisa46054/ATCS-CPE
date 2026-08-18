
# Connect all RAG components into a single pipeline.
#
# User Query
#   → query_transform   Improve the query before retrieval      (config.USE_QUERY_TRANSFORM)
#   → hybrid_retriever  Combine BM25 and dense retrieval        (config.USE_HYBRID)
#   → rerankers         Re-rank the retrieved results           (config.USE_RERANK)
#   → generator         Generate an answer with citations       (config.USE_LLM)
#   → memory            Store conversation history              (config.USE_MEMORY)
#
# Each stage can be enabled or disabled in config.py.
# Makes it easy to compare different RAG configurations.
#
# Usage:
#   rag = RAGPipeline()
#   print(rag.ask("What should I do if a condom breaks?")["answer"])

import time

import config
from src.generator import Generator, get_llm
from src.hybrid_retriever import HybridRetriever
from src.memory import ConversationMemory
from src.query_transform import QueryTransformer
from src.rerankers import get_reranker


class RAGPipeline:
    def __init__(self):
        llm = get_llm()

        self.retriever = HybridRetriever(reranker=get_reranker())
        self.transformer = QueryTransformer(llm)
        self.generator = Generator(llm)
        self.memory = ConversationMemory()

    def ask(self, query, top_k=config.TOP_K):
        """
        Ask 1 question, return dict with answer, sources, retrieved, timings

        Read code here top-down to see the whole RAG workflow
        """
        start_time = time.time()

        # ---- Step 1: Query Transform ----
        history = self.memory.get_context() if config.USE_MEMORY else ""

        # Pass history to transformer only if it is a follow-up question
        transform_history = history if self.memory.is_followup(query) else ""
        queries = self.transformer.transform(query, transform_history)
        time_after_transform = time.time()

        # ---- Step 2: Retrieve (+ rerank if enabled) ----
        chunks = self.retriever.retrieve(
            queries[0],
            top_k=top_k,
            extra_queries=queries[1:],
        )
        time_after_retrieve = time.time()

        # ---- Step 3: Generate Answer ----
        result = self.generator.generate(query, chunks, history)
        time_after_generate = time.time()

        # ---- Step 4: Remember for next turn ----
        if config.USE_MEMORY:
            self.memory.add_user(query)
            self.memory.add_assistant(result["answer"])

        result["queries_used"] = queries
        result["retrieved"] = chunks
        result["timings"] = {
            "Transform": round(time_after_transform - start_time, 2),
            "Retrieve": round(time_after_retrieve - time_after_transform, 2),
            "Generate": round(time_after_generate - time_after_retrieve, 2),
            "Total": round(time_after_generate - start_time, 2),
        }
        return result

    def search_only(self, query, top_k=config.TOP_K):
        """Search only without answering — used for retrieval evaluation"""
        queries = self.transformer.transform(query)
        return self.retriever.retrieve(queries[0], top_k, extra_queries=queries[1:])

    def reset(self):
        self.memory.clear()

    def show_settings(self):
        """Print current active stages"""
        settings = [
            ("Hybrid search BM25 + dense", config.USE_HYBRID),
            ("Rerank", config.USE_RERANK),
            ("Query transform before search", config.USE_QUERY_TRANSFORM),
            ("Remember conversation", config.USE_MEMORY),
            ("Use LLM to generate answer", config.USE_LLM),
        ]
        print("Settings (can be edited in config.py):")
        for name, enabled in settings:
            print(f"  {'On ' if enabled else 'Off '}  {name}")
