
# Re-rank retrieved documents using a cross-encoder model.
#
# Why rerank?
# FAISS uses a bi-encoder for fast retrieval but may confuse similar topics.
# A cross-encoder scores each query-document pair together for higher accuracy.
#
# In practice:
# Retrieve the top 20 chunks, rerank them, then keep the best 3 results.
#
# Enable with config.USE_RERANK = True.
# Disabled by default because the model is large and slower than retrieval.



from sentence_transformers import CrossEncoder

import config


class Reranker:
    def __init__(self):
        print(f"[rerank] Loading model {config.RERANK_MODEL_NAME} (first time may take a while) ...")
        self.model = CrossEncoder(config.RERANK_MODEL_NAME)

    def rerank(self, query, chunks, top_k=config.TOP_K):
        """
        Rerank chunks based on relevance to query and return only top_k chunks

        Original retrieval score is kept in retrieval_score key
        to compare how reranking changed the order
        """
        if not chunks:
            return []

        # Create pairs (query, text) for the model to score
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        results = []
        for chunk, score in zip(chunks, scores):
            chunk = dict(chunk)
            chunk["retrieval_score"] = chunk["score"]   # Keep original score
            chunk["score"] = float(score)               # Use new score instead
            results.append(chunk)

        results.sort(key=lambda chunk: chunk["score"], reverse=True)
        return results[:top_k]


def get_reranker():
    """
    Return Reranker if config.USE_RERANK = True otherwise return None

    If model fails to load (offline / not enough RAM), return None instead of crashing
    The system can still continue without reranking
    """
    if not config.USE_RERANK:
        return None

    try:
        return Reranker()
    except Exception as error:
        print(f"[rerank] Failed to load model ({error}) — skipping rerank step")
        return None
