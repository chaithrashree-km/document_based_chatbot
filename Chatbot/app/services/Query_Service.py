import logging
import redis
import json
import numpy as np
import asyncio
import hashlib
import nltk
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from app.db.Vector_Database import VectorDatabase
from app.Config import Config
from app.services.LLM_Service import Response
from app.services.Metrics_Service import (MetricsService, LatencyRecord, RetrievalMetrics,compute_retrieval_metrics)
from nltk.corpus import wordnet

class Retrieve:
    config = Config()
    database = VectorDatabase()
    response = Response()

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    metrics = MetricsService(persist_to_db=True)

    def __init__(self):
        try:
          self.documents = self.load_documents()
        except Exception as e:
          logging.warning(f"Could not load documents on startup as collection may not exist yet: {e}")
          self.documents = []

        corpus = [doc["doc"].payload["text"] for doc in self.documents]
        tokenized = [text.split() for text in corpus]

        try:
          self.bm25 = BM25Okapi(tokenized)
        except Exception as e:
          logging.warning(f"Could not tokenize using bm25 as its empty.")
        

        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

        logging.info("Pre-warming embedding model...")
        self.embedding_model.encode("warmup")
        logging.info("Embedding model warmed up and ready.")
        

    def load_documents(self, limit=1000):
        results = self.database.client.scroll(
            collection_name=self.config.COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        docs = []
        for doc in results[0]:
            docs.append({
                "doc": doc,
                "vector_score": 0,
                "keyword_score": 0
            })
        return docs
    
    def refresh_bm25(self):
       try: 
         logging.info("Refreshing BM25 index after ingestion...")
         self.documents = self.load_documents()
         corpus = [doc["doc"].payload["text"] for doc in self.documents]
         tokenized = [text.split() for text in corpus]
         self.bm25 = BM25Okapi(tokenized)
         logging.info(f"BM25 index refreshed with {len(self.documents)} documents.")
       except Exception as e: 
         logging.error(f"Failed to refresh BM25 index: {e}")
         logging.warning("BM25 not initialized because corpus is empty.")
         self.bm25 = None


    def keyword_search(self, question):
        if not self.bm25:
         logging.warning("BM25 search skipped because index is not initialized.")
         return []

        tokens = question.lower().split()
    
        expanded_tokens = set(tokens)
        for token in tokens:
          for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()
                expanded_tokens.add(synonym)
    
        tokenized_query = list(expanded_tokens)
        scores = self.bm25.get_scores(tokenized_query)
        docs = []
        for i, score in enumerate(scores):
          docs.append({
            "doc": self.documents[i]["doc"],
            "vector_score": 0,
            "keyword_score": float(score)
        })
        ranked = sorted(docs, key=lambda x: x["keyword_score"], reverse=True)
        return ranked[:10]

    def vector_search(self, query_embedding, limit=10):
        results = self.database.client.query_points(
            collection_name=self.config.COLLECTION_NAME,
            query=query_embedding,
            limit=limit
        )

        docs = []
        for doc in results.points:
            docs.append({
                "doc": doc,
                "vector_score": float(doc.score),
                "keyword_score": 0
            })
        return docs

    def merge_results(self, results):
        merged = {}
        for item in results:
            doc_id = item["doc"].id
            if doc_id not in merged:
                merged[doc_id] = item
            else:
                merged[doc_id]["vector_score"] = max(
                    merged[doc_id]["vector_score"],
                    item["vector_score"]
                )
        return list(merged.values())

    def combine_scores(self, docs):
        for item in docs:
            item["hybrid_score"] = (
                0.75 * item["vector_score"] + 
                0.25 * item["keyword_score"]
            )

        ranked = sorted(
            docs,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        return ranked[:4]

    def rerank(self, question, docs):
        if len(docs) <= 2:
            for doc in docs:
              doc["rerank_score"] = 0.0
            return docs

        pairs = [
            [question, item["doc"].payload["text"]]
            for item in docs
        ]

        scores = self.reranker.predict(pairs, batch_size=8)
        for i, score in enumerate(scores):
            docs[i]["rerank_score"] = float(score)
        ranked = sorted(
            docs,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        return ranked[:8]
    
    def _is_summary_request(self, question: str) -> bool:
        summary_keywords = [
            "summarize", "summarise", "summary", "summarization",
            "overview of document", "what does the document say",
            "what does the file say", "tell me about the document",
            "give me a summary", "describe the document"
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in summary_keywords)

    def normalize_question(self,question: str) -> str:
      return " ".join(question.lower().strip().split())

    def make_cache_key(self,question: str) -> str:
        normalized = self.normalize_question(question)
        return f"semantic_cache:{hashlib.sha256(normalized.encode()).hexdigest()}"

    def check_cache(self, question: str):
        cache_key = self.make_cache_key(question)
        value = self.redis_client.get(cache_key)
        if value:
           logging.info("Response found in cache")
           data = json.loads(value)
           return data["response"]
        return None

    def save_cache(self, question: str, response: str):
        invalid_response = "The documents does not have a specific answer to your question."
        if response != invalid_response:
          cache_key = self.make_cache_key(question)
          cache_data = {
            "response": response
          }
          self.redis_client.setex(cache_key, self.config.CACHE_TTL, json.dumps(cache_data))

    def _build_context(self, final_docs: list, is_summary: bool):
        context_parts = []
        sources_seen = []

        for item in final_docs:
           doc = item["doc"]
           text = doc.payload["text"]
           source = doc.payload.get("source", "Unknown")
           page = doc.payload.get("page", "N/A")

           if is_summary:
             context_parts.append(text)
             fname = source.split("_", 1)[-1] if "_" in source else source
             if fname not in sources_seen:
                sources_seen.append(fname)
           else:
             context_parts.append(f"{text} (Source: {source}, Page: {page})")

        context = "\n\n".join(context_parts)
        return context, sources_seen      

    async def query_docs(self, question: str):
        pipeline_start = time.perf_counter()
        lat = LatencyRecord()
        question_hash = hashlib.sha256(
            self.normalize_question(question).encode()
        ).hexdigest()[:16]

        logging.info("Running embedding and cache check concurrently...")

        t0 = time.perf_counter()
        embed_task = asyncio.to_thread(self.embedding_model.encode, question)
        cache_task = asyncio.to_thread(self.check_cache, question)

        embedding, cached_response = await asyncio.gather(embed_task, cache_task)
        stage1_ms = (time.perf_counter() - t0) * 1000
        lat.embed_cache_ms = stage1_ms

        logging.info("Checking Response from Semantic Cache")
        if cached_response:
           logging.info("Response found in Semantic Cache")
           lat.total_ms = (time.perf_counter() - pipeline_start) * 1000
           self.metrics.record(
                question_hash    = question_hash,
                served_from_cache= True,
                latency          = lat,
                retrieval        = RetrievalMetrics(),
            )
           return cached_response

        logging.info("Response not found from cache!")
        logging.info("Starting vector and keyword search asynchronously...")

        t1 = time.perf_counter()
        vector_task = asyncio.to_thread(self.vector_search, embedding.tolist())
        keyword_task = asyncio.to_thread(self.keyword_search, question)

        vector_docs, keyword_docs = await asyncio.gather(vector_task, keyword_task)
        search_ms = (time.perf_counter() - t1) * 1000
        lat.search_ms  = search_ms

        logging.info("Merging vector and keyword search results")
        merged_docs = self.merge_results(vector_docs + keyword_docs)
        logging.info("Generating scores for the merged results")
        hybrid_docs = self.combine_scores(merged_docs)

        logging.info("Starting reranking")
        t2 = time.perf_counter()
        reranked_docs = self.rerank(question, hybrid_docs)
        lat.rerank_ms = (time.perf_counter() - t2) * 1000

        final_docs = reranked_docs[:4]
        logging.info("Reranking completed")

        retrieval_metrics = compute_retrieval_metrics(final_docs, k=3)

        logging.info("Checking the query intent is summary?")
        is_summary = self._is_summary_request(question)
        logging.info("Building the context")
        context, sources = self._build_context(final_docs, is_summary)

        logging.info("Sending Query along with context for llm to generate response") 
        t3 = time.perf_counter()
        answer = self.response.generate_response(question, context, sources=sources if is_summary else None)
        lat.llm_ms = (time.perf_counter() - t3) * 1000
        logging.info("Saving the response to cache")
        self.save_cache(question, answer)
        lat.total_ms = (time.perf_counter() - pipeline_start) * 1000
        self.metrics.record(
            question_hash    = question_hash,
            served_from_cache= False,
            latency          = lat,
            retrieval        = retrieval_metrics,
        )
        return answer
