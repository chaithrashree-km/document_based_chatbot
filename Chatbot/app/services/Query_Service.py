import logging
import redis
import json
import threading
import numpy as np
import asyncio
import hashlib
import nltk
import time
from datetime import datetime
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from app.db.Vector_Database import VectorDatabase
from app.Config import Config
from app.services.LLM_Service import Response
from app.services.Metrics_Service import (MetricsService, LatencyRecord, RetrievalMetrics,compute_retrieval_metrics)
from nltk.corpus import wordnet

_config = Config()
_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("app.log"),])
logging.info("Pre-warming embedding model at module load...")
_embedding_model.encode("warmup")
logging.info("Embedding model warmed up.")

class Retrieve:
    def __init__(self):
        self.config = _config
        self.embedding_model = _embedding_model
        self.reranker = _reranker
 
        self.database = VectorDatabase()
        self.response = Response()
        self.redis_client = redis.Redis.from_url(self.config.REDIS_URL, decode_responses=True)
        self.metrics = MetricsService(persist_to_db=True)
        self._stemmer = PorterStemmer()
        self._bm25_lock = threading.RLock()
 
        try:
            self.documents = self.load_documents()
        except Exception as e:
            logging.warning(f"Could not load documents on startup: {e}")
            self.documents = []
 
        corpus = [doc["doc"].payload["text"] for doc in self.documents]
        tokenized = [self._stem_tokenize(text) for text in corpus]
 
        try:
            self.bm25 = BM25Okapi(tokenized)
        except Exception as e:
            logging.warning(f"Could not initialize BM25: {e}")
            self.bm25 = None
 
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)      
        
    def _stem_tokenize(self, text: str) -> list:
        return [self._stemmer.stem(token) for token in text.lower().split()]    

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
        with self._bm25_lock:
            try:
                logging.info("Refreshing BM25 index after ingestion...")
                self.documents = self.load_documents()
                corpus = [doc["doc"].payload["text"] for doc in self.documents]
                tokenized = [self._stem_tokenize(text) for text in corpus]
                self.bm25 = BM25Okapi(tokenized)
                logging.info(f"BM25 index refreshed with {len(self.documents)} documents.")
            except Exception as e:
                logging.error(f"Failed to refresh BM25 index: {e}")
                self.bm25 = None

    def keyword_search(self, question):
        logging.info("starting keyword search")
        with self._bm25_lock:
          if not self.bm25:
            logging.info("BM25 not initialized, building now...")
            try:
                self.documents = self.load_documents()
                corpus = [doc["doc"].payload["text"] for doc in self.documents]
                tokenized = [self._stem_tokenize(t) for t in corpus]
                self.bm25 = BM25Okapi(tokenized)
                logging.info("Keyword seach completed without any error")
            except Exception as e:
                logging.error(f"Lazy BM25 init failed: {e}")
                return []
          bm25_snapshot = self.bm25
          docs_snapshot = list(self.documents)

        stemmed_tokens = set(self._stem_tokenize(question))
        for token in question.lower().split():
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    stemmed_tokens.add(
                        self._stemmer.stem(lemma.name().replace("_", " ").lower())
                    )
 
        scores = bm25_snapshot.get_scores(list(stemmed_tokens))
        docs = []
        for i, score in enumerate(scores):
            docs.append({
                "doc": docs_snapshot[i]["doc"],
                "vector_score": 0,
                "keyword_score": float(score)
            })
        ranked = sorted(docs, key=lambda x: x["keyword_score"], reverse=True)
        logging.info("keyword search finsihed. returning results")
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
                merged[doc_id]["keyword_score"] = max(
                    merged[doc_id]["keyword_score"],
                    item["keyword_score"]
                )
        return list(merged.values())

    def combine_scores(self, docs):
        for item in docs:
            item["hybrid_score"] = (
                0.60 * item["vector_score"] + 
                0.40 * item["keyword_score"]
            )

        ranked = sorted(
            docs,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        return ranked[:4]

    def rerank(self, question, docs):
        if not docs:
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
    
    def _is_inventory_request(self, question: str) -> bool:
        inventory_keywords = [
        "what documents do you have", "what files do you have", "list all the documents you have","what documents you have?"
        "what information do you have", "list all documents", "list all the data you have", "what information do you contain"
        "list all files", "list all the documents", "show all documents", "what data do you have"
        "what do you have", "what's available", "what is available", "What are all the documents do you have"
         ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in inventory_keywords)

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
        not_found_phrases = [
            "the documents does not have a specific answer",
            "the documents do not have a specific answer",
        ]
        if any(phrase in response.lower() for phrase in not_found_phrases):
            return
        cache_key = self.make_cache_key(question)
        self.redis_client.setex(cache_key, self.config.CACHE_TTL, json.dumps({"response": response}))

    def _build_context(self, final_docs: list):
        context_parts = []
        for item in final_docs:
          doc = item["doc"]
          text = doc.payload["text"]
          source = doc.payload.get("source", "Unknown")
          page = doc.payload.get("page", "N/A")
          context_parts.append(f"{text} (Source: {source}, Page: {page})")
        return "\n\n".join(context_parts)

    def _build_summary_context(self, final_docs: list):
        context_parts = []
        sources_seen = []
        for item in final_docs:
          doc = item["doc"]
          text = doc.payload["text"]
          source = doc.payload.get("source", "Unknown")
          fname = source.split("_", 1)[-1] if "_" in source else source
          context_parts.append(text)
          if fname not in sources_seen:
             sources_seen.append(fname)
        return "\n\n".join(context_parts), sources_seen    

    def _build_inventory_context(self) -> str:
        docs = self.load_documents()

        source_texts: dict[str, list[str]] = {}
        for item in docs:
           payload = item["doc"].payload
           raw_source = payload.get("source", "Unknown")
           fname = raw_source.split("_", 1)[-1] if "_" in raw_source else raw_source
           text = payload.get("text", "")
           if fname not in source_texts:
              source_texts[fname] = []
           if text:
              source_texts[fname].append(text[:300])

        if not source_texts:
           return ""

        context_parts = []
        for fname, snippets in source_texts.items():
           combined = " ".join(snippets[:3])
           context_parts.append(f"File: {fname}\nContent preview: {combined}")
        return "\n\n".join(context_parts)   

    async def query_docs(self, question: str):
        pipeline_start = time.perf_counter()
        lat = LatencyRecord()
        question_hash = hashlib.sha256(self.normalize_question(question).encode()).hexdigest()[:16]
        
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
           self.metrics.record(question_hash = question_hash, served_from_cache= True, latency = lat, retrieval = RetrievalMetrics())
           return cached_response

        logging.info("Response not found from cache!")
        logging.info("Starting vector and keyword search asynchronously...")

        if self._is_inventory_request(question):
           logging.info("Query is an inventory request. Building context from loaded documents.")
           context = self._build_inventory_context()
           if not context:
               return "No documents are currently available in the knowledge base."

           logging.info("Sending Query along with context for llm to generate response") 
           answer = await asyncio.to_thread(self.response.generate_inventory_response, question, context)
           lat.total_ms = (time.perf_counter() - pipeline_start) * 1000
           self.metrics.record(question_hash=question_hash, served_from_cache=False, latency=lat, retrieval=RetrievalMetrics())
           return answer

        else:
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

           if self._is_summary_request(question):
                logging.info("Query/Question is a summary request!!!")
                logging.info("Building the context")
                summary_context = self._build_summary_context(final_docs)
                if not summary_context:
                   return "No documents are currently available in the knowledge base."
            
                logging.info("Sending Query along with context for llm to generate response") 
                answer = await asyncio.to_thread(self.response.generate_summary_response, question, summary_context)
                self.save_cache(question, answer) 
                lat.total_ms = (time.perf_counter() - pipeline_start) * 1000
                self.metrics.record(question_hash=question_hash, served_from_cache=False, latency=lat, retrieval=RetrievalMetrics())
                return answer

           context = self._build_context(final_docs)
           logging.info("Sending Query along with context for llm to generate response") 
           t3 = time.perf_counter()
           answer = self.response.generate_response(question, context)
           lat.llm_ms = (time.perf_counter() - t3) * 1000
           logging.info("Saving the response to cache")
           self.save_cache(question, answer)
           lat.total_ms = (time.perf_counter() - pipeline_start) * 1000
           self.metrics.record(question_hash  = question_hash, served_from_cache= False, latency = lat, retrieval = retrieval_metrics,)
           return answer
