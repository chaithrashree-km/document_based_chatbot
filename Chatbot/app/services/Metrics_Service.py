import asyncio
import logging
import time
import uuid
import math
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from app.db.Postgres_Database import Database

logger = logging.getLogger(__name__)

@dataclass
class LatencyRecord:
    embed_cache_ms: Optional[float] = None
    search_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    llm_ms: Optional[float] = None
    total_ms: Optional[float] = None


@dataclass
class RetrievalMetrics:
    docs_retrieved: int = 0
    hit_rate: float = 0.0
    mrr: float = 0.0
    ndcg_at_3: float = 0.0
    avg_rerank_score: float = 0.0
    top_rerank_score: float = 0.0
    avg_context_len: float = 0.0


@dataclass
class QueryRecord:
    query_id: str
    question_hash: str
    created_at: datetime
    served_from_cache: bool
    latency: LatencyRecord
    retrieval: RetrievalMetrics

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def compute_retrieval_metrics(docs: list, k: int = 3) -> RetrievalMetrics:
    if not docs:
        return RetrievalMetrics()

    scores = [sigmoid(d.get("rerank_score", 0.0)) for d in docs]
    texts  = [d["doc"].payload.get("text", "") for d in docs]

    RELEVANCE_THRESHOLD = 0.5  # after sigmoid

    hit_rate = sum(1 for s in scores if s > RELEVANCE_THRESHOLD) / len(scores)

    mrr = 0.0
    for rank, score in enumerate(scores, start=1):
     if score > RELEVANCE_THRESHOLD:
        mrr = 1.0 / rank
        break

    k_eff = min(k, len(scores))
    rel = np.clip(scores[:k_eff], 0, None)
    dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    avg_score = float(np.mean(scores)) if scores else 0.0
    top_score = float(scores[0]) if scores else 0.0
    avg_len   = float(np.mean([len(t) for t in texts])) if texts else 0.0

    return RetrievalMetrics(
        docs_retrieved  = len(docs),
        hit_rate        = round(hit_rate, 4),
        mrr             = round(mrr, 4),
        ndcg_at_3       = round(ndcg, 4),
        avg_rerank_score= round(avg_score, 4),
        top_rerank_score= round(top_score,4),
        avg_context_len = round(avg_len, 2),
    )

class _StopWatch:
    def __init__(self):
        self._start: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        if self._start is not None:
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000


@contextmanager
def sync_timer():
    sw = _StopWatch()
    sw.start()
    try:
        yield sw
    finally:
        sw.stop()


@asynccontextmanager
async def async_timer():
    sw = _StopWatch()
    sw.start()
    try:
        yield sw
    finally:
        sw.stop()


class MetricsService:

    _buffer: deque = deque(maxlen=1_000)
    _persist_to_db: bool = True

    def __init__(self, persist_to_db: bool = True):
        MetricsService._persist_to_db = persist_to_db

    def record(
        self,
        question_hash: str,
        served_from_cache: bool,
        latency: LatencyRecord,
        retrieval: RetrievalMetrics,
    ) -> str:
        query_id = str(uuid.uuid4())
        rec = QueryRecord(
            query_id         = query_id,
            question_hash    = question_hash,
            created_at       = datetime.now(timezone.utc),
            served_from_cache= served_from_cache,
            latency          = latency,
            retrieval        = retrieval,
        )
        MetricsService._buffer.append(rec)

        self._log(rec)

        if self._persist_to_db:
            try:
               loop = asyncio.get_running_loop()
               loop.run_in_executor(None, self._persist, rec)
            except RuntimeError:
               pass  
        return query_id

    def summary(self, last_n: int = 100) -> dict:
        records = list(MetricsService._buffer)[-last_n:]
        if not records:
            return {"message": "No metrics recorded yet."}

        non_cached = [r for r in records if not r.served_from_cache]

        def avg(vals):
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        def p95(vals):
            vals = [v for v in vals if v is not None]
            return round(float(np.percentile(vals, 95)), 2) if vals else None

        total_ms_all   = [r.latency.total_ms for r in records]
        total_ms_nc    = [r.latency.total_ms for r in non_cached]

        valid = [v for v in total_ms_all if v is not None]
        return {
            "window": last_n,
            "total_queries": len(records),
            "cache_hit_rate": round(
                sum(1 for r in records if r.served_from_cache) / len(records), 4
            ),
            "latency": {
                "all_queries": {
                    "avg_total_ms":  avg(total_ms_all),
                    "p95_total_ms":  p95(total_ms_all),
                    "sla_ok_pct" : round(
                        sum(1 for v in valid if v < 2000) / len(valid) * 100, 1
                     ) if valid else None,
                },
                "non_cached_only": {
                    "avg_total_ms":   avg(total_ms_nc),
                    "p95_total_ms":   p95(total_ms_nc),
                    "avg_embed_cache_ms": avg([r.latency.embed_cache_ms for r in non_cached]),
                    "avg_search_ms":  avg([r.latency.search_ms for r in non_cached]),
                    "avg_rerank_ms":  avg([r.latency.rerank_ms for r in non_cached]),
                    "avg_llm_ms":     avg([r.latency.llm_ms for r in non_cached]),
                },
            },
            "retrieval": {
                "avg_docs_retrieved":   avg([r.retrieval.docs_retrieved for r in non_cached]),
                "avg_hit_rate":         avg([r.retrieval.hit_rate for r in non_cached]),
                "avg_mrr":              avg([r.retrieval.mrr for r in non_cached]),
                "avg_ndcg_at_3":        avg([r.retrieval.ndcg_at_3 for r in non_cached]),
                "avg_rerank_score":     avg([r.retrieval.avg_rerank_score for r in non_cached]),
                "top_rerank_score":     avg([r.retrieval.top_rerank_score for r in non_cached]),
                "avg_context_len_chars":avg([r.retrieval.avg_context_len for r in non_cached]),
            },
        }

    def recent(self, n: int = 20) -> list:
        records = list(MetricsService._buffer)[-n:]
        out = []
        for r in records:
            out.append({
                "query_id":          r.query_id,
                "created_at":        r.created_at.isoformat(),
                "served_from_cache": r.served_from_cache,
                "latency_ms": {
                    "embed_cache": r.latency.embed_cache_ms,
                    "search": r.latency.search_ms,
                    "rerank": r.latency.rerank_ms,
                    "llm": r.latency.llm_ms,
                    "total": r.latency.total_ms,
                },
                "retrieval": {
                    "docs_retrieved":   r.retrieval.docs_retrieved,
                    "hit_rate":         r.retrieval.hit_rate,
                    "mrr":              r.retrieval.mrr,
                    "ndcg_at_3":        r.retrieval.ndcg_at_3,
                    "avg_rerank_score": r.retrieval.avg_rerank_score,
                    "top_rerank_score": r.retrieval.top_rerank_score,
                    "avg_context_len":  r.retrieval.avg_context_len,
                },
            })
        return out

    @staticmethod
    def _log(rec: QueryRecord):
        lat = rec.latency
        ret = rec.retrieval
        if rec.served_from_cache:
            logger.info(
                "[METRICS] query_id=%s CACHE_HIT total=%.1fms",
                rec.query_id, lat.total_ms or 0,
            )
        else:
            logger.info(
    "[METRICS] query_id=%s "
    "embed_cache=%.1f search=%.1f rerank=%.1f llm=%.1f total=%.1f ms | "
    "docs=%d hit=%.2f mrr=%.2f ndcg3=%.2f top_rerank=%.3f rerank_avg=%.3f",
    rec.query_id,
    lat.embed_cache_ms or 0, lat.search_ms or 0,
    lat.rerank_ms or 0, lat.llm_ms or 0, lat.total_ms or 0,
    ret.docs_retrieved, ret.hit_rate,
    ret.mrr, ret.ndcg_at_3,
    ret.top_rerank_score, 
    ret.avg_rerank_score,
)


    @staticmethod
    def _persist(rec: QueryRecord):
        sql = """
        INSERT INTO query_metrics (
            query_id, created_at, question_hash, served_from_cache,
            lat_embed_cache_ms, lat_search_ms,
            lat_rerank_ms, lat_llm_ms, lat_total_ms,
            docs_retrieved, hit_rate, mrr, ndcg_at_3, avg_rerank_score,top_rerank_score, avg_context_len
        ) VALUES (
            %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s,%s
        )
        """
        lat, ret = rec.latency, rec.retrieval

        def _f(v):
           return float(v) if v is not None else None

        values = (
                 rec.query_id, rec.created_at, rec.question_hash, rec.served_from_cache,
                 _f(lat.embed_cache_ms), _f(lat.search_ms),
                 _f(lat.rerank_ms),    _f(lat.llm_ms),   _f(lat.total_ms),
                 int(ret.docs_retrieved), _f(ret.hit_rate), _f(ret.mrr),
                 _f(ret.ndcg_at_3),    _f(ret.avg_rerank_score), _f(ret.top_rerank_score), _f(ret.avg_context_len),
                )
        try:
            db = Database()
            try:
                db.cursor.execute(sql, values)
                db.conn.commit()
            finally:
                db.return_to_pool()
        except Exception as exc:
            logger.warning("Failed to persist metrics: %s", exc)
