import asyncio
import unittest
from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np

from app.services.Metrics_Service import (
    LatencyRecord,
    RetrievalMetrics,
    QueryRecord,
    MetricsService,
    compute_retrieval_metrics,
    sync_timer,
    async_timer,
)

def _make_doc(rerank_score: float, text: str = "sample text") -> dict:
    payload_mock = MagicMock()
    payload_mock.get.return_value = text
    doc_mock = MagicMock()
    doc_mock.payload = payload_mock
    return {"rerank_score": rerank_score, "doc": doc_mock}


def _make_latency(**kwargs) -> LatencyRecord:
    defaults = dict(embed_cache_ms=20.0, search_ms=2000.0, rerank_ms=100.0, llm_ms=1500.0, total_ms=3620.0)
    defaults.update(kwargs)
    return LatencyRecord(**defaults)


def _make_retrieval(**kwargs) -> RetrievalMetrics:
    defaults = dict(docs_retrieved=4, hit_rate=0.5, mrr=1.0, ndcg_at_3=1.0, avg_rerank_score=-2.5, avg_context_len=400.0)
    defaults.update(kwargs)
    return RetrievalMetrics(**defaults)

class TestMetricsService(unittest.TestCase):

    def setUp(self):
        MetricsService._buffer = deque(maxlen=1_000)
        self.svc = MetricsService(persist_to_db=False)

    def test_compute_retrieval_metrics_all_positive_scores(self):
        docs = [_make_doc(2.0, "text one"), _make_doc(1.5, "text two")]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.docs_retrieved, 2)
        self.assertEqual(result.hit_rate, 1.0)
        self.assertEqual(result.mrr, 1.0)
        self.assertGreater(result.ndcg_at_3, 0.0)

    def test_compute_retrieval_metrics_partial_positive_scores(self):
        docs = [_make_doc(-1.0), _make_doc(2.0), _make_doc(-0.5), _make_doc(1.0)]
        result = compute_retrieval_metrics(docs)
        self.assertAlmostEqual(result.hit_rate, 0.5, places=2)
        self.assertAlmostEqual(result.mrr, 0.5, places=4)   # first hit at rank 2

    def test_compute_retrieval_metrics_avg_context_len(self):
        docs = [_make_doc(1.0, "ab"), _make_doc(1.0, "abcd")]   # 2 + 4 = mean 3
        result = compute_retrieval_metrics(docs)
        self.assertAlmostEqual(result.avg_context_len, 3.0, places=2)

    def test_compute_retrieval_metrics_avg_rerank_score(self):
        docs = [_make_doc(-2.0), _make_doc(2.0)]
        result = compute_retrieval_metrics(docs)
        self.assertAlmostEqual(result.avg_rerank_score, 0.0, places=4)

    def test_compute_retrieval_metrics_single_doc_positive(self):
        docs = [_make_doc(3.0, "hello world")]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.docs_retrieved, 1)
        self.assertEqual(result.hit_rate, 1.0)
        self.assertEqual(result.mrr, 1.0)

    def test_compute_retrieval_metrics_ndcg_k_clamp(self):
        docs = [_make_doc(3.0), _make_doc(2.0), _make_doc(1.0), _make_doc(0.5)]
        result_k3 = compute_retrieval_metrics(docs, k=3)
        result_k4 = compute_retrieval_metrics(docs, k=4)
        self.assertIsNotNone(result_k3.ndcg_at_3)
        self.assertIsNotNone(result_k4.ndcg_at_3)

    def test_compute_retrieval_metrics_empty_docs(self):
        result = compute_retrieval_metrics([])
        self.assertEqual(result.docs_retrieved, 0)
        self.assertEqual(result.hit_rate, 0.0)
        self.assertEqual(result.mrr, 0.0)
        self.assertEqual(result.ndcg_at_3, 0.0)

    def test_compute_retrieval_metrics_all_negative_scores(self):
        docs = [_make_doc(-5.0), _make_doc(-3.0)]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.hit_rate, 0.0)
        self.assertEqual(result.mrr, 0.0)
        self.assertEqual(result.ndcg_at_3, 0.0)

    def test_compute_retrieval_metrics_zero_score_not_counted_as_hit(self):
        docs = [_make_doc(0.0)]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.hit_rate, 0.0)
        self.assertEqual(result.mrr, 0.0)

    def test_compute_retrieval_metrics_single_doc_negative(self):
        docs = [_make_doc(-1.0)]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.hit_rate, 0.0)
        self.assertEqual(result.mrr, 0.0)

    def test_sync_timer_measures_elapsed(self):
        import time
        with sync_timer() as sw:
            time.sleep(0.01)
        self.assertIsNotNone(sw.elapsed_ms)
        self.assertGreater(sw.elapsed_ms, 5.0)

    def test_sync_timer_elapsed_on_instant_block(self):
        with sync_timer() as sw:
            pass
        self.assertIsNotNone(sw.elapsed_ms)
        self.assertGreaterEqual(sw.elapsed_ms, 0.0)

    def test_sync_timer_elapsed_not_set_before_exit(self):
        values_inside = []
        with sync_timer() as sw:
            values_inside.append(sw.elapsed_ms)
        self.assertIsNone(values_inside[0])

    def test_async_timer_measures_elapsed(self):
        import asyncio, time

        async def _run():
            async with async_timer() as sw:
                await asyncio.sleep(0.01)
            return sw

        sw = asyncio.get_event_loop().run_until_complete(_run())
        self.assertIsNotNone(sw.elapsed_ms)
        self.assertGreater(sw.elapsed_ms, 5.0)

    def test_async_timer_elapsed_on_instant_block(self):
        async def _run():
            async with async_timer() as sw:
                pass
            return sw

        sw = asyncio.get_event_loop().run_until_complete(_run())
        self.assertGreaterEqual(sw.elapsed_ms, 0.0)

    def test_record_returns_valid_uuid(self):
        import uuid
        qid = self.svc.record("hash1", False, _make_latency(), _make_retrieval())
        self.assertIsInstance(qid, str)
        self.assertTrue(len(qid) > 0)
        uuid.UUID(qid) 

    def test_record_appends_to_buffer(self):
        self.svc.record("h1", False, _make_latency(), _make_retrieval())
        self.svc.record("h2", False, _make_latency(), _make_retrieval())
        self.assertEqual(len(MetricsService._buffer), 2)

    def test_record_cached_query_stored_correctly(self):
        self.svc.record("hash_cache", True, _make_latency(total_ms=25.0), RetrievalMetrics())
        rec = MetricsService._buffer[-1]
        self.assertTrue(rec.served_from_cache)

    def test_record_non_cached_query_stored_correctly(self):
        self.svc.record("hash_nc", False, _make_latency(), _make_retrieval())
        rec = MetricsService._buffer[-1]
        self.assertFalse(rec.served_from_cache)

    def test_record_latency_values_preserved(self):
        lat = _make_latency(embed_cache_ms=15.5, llm_ms=3000.0, total_ms=5200.0)
        self.svc.record("h", False, lat, _make_retrieval())
        rec = MetricsService._buffer[-1]
        self.assertAlmostEqual(rec.latency.embed_cache_ms, 15.5)
        self.assertAlmostEqual(rec.latency.llm_ms, 3000.0)
        self.assertAlmostEqual(rec.latency.total_ms, 5200.0)

    def test_record_retrieval_values_preserved(self):
        ret = _make_retrieval(hit_rate=0.75, mrr=0.5, docs_retrieved=4)
        self.svc.record("h", False, _make_latency(), ret)
        rec = MetricsService._buffer[-1]
        self.assertAlmostEqual(rec.retrieval.hit_rate, 0.75)
        self.assertAlmostEqual(rec.retrieval.mrr, 0.5)
        self.assertEqual(rec.retrieval.docs_retrieved, 4)

    def test_record_created_at_is_utc(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        rec = MetricsService._buffer[-1]
        self.assertIsNotNone(rec.created_at.tzinfo)

    def test_record_with_none_latency_fields(self):
        lat = LatencyRecord()  
        self.svc.record("h", False, lat, _make_retrieval())
        self.assertEqual(len(MetricsService._buffer), 1)

    def test_record_buffer_maxlen_evicts_oldest(self):
        MetricsService._buffer = deque(maxlen=5)
        for i in range(7):
            self.svc.record(f"hash{i}", False, _make_latency(), _make_retrieval())
        self.assertEqual(len(MetricsService._buffer), 5)
        self.assertEqual(MetricsService._buffer[0].question_hash, "hash2")

    def test_record_persist_failure_does_not_raise(self):
        svc = MetricsService(persist_to_db=True)
        with patch.object(MetricsService, "_persist", side_effect=Exception("DB down")):
            try:
                svc.record("h", False, _make_latency(), _make_retrieval())
            except Exception:
                self.fail("record() raised an exception when _persist failed")

    def test_summary_empty_buffer(self):
        result = self.svc.summary()
        self.assertIn("message", result)

    def test_summary_counts_total_queries(self):
        for _ in range(5):
            self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.summary(last_n=10)
        self.assertEqual(result["total_queries"], 5)

    def test_summary_cache_hit_rate_all_cached(self):
        for _ in range(4):
            self.svc.record("h", True, _make_latency(total_ms=25.0), RetrievalMetrics())
        result = self.svc.summary()
        self.assertAlmostEqual(result["cache_hit_rate"], 1.0)

    def test_summary_cache_hit_rate_no_cached(self):
        for _ in range(3):
            self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["cache_hit_rate"], 0.0)

    def test_summary_cache_hit_rate_mixed(self):
        for _ in range(3):
            self.svc.record("h", False, _make_latency(), _make_retrieval())
        for _ in range(3):
            self.svc.record("h", True, _make_latency(total_ms=25.0), RetrievalMetrics())
        result = self.svc.summary()
        self.assertAlmostEqual(result["cache_hit_rate"], 0.5)

    def test_summary_avg_total_ms(self):
        self.svc.record("h", False, _make_latency(total_ms=2000.0), _make_retrieval())
        self.svc.record("h", False, _make_latency(total_ms=4000.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["all_queries"]["avg_total_ms"], 3000.0)

    def test_summary_avg_llm_ms_only_non_cached(self):
        self.svc.record("h", False, _make_latency(llm_ms=1000.0, total_ms=3000.0), _make_retrieval())
        self.svc.record("h", False, _make_latency(llm_ms=3000.0, total_ms=5000.0), _make_retrieval())
        self.svc.record("h", True,  _make_latency(llm_ms=None,   total_ms=25.0),   RetrievalMetrics())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["non_cached_only"]["avg_llm_ms"], 2000.0)

    def test_summary_sla_ok_pct_all_within(self):
        for _ in range(4):
            self.svc.record("h", False, _make_latency(total_ms=1500.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["all_queries"]["sla_ok_pct"], 100.0)

    def test_summary_sla_ok_pct_none_within(self):
        for _ in range(4):
            self.svc.record("h", False, _make_latency(total_ms=5000.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["all_queries"]["sla_ok_pct"], 0.0)

    def test_summary_avg_hit_rate(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval(hit_rate=1.0))
        self.svc.record("h", False, _make_latency(), _make_retrieval(hit_rate=0.0))
        result = self.svc.summary()
        self.assertAlmostEqual(result["retrieval"]["avg_hit_rate"], 0.5)

    def test_summary_last_n_respects_window(self):
        for i in range(10):
            self.svc.record("h", False, _make_latency(total_ms=float(i * 1000)), _make_retrieval())
        result = self.svc.summary(last_n=3)
        self.assertEqual(result["total_queries"], 3)

    def test_summary_last_n_larger_than_buffer(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.summary(last_n=100)
        self.assertEqual(result["total_queries"], 2)

    def test_summary_all_cached_non_cached_averages_are_none_or_empty(self):
        for _ in range(3):
            self.svc.record("h", True, _make_latency(total_ms=25.0), RetrievalMetrics())
        result = self.svc.summary()
        nc = result["latency"]["non_cached_only"]
        self.assertIsNone(nc["avg_llm_ms"])
        self.assertIsNone(nc["avg_search_ms"])

    def test_summary_none_total_ms_excluded_from_sla(self):
        self.svc.record("h", False, LatencyRecord(total_ms=None), _make_retrieval())
        result = self.svc.summary()
        self.assertIn("sla_ok_pct", result["latency"]["all_queries"])

    def test_recent_returns_list(self):
        result = self.svc.recent(n=10)
        self.assertIsInstance(result, list)

    def test_recent_empty_buffer_returns_empty_list(self):
        result = self.svc.recent(n=10)
        self.assertEqual(result, [])

    def test_recent_returns_last_n_records(self):
        for _ in range(5):
            self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.recent(n=2)
        self.assertEqual(len(result), 2)

    def test_recent_record_has_required_keys(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.recent(n=1)
        self.assertIn("query_id", result[0])
        self.assertIn("created_at", result[0])
        self.assertIn("served_from_cache", result[0])
        self.assertIn("latency_ms", result[0])
        self.assertIn("retrieval", result[0])

    def test_recent_latency_ms_has_required_keys(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        lat = self.svc.recent(n=1)[0]["latency_ms"]
        for key in ("embed_cache", "search", "rerank", "llm", "total"):
            self.assertIn(key, lat)

    def test_recent_retrieval_has_required_keys(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        ret = self.svc.recent(n=1)[0]["retrieval"]
        for key in ("docs_retrieved", "hit_rate", "mrr", "ndcg_at_3", "avg_rerank_score", "avg_context_len"):
            self.assertIn(key, ret)

    def test_recent_served_from_cache_flag_correct(self):
        self.svc.record("h", True, _make_latency(total_ms=25.0), RetrievalMetrics())
        result = self.svc.recent(n=1)
        self.assertTrue(result[0]["served_from_cache"])

    def test_recent_created_at_is_iso_string(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        created_at_str = self.svc.recent(n=1)[0]["created_at"]
        parsed = datetime.fromisoformat(created_at_str)
        self.assertIsNotNone(parsed)

    def test_recent_n_larger_than_buffer_returns_all(self):
        for _ in range(3):
            self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.recent(n=50)
        self.assertEqual(len(result), 3)

    def test_recent_none_latency_fields_serialised_as_none(self):
        self.svc.record("h", False, LatencyRecord(total_ms=500.0), _make_retrieval())
        lat = self.svc.recent(n=1)[0]["latency_ms"]
        self.assertIsNone(lat["embed_cache"])
        self.assertIsNone(lat["llm"])

    def test_persist_calls_db_execute_and_commit(self):
        import app.services.Metrics_Service as _ms_module

        rec = QueryRecord(
            query_id="test-id",
            question_hash="hash",
            created_at=datetime.now(timezone.utc),
            served_from_cache=False,
            latency=_make_latency(),
            retrieval=_make_retrieval(),
        )
        mock_db = MagicMock()
        with patch.object(_ms_module, "Database", return_value=mock_db):
            MetricsService._persist(rec)
        mock_db.cursor.execute.assert_called_once()
        mock_db.conn.commit.assert_called_once()
        mock_db.return_to_pool.assert_called_once()

    def test_persist_swallows_db_exception(self):
        import app.services.Metrics_Service as _ms_module

        rec = QueryRecord(
            query_id="test-id",
            question_hash="hash",
            created_at=datetime.now(timezone.utc),
            served_from_cache=False,
            latency=_make_latency(),
            retrieval=_make_retrieval(),
        )
        mock_db = MagicMock()
        mock_db.cursor.execute.side_effect = Exception("Connection refused")
        with patch.object(_ms_module, "Database", return_value=mock_db):
            try:
                MetricsService._persist(rec)
            except Exception:
                self.fail("_persist raised an unexpected exception")

    def test_persist_return_to_pool_called_even_on_execute_failure(self):
        import app.services.Metrics_Service as _ms_module

        rec = QueryRecord(
            query_id="test-id",
            question_hash="hash",
            created_at=datetime.now(timezone.utc),
            served_from_cache=False,
            latency=_make_latency(),
            retrieval=_make_retrieval(),
        )
        mock_db = MagicMock()
        mock_db.cursor.execute.side_effect = RuntimeError("boom")
        with patch.object(_ms_module, "Database", return_value=mock_db):
            MetricsService._persist(rec)
        mock_db.return_to_pool.assert_called_once()

    def test_persist_none_latency_fields_do_not_crash(self):
        import app.services.Metrics_Service as _ms_module

        rec = QueryRecord(
            query_id="test-id-2",
            question_hash="hash2",
            created_at=datetime.now(timezone.utc),
            served_from_cache=True,
            latency=LatencyRecord(),
            retrieval=RetrievalMetrics(),
        )
        mock_db = MagicMock()
        with patch.object(_ms_module, "Database", return_value=mock_db):
            try:
                MetricsService._persist(rec)
            except Exception as e:
                self.fail(f"_persist raised with None fields: {e}")

    def test_latency_record_defaults_are_none(self):
        lat = LatencyRecord()
        self.assertIsNone(lat.embed_cache_ms)
        self.assertIsNone(lat.search_ms)
        self.assertIsNone(lat.rerank_ms)
        self.assertIsNone(lat.llm_ms)
        self.assertIsNone(lat.total_ms)

    def test_retrieval_metrics_defaults_are_zero(self):
        ret = RetrievalMetrics()
        self.assertEqual(ret.docs_retrieved, 0)
        self.assertEqual(ret.hit_rate, 0.0)
        self.assertEqual(ret.mrr, 0.0)
        self.assertEqual(ret.ndcg_at_3, 0.0)
        self.assertEqual(ret.avg_rerank_score, 0.0)
        self.assertEqual(ret.avg_context_len, 0.0)
