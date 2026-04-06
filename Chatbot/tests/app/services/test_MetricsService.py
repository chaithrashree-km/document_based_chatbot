import asyncio
import math
import time
import uuid
import unittest
from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np

from app.services.Metrics_Service import (
    LatencyRecord,
    MetricsService,
    QueryRecord,
    RetrievalMetrics,
    _StopWatch,
    async_timer,
    compute_retrieval_metrics,
    sigmoid,
    sync_timer,
)

def _make_doc(rerank_score: float, text: str = "sample text") -> dict:
    payload_mock = MagicMock()
    payload_mock.get.return_value = text
    doc_mock = MagicMock()
    doc_mock.payload = payload_mock
    return {"rerank_score": rerank_score, "doc": doc_mock}


def _make_latency(**kwargs) -> LatencyRecord:
    defaults = dict(
        embed_cache_ms=20.0,
        search_ms=200.0,
        rerank_ms=100.0,
        llm_ms=1500.0,
        total_ms=1820.0,
    )
    defaults.update(kwargs)
    return LatencyRecord(**defaults)


def _make_retrieval(**kwargs) -> RetrievalMetrics:
    defaults = dict(
        docs_retrieved=4,
        hit_rate=0.5,
        mrr=1.0,
        ndcg_at_3=1.0,
        avg_rerank_score=0.5,
        top_rerank_score=0.8,
        avg_context_len=400.0,
    )
    defaults.update(kwargs)
    return RetrievalMetrics(**defaults)


class TestMetricsService(unittest.TestCase):

    def setUp(self):
        MetricsService._buffer = deque(maxlen=1_000)
        self.svc = MetricsService(persist_to_db=False)

    def test_sigmoid_zero_returns_half(self):
        self.assertAlmostEqual(sigmoid(0), 0.5, places=6)

    def test_sigmoid_positive_returns_above_half(self):
        self.assertGreater(sigmoid(2.0), 0.5)

    def test_sigmoid_negative_returns_below_half(self):
        self.assertLess(sigmoid(-2.0), 0.5)

    def test_sigmoid_large_positive_approaches_one(self):
        self.assertAlmostEqual(sigmoid(100), 1.0, places=5)

    def test_sigmoid_large_negative_approaches_zero(self):
        self.assertAlmostEqual(sigmoid(-100), 0.0, places=5)

    def test_sigmoid_symmetry(self):
        """sigmoid(x) + sigmoid(-x) == 1.0 for all x."""
        for x in [0.5, 1.0, 2.0, 5.0]:
            self.assertAlmostEqual(sigmoid(x) + sigmoid(-x), 1.0, places=10)

    def test_compute_retrieval_metrics_empty_docs_returns_defaults(self):
        result = compute_retrieval_metrics([])
        self.assertEqual(result.docs_retrieved, 0)
        self.assertEqual(result.hit_rate, 0.0)
        self.assertEqual(result.mrr, 0.0)
        self.assertEqual(result.ndcg_at_3, 0.0)
        self.assertEqual(result.avg_rerank_score, 0.0)
        self.assertEqual(result.top_rerank_score, 0.0)
        self.assertEqual(result.avg_context_len, 0.0)

    def test_compute_retrieval_metrics_all_positive_scores(self):
        docs = [_make_doc(2.0, "text one"), _make_doc(1.5, "text two")]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.docs_retrieved, 2)
        self.assertEqual(result.hit_rate, 1.0)
        self.assertEqual(result.mrr, 1.0)
        self.assertGreater(result.ndcg_at_3, 0.0)

    def test_compute_retrieval_metrics_all_negative_scores_hit_rate_zero(self):
        """
        sigmoid(-5) ≈ 0.007 and sigmoid(-3) ≈ 0.047, both < 0.5 threshold.
        hit_rate and mrr must be 0.0 but ndcg is non-zero because sigmoid > 0.
        """
        docs = [_make_doc(-5.0), _make_doc(-3.0)]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.hit_rate, 0.0)
        self.assertEqual(result.mrr, 0.0)

    def test_compute_retrieval_metrics_all_negative_scores_ndcg_nonzero(self):
        """
        ndcg is computed from clipped sigmoid values which are > 0 even for negative
        rerank_scores, so ndcg_at_3 will be non-zero (close to 1.0 for proportional
        DCG/IDCG).  The old assertion ndcg==0.0 was wrong.
        """
        docs = [_make_doc(-5.0), _make_doc(-3.0)]
        result = compute_retrieval_metrics(docs)
        self.assertGreater(result.ndcg_at_3, 0.0)

    def test_compute_retrieval_metrics_all_negative_avg_rerank_score(self):
        """avg_rerank_score = mean of sigmoid values, NOT mean of raw scores."""
        docs = [_make_doc(-5.0), _make_doc(-3.0)]
        expected_avg = round(
            (sigmoid(-5.0) + sigmoid(-3.0)) / 2, 4
        )
        result = compute_retrieval_metrics(docs)
        self.assertAlmostEqual(result.avg_rerank_score, expected_avg, places=4)

    def test_compute_retrieval_metrics_avg_rerank_score_symmetric(self):
        """
        For docs=[-2.0, 2.0]: sigmoid(-2)+sigmoid(2) = 1.0, mean = 0.5.
        The old test wrongly asserted 0.0 (mean of raw scores, not sigmoid values).
        """
        docs = [_make_doc(-2.0), _make_doc(2.0)]
        result = compute_retrieval_metrics(docs)
        self.assertAlmostEqual(result.avg_rerank_score, 0.5, places=4)

    def test_compute_retrieval_metrics_zero_score_is_not_a_hit(self):
        """sigmoid(0) = 0.5 which is NOT > 0.5, so it should not count as a hit."""
        docs = [_make_doc(0.0)]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.hit_rate, 0.0)
        self.assertEqual(result.mrr, 0.0)

    def test_compute_retrieval_metrics_mrr_first_hit_at_rank_2(self):
        """First doc below threshold, second above → mrr = 1/2 = 0.5."""
        docs = [_make_doc(-1.0), _make_doc(2.0)]
        result = compute_retrieval_metrics(docs)
        self.assertAlmostEqual(result.mrr, 0.5, places=4)

    def test_compute_retrieval_metrics_partial_positive_hit_rate(self):
        docs = [_make_doc(-1.0), _make_doc(2.0), _make_doc(-0.5), _make_doc(1.0)]
        result = compute_retrieval_metrics(docs)
        self.assertAlmostEqual(result.hit_rate, 0.5, places=2)

    def test_compute_retrieval_metrics_single_positive_doc(self):
        docs = [_make_doc(3.0, "hello world")]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.docs_retrieved, 1)
        self.assertEqual(result.hit_rate, 1.0)
        self.assertEqual(result.mrr, 1.0)

    def test_compute_retrieval_metrics_single_negative_doc(self):
        docs = [_make_doc(-1.0)]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.hit_rate, 0.0)
        self.assertEqual(result.mrr, 0.0)

    def test_compute_retrieval_metrics_top_rerank_score_is_first_doc_sigmoid(self):
        docs = [_make_doc(2.0, "first"), _make_doc(0.5, "second")]
        result = compute_retrieval_metrics(docs)
        self.assertAlmostEqual(result.top_rerank_score, round(sigmoid(2.0), 4), places=4)

    def test_compute_retrieval_metrics_avg_context_len(self):
        """avg_context_len = mean character length of text payloads."""
        docs = [_make_doc(1.0, "ab"), _make_doc(1.0, "abcd")]
        result = compute_retrieval_metrics(docs)
        self.assertAlmostEqual(result.avg_context_len, 3.0, places=2)

    def test_compute_retrieval_metrics_docs_retrieved_count(self):
        docs = [_make_doc(float(i)) for i in range(6)]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.docs_retrieved, 6)

    def test_compute_retrieval_metrics_k_clamped_to_doc_count(self):
        """k larger than number of docs should not raise."""
        docs = [_make_doc(1.0), _make_doc(2.0)]
        result = compute_retrieval_metrics(docs, k=10)
        self.assertIsNotNone(result.ndcg_at_3)

    def test_compute_retrieval_metrics_k3_vs_k4_differ_appropriately(self):
        docs = [_make_doc(3.0), _make_doc(2.0), _make_doc(1.0), _make_doc(0.5)]
        r3 = compute_retrieval_metrics(docs, k=3)
        r4 = compute_retrieval_metrics(docs, k=4)
        self.assertIsNotNone(r3.ndcg_at_3)
        self.assertIsNotNone(r4.ndcg_at_3)

    def test_compute_retrieval_metrics_values_are_rounded(self):
        docs = [_make_doc(1.234567, "text")]
        result = compute_retrieval_metrics(docs)
        self.assertEqual(result.hit_rate, round(result.hit_rate, 4))
        self.assertEqual(result.avg_context_len, round(result.avg_context_len, 2))

    def test_stopwatch_elapsed_none_before_stop(self):
        sw = _StopWatch()
        sw.start()
        self.assertIsNone(sw.elapsed_ms)

    def test_stopwatch_elapsed_set_after_stop(self):
        sw = _StopWatch()
        sw.start()
        sw.stop()
        self.assertIsNotNone(sw.elapsed_ms)
        self.assertGreaterEqual(sw.elapsed_ms, 0.0)

    def test_stopwatch_stop_without_start_does_nothing(self):
        """stop() with _start=None must not crash or set elapsed_ms."""
        sw = _StopWatch()
        sw.stop()
        self.assertIsNone(sw.elapsed_ms)

    def test_stopwatch_measures_real_time(self):
        sw = _StopWatch()
        sw.start()
        time.sleep(0.01)
        sw.stop()
        self.assertGreater(sw.elapsed_ms, 5.0)

    def test_sync_timer_elapsed_none_inside_block(self):
        captured = []
        with sync_timer() as sw:
            captured.append(sw.elapsed_ms)
        self.assertIsNone(captured[0])

    def test_sync_timer_elapsed_set_after_block(self):
        with sync_timer() as sw:
            pass
        self.assertIsNotNone(sw.elapsed_ms)
        self.assertGreaterEqual(sw.elapsed_ms, 0.0)

    def test_sync_timer_measures_real_time(self):
        with sync_timer() as sw:
            time.sleep(0.01)
        self.assertGreater(sw.elapsed_ms, 5.0)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_async_timer_elapsed_set_after_block(self):
        async def _inner():
            async with async_timer() as sw:
                pass
            return sw

        sw = self._run(_inner())
        self.assertIsNotNone(sw.elapsed_ms)
        self.assertGreaterEqual(sw.elapsed_ms, 0.0)

    def test_async_timer_measures_real_time(self):
        async def _inner():
            async with async_timer() as sw:
                await asyncio.sleep(0.01)
            return sw

        sw = self._run(_inner())
        self.assertGreater(sw.elapsed_ms, 5.0)

    def test_init_persist_to_db_true_sets_class_var(self):
        MetricsService(persist_to_db=True)
        self.assertTrue(MetricsService._persist_to_db)

    def test_init_persist_to_db_false_sets_class_var(self):
        MetricsService(persist_to_db=False)
        self.assertFalse(MetricsService._persist_to_db)

    def test_record_returns_valid_uuid(self):
        qid = self.svc.record("hash1", False, _make_latency(), _make_retrieval())
        self.assertIsInstance(qid, str)
        uuid.UUID(qid)  # raises ValueError if invalid

    def test_record_appends_to_buffer(self):
        self.svc.record("h1", False, _make_latency(), _make_retrieval())
        self.svc.record("h2", False, _make_latency(), _make_retrieval())
        self.assertEqual(len(MetricsService._buffer), 2)

    def test_record_stores_question_hash(self):
        self.svc.record("myhash", False, _make_latency(), _make_retrieval())
        self.assertEqual(MetricsService._buffer[-1].question_hash, "myhash")

    def test_record_stores_served_from_cache_true(self):
        self.svc.record("h", True, _make_latency(total_ms=25.0), RetrievalMetrics())
        self.assertTrue(MetricsService._buffer[-1].served_from_cache)

    def test_record_stores_served_from_cache_false(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        self.assertFalse(MetricsService._buffer[-1].served_from_cache)

    def test_record_stores_latency_values(self):
        lat = _make_latency(embed_cache_ms=15.5, llm_ms=3000.0, total_ms=5200.0)
        self.svc.record("h", False, lat, _make_retrieval())
        rec = MetricsService._buffer[-1]
        self.assertAlmostEqual(rec.latency.embed_cache_ms, 15.5)
        self.assertAlmostEqual(rec.latency.llm_ms, 3000.0)
        self.assertAlmostEqual(rec.latency.total_ms, 5200.0)

    def test_record_stores_retrieval_values(self):
        ret = _make_retrieval(hit_rate=0.75, mrr=0.5, docs_retrieved=4)
        self.svc.record("h", False, _make_latency(), ret)
        rec = MetricsService._buffer[-1]
        self.assertAlmostEqual(rec.retrieval.hit_rate, 0.75)
        self.assertAlmostEqual(rec.retrieval.mrr, 0.5)
        self.assertEqual(rec.retrieval.docs_retrieved, 4)

    def test_record_created_at_has_utc_timezone(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        rec = MetricsService._buffer[-1]
        self.assertIsNotNone(rec.created_at.tzinfo)
        self.assertEqual(rec.created_at.tzinfo, timezone.utc)

    def test_record_none_latency_fields_do_not_crash(self):
        lat = LatencyRecord()
        self.svc.record("h", False, lat, _make_retrieval())
        self.assertEqual(len(MetricsService._buffer), 1)

    def test_record_buffer_maxlen_evicts_oldest(self):
        MetricsService._buffer = deque(maxlen=5)
        for i in range(7):
            self.svc.record(f"hash{i}", False, _make_latency(), _make_retrieval())
        self.assertEqual(len(MetricsService._buffer), 5)
        self.assertEqual(MetricsService._buffer[0].question_hash, "hash2")

    def test_record_persist_false_does_not_call_persist(self):
        svc = MetricsService(persist_to_db=False)
        with patch.object(MetricsService, "_persist") as mock_persist:
            svc.record("h", False, _make_latency(), _make_retrieval())
        mock_persist.assert_not_called()

    def test_record_persist_true_runtime_error_no_loop_swallowed(self):
        svc = MetricsService(persist_to_db=True)
        with patch("app.services.Metrics_Service.asyncio.get_running_loop",
                   side_effect=RuntimeError("no running loop")):
            try:
                svc.record("h", False, _make_latency(), _make_retrieval())
            except RuntimeError:
                self.fail("record() should swallow RuntimeError from get_running_loop")

    def test_record_persist_true_with_running_loop_calls_run_in_executor(self):
        svc = MetricsService(persist_to_db=True)
        mock_loop = MagicMock()
        with patch("app.services.Metrics_Service.asyncio.get_running_loop",
                   return_value=mock_loop):
            svc.record("h", False, _make_latency(), _make_retrieval())
        mock_loop.run_in_executor.assert_called_once()
        call_args = mock_loop.run_in_executor.call_args[0]
        self.assertIsNone(call_args[0])       
        self.assertEqual(call_args[1], svc._persist)

    def test_summary_empty_buffer_returns_message(self):
        result = self.svc.summary()
        self.assertIn("message", result)

    def test_summary_total_queries_count(self):
        for _ in range(5):
            self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.summary(last_n=10)
        self.assertEqual(result["total_queries"], 5)

    def test_summary_last_n_respected(self):
        for i in range(10):
            self.svc.record("h", False, _make_latency(total_ms=float(i * 100)), _make_retrieval())
        result = self.svc.summary(last_n=3)
        self.assertEqual(result["total_queries"], 3)

    def test_summary_last_n_larger_than_buffer_returns_all(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.summary(last_n=100)
        self.assertEqual(result["total_queries"], 2)

    def test_summary_cache_hit_rate_all_cached(self):
        for _ in range(4):
            self.svc.record("h", True, _make_latency(total_ms=25.0), RetrievalMetrics())
        result = self.svc.summary()
        self.assertAlmostEqual(result["cache_hit_rate"], 1.0)

    def test_summary_cache_hit_rate_none_cached(self):
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
        self.svc.record("h", False, _make_latency(total_ms=1000.0), _make_retrieval())
        self.svc.record("h", False, _make_latency(total_ms=3000.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["all_queries"]["avg_total_ms"], 2000.0)

    def test_summary_p95_total_ms(self):
        for i in range(20):
            self.svc.record("h", False, _make_latency(total_ms=float(i * 100)), _make_retrieval())
        result = self.svc.summary()
        self.assertIsNotNone(result["latency"]["all_queries"]["p95_total_ms"])

    def test_summary_sla_ok_pct_all_within_2000ms(self):
        for _ in range(4):
            self.svc.record("h", False, _make_latency(total_ms=1500.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["all_queries"]["sla_ok_pct"], 100.0)

    def test_summary_sla_ok_pct_none_within_2000ms(self):
        for _ in range(4):
            self.svc.record("h", False, _make_latency(total_ms=5000.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["all_queries"]["sla_ok_pct"], 0.0)

    def test_summary_sla_ok_pct_all_none_total_ms_returns_none(self):
        """When all total_ms values are None, sla_ok_pct should be None."""
        self.svc.record("h", False, LatencyRecord(total_ms=None), _make_retrieval())
        result = self.svc.summary()
        self.assertIsNone(result["latency"]["all_queries"]["sla_ok_pct"])

    def test_summary_sla_ok_pct_mixed_none_total_ms(self):
        """None total_ms values are excluded from the SLA calculation."""
        self.svc.record("h", False, LatencyRecord(total_ms=None), _make_retrieval())
        self.svc.record("h", False, _make_latency(total_ms=1000.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["all_queries"]["sla_ok_pct"], 100.0)

    def test_summary_avg_llm_ms_non_cached_only(self):
        self.svc.record("h", False, _make_latency(llm_ms=1000.0, total_ms=1500.0), _make_retrieval())
        self.svc.record("h", False, _make_latency(llm_ms=3000.0, total_ms=3500.0), _make_retrieval())
        self.svc.record("h", True,  _make_latency(llm_ms=None,   total_ms=25.0),   RetrievalMetrics())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["non_cached_only"]["avg_llm_ms"], 2000.0)

    def test_summary_avg_search_ms_non_cached_only(self):
        self.svc.record("h", False, _make_latency(search_ms=100.0), _make_retrieval())
        self.svc.record("h", False, _make_latency(search_ms=300.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["non_cached_only"]["avg_search_ms"], 200.0)

    def test_summary_avg_rerank_ms_non_cached_only(self):
        self.svc.record("h", False, _make_latency(rerank_ms=50.0),  _make_retrieval())
        self.svc.record("h", False, _make_latency(rerank_ms=150.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["non_cached_only"]["avg_rerank_ms"], 100.0)

    def test_summary_avg_embed_cache_ms_non_cached_only(self):
        self.svc.record("h", False, _make_latency(embed_cache_ms=10.0), _make_retrieval())
        self.svc.record("h", False, _make_latency(embed_cache_ms=30.0), _make_retrieval())
        result = self.svc.summary()
        self.assertAlmostEqual(result["latency"]["non_cached_only"]["avg_embed_cache_ms"], 20.0)

    def test_summary_all_cached_non_cached_averages_none(self):
        for _ in range(3):
            self.svc.record("h", True, _make_latency(total_ms=25.0), RetrievalMetrics())
        result = self.svc.summary()
        nc = result["latency"]["non_cached_only"]
        self.assertIsNone(nc["avg_llm_ms"])
        self.assertIsNone(nc["avg_search_ms"])
        self.assertIsNone(nc["avg_rerank_ms"])
        self.assertIsNone(nc["avg_embed_cache_ms"])

    def test_summary_retrieval_avg_hit_rate(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval(hit_rate=1.0))
        self.svc.record("h", False, _make_latency(), _make_retrieval(hit_rate=0.0))
        result = self.svc.summary()
        self.assertAlmostEqual(result["retrieval"]["avg_hit_rate"], 0.5)

    def test_summary_retrieval_avg_mrr(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval(mrr=1.0))
        self.svc.record("h", False, _make_latency(), _make_retrieval(mrr=0.0))
        result = self.svc.summary()
        self.assertAlmostEqual(result["retrieval"]["avg_mrr"], 0.5)

    def test_summary_retrieval_avg_ndcg_at_3(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval(ndcg_at_3=1.0))
        self.svc.record("h", False, _make_latency(), _make_retrieval(ndcg_at_3=0.0))
        result = self.svc.summary()
        self.assertAlmostEqual(result["retrieval"]["avg_ndcg_at_3"], 0.5)

    def test_summary_retrieval_avg_rerank_score(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval(avg_rerank_score=0.8))
        self.svc.record("h", False, _make_latency(), _make_retrieval(avg_rerank_score=0.2))
        result = self.svc.summary()
        self.assertAlmostEqual(result["retrieval"]["avg_rerank_score"], 0.5)

    def test_summary_retrieval_top_rerank_score(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval(top_rerank_score=0.9))
        self.svc.record("h", False, _make_latency(), _make_retrieval(top_rerank_score=0.1))
        result = self.svc.summary()
        self.assertAlmostEqual(result["retrieval"]["top_rerank_score"], 0.5)

    def test_summary_retrieval_avg_context_len(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval(avg_context_len=200.0))
        self.svc.record("h", False, _make_latency(), _make_retrieval(avg_context_len=400.0))
        result = self.svc.summary()
        self.assertAlmostEqual(result["retrieval"]["avg_context_len_chars"], 300.0)

    def test_summary_retrieval_avg_docs_retrieved(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval(docs_retrieved=2))
        self.svc.record("h", False, _make_latency(), _make_retrieval(docs_retrieved=4))
        result = self.svc.summary()
        self.assertAlmostEqual(result["retrieval"]["avg_docs_retrieved"], 3.0)

    def test_summary_window_key_matches_last_n(self):
        for _ in range(5):
            self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.summary(last_n=3)
        self.assertEqual(result["window"], 3)

    def test_recent_empty_buffer_returns_empty_list(self):
        result = self.svc.recent(n=10)
        self.assertEqual(result, [])

    def test_recent_returns_list_type(self):
        result = self.svc.recent(n=10)
        self.assertIsInstance(result, list)

    def test_recent_returns_last_n_records(self):
        for _ in range(5):
            self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.recent(n=2)
        self.assertEqual(len(result), 2)

    def test_recent_n_larger_than_buffer_returns_all(self):
        for _ in range(3):
            self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.recent(n=50)
        self.assertEqual(len(result), 3)

    def test_recent_record_has_required_top_level_keys(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        rec = self.svc.recent(n=1)[0]
        for key in ("query_id", "created_at", "served_from_cache", "latency_ms", "retrieval"):
            self.assertIn(key, rec)

    def test_recent_latency_ms_has_required_sub_keys(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        lat = self.svc.recent(n=1)[0]["latency_ms"]
        for key in ("embed_cache", "search", "rerank", "llm", "total"):
            self.assertIn(key, lat)

    def test_recent_retrieval_has_required_sub_keys(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        ret = self.svc.recent(n=1)[0]["retrieval"]
        for key in ("docs_retrieved", "hit_rate", "mrr", "ndcg_at_3",
                    "avg_rerank_score", "top_rerank_score", "avg_context_len"):
            self.assertIn(key, ret)

    def test_recent_served_from_cache_true(self):
        self.svc.record("h", True, _make_latency(total_ms=25.0), RetrievalMetrics())
        result = self.svc.recent(n=1)
        self.assertTrue(result[0]["served_from_cache"])

    def test_recent_served_from_cache_false(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        result = self.svc.recent(n=1)
        self.assertFalse(result[0]["served_from_cache"])

    def test_recent_created_at_is_parseable_iso_string(self):
        self.svc.record("h", False, _make_latency(), _make_retrieval())
        created_at_str = self.svc.recent(n=1)[0]["created_at"]
        parsed = datetime.fromisoformat(created_at_str)
        self.assertIsNotNone(parsed)

    def test_recent_none_latency_fields_serialised_as_none(self):
        self.svc.record("h", False, LatencyRecord(total_ms=500.0), _make_retrieval())
        lat = self.svc.recent(n=1)[0]["latency_ms"]
        self.assertIsNone(lat["embed_cache"])
        self.assertIsNone(lat["llm"])
        self.assertIsNone(lat["search"])

    def test_recent_latency_values_match_recorded(self):
        self.svc.record("h", False, _make_latency(total_ms=999.0, llm_ms=800.0), _make_retrieval())
        lat = self.svc.recent(n=1)[0]["latency_ms"]
        self.assertAlmostEqual(lat["total"], 999.0)
        self.assertAlmostEqual(lat["llm"], 800.0)

    def test_recent_retrieval_values_match_recorded(self):
        self.svc.record("h", False, _make_latency(),
                        _make_retrieval(hit_rate=0.75, docs_retrieved=3))
        ret = self.svc.recent(n=1)[0]["retrieval"]
        self.assertAlmostEqual(ret["hit_rate"], 0.75)
        self.assertEqual(ret["docs_retrieved"], 3)

    def test_log_cache_hit_branch_does_not_raise(self):
        rec = QueryRecord(
            query_id="qid",
            question_hash="hash",
            created_at=datetime.now(timezone.utc),
            served_from_cache=True,
            latency=_make_latency(total_ms=25.0),
            retrieval=RetrievalMetrics(),
        )
        try:
            MetricsService._log(rec)
        except Exception as e:
            self.fail(f"_log raised for cache hit: {e}")

    def test_log_cache_miss_branch_does_not_raise(self):
        rec = QueryRecord(
            query_id="qid",
            question_hash="hash",
            created_at=datetime.now(timezone.utc),
            served_from_cache=False,
            latency=_make_latency(),
            retrieval=_make_retrieval(),
        )
        try:
            MetricsService._log(rec)
        except Exception as e:
            self.fail(f"_log raised for cache miss: {e}")

    def test_log_none_latency_fields_do_not_crash(self):
        rec = QueryRecord(
            query_id="qid",
            question_hash="hash",
            created_at=datetime.now(timezone.utc),
            served_from_cache=False,
            latency=LatencyRecord(),
            retrieval=RetrievalMetrics(),
        )
        try:
            MetricsService._log(rec)
        except Exception as e:
            self.fail(f"_log raised with None latency fields: {e}")

    def _make_query_record(self, latency=None, retrieval=None) -> QueryRecord:
        return QueryRecord(
            query_id="test-id",
            question_hash="hash",
            created_at=datetime.now(timezone.utc),
            served_from_cache=False,
            latency=latency or _make_latency(),
            retrieval=retrieval or _make_retrieval(),
        )

    def test_persist_calls_execute_and_commit(self):
        import app.services.Metrics_Service as ms_mod
        rec = self._make_query_record()
        mock_db = MagicMock()
        with patch.object(ms_mod, "Database", return_value=mock_db):
            MetricsService._persist(rec)
        mock_db.cursor.execute.assert_called_once()
        mock_db.conn.commit.assert_called_once()

    def test_persist_calls_return_to_pool(self):
        import app.services.Metrics_Service as ms_mod
        rec = self._make_query_record()
        mock_db = MagicMock()
        with patch.object(ms_mod, "Database", return_value=mock_db):
            MetricsService._persist(rec)
        mock_db.return_to_pool.assert_called_once()

    def test_persist_return_to_pool_called_even_on_execute_failure(self):
        import app.services.Metrics_Service as ms_mod
        rec = self._make_query_record()
        mock_db = MagicMock()
        mock_db.cursor.execute.side_effect = RuntimeError("DB error")
        with patch.object(ms_mod, "Database", return_value=mock_db):
            MetricsService._persist(rec)   # must not raise
        mock_db.return_to_pool.assert_called_once()

    def test_persist_swallows_db_exception(self):
        import app.services.Metrics_Service as ms_mod
        rec = self._make_query_record()
        mock_db = MagicMock()
        mock_db.cursor.execute.side_effect = Exception("Connection refused")
        with patch.object(ms_mod, "Database", return_value=mock_db):
            try:
                MetricsService._persist(rec)
            except Exception:
                self.fail("_persist raised an unexpected exception")

    def test_persist_none_latency_fields_do_not_crash(self):
        import app.services.Metrics_Service as ms_mod
        rec = self._make_query_record(latency=LatencyRecord(), retrieval=RetrievalMetrics())
        mock_db = MagicMock()
        with patch.object(ms_mod, "Database", return_value=mock_db):
            try:
                MetricsService._persist(rec)
            except Exception as e:
                self.fail(f"_persist raised with None fields: {e}")

    def test_persist_values_contain_correct_query_id(self):
        import app.services.Metrics_Service as ms_mod
        rec = self._make_query_record()
        rec.query_id = "unique-query-id"
        mock_db = MagicMock()
        with patch.object(ms_mod, "Database", return_value=mock_db):
            MetricsService._persist(rec)
        call_args = mock_db.cursor.execute.call_args[0]
        values = call_args[1]
        self.assertEqual(values[0], "unique-query-id")

    def test_persist_boolean_served_from_cache_passed_correctly(self):
        import app.services.Metrics_Service as ms_mod
        rec = self._make_query_record()
        rec.served_from_cache = True
        mock_db = MagicMock()
        with patch.object(ms_mod, "Database", return_value=mock_db):
            MetricsService._persist(rec)
        call_args = mock_db.cursor.execute.call_args[0]
        values = call_args[1]
        self.assertTrue(values[3])

    def test_latency_record_all_defaults_none(self):
        lat = LatencyRecord()
        self.assertIsNone(lat.embed_cache_ms)
        self.assertIsNone(lat.search_ms)
        self.assertIsNone(lat.rerank_ms)
        self.assertIsNone(lat.llm_ms)
        self.assertIsNone(lat.total_ms)

    def test_retrieval_metrics_all_defaults_zero(self):
        ret = RetrievalMetrics()
        self.assertEqual(ret.docs_retrieved, 0)
        self.assertEqual(ret.hit_rate, 0.0)
        self.assertEqual(ret.mrr, 0.0)
        self.assertEqual(ret.ndcg_at_3, 0.0)
        self.assertEqual(ret.avg_rerank_score, 0.0)
        self.assertEqual(ret.top_rerank_score, 0.0)
        self.assertEqual(ret.avg_context_len, 0.0)

    def test_query_record_fields_stored_correctly(self):
        lat = _make_latency()
        ret = _make_retrieval()
        now = datetime.now(timezone.utc)
        rec = QueryRecord(
            query_id="qid",
            question_hash="hsh",
            created_at=now,
            served_from_cache=True,
            latency=lat,
            retrieval=ret,
        )
        self.assertEqual(rec.query_id, "qid")
        self.assertEqual(rec.question_hash, "hsh")
        self.assertEqual(rec.created_at, now)
        self.assertTrue(rec.served_from_cache)
        self.assertIs(rec.latency, lat)
        self.assertIs(rec.retrieval, ret)