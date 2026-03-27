import pytest
import json
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

from app.services.Query_Service import Retrieve

def _make_doc(doc_id="doc1",
              text="Artificial Intelligence is the simulation of human intelligence.",
              source="test_doc.pdf",
              page=1):
    """Return a minimal mock Qdrant point with a payload."""
    doc = MagicMock()
    doc.id = doc_id
    doc.payload = {"text": text, "source": source, "page": page}
    return doc


def _make_retrieve():
    mock_doc = _make_doc()

    with patch("app.services.Query_Service.redis.Redis.from_url"), \
         patch("app.services.Query_Service.SentenceTransformer"), \
         patch("app.services.Query_Service.CrossEncoder"), \
         patch("app.services.Query_Service.nltk.download"), \
         patch.object(Retrieve, "load_documents", return_value=[{
             "doc": mock_doc,
             "vector_score": 0,
             "keyword_score": 0
         }]):
        retrieve = Retrieve()

    retrieve.redis_client = MagicMock()
    retrieve.embedding_model = MagicMock()
    retrieve.reranker = MagicMock()
    retrieve.database = MagicMock()
    retrieve.database.client = MagicMock()
    retrieve.response = MagicMock()
    retrieve.config = MagicMock()
    retrieve.config.COLLECTION_NAME = "test_collection"
    retrieve.config.CACHE_THRESHOLD = 0.95
    retrieve.config.CACHE_TTL = 300

    return retrieve, mock_doc


class TestRetrieveService:

    def setup_method(self):
        self.retrieve, self.mock_doc = _make_retrieve()
        self.embedding = np.random.rand(384).astype(np.float32)


    def test_load_documents_returns_correct_structure(self):
        """Each entry must have doc, vector_score=0, keyword_score=0."""
        mock_pt = _make_doc("d1", "some text")
        self.retrieve.database.client.scroll.return_value = ([mock_pt], None)

        docs = self.retrieve.load_documents(limit=10)

        assert len(docs) == 1
        assert docs[0]["doc"] is mock_pt
        assert docs[0]["vector_score"] == 0
        assert docs[0]["keyword_score"] == 0

    def test_load_documents_calls_scroll_with_correct_args(self):
        """scroll must be called with the configured collection name and limit."""
        self.retrieve.database.client.scroll.return_value = ([], None)

        self.retrieve.load_documents(limit=42)

        self.retrieve.database.client.scroll.assert_called_once_with(
            collection_name="test_collection",
            limit=42,
            with_payload=True,
            with_vectors=False
        )

    def test_load_documents_empty_collection(self):
        """Empty scroll result → empty list returned."""
        self.retrieve.database.client.scroll.return_value = ([], None)
        docs = self.retrieve.load_documents()
        assert docs == []

    def test_load_documents_multiple_points(self):
        """Multiple points are all wrapped correctly."""
        pts = [_make_doc(f"d{i}") for i in range(5)]
        self.retrieve.database.client.scroll.return_value = (pts, None)

        docs = self.retrieve.load_documents()
        assert len(docs) == 5

    def test_refresh_bm25_reloads_documents(self):
        """refresh_bm25 should call load_documents and rebuild self.bm25."""
        new_doc = _make_doc("d2", "new document text")
        self.retrieve.database.client.scroll.return_value = ([new_doc], None)

        with patch.object(self.retrieve, "load_documents",
                          return_value=[{"doc": new_doc, "vector_score": 0, "keyword_score": 0}]) as mock_load:
            self.retrieve.refresh_bm25()

        mock_load.assert_called_once()
        assert self.retrieve.documents[0]["doc"] is new_doc

    def test_refresh_bm25_handles_exception_gracefully(self):
        """If load_documents raises, refresh_bm25 should swallow the error."""
        with patch.object(self.retrieve, "load_documents", side_effect=Exception("DB down")):
            # should not raise
            self.retrieve.refresh_bm25()

    def test_refresh_bm25_rebuilds_bm25_index(self):
        """After refresh, self.bm25 must be a new BM25Okapi instance."""
        from rank_bm25 import BM25Okapi
        old_bm25 = self.retrieve.bm25
        new_doc = _make_doc("d3", "refreshed content here")

        with patch.object(self.retrieve, "load_documents",
                          return_value=[{"doc": new_doc, "vector_score": 0, "keyword_score": 0}]):
            self.retrieve.refresh_bm25()

        assert self.retrieve.bm25 is not old_bm25
        assert isinstance(self.retrieve.bm25, BM25Okapi)

    def test_keyword_search_returns_list(self):
        result = self.retrieve.keyword_search("Artificial Intelligence")
        assert isinstance(result, list)

    def test_keyword_search_max_10_results(self):
        """keyword_search should return at most 10 results."""
        result = self.retrieve.keyword_search("something")
        assert len(result) <= 10

    def test_keyword_search_result_structure(self):
        """Each result must have doc, vector_score=0, keyword_score keys."""
        result = self.retrieve.keyword_search("AI")
        if result:
            assert "doc" in result[0]
            assert result[0]["vector_score"] == 0
            assert "keyword_score" in result[0]

    def test_keyword_search_sorted_descending(self):
        """Results must be sorted by keyword_score descending."""
        result = self.retrieve.keyword_search("Artificial Intelligence")
        scores = [r["keyword_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_keyword_search_empty_query(self):
        """Empty query string should not raise."""
        result = self.retrieve.keyword_search("")
        assert isinstance(result, list)

    def test_keyword_search_expands_synonyms(self):
        """keyword_search should use expanded token set (doesn't raise with synonyms)."""
        with patch("app.services.Query_Service.wordnet") as mock_wn:
            mock_syn = MagicMock()
            mock_lemma = MagicMock()
            mock_lemma.name.return_value = "machine_learning"
            mock_syn.lemmas.return_value = [mock_lemma]
            mock_wn.synsets.return_value = [mock_syn]

            result = self.retrieve.keyword_search("AI")

        assert isinstance(result, list)

    def test_vector_search_returns_list(self):
        mock_pt = MagicMock()
        mock_pt.score = 0.9
        mock_pt.id = "d1"
        mock_pt.payload = {"text": "text", "source": "src", "page": 1}
        self.retrieve.database.client.query_points.return_value = MagicMock(points=[mock_pt])

        docs = self.retrieve.vector_search(self.embedding.tolist())

        assert isinstance(docs, list)
        assert len(docs) == 1

    def test_vector_search_score_stored_correctly(self):
        mock_pt = MagicMock()
        mock_pt.score = 0.87
        mock_pt.id = "d1"
        self.retrieve.database.client.query_points.return_value = MagicMock(points=[mock_pt])

        docs = self.retrieve.vector_search(self.embedding.tolist())
        assert docs[0]["vector_score"] == pytest.approx(0.87)

    def test_vector_search_keyword_score_is_zero(self):
        mock_pt = MagicMock()
        mock_pt.score = 0.5
        mock_pt.id = "d1"
        self.retrieve.database.client.query_points.return_value = MagicMock(points=[mock_pt])

        docs = self.retrieve.vector_search(self.embedding.tolist())
        assert docs[0]["keyword_score"] == 0

    def test_vector_search_empty_results(self):
        self.retrieve.database.client.query_points.return_value = MagicMock(points=[])
        docs = self.retrieve.vector_search(self.embedding.tolist())
        assert docs == []

    def test_vector_search_passes_limit(self):
        """query_points must be called with the given limit."""
        self.retrieve.database.client.query_points.return_value = MagicMock(points=[])
        self.retrieve.vector_search(self.embedding.tolist(), limit=5)

        call_kwargs = self.retrieve.database.client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 5

    def test_vector_search_passes_collection_name(self):
        self.retrieve.database.client.query_points.return_value = MagicMock(points=[])
        self.retrieve.vector_search(self.embedding.tolist())

        call_kwargs = self.retrieve.database.client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_collection"

    def test_merge_results_deduplicates_by_doc_id(self):
        """Two entries with the same doc id should be merged into one."""
        results = [
            {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0},
            {"doc": self.mock_doc, "vector_score": 0.5, "keyword_score": 0.8},
        ]
        merged = self.retrieve.merge_results(results)
        assert len(merged) == 1

    def test_merge_results_keeps_max_vector_score(self):
        """After merge, the higher vector_score must be kept."""
        results = [
            {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0},
            {"doc": self.mock_doc, "vector_score": 0.4, "keyword_score": 0},
        ]
        merged = self.retrieve.merge_results(results)
        assert merged[0]["vector_score"] == pytest.approx(0.9)

    def test_merge_results_different_docs_both_kept(self):
        doc2 = _make_doc("doc2", "second doc")
        results = [
            {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0},
            {"doc": doc2, "vector_score": 0.7, "keyword_score": 0},
        ]
        merged = self.retrieve.merge_results(results)
        assert len(merged) == 2

    def test_merge_results_empty_input(self):
        merged = self.retrieve.merge_results([])
        assert merged == []

    def test_merge_results_single_item(self):
        results = [{"doc": self.mock_doc, "vector_score": 0.6, "keyword_score": 0.3}]
        merged = self.retrieve.merge_results(results)
        assert len(merged) == 1
        assert merged[0]["vector_score"] == pytest.approx(0.6)

    def test_combine_scores_computes_hybrid_score(self):
        docs = [{"doc": self.mock_doc, "vector_score": 0.8, "keyword_score": 0.4}]
        ranked = self.retrieve.combine_scores(docs)
        expected = 0.75 * 0.8 + 0.25 * 0.4
        assert ranked[0]["hybrid_score"] == pytest.approx(expected)

    def test_combine_scores_returns_at_most_4(self):
        docs = [
            {"doc": _make_doc(str(i)), "vector_score": float(i) / 10, "keyword_score": 0}
            for i in range(10)
        ]
        ranked = self.retrieve.combine_scores(docs)
        assert len(ranked) <= 4

    def test_combine_scores_sorted_descending(self):
        docs = [
            {"doc": _make_doc("a"), "vector_score": 0.2, "keyword_score": 0.1},
            {"doc": _make_doc("b"), "vector_score": 0.9, "keyword_score": 0.8},
            {"doc": _make_doc("c"), "vector_score": 0.5, "keyword_score": 0.3},
        ]
        ranked = self.retrieve.combine_scores(docs)
        scores = [r["hybrid_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_combine_scores_zero_scores(self):
        docs = [{"doc": self.mock_doc, "vector_score": 0.0, "keyword_score": 0.0}]
        ranked = self.retrieve.combine_scores(docs)
        assert ranked[0]["hybrid_score"] == pytest.approx(0.0)

    def test_combine_scores_empty_input(self):
        ranked = self.retrieve.combine_scores([])
        assert ranked == []

    def test_rerank_single_doc_returned_as_is(self):
        """With only 1 doc, reranker.predict must NOT be called."""
        docs = [{"doc": self.mock_doc}]
        result = self.retrieve.rerank("question", docs)
        assert result == docs
        self.retrieve.reranker.predict.assert_not_called()

    def test_rerank_two_or_more_docs_calls_predict(self):
        docs = [
               {"doc": self.mock_doc},
               {"doc": _make_doc("d2", "another doc")},
               {"doc": _make_doc("d3", "third doc")}
            ]
        
        self.retrieve.reranker.predict.return_value = [0.9, 0.6, 0.5]
        self.retrieve.rerank("question", docs)
        self.retrieve.reranker.predict.assert_called_once()

    def test_rerank_assigns_rerank_score(self):
        docs = [{"doc": self.mock_doc}, {"doc": _make_doc("d2", "second")}]
        self.retrieve.reranker.predict.return_value = [0.8, 0.5]

        ranked = self.retrieve.rerank("question", docs)
        assert "rerank_score" in ranked[0]

    def test_rerank_sorted_by_rerank_score_descending(self):
        docs = [
            {"doc": self.mock_doc},
            {"doc": _make_doc("d2", "second doc")},
            {"doc": _make_doc("d3", "third doc")},
        ]
        self.retrieve.reranker.predict.return_value = [0.3, 0.9, 0.6]

        ranked = self.retrieve.rerank("question", docs)
        scores = [r["rerank_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_returns_at_most_8(self):
        docs = [{"doc": _make_doc(str(i), f"text {i}")} for i in range(12)]
        self.retrieve.reranker.predict.return_value = [float(i) / 12 for i in range(12)]

        ranked = self.retrieve.rerank("question", docs)
        assert len(ranked) <= 8

    def test_rerank_uses_batch_size_8(self):
        docs = [
        {"doc": self.mock_doc},
        {"doc": _make_doc("d2", "text")},
        {"doc": _make_doc("d3", "another")}
    ]
        
        self.retrieve.reranker.predict.return_value = [0.7, 0.4, 0.3]
        self.retrieve.rerank("question", docs)
        call_kwargs = self.retrieve.reranker.predict.call_args[1]
        assert call_kwargs.get("batch_size") == 8

    @pytest.mark.parametrize("question", [
        "summarize the document",
        "give me a summary",
        "provide an overview of document",
        "what does the document say",
        "what does the file say",
        "tell me about the document",
        "summarise the report",
        "describe the document",
    ])
    def test_is_summary_request_positive(self, question):
        assert self.retrieve._is_summary_request(question) is True

    @pytest.mark.parametrize("question", [
        "What is machine learning?",
        "How does AI work?",
        "Who wrote the paper?",
        "List the key findings",
    ])
    def test_is_summary_request_negative(self, question):
        assert self.retrieve._is_summary_request(question) is False

    def test_is_summary_request_case_insensitive(self):
        assert self.retrieve._is_summary_request("SUMMARIZE this") is True
        assert self.retrieve._is_summary_request("Give Me A Summary") is True

    def test_normalize_question_lowercases(self):
        assert self.retrieve.normalize_question("What Is AI?") == "what is ai?"

    def test_normalize_question_strips_whitespace(self):
        assert self.retrieve.normalize_question("  hello world  ") == "hello world"

    def test_normalize_question_collapses_spaces(self):
        assert self.retrieve.normalize_question("hello   world") == "hello world"

    def test_normalize_question_empty_string(self):
        assert self.retrieve.normalize_question("") == ""

    def test_normalize_question_combined(self):
        assert self.retrieve.normalize_question("  What  IS   AI?  ") == "what is ai?"

    def test_make_cache_key_prefix(self):
        key = self.retrieve.make_cache_key("any question")
        assert key.startswith("semantic_cache:")

    def test_make_cache_key_deterministic(self):
        key1 = self.retrieve.make_cache_key("hello")
        key2 = self.retrieve.make_cache_key("hello")
        assert key1 == key2

    def test_make_cache_key_different_questions_different_keys(self):
        key1 = self.retrieve.make_cache_key("question one")
        key2 = self.retrieve.make_cache_key("question two")
        assert key1 != key2

    def test_make_cache_key_normalizes_before_hashing(self):
        key1 = self.retrieve.make_cache_key("What is AI?")
        key2 = self.retrieve.make_cache_key("  what  is  ai?  ")
        assert key1 == key2

    def test_make_cache_key_uses_sha256(self):
        import hashlib
        question = "hello"
        normalized = self.retrieve.normalize_question(question)
        expected_hash = hashlib.sha256(normalized.encode()).hexdigest()
        expected_key = f"semantic_cache:{expected_hash}"
        assert self.retrieve.make_cache_key(question) == expected_key

    def test_check_cache_hit_returns_response(self):
        cache_data = {"response": "cached answer"}
        self.retrieve.redis_client.get.return_value = json.dumps(cache_data)

        result = self.retrieve.check_cache("any question")
        assert result == "cached answer"

    def test_check_cache_miss_returns_none(self):
        self.retrieve.redis_client.get.return_value = None
        result = self.retrieve.check_cache("unknown question")
        assert result is None

    def test_check_cache_calls_redis_with_correct_key(self):
        self.retrieve.redis_client.get.return_value = None
        self.retrieve.check_cache("my question")

        expected_key = self.retrieve.make_cache_key("my question")
        self.retrieve.redis_client.get.assert_called_once_with(expected_key)

    def test_check_cache_normalizes_question(self):
        cache_data = {"response": "answer"}
        self.retrieve.redis_client.get.return_value = json.dumps(cache_data)

        r1 = self.retrieve.check_cache("What is AI?")
        r2 = self.retrieve.check_cache("  what  is  ai?  ")
        # Both calls should use the same normalized key
        calls = self.retrieve.redis_client.get.call_args_list
        assert calls[0][0][0] == calls[1][0][0]

    def test_save_cache_valid_response_stores_in_redis(self):
        self.retrieve.save_cache("question", "valid answer")
        self.retrieve.redis_client.setex.assert_called_once()

    def test_save_cache_invalid_response_not_stored(self):
        self.retrieve.save_cache(
            "question",
            "The documents does not have a specific answer to your question."
        )
        self.retrieve.redis_client.setex.assert_not_called()

    def test_save_cache_uses_configured_ttl(self):
        self.retrieve.config.CACHE_TTL = 600
        self.retrieve.save_cache("q", "answer")
        call_args = self.retrieve.redis_client.setex.call_args[0]
        assert call_args[1] == 600

    def test_save_cache_key_matches_make_cache_key(self):
        question = "what is AI?"
        self.retrieve.save_cache(question, "answer")
        expected_key = self.retrieve.make_cache_key(question)
        call_args = self.retrieve.redis_client.setex.call_args[0]
        assert call_args[0] == expected_key

    def test_save_cache_payload_contains_response(self):
        self.retrieve.save_cache("q", "my answer")
        call_args = self.retrieve.redis_client.setex.call_args[0]
        payload = json.loads(call_args[2])
        assert payload["response"] == "my answer"

    def test_build_context_non_summary_includes_source_and_page(self):
        docs = [{"doc": self.mock_doc}]
        context, sources = self.retrieve._build_context(docs, is_summary=False)
        assert "Source:" in context
        assert "Page:" in context

    def test_build_context_summary_mode_no_source_inline(self):
        docs = [{"doc": self.mock_doc}]
        context, sources = self.retrieve._build_context(docs, is_summary=True)
        assert "Source:" not in context

    def test_build_context_summary_returns_sources_list(self):
        docs = [{"doc": self.mock_doc}]
        context, sources = self.retrieve._build_context(docs, is_summary=True)
        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_build_context_non_summary_sources_list_empty(self):
        docs = [{"doc": self.mock_doc}]
        context, sources = self.retrieve._build_context(docs, is_summary=False)
        assert sources == []

    def test_build_context_multiple_docs_joined_with_double_newline(self):
        doc2 = _make_doc("d2", "second document text", source="other.pdf", page=2)
        docs = [{"doc": self.mock_doc}, {"doc": doc2}]
        context, _ = self.retrieve._build_context(docs, is_summary=False)
        assert "\n\n" in context

    def test_build_context_deduplicates_sources_in_summary_mode(self):
        doc2 = _make_doc("d2", "more text", source="test_doc.pdf", page=2)
        docs = [{"doc": self.mock_doc}, {"doc": doc2}]
        _, sources = self.retrieve._build_context(docs, is_summary=True)
        assert sources.count(sources[0]) == 1

    def test_build_context_empty_docs(self):
        context, sources = self.retrieve._build_context([], is_summary=False)
        assert context == ""
        assert sources == []

    def test_build_context_missing_source_defaults_to_unknown(self):
        doc = _make_doc()
        doc.payload = {"text": "text"}  # no 'source' key
        docs = [{"doc": doc}]
        context, _ = self.retrieve._build_context(docs, is_summary=False)
        assert "Unknown" in context

    def test_build_context_source_strips_prefix_in_summary_mode(self):
        doc = _make_doc(source="prefix_filename.pdf")
        docs = [{"doc": doc}]
        _, sources = self.retrieve._build_context(docs, is_summary=True)
        assert "filename.pdf" in sources[0]

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_query_docs_returns_cached_response_when_cache_hit(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value="from cache")
        self.retrieve.vector_search = MagicMock()
        self.retrieve.keyword_search = MagicMock()

        result = self._run(self.retrieve.query_docs("What is AI?"))

        assert result == "from cache"
        self.retrieve.vector_search.assert_not_called()
        self.retrieve.keyword_search.assert_not_called()

    def test_query_docs_full_pipeline_on_cache_miss(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)

        vector_doc = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        keyword_doc = {"doc": self.mock_doc, "vector_score": 0, "keyword_score": 0.6}

        self.retrieve.vector_search = MagicMock(return_value=[vector_doc])
        self.retrieve.keyword_search = MagicMock(return_value=[keyword_doc])
        self.retrieve.merge_results = MagicMock(return_value=[vector_doc])
        self.retrieve.combine_scores = MagicMock(return_value=[vector_doc])
        self.retrieve.rerank = MagicMock(return_value=[vector_doc])
        self.retrieve.response.generate_response = MagicMock(return_value="final answer")
        self.retrieve.save_cache = MagicMock()

        result = self._run(self.retrieve.query_docs("What is AI?"))

        assert result == "final answer"

    def test_query_docs_calls_save_cache_after_generating_answer(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        self.retrieve.response.generate_response = MagicMock(return_value="answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("question"))

        self.retrieve.save_cache.assert_called_once_with("question", "answer")

    def test_query_docs_summary_request_passes_sources_to_generate(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        self.retrieve.response.generate_response = MagicMock(return_value="summary answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("summarize the document"))

        call_kwargs = self.retrieve.response.generate_response.call_args[1]
        assert call_kwargs.get("sources") is not None

    def test_query_docs_non_summary_passes_none_sources(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        self.retrieve.response.generate_response = MagicMock(return_value="answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("What is AI?"))

        call_kwargs = self.retrieve.response.generate_response.call_args[1]
        assert call_kwargs.get("sources") is None

    def test_query_docs_slices_final_docs_to_6(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)

        many_docs = [
            {"doc": _make_doc(str(i), f"text {i}"), "vector_score": 0.9, "keyword_score": 0}
            for i in range(10)
        ]
        self.retrieve.vector_search = MagicMock(return_value=many_docs)
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=many_docs)
        self.retrieve.combine_scores = MagicMock(return_value=many_docs)
        self.retrieve.rerank = MagicMock(return_value=many_docs)

        captured = {}

        def fake_build_context(final_docs, is_summary):
            captured["count"] = len(final_docs)
            return "context", []

        self.retrieve._build_context = fake_build_context
        self.retrieve.response.generate_response = MagicMock(return_value="answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("What is AI?"))

        assert captured["count"] == 4

    def test_query_docs_embedding_used_for_vector_search(self):
        fixed_embedding = np.ones(384, dtype=np.float32)
        self.retrieve.embedding_model.encode.return_value = fixed_embedding
        self.retrieve.check_cache = MagicMock(return_value=None)

        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        self.retrieve.response.generate_response = MagicMock(return_value="answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("What is AI?"))

        call_args = self.retrieve.vector_search.call_args[0][0]
        assert call_args == fixed_embedding.tolist()
