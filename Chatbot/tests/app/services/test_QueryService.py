import json
import asyncio
import hashlib
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from rank_bm25 import BM25Okapi

from app.services.Query_Service import Retrieve

def _make_doc(doc_id="doc1",
              text="Artificial Intelligence is the simulation of human intelligence.",
              source="test_doc.pdf",
              page=1):
    doc = MagicMock()
    doc.id = doc_id
    doc.payload = {"text": text, "source": source, "page": page}
    return doc


def _make_retrieve():
    """
    Construct a Retrieve instance with all heavy I/O mocked out.
    Returns (retrieve, mock_doc).
    """
    mock_doc = _make_doc()

    with patch("app.services.Query_Service.redis.Redis.from_url"), \
         patch("app.services.Query_Service.SentenceTransformer"), \
         patch("app.services.Query_Service.CrossEncoder"), \
         patch("app.services.Query_Service.nltk.download"), \
         patch.object(Retrieve, "load_documents", return_value=[{
             "doc": mock_doc,
             "vector_score": 0,
             "keyword_score": 0,
         }]):
        retrieve = Retrieve()

    retrieve.redis_client = MagicMock()
    retrieve.embedding_model = MagicMock()
    retrieve.reranker = MagicMock()
    retrieve.database = MagicMock()
    retrieve.database.client = MagicMock()
    retrieve.response = MagicMock()
    retrieve.metrics = MagicMock()
    retrieve.config = MagicMock()
    retrieve.config.COLLECTION_NAME = "test_collection"
    retrieve.config.CACHE_TTL = 300

    return retrieve, mock_doc

class TestRetrieveService:

    def setup_method(self):
        self.retrieve, self.mock_doc = _make_retrieve()
        self.embedding = np.random.rand(384).astype(np.float32)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_init_sets_documents_from_load(self):
        assert len(self.retrieve.documents) == 1
        assert self.retrieve.documents[0]["doc"] is self.mock_doc

    def test_init_load_documents_fails_sets_empty_list(self):
        mock_doc = _make_doc()
        with patch("app.services.Query_Service.redis.Redis.from_url"), \
             patch("app.services.Query_Service.SentenceTransformer"), \
             patch("app.services.Query_Service.CrossEncoder"), \
             patch("app.services.Query_Service.nltk.download"), \
             patch.object(Retrieve, "load_documents",
                          side_effect=Exception("DB unavailable")):
            retrieve = Retrieve()
        assert retrieve.documents == []

    def test_init_bm25_is_created(self):
        assert isinstance(self.retrieve.bm25, BM25Okapi)

    def test_init_bm25_is_none_when_corpus_fails(self):
        """If BM25Okapi construction raises, self.bm25 is None."""
        mock_doc = _make_doc()
        with patch("app.services.Query_Service.redis.Redis.from_url"), \
             patch("app.services.Query_Service.SentenceTransformer"), \
             patch("app.services.Query_Service.CrossEncoder"), \
             patch("app.services.Query_Service.nltk.download"), \
             patch.object(Retrieve, "load_documents", return_value=[{
                 "doc": mock_doc, "vector_score": 0, "keyword_score": 0
             }]), \
             patch("app.services.Query_Service.BM25Okapi",
                   side_effect=Exception("BM25 error")):
            retrieve = Retrieve()
        assert retrieve.bm25 is None

    def test_load_documents_returns_correct_structure(self):
        mock_pt = _make_doc("d1", "some text")
        self.retrieve.database.client.scroll.return_value = ([mock_pt], None)

        docs = self.retrieve.load_documents(limit=10)

        assert len(docs) == 1
        assert docs[0]["doc"] is mock_pt
        assert docs[0]["vector_score"] == 0
        assert docs[0]["keyword_score"] == 0

    def test_load_documents_calls_scroll_with_correct_args(self):
        self.retrieve.database.client.scroll.return_value = ([], None)

        self.retrieve.load_documents(limit=42)

        self.retrieve.database.client.scroll.assert_called_once_with(
            collection_name="test_collection",
            limit=42,
            with_payload=True,
            with_vectors=False,
        )

    def test_load_documents_empty_collection(self):
        self.retrieve.database.client.scroll.return_value = ([], None)
        docs = self.retrieve.load_documents()
        assert docs == []

    def test_load_documents_multiple_points(self):
        pts = [_make_doc(f"d{i}") for i in range(5)]
        self.retrieve.database.client.scroll.return_value = (pts, None)

        docs = self.retrieve.load_documents()
        assert len(docs) == 5

    def test_refresh_bm25_reloads_documents(self):
        new_doc = _make_doc("d2", "new document text")
        with patch.object(self.retrieve, "load_documents",
                          return_value=[{"doc": new_doc, "vector_score": 0, "keyword_score": 0}]) as mock_load:
            self.retrieve.refresh_bm25()

        mock_load.assert_called_once()
        assert self.retrieve.documents[0]["doc"] is new_doc

    def test_refresh_bm25_rebuilds_bm25_index(self):
        old_bm25 = self.retrieve.bm25
        new_doc = _make_doc("d3", "refreshed content here")

        with patch.object(self.retrieve, "load_documents",
                          return_value=[{"doc": new_doc, "vector_score": 0, "keyword_score": 0}]):
            self.retrieve.refresh_bm25()

        assert self.retrieve.bm25 is not old_bm25
        assert isinstance(self.retrieve.bm25, BM25Okapi)

    def test_refresh_bm25_handles_exception_sets_bm25_none(self):
        """If load_documents raises, refresh_bm25 swallows and sets bm25=None."""
        with patch.object(self.retrieve, "load_documents", side_effect=Exception("DB down")):
            self.retrieve.refresh_bm25()

        assert self.retrieve.bm25 is None

    def test_keyword_search_returns_list(self):
        result = self.retrieve.keyword_search("Artificial Intelligence")
        assert isinstance(result, list)

    def test_keyword_search_max_10_results(self):
        result = self.retrieve.keyword_search("something")
        assert len(result) <= 10

    def test_keyword_search_result_structure(self):
        result = self.retrieve.keyword_search("AI")
        if result:
            assert "doc" in result[0]
            assert result[0]["vector_score"] == 0
            assert "keyword_score" in result[0]

    def test_keyword_search_sorted_descending(self):
        result = self.retrieve.keyword_search("Artificial Intelligence")
        scores = [r["keyword_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_keyword_search_empty_query(self):
        result = self.retrieve.keyword_search("")
        assert isinstance(result, list)

    def test_keyword_search_lazy_init_when_bm25_is_none(self):
        """When bm25 is None, keyword_search re-initialises it from load_documents."""
        self.retrieve.bm25 = None
        new_doc = _make_doc("d9", "lazy init text")

        with patch.object(self.retrieve, "load_documents",
                          return_value=[{"doc": new_doc, "vector_score": 0, "keyword_score": 0}]):
            result = self.retrieve.keyword_search("lazy")

        assert isinstance(result, list)
        assert self.retrieve.bm25 is not None

    def test_keyword_search_lazy_init_failure_returns_empty(self):
        """When bm25 is None and lazy init raises, returns []."""
        self.retrieve.bm25 = None

        with patch.object(self.retrieve, "load_documents", side_effect=Exception("fail")):
            result = self.retrieve.keyword_search("query")

        assert result == []

    def test_keyword_search_expands_synonyms(self):
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
        results = [
            {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0},
            {"doc": self.mock_doc, "vector_score": 0.5, "keyword_score": 0.8},
        ]
        merged = self.retrieve.merge_results(results)
        assert len(merged) == 1

    def test_merge_results_keeps_max_vector_score(self):
        results = [
            {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0},
            {"doc": self.mock_doc, "vector_score": 0.4, "keyword_score": 0},
        ]
        merged = self.retrieve.merge_results(results)
        assert merged[0]["vector_score"] == pytest.approx(0.9)

    def test_merge_results_keeps_max_keyword_score(self):
        results = [
            {"doc": self.mock_doc, "vector_score": 0, "keyword_score": 0.3},
            {"doc": self.mock_doc, "vector_score": 0, "keyword_score": 0.9},
        ]
        merged = self.retrieve.merge_results(results)
        assert merged[0]["keyword_score"] == pytest.approx(0.9)

    def test_merge_results_different_docs_both_kept(self):
        doc2 = _make_doc("doc2", "second doc")
        results = [
            {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0},
            {"doc": doc2, "vector_score": 0.7, "keyword_score": 0},
        ]
        merged = self.retrieve.merge_results(results)
        assert len(merged) == 2

    def test_merge_results_empty_input(self):
        assert self.retrieve.merge_results([]) == []

    def test_merge_results_single_item(self):
        results = [{"doc": self.mock_doc, "vector_score": 0.6, "keyword_score": 0.3}]
        merged = self.retrieve.merge_results(results)
        assert len(merged) == 1
        assert merged[0]["vector_score"] == pytest.approx(0.6)

    def test_combine_scores_computes_hybrid_score(self):
        """hybrid = 0.60 * vector_score + 0.40 * keyword_score"""
        docs = [{"doc": self.mock_doc, "vector_score": 0.8, "keyword_score": 0.4}]
        ranked = self.retrieve.combine_scores(docs)
        expected = 0.60 * 0.8 + 0.40 * 0.4
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
        assert self.retrieve.combine_scores([]) == []

    def test_rerank_empty_docs_returns_empty(self):
        result = self.retrieve.rerank("question", [])
        assert result == []
        self.retrieve.reranker.predict.assert_not_called()

    def test_rerank_calls_predict(self):
        docs = [
            {"doc": self.mock_doc},
            {"doc": _make_doc("d2", "another doc")},
        ]
        self.retrieve.reranker.predict.return_value = [0.9, 0.6]
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
        ]
        self.retrieve.reranker.predict.return_value = [0.7, 0.4]
        self.retrieve.rerank("question", docs)

        call_kwargs = self.retrieve.reranker.predict.call_args[1]
        assert call_kwargs.get("batch_size") == 8

    @pytest.mark.parametrize("question", [
        "summarize the document",
        "summarise the report",
        "give me a summary",
        "provide a summarization",
        "overview of document",
        "what does the document say",
        "what does the file say",
        "tell me about the document",
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

    @pytest.mark.parametrize("question", [
        "what documents do you have",
        "what files do you have",
        "what information do you have",
        "list all documents",
        "list all files",
        "list the documents",
        "show all documents",
        "what do you have",
        "what's available",
        "what is available",
    ])
    def test_is_inventory_request_positive(self, question):
        assert self.retrieve._is_inventory_request(question) is True

    @pytest.mark.parametrize("question", [
        "What is AI?",
        "How does the model work?",
        "Summarize the document",
    ])
    def test_is_inventory_request_negative(self, question):
        assert self.retrieve._is_inventory_request(question) is False

    def test_is_inventory_request_case_insensitive(self):
        assert self.retrieve._is_inventory_request("What Documents Do You Have") is True
        assert self.retrieve._is_inventory_request("LIST ALL FILES") is True

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
        assert self.retrieve.make_cache_key("hello") == self.retrieve.make_cache_key("hello")

    def test_make_cache_key_different_questions_different_keys(self):
        assert self.retrieve.make_cache_key("question one") != self.retrieve.make_cache_key("question two")

    def test_make_cache_key_normalizes_before_hashing(self):
        assert self.retrieve.make_cache_key("What is AI?") == self.retrieve.make_cache_key("  what  is  ai?  ")

    def test_make_cache_key_uses_sha256(self):
        question = "hello"
        normalized = self.retrieve.normalize_question(question)
        expected = f"semantic_cache:{hashlib.sha256(normalized.encode()).hexdigest()}"
        assert self.retrieve.make_cache_key(question) == expected

    def test_check_cache_hit_returns_response(self):
        self.retrieve.redis_client.get.return_value = json.dumps({"response": "cached answer"})
        assert self.retrieve.check_cache("any question") == "cached answer"

    def test_check_cache_miss_returns_none(self):
        self.retrieve.redis_client.get.return_value = None
        assert self.retrieve.check_cache("unknown question") is None

    def test_check_cache_calls_redis_with_correct_key(self):
        self.retrieve.redis_client.get.return_value = None
        self.retrieve.check_cache("my question")

        expected_key = self.retrieve.make_cache_key("my question")
        self.retrieve.redis_client.get.assert_called_once_with(expected_key)

    def test_check_cache_normalizes_question_before_lookup(self):
        self.retrieve.redis_client.get.return_value = json.dumps({"response": "ans"})
        self.retrieve.check_cache("What is AI?")
        self.retrieve.check_cache("  what  is  ai?  ")

        calls = self.retrieve.redis_client.get.call_args_list
        assert calls[0][0][0] == calls[1][0][0]

    def test_save_cache_valid_response_stores_in_redis(self):
        self.retrieve.save_cache("question", "valid answer")
        self.retrieve.redis_client.setex.assert_called_once()

    def test_save_cache_not_found_phrase_one_not_stored(self):
        self.retrieve.save_cache(
            "question",
            "The documents does not have a specific answer to your question."
        )
        self.retrieve.redis_client.setex.assert_not_called()

    def test_save_cache_not_found_phrase_two_not_stored(self):
        self.retrieve.save_cache(
            "question",
            "The documents do not have a specific answer to your question."
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

    def test_build_context_includes_source_and_page(self):
        docs = [{"doc": self.mock_doc}]
        result = self.retrieve._build_context(docs)
        assert "Source:" in result
        assert "Page:" in result

    def test_build_context_includes_doc_text(self):
        docs = [{"doc": self.mock_doc}]
        result = self.retrieve._build_context(docs)
        assert self.mock_doc.payload["text"] in result

    def test_build_context_multiple_docs_joined_with_double_newline(self):
        doc2 = _make_doc("d2", "second document text", source="other.pdf", page=2)
        docs = [{"doc": self.mock_doc}, {"doc": doc2}]
        result = self.retrieve._build_context(docs)
        assert "\n\n" in result

    def test_build_context_missing_source_defaults_to_unknown(self):
        doc = _make_doc()
        doc.payload = {"text": "text"} 
        docs = [{"doc": doc}]
        result = self.retrieve._build_context(docs)
        assert "Unknown" in result

    def test_build_context_missing_page_defaults_to_na(self):
        doc = _make_doc()
        doc.payload = {"text": "text", "source": "src.pdf"} 
        docs = [{"doc": doc}]
        result = self.retrieve._build_context(docs)
        assert "N/A" in result

    def test_build_context_empty_docs_returns_empty_string(self):
        result = self.retrieve._build_context([])
        assert result == ""

    def test_build_context_returns_string(self):
        result = self.retrieve._build_context([{"doc": self.mock_doc}])
        assert isinstance(result, str)

    def test_build_summary_context_returns_tuple(self):
        result = self.retrieve._build_summary_context([{"doc": self.mock_doc}])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_build_summary_context_context_contains_text(self):
        context, _ = self.retrieve._build_summary_context([{"doc": self.mock_doc}])
        assert self.mock_doc.payload["text"] in context

    def test_build_summary_context_sources_list(self):
        _, sources = self.retrieve._build_summary_context([{"doc": self.mock_doc}])
        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_build_summary_context_source_strips_prefix(self):
        doc = _make_doc(source="prefix_filename.pdf")
        _, sources = self.retrieve._build_summary_context([{"doc": doc}])
        assert "filename.pdf" in sources[0]

    def test_build_summary_context_no_prefix_source_kept_as_is(self):
        doc = _make_doc(source="nodot.pdf")
        _, sources = self.retrieve._build_summary_context([{"doc": doc}])
        assert "nodot.pdf" in sources[0]

    def test_build_summary_context_deduplicates_sources(self):
        doc2 = _make_doc("d2", "more text", source="test_doc.pdf", page=2)
        docs = [{"doc": self.mock_doc}, {"doc": doc2}]
        _, sources = self.retrieve._build_summary_context(docs)
        assert len(sources) == len(set(sources))

    def test_build_summary_context_multiple_docs_joined(self):
        doc2 = _make_doc("d2", "second text", source="other.pdf")
        docs = [{"doc": self.mock_doc}, {"doc": doc2}]
        context, _ = self.retrieve._build_summary_context(docs)
        assert "\n\n" in context

    def test_build_summary_context_empty_docs(self):
        context, sources = self.retrieve._build_summary_context([])
        assert context == ""
        assert sources == []

    def test_build_inventory_context_returns_string(self):
        doc = _make_doc(text="some content", source="myfile.pdf")
        with patch.object(self.retrieve, "load_documents",
                          return_value=[{"doc": doc, "vector_score": 0, "keyword_score": 0}]):
            result = self.retrieve._build_inventory_context()
        assert isinstance(result, str)

    def test_build_inventory_context_contains_file_label(self):
        doc = _make_doc(text="some content", source="myfile.pdf")
        with patch.object(self.retrieve, "load_documents",
                          return_value=[{"doc": doc, "vector_score": 0, "keyword_score": 0}]):
            result = self.retrieve._build_inventory_context()
        assert "File:" in result

    def test_build_inventory_context_strips_source_prefix(self):
        doc = _make_doc(text="content", source="uuid_realname.pdf")
        with patch.object(self.retrieve, "load_documents",
                          return_value=[{"doc": doc, "vector_score": 0, "keyword_score": 0}]):
            result = self.retrieve._build_inventory_context()
        assert "realname.pdf" in result

    def test_build_inventory_context_empty_docs_returns_empty_string(self):
        with patch.object(self.retrieve, "load_documents", return_value=[]):
            result = self.retrieve._build_inventory_context()
        assert result == ""

    def test_build_inventory_context_doc_with_no_text_excluded(self):
        doc = _make_doc(text="", source="empty.pdf")
        doc.payload["text"] = ""
        with patch.object(self.retrieve, "load_documents",
                          return_value=[{"doc": doc, "vector_score": 0, "keyword_score": 0}]):
            result = self.retrieve._build_inventory_context()
        assert isinstance(result, str)

    def test_query_docs_returns_cached_response_when_cache_hit(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value="from cache")
        self.retrieve.vector_search = MagicMock()
        self.retrieve.keyword_search = MagicMock()

        result = self._run(self.retrieve.query_docs("What is AI?"))

        assert result == "from cache"
        self.retrieve.vector_search.assert_not_called()
        self.retrieve.keyword_search.assert_not_called()

    def test_query_docs_cache_hit_records_metrics(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value="cached")

        self._run(self.retrieve.query_docs("What is AI?"))

        self.retrieve.metrics.record.assert_called_once()
        call_kwargs = self.retrieve.metrics.record.call_args.kwargs
        assert call_kwargs["served_from_cache"] is True

    def test_query_docs_inventory_request_calls_generate_inventory_response(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        self.retrieve._build_inventory_context = MagicMock(return_value="inventory context")
        self.retrieve.response.generate_inventory_response = MagicMock(return_value="inventory answer")

        result = self._run(self.retrieve.query_docs("what documents do you have"))

        self.retrieve.response.generate_inventory_response.assert_called_once()
        assert result == "inventory answer"

    def test_query_docs_inventory_request_empty_context_returns_fallback(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        self.retrieve._build_inventory_context = MagicMock(return_value="")

        result = self._run(self.retrieve.query_docs("list all documents"))

        assert "No documents" in result

    def test_query_docs_full_pipeline_on_cache_miss(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)

        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[doc_item])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        self.retrieve._build_context = MagicMock(return_value=("context text", ["source1"]))
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
        self.retrieve._build_context = MagicMock(return_value=("ctx", ["src"]))
        self.retrieve.response.generate_response = MagicMock(return_value="answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("question"))

        self.retrieve.save_cache.assert_called_once_with("question", "answer")

    def test_query_docs_cache_miss_records_metrics(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        self.retrieve._build_context = MagicMock(return_value=("ctx", ["src"]))
        self.retrieve.response.generate_response = MagicMock(return_value="answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("What is AI?"))

        self.retrieve.metrics.record.assert_called_once()
        call_kwargs = self.retrieve.metrics.record.call_args.kwargs
        assert call_kwargs["served_from_cache"] is False

    def test_query_docs_summary_request_calls_generate_summary_response(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        self.retrieve._build_summary_context = MagicMock(return_value=("summary ctx", ["src.pdf"]))
        self.retrieve.response.generate_summary_response = MagicMock(return_value="summary answer")
        self.retrieve.save_cache = MagicMock()

        result = self._run(self.retrieve.query_docs("summarize the document"))

        self.retrieve.response.generate_summary_response.assert_called_once()
        assert result == "summary answer"

    def test_query_docs_summary_request_saves_cache(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        self.retrieve._build_summary_context = MagicMock(return_value=("ctx", ["src"]))
        self.retrieve.response.generate_summary_response = MagicMock(return_value="sum")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("summarize the document"))

        self.retrieve.save_cache.assert_called_once()

    def test_query_docs_summary_no_docs_returns_fallback(self):
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        # Return a falsy value so `if not summary_context` fires
        self.retrieve._build_summary_context = MagicMock(return_value="")
        self.retrieve.response.generate_summary_response = MagicMock(return_value="sum")
        self.retrieve.save_cache = MagicMock()

        result = self._run(self.retrieve.query_docs("summarize the document"))

        assert "No documents" in result
        self.retrieve.response.generate_summary_response.assert_not_called()

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
        self.retrieve._build_context = MagicMock(return_value=("ctx", ["src"]))
        self.retrieve.response.generate_response = MagicMock(return_value="answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("What is AI?"))

        call_args = self.retrieve.vector_search.call_args[0][0]
        assert call_args == fixed_embedding.tolist()

    def test_query_docs_final_docs_sliced_to_4(self):
        """reranked_docs[:4] are passed to _build_context."""
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

        def fake_build_context(final_docs):
            captured["count"] = len(final_docs)
            return ("context", [])

        self.retrieve._build_context = fake_build_context
        self.retrieve.response.generate_response = MagicMock(return_value="answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("What is AI?"))

        assert captured["count"] == 4

    def test_query_docs_generate_response_called_with_sources(self):
        """generate_response must receive the sources kwarg from _build_context."""
        self.retrieve.embedding_model.encode.return_value = self.embedding
        self.retrieve.check_cache = MagicMock(return_value=None)
        doc_item = {"doc": self.mock_doc, "vector_score": 0.9, "keyword_score": 0}
        self.retrieve.vector_search = MagicMock(return_value=[doc_item])
        self.retrieve.keyword_search = MagicMock(return_value=[])
        self.retrieve.merge_results = MagicMock(return_value=[doc_item])
        self.retrieve.combine_scores = MagicMock(return_value=[doc_item])
        self.retrieve.rerank = MagicMock(return_value=[doc_item])
        self.retrieve._build_context = MagicMock(return_value=("ctx", ["src.pdf"]))
        self.retrieve.response.generate_response = MagicMock(return_value="answer")
        self.retrieve.save_cache = MagicMock()

        self._run(self.retrieve.query_docs("What is AI?"))

        call_kwargs = self.retrieve.response.generate_response.call_args[1]
        assert call_kwargs.get("sources") == ["src.pdf"]