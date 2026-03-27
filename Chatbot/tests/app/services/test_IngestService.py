import numpy as np
from unittest.mock import patch, MagicMock, call
from app.services.Ingest_Service import Upload

class TestUpload:

    def setup_method(self):
        self.upload = Upload()

        self.upload.database = MagicMock()
        self.upload.database.client = MagicMock()
        self.upload.database.create_collection = MagicMock()

        self.upload.model = MagicMock()
        self.upload.model.get_sentence_embedding_dimension.return_value = 384
        self.upload.model.encode.return_value = np.array([[0.1] * 384])

        self.upload.embeddings_model = MagicMock()

        self.upload.splitter = MagicMock()
        self.upload.splitter.split_text.return_value = ["chunk1"]

        self.upload.config = MagicMock()
        self.upload.config.COLLECTION_NAME = "test_collection"

    def _make_element(self, text: str, page_number=1):
        el = MagicMock()
        el.__str__ = MagicMock(return_value=text)
        el.metadata = MagicMock()
        el.metadata.page_number = page_number
        return el

    def test_extract_single_page_returns_one_tuple(self):
        elements = [self._make_element("hello world", page_number=1)]

        with patch("app.services.Ingest_Service.partition", return_value=elements):
            result = self.upload._extract("file.pdf")

        assert len(result) == 1
        assert result[0] == ("hello world", 1)

    def test_extract_multiple_pages_returns_multiple_tuples(self):
        elements = [
            self._make_element("page one text", page_number=1),
            self._make_element("page two text", page_number=2),
        ]

        with patch("app.services.Ingest_Service.partition", return_value=elements):
            result = self.upload._extract("file.pdf")

        assert len(result) == 2
        pages = {r[1]: r[0] for r in result}
        assert pages[1] == "page one text"
        assert pages[2] == "page two text"

    def test_extract_multiple_elements_same_page_joined(self):
        elements = [
            self._make_element("first", page_number=1),
            self._make_element("second", page_number=1),
        ]

        with patch("app.services.Ingest_Service.partition", return_value=elements):
            result = self.upload._extract("file.pdf")

        assert len(result) == 1
        assert result[0][0] == "first\nsecond"

    def test_extract_element_with_none_page_number_defaults_to_1(self):
        el = self._make_element("no page info")
        el.metadata.page_number = None

        with patch("app.services.Ingest_Service.partition", return_value=[el]):
            result = self.upload._extract("file.txt")

        assert result[0][1] == 1

    def test_extract_element_without_metadata_defaults_to_page_1(self):
        el = MagicMock()
        el.__str__ = MagicMock(return_value="no metadata")
        el.metadata = None

        with patch("app.services.Ingest_Service.partition", return_value=[el]):
            result = self.upload._extract("file.txt")

        assert result[0][1] == 1

    def test_extract_returns_pages_in_sorted_order(self):
        elements = [
            self._make_element("third", page_number=3),
            self._make_element("first", page_number=1),
            self._make_element("second", page_number=2),
        ]

        with patch("app.services.Ingest_Service.partition", return_value=elements):
            result = self.upload._extract("file.pdf")

        assert [r[1] for r in result] == [1, 2, 3]

    def test_extract_various_supported_extensions(self):
        for ext in [".pdf", ".txt", ".csv", ".xlsx", ".jpg", ".docx", ".pptx", ".html", ".md"]:
            elements = [self._make_element(f"content for {ext}")]
            with patch("app.services.Ingest_Service.partition", return_value=elements):
                result = self.upload._extract(f"file{ext}")
            assert len(result) == 1

    def test_extract_empty_elements_returns_empty_list(self):
        with patch("app.services.Ingest_Service.partition", return_value=[]):
            result = self.upload._extract("file.pdf")

        assert result == []

    def test_extract_all_whitespace_elements_excluded(self):
        el = self._make_element("   ")

        with patch("app.services.Ingest_Service.partition", return_value=[el]):
            result = self.upload._extract("file.txt")

        assert result == []

    def test_extract_partition_raises_exception_returns_empty(self):
        with patch("app.services.Ingest_Service.partition", side_effect=Exception("boom")):
            result = self.upload._extract("broken.pdf")

        assert result == []

    def test_extract_mixed_blank_and_real_elements(self):
        elements = [
            self._make_element(""),
            self._make_element("real text", page_number=2),
            self._make_element("  "),
        ]

        with patch("app.services.Ingest_Service.partition", return_value=elements):
            result = self.upload._extract("file.txt")

        assert len(result) == 1
        assert result[0] == ("real text", 2)

    def test_upload_documents_invalid_path_returns_error_message(self):
        with patch("app.services.Ingest_Service.os.path.isfile", return_value=False), \
             patch("app.services.Ingest_Service.os.path.isdir", return_value=False):
            result = self.upload.upload_documents("nonexistent")

        assert result["message"] == "Invalid path provided."

    def test_upload_documents_single_file_success(self):
        self.upload._extract = MagicMock(return_value=[("sample text", 1)])
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("file", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="file.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/some/folder"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            result = self.upload.upload_documents("file.txt")

        assert result["message"] == "Chunks ingested successfully."
        self.upload.database.create_collection.assert_called_once()
        self.upload.database.client.upsert.assert_called()

    def test_upload_documents_upsert_payload_contains_correct_keys(self):
        self.upload._extract = MagicMock(return_value=[("some content", 1)])
        self.upload.model.encode.return_value = np.array([[0.2] * 384])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".pdf")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.pdf"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/dir"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            self.upload.upload_documents("f.pdf")

        call_kwargs = self.upload.database.client.upsert.call_args.kwargs
        points = call_kwargs["points"]
        assert len(points) == 1
        p = points[0]
        assert "id" in p
        assert "vector" in p
        assert p["payload"]["source"] == "f.pdf"
        assert p["payload"]["page"] == 1
        assert p["payload"]["folder"] == "/dir"
        assert p["payload"]["text"] == "chunk1"

    def test_upload_documents_collection_name_passed_to_upsert(self):
        self.upload._extract = MagicMock(return_value=[("text", 1)])
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            self.upload.upload_documents("f.txt")

        call_kwargs = self.upload.database.client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_collection"

    def test_upload_documents_skips_unsupported_extension(self):
        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("file", ".exe")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="file.exe"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"):
            result = self.upload.upload_documents("file.exe")

        assert result["message"] == "Chunks ingested successfully."
        self.upload.database.client.upsert.assert_not_called()

    def test_upload_documents_skips_multiple_unsupported_extensions(self):
        for bad_ext in [".exe", ".bin", ".so", ".dll", ".zip"]:
            self.upload.database.client.upsert.reset_mock()
            with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
                 patch("app.services.Ingest_Service.os.path.splitext", return_value=("file", bad_ext)):
                self.upload.upload_documents(f"file{bad_ext}")

            self.upload.database.client.upsert.assert_not_called()

    def test_upload_documents_no_content_extracted_no_upsert(self):
        self.upload._extract = MagicMock(return_value=[])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"):
            result = self.upload.upload_documents("f.txt")

        assert result["message"] == "Chunks ingested successfully."
        self.upload.database.client.upsert.assert_not_called()

    def test_upload_documents_batch_flush_at_exactly_100_chunks(self):
        self.upload._extract = MagicMock(return_value=[("page text", 1)])
        self.upload.splitter.split_text.return_value = ["chunk"] * 100
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            result = self.upload.upload_documents("f.txt")

        assert result["message"] == "Chunks ingested successfully."
        self.upload.database.client.upsert.assert_called()

    def test_upload_documents_batch_flush_more_than_100_chunks(self):
        self.upload._extract = MagicMock(return_value=[("page text", 1)])
        self.upload.splitter.split_text.return_value = ["chunk"] * 150
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            result = self.upload.upload_documents("f.txt")

        assert self.upload.database.client.upsert.call_count >= 2

    def test_upload_documents_final_flush_called_after_loop(self):
        self.upload._extract = MagicMock(return_value=[("page text", 1)])
        self.upload.splitter.split_text.return_value = ["chunk"] * 42
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            result = self.upload.upload_documents("f.txt")

        assert result["message"] == "Chunks ingested successfully."
        upsert_call = self.upload.database.client.upsert.call_args.kwargs
        assert len(upsert_call["points"]) == 42

    def test_upload_documents_directory_processes_all_files(self):
        self.upload._extract = MagicMock(return_value=[("data", 1)])
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=False), \
             patch("app.services.Ingest_Service.os.path.isdir", return_value=True), \
             patch("app.services.Ingest_Service.os.walk") as mock_walk, \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/dir"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            mock_walk.return_value = [("/dir", [], ["f1.txt", "f2.txt"])]
            result = self.upload.upload_documents("/dir")

        assert result["message"] == "Chunks ingested successfully."
        assert self.upload._extract.call_count == 2

    def test_upload_documents_directory_skips_unsupported_files(self):
        self.upload._extract = MagicMock(return_value=[("data", 1)])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=False), \
             patch("app.services.Ingest_Service.os.path.isdir", return_value=True), \
             patch("app.services.Ingest_Service.os.walk") as mock_walk, \
             patch("app.services.Ingest_Service.os.path.splitext", side_effect=lambda p: (p, ".exe")):
            mock_walk.return_value = [("/dir", [], ["a.exe", "b.exe"])]
            result = self.upload.upload_documents("/dir")

        assert result["message"] == "Chunks ingested successfully."
        self.upload._extract.assert_not_called()

    def test_upload_documents_directory_mixed_supported_unsupported(self):
        self.upload._extract = MagicMock(return_value=[("data", 1)])
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        def fake_splitext(path):
            return (path, ".txt") if path.endswith(".txt") else (path, ".exe")

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=False), \
             patch("app.services.Ingest_Service.os.path.isdir", return_value=True), \
             patch("app.services.Ingest_Service.os.walk") as mock_walk, \
             patch("app.services.Ingest_Service.os.path.splitext", side_effect=fake_splitext), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            mock_walk.return_value = [("/d", [], ["good.txt", "bad.exe"])]
            result = self.upload.upload_documents("/d")

        assert result["message"] == "Chunks ingested successfully."
        assert self.upload._extract.call_count == 1

    def test_upload_documents_directory_empty_no_upsert(self):
        with patch("app.services.Ingest_Service.os.path.isfile", return_value=False), \
             patch("app.services.Ingest_Service.os.path.isdir", return_value=True), \
             patch("app.services.Ingest_Service.os.walk") as mock_walk:
            mock_walk.return_value = [("/empty", [], [])]
            result = self.upload.upload_documents("/empty")

        assert result["message"] == "Chunks ingested successfully."
        self.upload.database.client.upsert.assert_not_called()

    def test_upload_documents_create_collection_always_called(self):
        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".exe")):
            self.upload.upload_documents("f.exe")

        self.upload.database.create_collection.assert_called_once_with(384)

    def test_upload_documents_create_collection_uses_embedding_dim(self):
        self.upload.model.get_sentence_embedding_dimension.return_value = 768

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".exe")):
            self.upload.upload_documents("f.exe")

        self.upload.database.create_collection.assert_called_once_with(768)

    def test_upload_documents_chunk_ids_are_unique(self):
        self.upload._extract = MagicMock(return_value=[("page text", 1)])
        self.upload.splitter.split_text.return_value = ["c1", "c2", "c3"]
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            self.upload.upload_documents("f.txt")

        points = self.upload.database.client.upsert.call_args.kwargs["points"]
        ids = [p["id"] for p in points]
        assert len(ids) == len(set(ids)), "Duplicate point IDs detected"

    def test_upload_documents_multipage_chunks_all_uploaded(self):
        self.upload._extract = MagicMock(
            return_value=[("page 1 content", 1), ("page 2 content", 2)]
        )
        self.upload.splitter.split_text.side_effect = lambda t: [t]  # one chunk per page
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".pdf")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.pdf"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            result = self.upload.upload_documents("f.pdf")

        assert result["message"] == "Chunks ingested successfully."
        points = self.upload.database.client.upsert.call_args.kwargs["points"]
        assert len(points) == 2

    def test_upload_documents_clean_text_is_called_per_page(self):
        self.upload._extract = MagicMock(
            return_value=[("raw1", 1), ("raw2", 2)]
        )
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", return_value="cleaned") as mock_clean:
            self.upload.upload_documents("f.txt")

        assert mock_clean.call_count == 2
        mock_clean.assert_any_call("raw1")
        mock_clean.assert_any_call("raw2")

    def test_upload_documents_directory_file_paths_built_correctly(self):
        self.upload._extract = MagicMock(return_value=[])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=False), \
             patch("app.services.Ingest_Service.os.path.isdir", return_value=True), \
             patch("app.services.Ingest_Service.os.walk") as mock_walk, \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.join", return_value="/root/file.txt") as mock_join, \
             patch("app.services.Ingest_Service.os.path.basename", return_value="file.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/root"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            mock_walk.return_value = [("/root", [], ["file.txt"])]
            self.upload.upload_documents("/root")

        mock_join.assert_called()

    def test_supported_extensions_contains_expected_types(self):
        expected = {".pdf", ".txt", ".csv", ".xls", ".xlsx", ".jpg", ".jpeg",
                    ".png", ".docx", ".pptx", ".html", ".md"}
        assert expected == Upload.SUPPORTED_EXTENSIONS

    def test_supported_extensions_does_not_contain_executables(self):
        for bad in [".exe", ".bin", ".sh", ".bat", ".dll"]:
            assert bad not in Upload.SUPPORTED_EXTENSIONS

    def test_extract_text_is_stripped(self):
        el = self._make_element("  padded text  ")

        with patch("app.services.Ingest_Service.partition", return_value=[el]):
            result = self.upload._extract("file.txt")

        assert result[0][0] == "padded text"

    def test_extract_only_whitespace_element_not_included(self):
        el = self._make_element("\n\t   \n")

        with patch("app.services.Ingest_Service.partition", return_value=[el]):
            result = self.upload._extract("file.txt")

        assert result == []

    def test_extract_passes_filename_to_partition(self):
        with patch("app.services.Ingest_Service.partition", return_value=[]) as mock_partition:
            self.upload._extract("/some/path/doc.pdf")

        mock_partition.assert_called_once_with(filename="/some/path/doc.pdf")

    def test_upload_documents_encode_called_with_chunk_texts(self):
        self.upload._extract = MagicMock(return_value=[("text", 1)])
        self.upload.splitter.split_text.return_value = ["alpha", "beta"]

        captured_texts = []

        def encode_side_effect(texts):
            captured_texts.extend(texts)
            return np.array([[0.1] * 384 for _ in texts])

        self.upload.model.encode.side_effect = encode_side_effect

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            self.upload.upload_documents("f.txt")

        assert "alpha" in captured_texts
        assert "beta" in captured_texts

    def test_upload_documents_isfile_used_when_true(self):
        self.upload._extract = MagicMock(return_value=[("data", 1)])
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.1] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.isdir", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            self.upload.upload_documents("f.txt")

        assert self.upload._extract.call_count == 1

    def test_upload_documents_vectors_stored_as_lists(self):
        self.upload._extract = MagicMock(return_value=[("text", 1)])
        self.upload.model.encode.side_effect = lambda texts: np.array([[0.5] * 384 for _ in texts])

        with patch("app.services.Ingest_Service.os.path.isfile", return_value=True), \
             patch("app.services.Ingest_Service.os.path.splitext", return_value=("f", ".txt")), \
             patch("app.services.Ingest_Service.os.path.basename", return_value="f.txt"), \
             patch("app.services.Ingest_Service.os.path.dirname", return_value="/d"), \
             patch("app.services.Ingest_Service.clean_text", side_effect=lambda x: x):
            self.upload.upload_documents("f.txt")

        points = self.upload.database.client.upsert.call_args.kwargs["points"]
        assert isinstance(points[0]["vector"], list)

