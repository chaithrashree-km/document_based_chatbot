import pytest
from unittest.mock import MagicMock, patch
from app.tasks.Async_Upload import async_upload


class TestAsyncUploadTask:

    def setup_method(self):
        async_upload.update_state = MagicMock()

    def test_async_upload_success(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            result = async_upload.run("sample.pdf")
            assert result["status"] == "SUCCESS"
            assert result["result"] == {"message": "Chunks ingested successfully."}

    def test_async_upload_calls_upload_documents_with_correct_path(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            async_upload.run("documents/report.pdf")
            mock_upload.upload_documents.assert_called_once_with("documents/report.pdf")

    def test_async_upload_calls_refresh_bm25_after_upload(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            async_upload.run("sample.pdf")
            mock_retriever.refresh_bm25.assert_called_once()

    def test_async_upload_refresh_bm25_called_after_upload_documents(self):
        call_order = []
        mock_upload = MagicMock()
        mock_upload.upload_documents.side_effect = lambda path: call_order.append("upload")
        mock_retriever = MagicMock()
        mock_retriever.refresh_bm25.side_effect = lambda: call_order.append("refresh")

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            async_upload.run("sample.pdf")
            assert call_order == ["upload", "refresh"]

    def test_async_upload_updates_state_to_progress_extracting(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            async_upload.run("sample.pdf")
            async_upload.update_state.assert_any_call(
                state="PROGRESS",
                meta={"status": "Extracting documents..."}
            )

    def test_async_upload_updates_state_to_progress_refreshing(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            async_upload.run("sample.pdf")
            async_upload.update_state.assert_any_call(
                state="PROGRESS",
                meta={"status": "Refreshing search index..."}
            )

    def test_async_upload_returns_success_status(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            result = async_upload.run("sample.pdf")
            assert result["status"] == "SUCCESS"

    def test_async_upload_with_txt_file(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            result = async_upload.run("notes.txt")
            mock_upload.upload_documents.assert_called_once_with("notes.txt")
            assert result["status"] == "SUCCESS"

    def test_async_upload_creates_fresh_upload_instance(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload) as MockUpload, \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            async_upload.run("sample.pdf")
            MockUpload.assert_called_once()

    def test_async_upload_creates_fresh_retrieve_instance(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever) as MockRetrieve:
            async_upload.run("sample.pdf")
            MockRetrieve.assert_called_once()

    def test_async_upload_with_empty_path(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Invalid path provided."}
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            result = async_upload.run("")
            mock_upload.upload_documents.assert_called_once_with("")
            assert result["status"] == "SUCCESS"
            assert result["result"] == {"message": "Invalid path provided."}

    def test_async_upload_raises_when_upload_documents_fails(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.side_effect = Exception("Ingestion failed")
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            with pytest.raises(Exception, match="Ingestion failed"):
                async_upload.run("sample.pdf")

    def test_async_upload_updates_state_to_failure_on_exception(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.side_effect = Exception("Ingestion failed")
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            with pytest.raises(Exception):
                async_upload.run("sample.pdf")
            async_upload.update_state.assert_any_call(
                state="FAILURE",
                meta={"status": "Ingestion failed"}
            )

    def test_async_upload_raises_when_refresh_bm25_fails(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()
        mock_retriever.refresh_bm25.side_effect = Exception("BM25 refresh failed")

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            with pytest.raises(Exception, match="BM25 refresh failed"):
                async_upload.run("sample.pdf")

    def test_async_upload_updates_failure_state_when_refresh_fails(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.return_value = {"message": "Chunks ingested successfully."}
        mock_retriever = MagicMock()
        mock_retriever.refresh_bm25.side_effect = Exception("BM25 refresh failed")

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            with pytest.raises(Exception):
                async_upload.run("sample.pdf")
            async_upload.update_state.assert_any_call(
                state="FAILURE",
                meta={"status": "BM25 refresh failed"}
            )

    def test_async_upload_refresh_bm25_not_called_when_upload_fails(self):
        mock_upload = MagicMock()
        mock_upload.upload_documents.side_effect = Exception("Ingestion failed")
        mock_retriever = MagicMock()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
            with pytest.raises(Exception):
                async_upload.run("sample.pdf")
            mock_retriever.refresh_bm25.assert_not_called()

