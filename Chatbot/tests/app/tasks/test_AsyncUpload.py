import pytest
from unittest.mock import MagicMock, patch, call
from app.tasks.Async_Upload import async_upload


# ------------------------------------------------------------------ #
#  Shared helpers                                                      #
# ------------------------------------------------------------------ #

def _make_mocks(upload_return=None, upload_side_effect=None,
                refresh_side_effect=None):
    """
    Build mock Upload and Retrieve instances.
    Returns (mock_upload, mock_retriever).
    """
    mock_upload = MagicMock()
    mock_upload.upload_documents.return_value = (
        upload_return if upload_return is not None
        else {"message": "Chunks ingested successfully."}
    )
    if upload_side_effect is not None:
        mock_upload.upload_documents.side_effect = upload_side_effect

    mock_retriever = MagicMock()
    if refresh_side_effect is not None:
        mock_retriever.refresh_bm25.side_effect = refresh_side_effect

    return mock_upload, mock_retriever


def _run(path, file_hash=None,
         mock_upload=None, mock_retriever=None,
         extra_patches=None):
    """
    Invoke async_upload.run() with Upload and Retrieve mocked.
    Returns the task result.
    """
    with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
         patch("app.services.Query_Service.Retrieve", return_value=mock_retriever):
        return async_upload.run(path, file_hash=file_hash)


# ------------------------------------------------------------------ #
#  Test class                                                          #
# ------------------------------------------------------------------ #

class TestAsyncUploadTask:

    def setup_method(self):
        async_upload.update_state = MagicMock()

    # ---------------------------------------------------------------- #
    #  Happy path — basic success                                       #
    # ---------------------------------------------------------------- #

    def test_async_upload_success_returns_correct_structure(self):
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            result = async_upload.run("sample.pdf")

        assert result["status"] == "SUCCESS"
        assert result["result"] == {"message": "Chunks ingested successfully."}

    def test_async_upload_returns_upload_result_verbatim(self):
        custom_result = {"message": "Custom ingestion result."}
        mock_upload, mock_retriever = _make_mocks(upload_return=custom_result)

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            result = async_upload.run("sample.pdf")

        assert result["result"] == custom_result

    # ---------------------------------------------------------------- #
    #  upload_documents call signature                                  #
    # ---------------------------------------------------------------- #

    def test_async_upload_calls_upload_documents_with_path_and_default_file_hash(self):
        """
        upload_documents must be called with (path, file_hash=None) by default.
        Old tests only checked the positional path arg — the new kwarg broke them.
        """
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            async_upload.run("documents/report.pdf")

        mock_upload.upload_documents.assert_called_once_with(
            "documents/report.pdf", file_hash=None
        )

    def test_async_upload_passes_explicit_file_hash_to_upload_documents(self):
        """When file_hash is supplied to the task, it must be forwarded."""
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            async_upload.run("sample.pdf", file_hash="abc123")

        mock_upload.upload_documents.assert_called_once_with(
            "sample.pdf", file_hash="abc123"
        )

    def test_async_upload_with_txt_file(self):
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            result = async_upload.run("notes.txt")

        mock_upload.upload_documents.assert_called_once_with("notes.txt", file_hash=None)
        assert result["status"] == "SUCCESS"

    def test_async_upload_with_empty_path(self):
        mock_upload, mock_retriever = _make_mocks(
            upload_return={"message": "Invalid path provided."}
        )

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            result = async_upload.run("")

        mock_upload.upload_documents.assert_called_once_with("", file_hash=None)
        assert result["status"] == "SUCCESS"
        assert result["result"] == {"message": "Invalid path provided."}

    # ---------------------------------------------------------------- #
    #  update_state — PROGRESS calls                                    #
    # ---------------------------------------------------------------- #

    def test_async_upload_updates_state_to_progress_extracting(self):
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            async_upload.run("sample.pdf")

        async_upload.update_state.assert_any_call(
            state="PROGRESS",
            meta={"status": "Extracting documents..."}
        )

    def test_async_upload_updates_state_to_progress_refreshing(self):
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            async_upload.run("sample.pdf")

        async_upload.update_state.assert_any_call(
            state="PROGRESS",
            meta={"status": "Refreshing search index..."}
        )

    def test_async_upload_extracting_state_called_before_refreshing_state(self):
        """PROGRESS Extracting must be sent before PROGRESS Refreshing."""
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            async_upload.run("sample.pdf")

        calls = async_upload.update_state.call_args_list
        states = [c.kwargs["state"] if c.kwargs else c[1]["state"] for c in calls
                  if (c.kwargs.get("state") if c.kwargs else c[1].get("state")) == "PROGRESS"]
        metas = [c.kwargs["meta"]["status"] if c.kwargs else c[1]["meta"]["status"]
                 for c in calls
                 if (c.kwargs.get("state") if c.kwargs else c[1].get("state")) == "PROGRESS"]
        assert metas.index("Extracting documents...") < metas.index("Refreshing search index...")

    # ---------------------------------------------------------------- #
    #  refresh_bm25 behaviour                                           #
    # ---------------------------------------------------------------- #

    def test_async_upload_calls_refresh_bm25(self):
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            async_upload.run("sample.pdf")

        mock_retriever.refresh_bm25.assert_called_once()

    def test_async_upload_refresh_bm25_called_after_upload_documents(self):
        call_order = []
        mock_upload, mock_retriever = _make_mocks()
        mock_upload.upload_documents.side_effect = lambda path, file_hash=None: call_order.append("upload")
        mock_retriever.refresh_bm25.side_effect = lambda: call_order.append("refresh")

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            async_upload.run("sample.pdf")

        assert call_order == ["upload", "refresh"]

    # ---------------------------------------------------------------- #
    #  Instance creation                                                #
    # ---------------------------------------------------------------- #

    def test_async_upload_creates_fresh_upload_instance(self):
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload) as MockUpload, \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            async_upload.run("sample.pdf")

        MockUpload.assert_called_once()

    def test_async_upload_creates_fresh_retrieve_instance(self):
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever) as MockRetrieve, \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            async_upload.run("sample.pdf")

        MockRetrieve.assert_called_once()

    # ---------------------------------------------------------------- #
    #  Failure — upload_documents raises                                #
    # ---------------------------------------------------------------- #

    def test_async_upload_raises_when_upload_documents_fails(self):
        mock_upload, mock_retriever = _make_mocks(
            upload_side_effect=Exception("Ingestion failed")
        )

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            with pytest.raises(Exception, match="Ingestion failed"):
                async_upload.run("sample.pdf")

    def test_async_upload_updates_state_to_failure_on_upload_exception(self):
        mock_upload, mock_retriever = _make_mocks(
            upload_side_effect=Exception("Ingestion failed")
        )

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            with pytest.raises(Exception):
                async_upload.run("sample.pdf")

        async_upload.update_state.assert_any_call(
            state="FAILURE",
            meta={"status": "Ingestion failed"}
        )

    def test_async_upload_refresh_bm25_not_called_when_upload_fails(self):
        mock_upload, mock_retriever = _make_mocks(
            upload_side_effect=Exception("Ingestion failed")
        )

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            with pytest.raises(Exception):
                async_upload.run("sample.pdf")

        mock_retriever.refresh_bm25.assert_not_called()

    # ---------------------------------------------------------------- #
    #  Failure — refresh_bm25 raises                                    #
    # ---------------------------------------------------------------- #

    def test_async_upload_raises_when_refresh_bm25_fails(self):
        mock_upload, mock_retriever = _make_mocks(
            refresh_side_effect=Exception("BM25 refresh failed")
        )

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            with pytest.raises(Exception, match="BM25 refresh failed"):
                async_upload.run("sample.pdf")

    def test_async_upload_updates_failure_state_when_refresh_fails(self):
        mock_upload, mock_retriever = _make_mocks(
            refresh_side_effect=Exception("BM25 refresh failed")
        )

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False):
            with pytest.raises(Exception):
                async_upload.run("sample.pdf")

        async_upload.update_state.assert_any_call(
            state="FAILURE",
            meta={"status": "BM25 refresh failed"}
        )

    # ---------------------------------------------------------------- #
    #  finally block — file cleanup                                     #
    #                                                                   #
    #  The finally block runs regardless of success or failure.         #
    #  It patches os/shutil inside app.tasks.Async_Upload namespace.    #
    # ---------------------------------------------------------------- #

    def test_cleanup_skipped_when_path_does_not_exist(self):
        """os.path.exists returns False → loop breaks immediately, no remove/rmtree."""
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=False), \
             patch("app.tasks.Async_Upload.os.remove") as mock_remove, \
             patch("app.tasks.Async_Upload.shutil.rmtree") as mock_rmtree:
            async_upload.run("sample.pdf")

        mock_remove.assert_not_called()
        mock_rmtree.assert_not_called()

    def test_cleanup_calls_os_remove_for_file(self):
        """When path exists and is a file, os.remove is called."""
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isfile", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isdir", return_value=False), \
             patch("app.tasks.Async_Upload.os.remove") as mock_remove, \
             patch("app.tasks.Async_Upload.shutil.rmtree") as mock_rmtree:
            async_upload.run("sample.pdf")

        mock_remove.assert_called_once_with("sample.pdf")
        mock_rmtree.assert_not_called()

    def test_cleanup_calls_shutil_rmtree_for_directory(self):
        """When path exists and is a directory, shutil.rmtree is called."""
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isfile", return_value=False), \
             patch("app.tasks.Async_Upload.os.path.isdir", return_value=True), \
             patch("app.tasks.Async_Upload.os.remove") as mock_remove, \
             patch("app.tasks.Async_Upload.shutil.rmtree") as mock_rmtree:
            async_upload.run("/uploads/docs")

        mock_rmtree.assert_called_once_with("/uploads/docs")
        mock_remove.assert_not_called()

    def test_cleanup_retries_on_permission_error(self):
        """PermissionError triggers retry; succeeds on second attempt."""
        mock_upload, mock_retriever = _make_mocks()
        call_count = {"n": 0}

        def flaky_remove(path):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise PermissionError("locked")

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isfile", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isdir", return_value=False), \
             patch("app.tasks.Async_Upload.os.remove", side_effect=flaky_remove), \
             patch("app.tasks.Async_Upload.time.sleep"):
            async_upload.run("sample.pdf")

        assert call_count["n"] == 2

    def test_cleanup_all_5_attempts_fail_does_not_raise(self):
        """After 5 failed PermissionError attempts, the task does not raise."""
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isfile", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isdir", return_value=False), \
             patch("app.tasks.Async_Upload.os.remove",
                   side_effect=PermissionError("always locked")), \
             patch("app.tasks.Async_Upload.time.sleep"):
            # Must NOT raise even though all retries failed
            result = async_upload.run("sample.pdf")

        assert result["status"] == "SUCCESS"

    def test_cleanup_sleeps_between_retries(self):
        """time.sleep(1) must be called between PermissionError retries."""
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isfile", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isdir", return_value=False), \
             patch("app.tasks.Async_Upload.os.remove",
                   side_effect=PermissionError("locked")), \
             patch("app.tasks.Async_Upload.time.sleep") as mock_sleep:
            async_upload.run("sample.pdf")

        # 5 PermissionErrors → 5 sleep(1) calls
        assert mock_sleep.call_count == 5
        mock_sleep.assert_called_with(1)

    def test_cleanup_runs_even_when_upload_raises(self):
        """finally block must execute cleanup even if upload_documents fails."""
        mock_upload, mock_retriever = _make_mocks(
            upload_side_effect=Exception("Ingestion failed")
        )

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isfile", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isdir", return_value=False), \
             patch("app.tasks.Async_Upload.os.remove") as mock_remove:
            with pytest.raises(Exception):
                async_upload.run("sample.pdf")

        mock_remove.assert_called_once_with("sample.pdf")

    def test_cleanup_runs_even_when_refresh_raises(self):
        """finally block must execute cleanup even if refresh_bm25 fails."""
        mock_upload, mock_retriever = _make_mocks(
            refresh_side_effect=Exception("BM25 refresh failed")
        )

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isfile", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isdir", return_value=False), \
             patch("app.tasks.Async_Upload.os.remove") as mock_remove:
            with pytest.raises(Exception):
                async_upload.run("sample.pdf")

        mock_remove.assert_called_once_with("sample.pdf")

    def test_cleanup_attempts_exactly_5_times_then_gives_up(self):
        """os.remove is called exactly 5 times before giving up."""
        mock_upload, mock_retriever = _make_mocks()

        with patch("app.services.Ingest_Service.Upload", return_value=mock_upload), \
             patch("app.services.Query_Service.Retrieve", return_value=mock_retriever), \
             patch("app.tasks.Async_Upload.os.path.exists", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isfile", return_value=True), \
             patch("app.tasks.Async_Upload.os.path.isdir", return_value=False), \
             patch("app.tasks.Async_Upload.os.remove",
                   side_effect=PermissionError("locked")) as mock_remove, \
             patch("app.tasks.Async_Upload.time.sleep"):
            async_upload.run("sample.pdf")

        assert mock_remove.call_count == 5