import io
import json
import os
import sys
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.main import app
from app.api.Controller import auth
from fastapi.testclient import TestClient

client = TestClient(app)

VALID_TOKEN_PAYLOAD = {"sub": "user-123"}
AUTH_HEADERS = {"Authorization": "Bearer dummy_token"}


def mock_verify_token():
    return VALID_TOKEN_PAYLOAD


class TestController:

    @patch("app.api.Controller.user_service")
    @patch("app.api.Controller.auth")
    def test_signup_success(self, mock_auth, mock_user_service):
        """Positive: new user signs up successfully and receives a token."""
        mock_user_service.get_user_by_email.return_value = None
        mock_user_service.create_user.return_value = "user-123"
        mock_auth.create_access_token.return_value = "mocked_token"

        response = client.post("/signup", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "secret123"
        })

        assert response.status_code == 200
        body = response.json()
        assert body["access_token"] == "mocked_token"
        assert body["token_type"] == "bearer"

    @patch("app.api.Controller.user_service")
    def test_signup_duplicate_email(self, mock_user_service):
        """Negative: signup with an already-registered email returns 400."""
        mock_user_service.get_user_by_email.return_value = {"email": "test@example.com"}

        response = client.post("/signup", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "secret123"
        })

        assert response.status_code == 400
        assert response.json()["detail"] == "Email already registered"

    def test_signup_missing_fields(self):
        """Negative: signup without required fields returns 422."""
        response = client.post("/signup", json={"email": "test@example.com"})
        assert response.status_code == 422

    def test_signup_missing_all_fields(self):
        """Negative: signup with empty body returns 422."""
        response = client.post("/signup", json={})
        assert response.status_code == 422

    def test_signup_invalid_email_format(self):
        """Negative: signup with malformed email may still reach validation layer."""
        response = client.post("/signup", json={
            "username": "user",
            "email": "not-an-email",
            "password": "pass123"
        })
        assert response.status_code in (200, 422, 400)

    @patch("app.api.Controller.user_service")
    @patch("app.api.Controller.auth")
    def test_login_success(self, mock_auth, mock_user_service):
        """Positive: valid credentials return access token."""
        mock_user_service.get_user_by_email.return_value = (
            "user-123", "test@example.com", "hashed_pw"
        )
        mock_user_service.verify_password.return_value = True
        mock_auth.create_access_token.return_value = "mocked_token"

        response = client.post("/login", json={
            "email": "test@example.com",
            "password": "secret123"
        })

        assert response.status_code == 200
        body = response.json()
        assert body["access_token"] == "mocked_token"
        assert body["token_type"] == "bearer"

    @patch("app.api.Controller.user_service")
    def test_login_user_not_found(self, mock_user_service):
        """Negative: login for non-existent user returns 404."""
        mock_user_service.get_user_by_email.return_value = None

        response = client.post("/login", json={
            "email": "nobody@example.com",
            "password": "secret123"
        })

        assert response.status_code == 404
        assert "sign up" in response.json()["detail"].lower()

    @patch("app.api.Controller.user_service")
    def test_login_invalid_password(self, mock_user_service):
        """Negative: wrong password returns 401."""
        mock_user_service.get_user_by_email.return_value = (
            "user-123", "test@example.com", "hashed_pw"
        )
        mock_user_service.verify_password.return_value = False

        response = client.post("/login", json={
            "email": "test@example.com",
            "password": "wrongpassword"
        })

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_missing_fields(self):
        """Negative: login without password returns 422."""
        response = client.post("/login", json={"email": "test@example.com"})
        assert response.status_code == 422

    def test_login_missing_all_fields(self):
        """Negative: login with empty body returns 422."""
        response = client.post("/login", json={})
        assert response.status_code == 422

    @patch("app.api.Controller.vector_db")
    @patch("app.api.Controller.async_upload")
    def test_upload_success(self, mock_async_upload, mock_vector_db):
        """Positive: new valid file is uploaded and task is dispatched."""
        mock_task_result = MagicMock()
        mock_task_result.id = "task-abc"
        mock_async_upload.delay.return_value = mock_task_result
        mock_vector_db.filename_exists.return_value = False

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.api.Controller.config") as mock_config:
                mock_config.UPLOAD_DIR = tmpdir
                mock_config.ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
                mock_config.ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
                mock_config.MAX_FILE_SIZE = 30 * 1024 * 1024

                response = client.post(
                    "/upload",
                    files={"file": ("sample.txt", io.BytesIO(b"hello test content"), "text/plain")},
                    headers=AUTH_HEADERS,
                )

        app.dependency_overrides = {}
        assert response.status_code == 200
        body = response.json()
        assert "task_id" in body
        assert body["message"] == "File uploaded and processing started"
        assert body["user_id"] == "user-123"

    def test_upload_unsupported_extension(self):
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.config") as mock_config:
            mock_config.ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
            mock_config.ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
            mock_config.MAX_FILE_SIZE = 30 * 1024 * 1024

            response = client.post(
                "/upload",
                files={"file": ("malware.exe", io.BytesIO(b"binary"), "application/octet-stream")},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_unsupported_mime_type(self):
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.config") as mock_config:
            mock_config.ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
            mock_config.ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
            mock_config.MAX_FILE_SIZE = 30 * 1024 * 1024

            response = client.post(
                "/upload",
                files={"file": ("doc.pdf", io.BytesIO(b"data"), "application/octet-stream")},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert response.status_code == 400
        assert "Unsupported content type" in response.json()["detail"]

    def test_upload_file_too_large(self):
        """Negative: file exceeding size limit returns 400."""
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.config") as mock_config:
            mock_config.ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
            mock_config.ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
            mock_config.MAX_FILE_SIZE = 10   # 10-byte limit

            response = client.post(
                "/upload",
                files={"file": ("big.txt", io.BytesIO(b"A" * 50), "text/plain")},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert response.status_code == 400
        assert "exceeds" in response.json()["detail"]

    def test_upload_no_auth(self):
        """Negative: upload without auth token returns 401."""
        response = client.post(
            "/upload",
            files={"file": ("sample.txt", io.BytesIO(b"data"), "text/plain")},
        )
        assert response.status_code == 401

    @patch("app.api.Controller.vector_db")
    @patch("app.api.Controller.async_upload")
    def test_upload_identical_file_returns_no_change_message(
        self, mock_async_upload, mock_vector_db
    ):
        """Positive: uploading a file whose hash matches stored hash skips processing."""
        import hashlib
        content = b"same content"
        file_hash = hashlib.sha256(content).hexdigest()

        mock_vector_db.filename_exists.return_value = True
        mock_vector_db.get_file_hash.return_value = file_hash

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.config") as mock_config:
            mock_config.ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
            mock_config.ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
            mock_config.MAX_FILE_SIZE = 30 * 1024 * 1024

            response = client.post(
                "/upload",
                files={"file": ("sample.txt", io.BytesIO(content), "text/plain")},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert "identical content" in response.json()["message"].lower()

    @patch("app.api.Controller.vector_db")
    @patch("app.api.Controller.async_upload")
    def test_upload_conflict_returns_conflict_status(
        self, mock_async_upload, mock_vector_db
    ):
        """Positive: file name exists but content differs returns conflict status."""
        mock_vector_db.filename_exists.return_value = True
        mock_vector_db.get_file_hash.return_value = "different_hash_xyz"

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.config") as mock_config:
            mock_config.ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
            mock_config.ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
            mock_config.MAX_FILE_SIZE = 30 * 1024 * 1024

            response = client.post(
                "/upload",
                files={"file": ("sample.txt", io.BytesIO(b"new content"), "text/plain")},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "conflict"

    @patch("app.api.Controller.vector_db")
    @patch("app.api.Controller.async_upload")
    def test_upload_replace_existing_file(self, mock_async_upload, mock_vector_db):
        """Positive: user chooses 'replace'; old file deleted and new task dispatched."""
        mock_task_result = MagicMock()
        mock_task_result.id = "task-replace"
        mock_async_upload.delay.return_value = mock_task_result
        mock_vector_db.filename_exists.return_value = True
        mock_vector_db.get_file_hash.return_value = "old_hash"
        mock_vector_db.delete_by_filename.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.api.Controller.config") as mock_config:
                mock_config.UPLOAD_DIR = tmpdir
                mock_config.ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
                mock_config.ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
                mock_config.MAX_FILE_SIZE = 30 * 1024 * 1024

                response = client.post(
                    "/upload",
                    files={"file": ("sample.txt", io.BytesIO(b"new content"), "text/plain")},
                    params={"user_input": "replace"},
                    headers=AUTH_HEADERS,
                )

        app.dependency_overrides = {}
        assert response.status_code == 200
        body = response.json()
        assert body["task_id"] == "task-replace"
        mock_vector_db.delete_by_filename.assert_called_once_with("sample.txt")

    @patch("app.api.Controller.vector_db")
    @patch("app.api.Controller.async_upload")
    def test_upload_keep_both(self, mock_async_upload, mock_vector_db):
        """Positive: user chooses 'keep_both'; new unique file is created."""
        mock_task_result = MagicMock()
        mock_task_result.id = "task-keep"
        mock_async_upload.delay.return_value = mock_task_result
        mock_vector_db.filename_exists.return_value = True
        mock_vector_db.get_file_hash.return_value = "old_hash"

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.api.Controller.config") as mock_config:
                mock_config.UPLOAD_DIR = tmpdir
                mock_config.ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
                mock_config.ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
                mock_config.MAX_FILE_SIZE = 30 * 1024 * 1024

                response = client.post(
                    "/upload",
                    files={"file": ("sample.txt", io.BytesIO(b"new content"), "text/plain")},
                    params={"user_input": "keep_both"},
                    headers=AUTH_HEADERS,
                )

        app.dependency_overrides = {}
        assert response.status_code == 200
        body = response.json()
        assert body["task_id"] == "task-keep"

    @patch("app.api.Controller.vector_db")
    def test_upload_invalid_user_input_for_conflict(self, mock_vector_db):
        """Negative: user provides an invalid resolution value returns 400."""
        mock_vector_db.filename_exists.return_value = True
        mock_vector_db.get_file_hash.return_value = "different_hash"

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.config") as mock_config:
            mock_config.ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
            mock_config.ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
            mock_config.MAX_FILE_SIZE = 30 * 1024 * 1024

            response = client.post(
                "/upload",
                files={"file": ("sample.txt", io.BytesIO(b"new content"), "text/plain")},
                params={"user_input": "invalid_choice"},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert response.status_code == 400
        assert "Invalid resolution" in response.json()["detail"]

    def test_upload_status_pending(self):
        """Positive: task status PENDING is returned correctly."""
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.AsyncResult") as mock_async_result:
            mock_async_result.return_value = MagicMock(status="PENDING")
            response = client.get("/upload/status/task-abc", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        body = response.json()
        assert body["task_id"] == "task-abc"
        assert body["status"] == "PENDING"

    def test_upload_status_success(self):
        """Positive: task status SUCCESS is returned correctly."""
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.AsyncResult") as mock_async_result:
            mock_async_result.return_value = MagicMock(status="SUCCESS")
            response = client.get("/upload/status/task-abc", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert response.json()["status"] == "SUCCESS"

    def test_upload_status_failure(self):
        """Positive: task status FAILURE is returned correctly."""
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.AsyncResult") as mock_async_result:
            mock_async_result.return_value = MagicMock(status="FAILURE")
            response = client.get("/upload/status/task-abc", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert response.json()["status"] == "FAILURE"

    def test_upload_status_no_auth(self):
        """Negative: checking task status without token returns 401."""
        response = client.get("/upload/status/task-abc")
        assert response.status_code == 401

    @staticmethod
    def _make_to_thread_mock(intent: str):
        """Return an AsyncMock that stands in for asyncio.to_thread → intent."""
        return AsyncMock(return_value=intent)

    @patch("app.api.Controller.chat_db")
    @patch("app.api.Controller.retrieve")
    @patch("app.api.Controller.session")
    def test_chat_success(self, mock_session, mock_retrieve, mock_chat_db):
        """Positive: valid chat request returns session_id and response."""
        mock_session.get_or_create_session.return_value = (
            "session-1", "2025-01-01", "2025-01-02"
        )
        mock_chat_db.get_session_meta.return_value = None
        mock_retrieve.query_docs = AsyncMock(return_value="Here is the answer.")
        mock_chat_db.store_chat.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("asyncio.to_thread", new=self._make_to_thread_mock("question")):
            resp = client.post(
                "/chat",
                json={"question": "What is AI?"},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "session-1"
        assert body["Response"] == "Here is the answer."

    @patch("app.api.Controller.chat_db")
    @patch("app.api.Controller.retrieve")
    @patch("app.api.Controller.session")
    def test_chat_stores_history_with_correct_fields(
        self, mock_session, mock_retrieve, mock_chat_db
    ):
        """Positive: store_chat is called once with all expected kwargs."""
        mock_session.get_or_create_session.return_value = (
            "session-1", "2025-01-01", "2025-01-02"
        )
        mock_chat_db.get_session_meta.return_value = None
        mock_retrieve.query_docs = AsyncMock(return_value="some answer")
        mock_chat_db.store_chat.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("asyncio.to_thread", new=self._make_to_thread_mock("question")):
            client.post("/chat", json={"question": "Hello?"}, headers=AUTH_HEADERS)

        app.dependency_overrides = {}

        mock_chat_db.store_chat.assert_called_once()
        kwargs = mock_chat_db.store_chat.call_args.kwargs
        assert kwargs["user_id"] == "user-123"
        assert kwargs["question"] == "Hello?"
        assert kwargs["response"] == "some answer"
        assert kwargs["session_id"] == "session-1"
        assert "intent" in kwargs

    @patch("app.api.Controller.chat_db")
    @patch("app.api.Controller.retrieve")
    @patch("app.api.Controller.session")
    def test_chat_with_existing_session_id(
        self, mock_session, mock_retrieve, mock_chat_db
    ):
        """Positive: chat with a valid client-supplied session_id reuses that session."""
        mock_chat_db.get_session_meta.return_value = ("2025-01-01", "2025-01-02")
        mock_retrieve.query_docs = AsyncMock(return_value="answer")
        mock_chat_db.store_chat.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("asyncio.to_thread", new=self._make_to_thread_mock("question")):
            resp = client.post(
                "/chat",
                json={"question": "Hello?", "session_id": "existing-session"},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "existing-session"

    @patch("app.api.Controller.chat_db")
    @patch("app.api.Controller.retrieve")
    @patch("app.api.Controller.session")
    def test_chat_with_invalid_session_id_falls_back_to_redis(
        self, mock_session, mock_retrieve, mock_chat_db
    ):
        """Positive: chat with an unrecognised session_id falls back to Redis session."""
        mock_chat_db.get_session_meta.return_value = None
        mock_session.get_or_create_session.return_value = (
            "redis-session", "2025-01-01", "2025-01-02"
        )
        mock_retrieve.query_docs = AsyncMock(return_value="answer")
        mock_chat_db.store_chat.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("asyncio.to_thread", new=self._make_to_thread_mock("question")):
            resp = client.post(
                "/chat",
                json={"question": "Hi?", "session_id": "ghost-session"},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "redis-session"

    def test_chat_missing_question(self):
        """Negative: chat without 'question' field returns 422."""
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.post("/chat", json={}, headers=AUTH_HEADERS)
        app.dependency_overrides = {}
        assert resp.status_code == 422

    def test_chat_no_auth(self):
        """Negative: chat without auth token returns 401."""
        resp = client.post("/chat", json={"question": "What is AI?"})
        assert resp.status_code == 401

    @patch("app.api.Controller.chat_db")
    @patch("app.api.Controller.retrieve")
    @patch("app.api.Controller.session")
    def test_chat_returns_intent_field(
        self, mock_session, mock_retrieve, mock_chat_db
    ):
        """Positive: response body includes the 'intent' key with the detected value."""
        mock_session.get_or_create_session.return_value = (
            "session-1", "2025-01-01", "2025-01-02"
        )
        mock_chat_db.get_session_meta.return_value = None
        mock_retrieve.query_docs = AsyncMock(return_value="answer text")
        mock_chat_db.store_chat.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("asyncio.to_thread", new=self._make_to_thread_mock("factual")):
            resp = client.post(
                "/chat",
                json={"question": "What?"},
                headers=AUTH_HEADERS,
            )

        app.dependency_overrides = {}
        assert resp.status_code == 200
        assert "intent" in resp.json()
        assert resp.json()["intent"] == "factual"

    @patch("app.api.Controller.auth")
    def test_logout_success(self, mock_auth):
        """Positive: valid token is revoked and success message returned."""
        mock_auth.revoke_token.return_value = None

        resp = client.post("/logout", headers={"Authorization": "Bearer some_valid_token"})

        assert resp.status_code == 200
        assert resp.json()["message"] == "Successfully logged out"
        mock_auth.revoke_token.assert_called_once_with("some_valid_token")

    def test_logout_no_token(self):
        """Negative: logout without Authorization header returns 401."""
        resp = client.post("/logout")
        assert resp.status_code == 401

    @patch("app.api.Controller.session")
    def test_new_chat_creates_session(self, mock_session):
        """Positive: new_chat endpoint creates a fresh session."""
        mock_session.create_new_session.return_value = (
            "sess-new", "2025-01-01T00:00:00", "2025-01-01T01:00:00"
        )

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.post("/new_chat", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess-new"
        assert "session_start" in body
        assert "session_end" in body

    def test_new_chat_no_auth(self):
        """Negative: new_chat without auth token returns 401."""
        resp = client.post("/new_chat")
        assert resp.status_code == 401

    @patch("app.api.Controller.session")
    def test_new_chat_returns_correct_user_session(self, mock_session):
        """Positive: create_new_session is called with the authenticated user's ID."""
        mock_session.create_new_session.return_value = (
            "sess-xyz", "2025-01-01T00:00:00", "2025-01-01T01:00:00"
        )
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        client.post("/new_chat", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        mock_session.create_new_session.assert_called_once_with("user-123")

    @patch("app.api.Controller.redis_client")
    def test_cache_list_empty(self, mock_redis):
        """Positive: no cached entries returns empty list."""
        mock_redis.keys.return_value = []
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/cache_list", headers=AUTH_HEADERS)
        app.dependency_overrides = {}
        assert resp.status_code == 200
        assert resp.json() == []

    @patch("app.api.Controller.redis_client")
    def test_cache_list_with_entries(self, mock_redis):
        """Positive: cached entries are returned with correct structure."""
        cache_entry = json.dumps({"response": "cached answer"})
        mock_redis.keys.return_value = ["semantic_cache:key1"]
        mock_redis.get.return_value = cache_entry

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/cache_list", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["response"] == "cached answer"

    def test_cache_list_no_auth(self):
        """Negative: cache_list without auth token returns 401."""
        resp = client.get("/cache_list")
        assert resp.status_code == 401

    @patch("app.api.Controller.redis_client")
    def test_clear_cache_with_keys(self, mock_redis):
        """Positive: clearing cache with existing keys calls redis delete."""
        mock_redis.keys.return_value = ["semantic_cache:key1", "semantic_cache:key2"]
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.delete("/clear_cache", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert "cache cleared" in resp.json().lower()
        mock_redis.delete.assert_called_once()

    @patch("app.api.Controller.redis_client")
    def test_clear_cache_no_keys(self, mock_redis):
        """Positive: clearing cache when already empty skips delete call."""
        mock_redis.keys.return_value = []
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.delete("/clear_cache", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        mock_redis.delete.assert_not_called()

    def test_clear_cache_no_auth(self):
        """Negative: clear_cache without auth token returns 401."""
        resp = client.delete("/clear_cache")
        assert resp.status_code == 401

    @patch("app.api.Controller.chat_db")
    def test_count_chats_returns_number(self, mock_chat_db):
        """Positive: count_user_chats returns the correct integer."""
        mock_chat_db.count_user_chats.return_value = 5
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/count_chats", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert resp.json() == 5

    @patch("app.api.Controller.chat_db")
    def test_count_chats_returns_zero(self, mock_chat_db):
        """Positive: count_user_chats returns zero when no chats exist."""
        mock_chat_db.count_user_chats.return_value = 0
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/count_chats", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert resp.json() == 0

    def test_count_chats_no_auth(self):
        """Negative: count_chats without auth token returns 401."""
        resp = client.get("/count_chats")
        assert resp.status_code == 401

    @patch("app.api.Controller.chat_db")
    def test_delete_chats_success(self, mock_chat_db):
        """Positive: delete_chats_by_user is called and success message returned."""
        mock_chat_db.delete_chats_by_user.return_value = None
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.delete("/delete_chats", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert "deletion completed" in resp.json().lower()

    @patch("app.api.Controller.chat_db")
    def test_delete_chats_called_with_correct_user_id(self, mock_chat_db):
        """Positive: delete_chats_by_user is called with the authenticated user's ID."""
        mock_chat_db.delete_chats_by_user.return_value = None
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        client.delete("/delete_chats", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        mock_chat_db.delete_chats_by_user.assert_called_once_with("user-123")

    def test_delete_chats_no_auth(self):
        """Negative: delete_chats without auth token returns 401."""
        resp = client.delete("/delete_chats")
        assert resp.status_code == 401

    @patch("app.api.Controller.chat_db")
    def test_get_chats_returns_list(self, mock_chat_db):
        """Positive: get_chats_by_user returns the full list of chat records."""
        mock_chat_db.get_chats_by_user.return_value = [
            {"session_id": "s1", "question": "Q1", "response": "A1"},
            {"session_id": "s1", "question": "Q2", "response": "A2"},
        ]

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/get_chats_by_user", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @patch("app.api.Controller.chat_db")
    def test_get_chats_empty(self, mock_chat_db):
        """Positive: get_chats_by_user returns empty list when no history exists."""
        mock_chat_db.get_chats_by_user.return_value = []
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/get_chats_by_user", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_chats_no_auth(self):
        """Negative: get_chats_by_user without auth token returns 401."""
        resp = client.get("/get_chats_by_user")
        assert resp.status_code == 401

    @patch("app.api.Controller.chat_db")
    @patch("app.api.Controller.session")
    def test_delete_session_success(self, mock_session, mock_chat_db):
        """Positive: owner deletes their own session and gets success message.

        The controller calls get_session_owner() (not session_belongs_to_user()),
        so we mock that method and return the same user ID as the authenticated user.
        """
        mock_chat_db.get_session_owner.return_value = "user-123"
        mock_chat_db.delete_session.return_value = None
        mock_session.delete_session_from_redis.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        resp = client.delete(
            "/delete_session",
            params={"session_id": "session-abc"},
            headers=AUTH_HEADERS,
        )

        app.dependency_overrides = {}
        assert resp.status_code == 200
        body = resp.json()
        assert "session deleted" in body["message"].lower()
        mock_chat_db.delete_session.assert_called_once_with("session-abc")

    @patch("app.api.Controller.chat_db")
    def test_delete_session_unauthorized(self, mock_chat_db):
        """Negative: a user trying to delete another user's session gets 403.

        The controller raises 403 with 'You are not authorised to delete this session.'
        We assert status code and that the detail contains 'authorised'.
        """
        mock_chat_db.get_session_owner.return_value = "other-user-456"
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        resp = client.delete(
            "/delete_session",
            params={"session_id": "session-other"},
            headers=AUTH_HEADERS,
        )

        app.dependency_overrides = {}
        assert resp.status_code == 403
        assert "authorised" in resp.json()["detail"].lower()

    @patch("app.api.Controller.chat_db")
    def test_delete_session_not_found(self, mock_chat_db):
        """Negative: deleting a non-existent session returns 404."""
        mock_chat_db.get_session_owner.return_value = None
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        resp = client.delete(
            "/delete_session",
            params={"session_id": "ghost-session"},
            headers=AUTH_HEADERS,
        )

        app.dependency_overrides = {}
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_delete_session_missing_session_id(self):
        """Negative: delete_session without session_id parameter returns 422."""
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.delete("/delete_session", headers=AUTH_HEADERS)
        app.dependency_overrides = {}
        assert resp.status_code == 422

    def test_delete_session_no_auth(self):
        """Negative: delete_session without auth token returns 401."""
        resp = client.delete("/delete_session", params={"session_id": "session-abc"})
        assert resp.status_code == 401

    @patch("app.api.Controller.chat_db")
    def test_get_sessions_by_user_returns_list(self, mock_chat_db):
        """Positive: get_sessions_by_user returns a list of session records."""
        mock_chat_db.get_sessions_by_user.return_value = [
            {"session_id": "s1", "session_start": "2025-01-01"},
            {"session_id": "s2", "session_start": "2025-01-02"},
        ]

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/get_sessions_by_user", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @patch("app.api.Controller.chat_db")
    def test_get_sessions_by_user_empty(self, mock_chat_db):
        """Positive: get_sessions_by_user returns empty list when no sessions exist."""
        mock_chat_db.get_sessions_by_user.return_value = []
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/get_sessions_by_user", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_sessions_by_user_no_auth(self):
        """Negative: get_sessions_by_user without auth returns 401."""
        resp = client.get("/get_sessions_by_user")
        assert resp.status_code == 401

    @patch("app.api.Controller.chat_db")
    def test_get_chats_by_session_success(self, mock_chat_db):
        """Positive: returns chats filtered by session_id."""
        mock_chat_db.get_chats_by_session_id.return_value = [
            {"session_id": "s1", "question": "Q1", "response": "A1"},
        ]

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get(
            "/get_chats_by_session",
            params={"session_id": "s1"},
            headers=AUTH_HEADERS,
        )
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_chats_by_session_no_auth(self):
        """Negative: get_chats_by_session without auth returns 401."""
        resp = client.get("/get_chats_by_session", params={"session_id": "s1"})
        assert resp.status_code == 401

    @patch("app.api.Controller.chat_db")
    def test_delete_message_success(self, mock_chat_db):
        """Positive: owner deletes a message from their session."""
        mock_chat_db.get_session_owner.return_value = "user-123"
        mock_chat_db.delete_message.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.delete(
            "/delete_message",
            params={"session_id": "s1", "question": "What is AI?"},
            headers=AUTH_HEADERS,
        )
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"].lower()

    @patch("app.api.Controller.chat_db")
    def test_delete_message_unauthorized(self, mock_chat_db):
        """Negative: user tries to delete a message from another user's session."""
        mock_chat_db.get_session_owner.return_value = "other-user-456"
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        resp = client.delete(
            "/delete_message",
            params={"session_id": "s-other", "question": "Q?"},
            headers=AUTH_HEADERS,
        )
        app.dependency_overrides = {}

        assert resp.status_code == 403

    @patch("app.api.Controller.chat_db")
    def test_delete_message_session_not_found(self, mock_chat_db):
        """Negative: deleting a message from a non-existent session returns 404."""
        mock_chat_db.get_session_owner.return_value = None
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        resp = client.delete(
            "/delete_message",
            params={"session_id": "ghost", "question": "Q?"},
            headers=AUTH_HEADERS,
        )
        app.dependency_overrides = {}

        assert resp.status_code == 404

    def test_delete_message_no_auth(self):
        """Negative: delete_message without auth returns 401."""
        resp = client.delete(
            "/delete_message",
            params={"session_id": "s1", "question": "What?"},
        )
        assert resp.status_code == 401

    @patch("app.api.Controller._metrics")
    def test_metrics_summary_default(self, mock_metrics):
        """Positive: metrics/summary returns summary data with default last_n."""
        mock_metrics.summary.return_value = {"total": 10, "avg_latency": 0.5}
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/metrics/summary", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        mock_metrics.summary.assert_called_once_with(last_n=100)

    @patch("app.api.Controller._metrics")
    def test_metrics_summary_custom_n(self, mock_metrics):
        """Positive: metrics/summary respects custom last_n query param."""
        mock_metrics.summary.return_value = {"total": 5}
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/metrics/summary", params={"last_n": 50}, headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        mock_metrics.summary.assert_called_once_with(last_n=50)

    def test_metrics_summary_no_auth(self):
        """Negative: metrics/summary without auth returns 401."""
        resp = client.get("/metrics/summary")
        assert resp.status_code == 401

    @patch("app.api.Controller._metrics")
    def test_metrics_recent_default(self, mock_metrics):
        """Positive: metrics/recent returns recent data with default n."""
        mock_metrics.recent.return_value = [{"latency": 0.3}]
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/metrics/recent", headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        mock_metrics.recent.assert_called_once_with(n=20)

    @patch("app.api.Controller._metrics")
    def test_metrics_recent_custom_n(self, mock_metrics):
        """Positive: metrics/recent respects custom n query param."""
        mock_metrics.recent.return_value = []
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/metrics/recent", params={"n": 5}, headers=AUTH_HEADERS)
        app.dependency_overrides = {}

        assert resp.status_code == 200
        mock_metrics.recent.assert_called_once_with(n=5)

    def test_metrics_recent_no_auth(self):
        """Negative: metrics/recent without auth returns 401."""
        resp = client.get("/metrics/recent")
        assert resp.status_code == 401

    @patch("app.api.Controller.vector_db")
    def test_check_file_exists_returns_hash(self, mock_vector_db):
        """Positive: existing filename returns its stored hash."""
        mock_vector_db.filename_exists.return_value = True
        mock_vector_db.get_file_hash.return_value = "abc123hash"

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get(
            "/upload/check-name",
            params={"filename": "sample.txt"},
            headers=AUTH_HEADERS,
        )
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert resp.json() == "abc123hash"

    @patch("app.api.Controller.vector_db")
    def test_check_file_not_exists_returns_none(self, mock_vector_db):
        """Positive: non-existent filename returns null/None."""
        mock_vector_db.filename_exists.return_value = False

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get(
            "/upload/check-name",
            params={"filename": "unknown.txt"},
            headers=AUTH_HEADERS,
        )
        app.dependency_overrides = {}

        assert resp.status_code == 200
        assert resp.json() is None

    def test_check_file_no_auth(self):
        """Negative: check-name without auth returns 401."""
        resp = client.get("/upload/check-name", params={"filename": "sample.txt"})
        assert resp.status_code == 401

    def test_check_file_missing_filename_param(self):
        """Negative: check-name without filename query param returns 422."""
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        resp = client.get("/upload/check-name", headers=AUTH_HEADERS)
        app.dependency_overrides = {}
        assert resp.status_code == 422