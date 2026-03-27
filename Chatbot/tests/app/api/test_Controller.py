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
        mock_user_service.get_user_by_email.return_value = {"email": "test@example.com"}

        response = client.post("/signup", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "secret123"
        })

        assert response.status_code == 400
        assert response.json()["detail"] == "Email already registered"

    def test_signup_missing_fields(self):
        response = client.post("/signup", json={"email": "test@example.com"})

        assert response.status_code == 422

    @patch("app.api.Controller.user_service")
    @patch("app.api.Controller.auth")
    def test_login_success(self, mock_auth, mock_user_service):
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
        mock_user_service.get_user_by_email.return_value = None

        response = client.post("/login", json={
            "email": "nobody@example.com",
            "password": "secret123"
        })

        assert response.status_code == 404
        assert "sign up" in response.json()["detail"].lower()

    @patch("app.api.Controller.user_service")
    def test_login_invalid_password(self, mock_user_service):
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
        response = client.post("/login", json={"email": "test@example.com"})

        assert response.status_code == 422

    @patch("app.api.Controller.async_upload")
    def test_upload_success(self, mock_async_upload):
        mock_task_result = MagicMock()
        mock_task_result.id = "task-abc"
        mock_async_upload.delay.return_value = mock_task_result

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
        response = client.post(
            "/upload",
            files={"file": ("sample.txt", io.BytesIO(b"data"), "text/plain")},
        )
        assert response.status_code == 401

    def test_upload_status_pending(self):
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
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.AsyncResult") as mock_async_result:
            mock_async_result.return_value = MagicMock(status="SUCCESS")

            response = client.get("/upload/status/task-abc", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert response.json()["status"] == "SUCCESS"

    def test_upload_status_failure(self):
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        with patch("app.api.Controller.AsyncResult") as mock_async_result:
            mock_async_result.return_value = MagicMock(status="FAILURE")

            response = client.get("/upload/status/task-abc", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert response.json()["status"] == "FAILURE"

    @patch("app.api.Controller.chat_db")
    @patch("app.api.Controller.retrieve")
    @patch("app.api.Controller.response")
    @patch("app.api.Controller.session")
    def test_chat_success(self, mock_session, mock_response, mock_retrieve, mock_chat_db):
        mock_session.get_or_create_session.return_value = (
            "session-1", "2025-01-01", "2025-01-02"
        )
        mock_response.detect_intent.return_value = "question"
        mock_retrieve.query_docs = AsyncMock(return_value="Here is the answer.")
        mock_chat_db.store_chat.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.post(
            "/chat",
            json={"question": "What is AI?"},
            headers=AUTH_HEADERS,
        )

        app.dependency_overrides = {}
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "session-1"
        assert body["Response"] == "Here is the answer."
        # assert body["intent"] == "question"

    @patch("app.api.Controller.chat_db")
    @patch("app.api.Controller.retrieve")
    @patch("app.api.Controller.session")
    def test_chat_stores_history_with_correct_fields(
        self, mock_session, mock_retrieve, mock_chat_db
    ):
        mock_session.get_or_create_session.return_value = (
            "session-1", "2025-01-01", "2025-01-02"
        )
        mock_retrieve.query_docs = AsyncMock(return_value="some answer")
        mock_chat_db.store_chat.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        client.post("/chat", json={"question": "Hello?"}, headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        mock_chat_db.store_chat.assert_called_once()
        kwargs = mock_chat_db.store_chat.call_args.kwargs
        assert kwargs["user_id"] == "user-123"
        assert kwargs["question"] == "Hello?"
        assert kwargs["response"] == "some answer"

    def test_chat_missing_question(self):
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.post("/chat", json={}, headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 422

    def test_chat_no_auth(self):
        response = client.post("/chat", json={"question": "What is AI?"})
        assert response.status_code == 401

    @patch("app.api.Controller.auth")
    def test_logout_success(self, mock_auth):
        mock_auth.revoke_token.return_value = None

        response = client.post("/logout", headers={"Authorization": "Bearer some_valid_token"})

        assert response.status_code == 200
        assert response.json()["message"] == "Successfully logged out"
        mock_auth.revoke_token.assert_called_once_with("some_valid_token")

    def test_logout_no_token(self):
        response = client.post("/logout")
        assert response.status_code == 401

    @patch("app.api.Controller.session")
    def test_new_chat_creates_session(self, mock_session):
        mock_session.create_new_session.return_value = (
            "sess-new", "2025-01-01T00:00:00", "2025-01-01T01:00:00"
        )

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.post("/new_chat", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "sess-new"
        assert "session_start" in body
        assert "session_end" in body

    def test_new_chat_no_auth(self):
        response = client.post("/new_chat")
        assert response.status_code == 401

    @patch("app.api.Controller.redis_client")
    def test_cache_list_empty(self, mock_redis):
        mock_redis.keys.return_value = []

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.get("/cache_list", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert response.json() == []

    @patch("app.api.Controller.redis_client")
    def test_cache_list_with_entries(self, mock_redis):
        cache_entry = json.dumps({
            "response": "cached answer"
        })
        mock_redis.keys.return_value = ["semantic_cache:key1"]
        mock_redis.get.return_value = cache_entry

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.get("/cache_list", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["response"] == "cached answer"

    @patch("app.api.Controller.redis_client")
    def test_clear_cache_with_keys(self, mock_redis):
        mock_redis.keys.return_value = ["semantic_cache:key1", "semantic_cache:key2"]

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.delete("/clear_cache", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert "cache cleared" in response.json().lower()
        mock_redis.delete.assert_called_once()

    @patch("app.api.Controller.redis_client")
    def test_clear_cache_no_keys(self, mock_redis):
        mock_redis.keys.return_value = []

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.delete("/clear_cache", headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        mock_redis.delete.assert_not_called()

    @patch("app.api.Controller.chat_db")
    def test_count_chats_returns_number(self, mock_chat_db):
        mock_chat_db.count_user_chats.return_value = 5

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.get("/count_chats",params={"user_id": "user-123"}, headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert response.json() == 5

    @patch("app.api.Controller.chat_db")
    def test_count_chats_returns_zero(self, mock_chat_db):
        mock_chat_db.count_user_chats.return_value = 0

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.get("/count_chats", params={"user_id": "user-123"}, headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert response.json() == 0

    def test_count_chats_no_auth(self):
        response = client.get("/count_chats")
        assert response.status_code == 401

    @patch("app.api.Controller.chat_db")
    def test_delete_chats_success(self, mock_chat_db):
        mock_chat_db.delete_chats_by_user.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.delete("/delete_chats", params={"user_id": "user-123"}, headers=AUTH_HEADERS)

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert "deletion completed" in response.json().lower()

    @patch("app.api.Controller.chat_db")
    def test_delete_chats_called_with_correct_user_id(self, mock_chat_db):
        mock_chat_db.delete_chats_by_user.return_value = None
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        client.delete("/delete_chats", params={"user_id": "user-123"}, headers=AUTH_HEADERS)
        app.dependency_overrides = {}
        mock_chat_db.delete_chats_by_user.assert_called_once_with("user-123")

    def test_delete_chats_no_auth(self):
        response = client.delete("/delete_chats")
        assert response.status_code == 401

    @patch("app.api.Controller.chat_db")
    def test_get_chats_returns_list(self, mock_chat_db):
        mock_chat_db.get_chats_by_user.return_value = [
            {"session_id": "s1", "question": "Q1", "response": "A1"},
            {"session_id": "s1", "question": "Q2", "response": "A2"},
        ]

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        response = client.get("/get_chats_by_user", params={"user_id": "user-123"}, headers=AUTH_HEADERS)
        app.dependency_overrides = {}
        assert response.status_code == 200
        assert len(response.json()) == 2

    @patch("app.api.Controller.chat_db")
    def test_get_chats_empty(self, mock_chat_db):
        mock_chat_db.get_chats_by_user.return_value = []

        app.dependency_overrides[auth.verify_token] = mock_verify_token
        response = client.get("/get_chats_by_user", params={"user_id": "user-123"}, headers=AUTH_HEADERS)
        app.dependency_overrides = {}
        assert response.status_code == 200
        assert response.json() == []

    def test_get_chats_no_auth(self):
        response = client.get("/get_chats_by_user")
        assert response.status_code == 401

    @patch("app.api.Controller.chat_db")
    def test_delete_session_success(self, mock_chat_db):
        mock_chat_db.session_belongs_to_user.return_value = True
        mock_chat_db.delete_session.return_value = None

        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.delete(
            "/delete_session",
            params={"session_id": "session-abc"},
            headers=AUTH_HEADERS,
        )

        app.dependency_overrides = {}
        assert response.status_code == 200
        assert "session deleted" in response.json().lower()
        mock_chat_db.delete_session.assert_called_once_with("session-abc")

    @patch("app.api.Controller.chat_db")
    def test_delete_session_unauthorized(self, mock_chat_db):
        mock_chat_db.session_belongs_to_user.return_value = False
        app.dependency_overrides[auth.verify_token] = mock_verify_token

        response = client.delete(
            "/delete_session",
            params={"session_id": "session-other"},
            headers=AUTH_HEADERS,
        )

        app.dependency_overrides = {}
        assert response.status_code == 403
        assert "Not authorized" in response.json()["detail"]

    @patch("app.api.Controller.chat_db")
    def test_delete_session_missing_session_id(self, mock_chat_db):
        app.dependency_overrides[auth.verify_token] = mock_verify_token
        response = client.delete("/delete_session", headers=AUTH_HEADERS)
        app.dependency_overrides = {}
        assert response.status_code == 422

    def test_delete_session_no_auth(self):
        response = client.delete("/delete_session", params={"session_id": "session-abc"})
        assert response.status_code == 401