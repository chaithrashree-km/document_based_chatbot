import pytest
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.services.Session_Service import SessionManagement


class TestSessionManagement:

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_reuses_existing_session(self, mock_redis):
        session_data = {
            "session_id": str(uuid.uuid4()),
            "start": datetime.now().isoformat(),
            "end": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        mock_redis.get.return_value = json.dumps(session_data)

        service = SessionManagement()
        session_id, start, end = service.get_or_create_session("user2")

        assert session_id == session_data["session_id"]
        assert start.isoformat() == session_data["start"]
        assert end.isoformat() == session_data["end"]

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_existing_returns_datetime_types(self, mock_redis):
        session_data = {
            "session_id": str(uuid.uuid4()),
            "start": datetime.now().isoformat(),
            "end": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        mock_redis.get.return_value = json.dumps(session_data)

        service = SessionManagement()
        session_id, start, end = service.get_or_create_session("user2")

        assert isinstance(start, datetime)
        assert isinstance(end, datetime)

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_existing_does_not_call_setex(self, mock_redis):
        """When a session already exists, no new session should be written."""
        session_data = {
            "session_id": str(uuid.uuid4()),
            "start": datetime.now().isoformat(),
            "end": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        mock_redis.get.return_value = json.dumps(session_data)

        service = SessionManagement()
        service.get_or_create_session("user2")

        mock_redis.setex.assert_not_called()

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_creates_new_when_none_exists(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()
        session_id, start, end = service.get_or_create_session("user1")

        assert isinstance(session_id, str)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert end > start

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_new_session_calls_setex(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()
        service.get_or_create_session("user1")

        mock_redis.setex.assert_called_once()

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_new_session_timeout_calculation(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()
        session_id, start, end = service.get_or_create_session("user3")

        expected_end = start + timedelta(seconds=service.SESSION_TIMEOUT)
        assert abs((end - expected_end).total_seconds()) < 1

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_new_session_redis_key_format(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()
        service.get_or_create_session("test_user")

        mock_redis.get.assert_called_with("user_session:test_user")

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_new_session_setex_uses_correct_ttl(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()
        service.get_or_create_session("user1")

        call_args = mock_redis.setex.call_args[0]
        assert call_args[1] == service.SESSION_TIMEOUT

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_new_session_setex_key_format(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()
        service.get_or_create_session("user1")

        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "user_session:user1"

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_new_session_payload_contains_session_id(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()
        session_id, start, end = service.get_or_create_session("user1")

        call_args = mock_redis.setex.call_args[0]
        stored = json.loads(call_args[2])
        assert stored["session_id"] == session_id

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_new_session_payload_contains_start_and_end(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()
        session_id, start, end = service.get_or_create_session("user1")

        call_args = mock_redis.setex.call_args[0]
        stored = json.loads(call_args[2])
        assert "start" in stored
        assert "end" in stored

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_new_session_id_is_valid_uuid(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()
        session_id, _, _ = service.get_or_create_session("user1")

        uuid.UUID(session_id)

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_corrupted_json_raises(self, mock_redis):
        mock_redis.get.return_value = "invalid-json"

        service = SessionManagement()

        with pytest.raises(json.JSONDecodeError):
            service.get_or_create_session("user4")

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_missing_start_field_raises(self, mock_redis):
        invalid_session = {"session_id": str(uuid.uuid4())}
        mock_redis.get.return_value = json.dumps(invalid_session)

        service = SessionManagement()

        with pytest.raises(KeyError):
            service.get_or_create_session("user5")

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_missing_end_field_raises(self, mock_redis):
        invalid_session = {
            "session_id": str(uuid.uuid4()),
            "start": datetime.now().isoformat()
        }
        mock_redis.get.return_value = json.dumps(invalid_session)

        service = SessionManagement()

        with pytest.raises(KeyError):
            service.get_or_create_session("user5")

    @patch.object(SessionManagement, "redis_client")
    def test_get_or_create_session_redis_get_raises_propagates(self, mock_redis):
        mock_redis.get.side_effect = Exception("Redis connection error")

        service = SessionManagement()

        with pytest.raises(Exception, match="Redis connection error"):
            service.get_or_create_session("user6")

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_returns_correct_types(self, mock_redis):
        service = SessionManagement()
        session_id, start, end = service.create_new_session("user1")

        assert isinstance(session_id, str)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_end_is_after_start(self, mock_redis):
        service = SessionManagement()
        session_id, start, end = service.create_new_session("user1")

        assert end > start

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_timeout_calculation(self, mock_redis):
        service = SessionManagement()
        session_id, start, end = service.create_new_session("user1")

        expected_end = start + timedelta(seconds=service.SESSION_TIMEOUT)
        assert abs((end - expected_end).total_seconds()) < 1

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_calls_setex(self, mock_redis):
        service = SessionManagement()
        service.create_new_session("user1")

        mock_redis.setex.assert_called_once()

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_redis_key_format(self, mock_redis):
        service = SessionManagement()
        service.create_new_session("user1")

        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "user_session:user1"

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_setex_uses_correct_ttl(self, mock_redis):
        service = SessionManagement()
        service.create_new_session("user1")

        call_args = mock_redis.setex.call_args[0]
        assert call_args[1] == service.SESSION_TIMEOUT

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_payload_contains_session_id(self, mock_redis):
        service = SessionManagement()
        session_id, _, _ = service.create_new_session("user1")

        call_args = mock_redis.setex.call_args[0]
        stored = json.loads(call_args[2])
        assert stored["session_id"] == session_id

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_payload_contains_start_and_end(self, mock_redis):
        service = SessionManagement()
        service.create_new_session("user1")

        call_args = mock_redis.setex.call_args[0]
        stored = json.loads(call_args[2])
        assert "start" in stored
        assert "end" in stored

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_id_is_valid_uuid(self, mock_redis):
        service = SessionManagement()
        session_id, _, _ = service.create_new_session("user1")

        uuid.UUID(session_id) 

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_ignores_any_existing_redis_key(self, mock_redis):
        """create_new_session always creates a new one — it never reads from redis."""
        service = SessionManagement()
        service.create_new_session("user1")

        mock_redis.get.assert_not_called()

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_always_generates_unique_ids(self, mock_redis):
        service = SessionManagement()
        id1, _, _ = service.create_new_session("user1")
        id2, _, _ = service.create_new_session("user1")

        assert id1 != id2

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_setex_raises_propagates(self, mock_redis):
        mock_redis.setex.side_effect = Exception("Redis write error")

        service = SessionManagement()

        with pytest.raises(Exception, match="Redis write error"):
            service.create_new_session("user1")

    @patch.object(SessionManagement, "redis_client")
    def test_delete_session_matching_id_calls_delete(self, mock_redis):
        """When the stored session_id matches, redis.delete must be called."""
        target_id = str(uuid.uuid4())
        session_data = {
            "session_id": target_id,
            "start": datetime.now().isoformat(),
            "end": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        mock_redis.get.return_value = json.dumps(session_data)

        service = SessionManagement()
        service.delete_session_from_redis("user1", target_id)

        mock_redis.delete.assert_called_once_with("user_session:user1")

    @patch.object(SessionManagement, "redis_client")
    def test_delete_session_checks_correct_redis_key(self, mock_redis):
        """redis.get must be called with user_session:{user_id}."""
        mock_redis.get.return_value = None

        service = SessionManagement()
        service.delete_session_from_redis("user99", "some-id")

        mock_redis.get.assert_called_once_with("user_session:user99")

    @patch.object(SessionManagement, "redis_client")
    def test_delete_session_non_matching_id_does_not_call_delete(self, mock_redis):
        """When the stored session_id differs, redis.delete must NOT be called."""
        session_data = {
            "session_id": str(uuid.uuid4()),
            "start": datetime.now().isoformat(),
            "end": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        mock_redis.get.return_value = json.dumps(session_data)

        service = SessionManagement()
        service.delete_session_from_redis("user1", "different-session-id")

        mock_redis.delete.assert_not_called()

    @patch.object(SessionManagement, "redis_client")
    def test_delete_session_no_existing_key_does_not_call_delete(self, mock_redis):
        """When redis.get returns None, redis.delete must NOT be called."""
        mock_redis.get.return_value = None

        service = SessionManagement()
        service.delete_session_from_redis("user1", "any-session-id")

        mock_redis.delete.assert_not_called()

    @patch.object(SessionManagement, "redis_client")
    def test_delete_session_redis_get_raises_propagates(self, mock_redis):
        mock_redis.get.side_effect = Exception("Redis error")

        service = SessionManagement()

        with pytest.raises(Exception, match="Redis error"):
            service.delete_session_from_redis("user1", "some-id")