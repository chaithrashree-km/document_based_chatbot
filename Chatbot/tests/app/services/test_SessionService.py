import pytest
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from app.services.Session_Service import SessionManagement


class TestSessionManagement:

    @patch.object(SessionManagement, "redis_client")
    def test_create_new_session_when_none_exists(self, mock_redis):
        mock_redis.get.return_value = None

        service = SessionManagement()

        session_id, start, end = service.get_or_create_session("user1")

        assert isinstance(session_id, str)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert end > start

        mock_redis.setex.assert_called_once()


    @patch.object(SessionManagement, "redis_client")
    def test_reuse_existing_session(self, mock_redis):

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
    def test_session_timeout_calculation(self, mock_redis):

        mock_redis.get.return_value = None

        service = SessionManagement()

        session_id, start, end = service.get_or_create_session("user3")

        expected = start + timedelta(seconds=service.SESSION_TIMEOUT)

        assert abs((end - expected).total_seconds()) < 1


    @patch.object(SessionManagement, "redis_client")
    def test_redis_key_format(self, mock_redis):

        mock_redis.get.return_value = None

        service = SessionManagement()

        service.get_or_create_session("test_user")

        mock_redis.get.assert_called_with("user_session:test_user")


    @patch.object(SessionManagement, "redis_client")
    def test_corrupted_json_in_redis(self, mock_redis):

        mock_redis.get.return_value = "invalid-json"

        service = SessionManagement()

        with pytest.raises(json.JSONDecodeError):
            service.get_or_create_session("user4")


    @patch.object(SessionManagement, "redis_client")
    def test_missing_fields_in_session_data(self, mock_redis):

        invalid_session = {
            "session_id": str(uuid.uuid4())
        }

        mock_redis.get.return_value = json.dumps(invalid_session)

        service = SessionManagement()

        with pytest.raises(KeyError):
            service.get_or_create_session("user5")


    @patch.object(SessionManagement, "redis_client")
    def test_redis_connection_failure(self, mock_redis):

        mock_redis.get.side_effect = Exception("Redis connection error")

        service = SessionManagement()

        with pytest.raises(Exception):
            service.get_or_create_session("user6")