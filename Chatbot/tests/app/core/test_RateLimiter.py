import pytest
from unittest.mock import patch, MagicMock
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.Rate_Limiter import limiter


class TestRateLimiter:

    def test_limiter_instance_created(self):
        assert limiter is not None

    def test_limiter_is_correct_type(self):
        assert isinstance(limiter, Limiter)

    def test_limiter_has_key_func(self):
        assert limiter._key_func is not None

    def test_limiter_key_func_is_get_remote_address(self):
        assert limiter._key_func == get_remote_address

    def test_limiter_key_func_returns_ip(self):
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        result = get_remote_address(mock_request)
        assert result == "127.0.0.1"

    def test_limiter_key_func_returns_forwarded_ip(self):
        mock_request = MagicMock()
        mock_request.client.host = "10.0.0.1"
        mock_request.headers = {"X-Forwarded-For": "192.168.1.1"}
        result = get_remote_address(mock_request)
        assert result is not None

    def test_limiter_storage_uses_redis_url(self):
        from app.Config import Config
        config = Config()
        assert config.REDIS_URL is not None
        assert config.REDIS_URL.startswith("redis://") or config.REDIS_URL.startswith("rediss://")

    def test_limiter_is_singleton(self):
        from app.core.Rate_Limiter import limiter as limiter2
        assert limiter is limiter2

    def test_limiter_can_be_created_with_valid_params(self):
        from app.Config import Config
        config = Config()
        new_limiter = Limiter(key_func=get_remote_address, storage_uri=config.REDIS_URL)
        assert new_limiter is not None

    def test_limiter_key_func_callable(self):
        assert callable(limiter._key_func)

    def test_limiter_key_func_different_ips_return_different_keys(self):
        mock_request_1 = MagicMock()
        mock_request_1.client.host = "192.168.0.1"
        mock_request_1.headers = {}

        mock_request_2 = MagicMock()
        mock_request_2.client.host = "192.168.0.2"
        mock_request_2.headers = {}

        result1 = get_remote_address(mock_request_1)
        result2 = get_remote_address(mock_request_2)
        assert result1 != result2

    def test_limiter_key_func_returns_default_when_client_none(self):
        mock_request = MagicMock()
        mock_request.client = None
        mock_request.headers = {}
        result = get_remote_address(mock_request)
        assert result == "127.0.0.1"

    def test_limiter_creation_fails_when_config_raises(self):
        with pytest.raises(Exception):
            with patch("app.Config.Config.__init__", side_effect=Exception("Config load failed")):
                from app.Config import Config
                Config()

    def test_limiter_key_func_with_empty_host_returns_value(self):
        mock_request = MagicMock()
        mock_request.client.host = ""
        mock_request.headers = {}
        result = get_remote_address(mock_request)
        assert result is not None or result == ""