import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.security.jwt_authentication import Authentication


class TestAuthentication:

    @patch("app.security.jwt_authentication.jwt.encode")
    @patch("app.security.jwt_authentication.Config")
    def test_create_access_token_success(self, mock_config, mock_encode):

        mock_config_instance = MagicMock()
        mock_config_instance.ACCESS_TOKEN_EXPIRE_MINUTES = 30
        mock_config_instance.SECRET_KEY = "secret"
        mock_config_instance.ALGORITHM = "HS256"
        mock_config.return_value = mock_config_instance

        mock_encode.return_value = "mocked_token"

        auth = Authentication()
        auth.config = mock_config_instance

        data = {"sub": "user-123"}
        token = auth.create_access_token(data)

        assert token == "mocked_token"
        mock_encode.assert_called_once()


    @patch("app.security.jwt_authentication.jwt.decode")
    @patch("app.security.jwt_authentication.Config")
    def test_verify_token_success(self, mock_config, mock_decode):
        mock_config_instance = MagicMock()
        mock_config_instance.SECRET_KEY = "secret"
        mock_config_instance.ALGORITHM = "HS256"
        mock_config.return_value = mock_config_instance

        mock_decode.return_value = {"sub": "user-123"}

        auth = Authentication()
        auth.config = mock_config_instance

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="valid_token"
        )

        result = auth.verify_token(credentials)

        assert result["sub"] == "user-123"    

    @patch("app.security.jwt_authentication.jwt.decode")
    @patch("app.security.jwt_authentication.Config")
    def test_verify_token_missing_sub(self, mock_config, mock_decode):
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        mock_decode.return_value = {}  # no "sub"

        auth = Authentication()
        auth.config = mock_config_instance

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="token"
        )

        with pytest.raises(HTTPException) as exc:
            auth.verify_token(credentials)

        assert exc.value.status_code == 401
        assert exc.value.detail == "Invalid token" 

    @patch("app.security.jwt_authentication.jwt.decode")
    @patch("app.security.jwt_authentication.Config")
    def test_verify_token_invalid(self, mock_config, mock_decode):
        from jose import JWTError
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        mock_decode.side_effect = JWTError("Invalid")

        auth = Authentication()
        auth.config = mock_config_instance

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="bad_token"
        )

        with pytest.raises(HTTPException) as exc:
            auth.verify_token(credentials)

        assert exc.value.status_code == 401
        assert exc.value.detail == "Token validation failed"  

    @patch("app.security.jwt_authentication.Config")
    def test_revoke_token_success(self, mock_config):
        mock_config_instance = MagicMock()
        mock_config_instance.ACCESS_TOKEN_EXPIRE_MINUTES = 30
        mock_config.return_value = mock_config_instance

        mock_redis = MagicMock()

        auth = Authentication()
        auth.config = mock_config_instance
        auth.redis_client = mock_redis

        token = "sample_token"
        auth.revoke_token(token)

        mock_redis.setex.assert_called_once_with(
            f"blacklist:{token}",
            30 * 60,
            "revoked"
        )         

    @patch("app.security.jwt_authentication.Config")
    def test_revoke_token_empty(self, mock_config):
        mock_config_instance = MagicMock()
        mock_config_instance.ACCESS_TOKEN_EXPIRE_MINUTES = 30
        mock_config.return_value = mock_config_instance

        mock_redis = MagicMock()

        auth = Authentication()
        auth.config = mock_config_instance
        auth.redis_client = mock_redis

        auth.revoke_token("")

        mock_redis.setex.assert_called_once()        