import os
from unittest.mock import patch
from app.Config import Config

class TestConfig:

    def test_qdrant_url_default_value(self):
        config = Config()
        assert config.QDRANT_URL == "http://localhost:6333"

    def test_collection_name(self):
        config = Config()
        assert config.COLLECTION_NAME == "documents"

    def test_redis_url(self):
        config = Config()
        assert config.REDIS_URL == "redis://localhost:6379/0"

    def test_cache_threshold_value(self):
        config = Config()
        assert config.CACHE_THRESHOLD == 0.90

    def test_cache_ttl_value(self):
        config = Config()
        assert config.CACHE_TTL == 86400

    def test_secret_key_value(self):
        config = Config()
        assert config.SECRET_KEY == "document_based_chatbot"

    def test_algorithm_value(self):
        config = Config()
        assert config.ALGORITHM == "HS256"

    def test_access_token_expiry(self):
        config = Config()
        assert config.ACCESS_TOKEN_EXPIRE_MINUTES == 60

    def test_postgres_user(self):
        config = Config()
        assert config.POSTGRES_USER == "postgres"

    def test_postgres_password(self):
        config = Config()
        assert config.POSTGRES_PASSWORD == "root"

    def test_postgres_host(self):
        config = Config()
        assert config.POSTGRES_HOST == "127.0.0.1"

    def test_postgres_port(self):
        config = Config()
        assert config.POSTGRES_PORT == 5432

    def test_postgres_database(self):
        config = Config()
        assert config.POSTGRES_DB == "chatbot_database"

    def test_logout_token_expiry(self):
        config = Config()
        assert config.LOGOUT_TOKEN_EXPIRE == 1
