import pytest
from unittest.mock import patch, MagicMock
from app.services.Health_Service import Health_Service

class TestHealthService:

    def setup_method(self):
        self.service = Health_Service()

    def test_check_redis_up(self):
        with patch("app.services.Health_Service.redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            result = self.service.check_redis()
        assert result == {"status": "up"}
        mock_client.ping.assert_called_once()
        mock_client.close.assert_called_once()

    def test_check_redis_down_on_ping_failure(self):
        with patch("app.services.Health_Service.redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.side_effect = Exception("Connection refused")
            mock_redis.return_value = mock_client
            result = self.service.check_redis()
        assert result["status"] == "down"
        assert result["error"] == "Connection refused"

    def test_check_redis_down_on_connection_failure(self):
        with patch("app.services.Health_Service.redis.from_url") as mock_redis:
            mock_redis.side_effect = Exception("Invalid Redis URL")
            result = self.service.check_redis()
        assert result["status"] == "down"
        assert result["error"] == "Invalid Redis URL"

    def test_check_redis_close_called_on_success(self):
        with patch("app.services.Health_Service.redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            self.service.check_redis()
        mock_client.close.assert_called_once()

    def test_check_redis_error_message_preserved(self):
        with patch("app.services.Health_Service.redis.from_url") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.side_effect = Exception("Timeout after 30s")
            mock_redis.return_value = mock_client
            result = self.service.check_redis()
        assert "Timeout after 30s" in result["error"]

    def test_check_postgres_up(self):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        with patch("app.services.Health_Service.get_pool", return_value=mock_pool):
            result = self.service.check_postgres()
        assert result == {"status": "up"}

    def test_check_postgres_executes_select_1(self):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        with patch("app.services.Health_Service.get_pool", return_value=mock_pool):
            self.service.check_postgres()
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    def test_check_postgres_cursor_closed_on_success(self):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        with patch("app.services.Health_Service.get_pool", return_value=mock_pool):
            self.service.check_postgres()
        mock_cursor.close.assert_called_once()

    def test_check_postgres_conn_returned_to_pool_on_success(self):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        with patch("app.services.Health_Service.get_pool", return_value=mock_pool):
            self.service.check_postgres()
        mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_check_postgres_down_on_getconn_failure(self):
        mock_pool = MagicMock()
        mock_pool.getconn.side_effect = Exception("Pool exhausted")
        with patch("app.services.Health_Service.get_pool", return_value=mock_pool):
            result = self.service.check_postgres()
        assert result["status"] == "down"
        assert result["error"] == "Pool exhausted"

    def test_check_postgres_down_on_execute_failure(self):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query failed")
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        with patch("app.services.Health_Service.get_pool", return_value=mock_pool):
            result = self.service.check_postgres()
        assert result["status"] == "down"
        assert result["error"] == "Query failed"

    def test_check_postgres_conn_returned_in_finally_on_failure(self):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query failed")
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        with patch("app.services.Health_Service.get_pool", return_value=mock_pool):
            self.service.check_postgres()
        mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_check_postgres_down_on_get_pool_failure(self):
        with patch("app.services.Health_Service.get_pool", side_effect=Exception("DB unavailable")):
            result = self.service.check_postgres()
        assert result["status"] == "down"
        assert result["error"] == "DB unavailable"

    def test_check_postgres_putconn_not_called_twice_on_success(self):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        with patch("app.services.Health_Service.get_pool", return_value=mock_pool):
            self.service.check_postgres()
        assert mock_pool.putconn.call_count == 1

    def test_check_qdrant_up(self):
        mock_vector_db = MagicMock()
        mock_collection_1 = MagicMock()
        mock_collection_2 = MagicMock()
        mock_vector_db.client.get_collections.return_value.collections = [mock_collection_1, mock_collection_2]
        with patch("app.services.Health_Service.VectorDatabase", return_value=mock_vector_db):
            result = self.service.check_qdrant()
        assert result["status"] == "up"
        assert result["collections"] == 2

    def test_check_qdrant_up_with_zero_collections(self):
        mock_vector_db = MagicMock()
        mock_vector_db.client.get_collections.return_value.collections = []
        with patch("app.services.Health_Service.VectorDatabase", return_value=mock_vector_db):
            result = self.service.check_qdrant()
        assert result["status"] == "up"
        assert result["collections"] == 0

    def test_check_qdrant_up_collection_count_matches(self):
        mock_vector_db = MagicMock()
        mock_vector_db.client.get_collections.return_value.collections = [MagicMock() for _ in range(5)]
        with patch("app.services.Health_Service.VectorDatabase", return_value=mock_vector_db):
            result = self.service.check_qdrant()
        assert result["collections"] == 5

    def test_check_qdrant_down_on_client_failure(self):
        mock_vector_db = MagicMock()
        mock_vector_db.client.get_collections.side_effect = Exception("Qdrant unreachable")
        with patch("app.services.Health_Service.VectorDatabase", return_value=mock_vector_db):
            result = self.service.check_qdrant()
        assert result["status"] == "down"
        assert result["error"] == "Qdrant unreachable"

    def test_check_qdrant_down_on_vector_db_init_failure(self):
        with patch("app.services.Health_Service.VectorDatabase", side_effect=Exception("Init failed")):
            result = self.service.check_qdrant()
        assert result["status"] == "down"
        assert result["error"] == "Init failed"

    def test_check_qdrant_error_message_preserved(self):
        with patch("app.services.Health_Service.VectorDatabase", side_effect=Exception("SSL handshake error")):
            result = self.service.check_qdrant()
        assert "SSL handshake error" in result["error"]