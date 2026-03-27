import pytest
from unittest.mock import patch, MagicMock, call
from app.db.Postgres_Database import Database, get_pool
import app.db.Postgres_Database as db_module


class TestPostgresDatabase:

    def setup_method(self):
        db_module._pool = None

    @patch("app.db.Postgres_Database.pool.ThreadedConnectionPool")
    @patch("app.db.Postgres_Database.Config")
    def test_get_pool_creates_pool_on_first_call(self, mock_config, mock_pool_cls):
        mock_config_instance = MagicMock()
        mock_config_instance.MIN_CONN = 2
        mock_config_instance.MAX_CONN = 10
        mock_config_instance.POSTGRES_DB = "test_db"
        mock_config_instance.POSTGRES_USER = "user"
        mock_config_instance.POSTGRES_PASSWORD = "pass"
        mock_config_instance.POSTGRES_HOST = "localhost"
        mock_config_instance.POSTGRES_PORT = 5432
        mock_config.return_value = mock_config_instance

        mock_pool_cls.return_value = MagicMock()

        get_pool()

        mock_pool_cls.assert_called_once_with(
            minconn=2,
            maxconn=10,
            dbname="test_db",
            user="user",
            password="pass",
            host="localhost",
            port=5432
        )

    @patch("app.db.Postgres_Database.pool.ThreadedConnectionPool")
    @patch("app.db.Postgres_Database.Config")
    def test_get_pool_returns_same_instance_on_repeated_calls(self, mock_config, mock_pool_cls):
        mock_config.return_value = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool_cls.return_value = mock_pool_instance

        pool1 = get_pool()
        pool2 = get_pool()

        # Pool class must only be instantiated once
        mock_pool_cls.assert_called_once()
        assert pool1 is pool2

    @patch("app.db.Postgres_Database.pool.ThreadedConnectionPool")
    @patch("app.db.Postgres_Database.Config")
    def test_get_pool_returns_pool_instance(self, mock_config, mock_pool_cls):
        mock_config.return_value = MagicMock()
        mock_pool_instance = MagicMock()
        mock_pool_cls.return_value = mock_pool_instance

        result = get_pool()

        assert result is mock_pool_instance

    @patch("app.db.Postgres_Database.pool.ThreadedConnectionPool")
    @patch("app.db.Postgres_Database.Config")
    def test_get_pool_raises_if_pool_creation_fails(self, mock_config, mock_pool_cls):
        mock_config.return_value = MagicMock()
        mock_pool_cls.side_effect = Exception("Could not connect to DB")

        with pytest.raises(Exception, match="Could not connect to DB"):
            get_pool()

    @patch("app.db.Postgres_Database.get_pool")
    def test_database_init_calls_getconn(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        Database()

        mock_pool.getconn.assert_called_once()

    @patch("app.db.Postgres_Database.get_pool")
    def test_database_init_sets_autocommit_true(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        Database()

        assert mock_conn.autocommit is True

    @patch("app.db.Postgres_Database.get_pool")
    def test_database_init_creates_cursor(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        db = Database()

        mock_conn.cursor.assert_called_once()
        assert db.cursor is mock_cursor

    @patch("app.db.Postgres_Database.get_pool")
    def test_database_init_stores_conn(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        db = Database()

        assert db.conn is mock_conn

    @patch("app.db.Postgres_Database.get_pool")
    def test_database_init_raises_if_getconn_fails(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_pool.getconn.side_effect = Exception("Pool exhausted")
        mock_get_pool.return_value = mock_pool

        with pytest.raises(Exception, match="Pool exhausted"):
            Database()

    @patch("app.db.Postgres_Database.get_pool")
    def test_database_init_raises_if_cursor_fails(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("Cursor error")
        mock_pool.getconn.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        with pytest.raises(Exception, match="Cursor error"):
            Database()

    @patch("app.db.Postgres_Database.get_pool")
    def test_return_to_pool_closes_cursor(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        db = Database()
        db.return_to_pool()

        mock_cursor.close.assert_called_once()

    @patch("app.db.Postgres_Database.get_pool")
    def test_return_to_pool_calls_putconn_with_connection(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        db = Database()
        db.return_to_pool()

        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch("app.db.Postgres_Database.get_pool")
    def test_return_to_pool_closes_cursor_before_putconn(self, mock_get_pool):
        """Cursor must be closed before the connection is returned to the pool."""
        call_order = []

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.close.side_effect = lambda: call_order.append("cursor_close")
        mock_pool.putconn.side_effect = lambda c: call_order.append("putconn")

        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        db = Database()
        db.return_to_pool()

        assert call_order == ["cursor_close", "putconn"]

    @patch("app.db.Postgres_Database.get_pool")
    def test_return_to_pool_raises_if_cursor_close_fails(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.close.side_effect = Exception("Cursor already closed")
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        mock_get_pool.return_value = mock_pool

        db = Database()

        with pytest.raises(Exception, match="Cursor already closed"):
            db.return_to_pool()

    @patch("app.db.Postgres_Database.get_pool")
    def test_multiple_instances_use_same_pool(self, mock_get_pool):
        mock_pool = MagicMock()
        mock_pool.getconn.side_effect = [MagicMock(), MagicMock()]
        mock_get_pool.return_value = mock_pool

        db1 = Database()
        db2 = Database()

        assert mock_pool.getconn.call_count == 2
        assert db1.conn is not db2.conn  # each gets its own connection from the pool

    @patch("app.db.Postgres_Database.get_pool")
    def test_return_to_pool_called_twice_for_two_instances(self, mock_get_pool):
        mock_pool = MagicMock()
        conn1, conn2 = MagicMock(), MagicMock()
        mock_pool.getconn.side_effect = [conn1, conn2]
        mock_get_pool.return_value = mock_pool

        db1 = Database()
        db2 = Database()
        db1.return_to_pool()
        db2.return_to_pool()

        assert mock_pool.putconn.call_count == 2
        mock_pool.putconn.assert_any_call(conn1)
        mock_pool.putconn.assert_any_call(conn2)