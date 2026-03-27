import pytest
from unittest.mock import MagicMock, patch
from app.services.ChatHistory_Service import ChatHistoryService


class TestChatHistoryService:

    @patch("app.services.ChatHistory_Service.Database")
    def test_store_chat_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_conn = MagicMock()

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db_instance.conn = mock_conn
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        service.store_chat(
            1,
            "session1",
            "2026-01-01",
            "2026-01-01",
            "Hello",
            "Hi there"
        )

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


    @patch("app.services.ChatHistory_Service.Database")
    def test_store_chat_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Insert error")

        mock_conn = MagicMock()

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db_instance.conn = mock_conn
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception):
            service.store_chat(1, "s1", "start", "end", "q", "r")


    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_user_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "session1", "Q1", "R1"),
            (1, "session2", "Q2", "R2")
        ]

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        result = service.get_chats_by_user(1)

        assert len(result) == 2
        mock_cursor.execute.assert_called_once()


    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_user_empty(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        result = service.get_chats_by_user(10)

        assert result == []


    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_user_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query error")

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception):
            service.get_chats_by_user(1)


    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_session_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("Hello", "Hi"),
            ("How are you?", "Good")
        ]

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        result = service.get_chats_by_session("session1")

        assert len(result) == 2


    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_chats_by_user_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_conn = MagicMock()

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db_instance.conn = mock_conn
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        service.delete_chats_by_user(1)

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_session_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_conn = MagicMock()

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db_instance.conn = mock_conn
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        service.delete_session("session1")

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_session_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Delete error")

        mock_conn = MagicMock()

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db_instance.conn = mock_conn
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception):
            service.delete_session("session1")


    @patch("app.services.ChatHistory_Service.Database")
    def test_count_user_chats_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5,)

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        result = service.count_user_chats(1)

        assert result == 5


    @patch("app.services.ChatHistory_Service.Database")
    def test_count_user_chats_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Count query failed")

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception):
            service.count_user_chats(1)