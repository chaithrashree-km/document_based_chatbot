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
            user_id=1,
            session_id="session1",
            session_start="2026-01-01",
            session_end="2026-01-01",
            question="Hello",
            response="Hi there",
            intent="greeting",
        )

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_store_chat_calls_return_to_pool(self, mock_db):

        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.store_chat(1, "s1", "start", "end", "q", "r", "intent")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_store_chat_execute_receives_correct_values(self, mock_db):

        mock_cursor = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.store_chat(
            user_id="u1",
            session_id="s1",
            session_start="2026-01-01",
            session_end="2026-01-02",
            question="What is AI?",
            response="An answer.",
            intent="factual",
        )

        args = mock_cursor.execute.call_args[0]
        assert args[1] == ("u1", "s1", "2026-01-01", "2026-01-02", "What is AI?", "An answer.", "factual")

    @patch("app.services.ChatHistory_Service.Database")
    def test_store_chat_database_error_propagates(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Insert error")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Insert error"):
            service.store_chat(1, "s1", "start", "end", "q", "r", "intent")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_store_chat_commit_error_propagates(self, mock_db):

        mock_conn = MagicMock()
        mock_conn.commit.side_effect = Exception("Commit failed")
        mock_db_instance = MagicMock()
        mock_db_instance.conn = mock_conn
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Commit failed"):
            service.store_chat(1, "s1", "start", "end", "q", "r", "intent")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_user_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "session1", "Q1", "R1"),
            (1, "session2", "Q2", "R2"),
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
        result = service.get_chats_by_user(99)

        assert result == []

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_user_calls_return_to_pool(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.get_chats_by_user(1)

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_user_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query error")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Query error"):
            service.get_chats_by_user(1)

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_session_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("Hello", "Hi"),
            ("How are you?", "Good"),
        ]
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_chats_by_session("session1")

        assert len(result) == 2
        assert result[0] == ("Hello", "Hi")

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_session_empty(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_chats_by_session("ghost-session")

        assert result == []

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_session_calls_return_to_pool(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.get_chats_by_session("s1")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_session_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query error")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Query error"):
            service.get_chats_by_session("s1")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_session_id_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("What is ML?", "Machine Learning is..."),
        ]
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_chats_by_session_id("session-abc")

        assert len(result) == 1
        assert result[0][0] == "What is ML?"

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_session_id_empty(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_chats_by_session_id("no-such-session")

        assert result == []

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_session_id_calls_return_to_pool(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.get_chats_by_session_id("s1")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_chats_by_session_id_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query error")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Query error"):
            service.get_chats_by_session_id("s1")

        mock_db_instance.return_to_pool.assert_called_once()

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
    def test_delete_chats_by_user_calls_return_to_pool(self, mock_db):

        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.delete_chats_by_user(1)

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_chats_by_user_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Delete error")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Delete error"):
            service.delete_chats_by_user(1)

        mock_db_instance.return_to_pool.assert_called_once()

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
    def test_delete_session_calls_return_to_pool(self, mock_db):

        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.delete_session("session1")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_session_called_with_correct_session_id(self, mock_db):

        mock_cursor = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.delete_session("target-session")

        args = mock_cursor.execute.call_args[0]
        assert args[1] == ("target-session",)

    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_session_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Delete error")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Delete error"):
            service.delete_session("session1")

        mock_db_instance.return_to_pool.assert_called_once()

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
    def test_count_user_chats_zero(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.count_user_chats(99)

        assert result == 0

    @patch("app.services.ChatHistory_Service.Database")
    def test_count_user_chats_calls_return_to_pool(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (3,)
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.count_user_chats(1)

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_count_user_chats_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Count query failed")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Count query failed"):
            service.count_user_chats(1)

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_session_owner_returns_user_id(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("user-42",)
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_session_owner("session-abc")

        assert result == "user-42"

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_session_owner_returns_none_when_not_found(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_session_owner("ghost-session")

        assert result is None

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_session_owner_returns_string(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (123,)
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_session_owner("s1")

        assert isinstance(result, str)
        assert result == "123"

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_session_owner_calls_return_to_pool(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("u1",)
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.get_session_owner("s1")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_session_owner_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query failed")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Query failed"):
            service.get_session_owner("s1")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_session_meta_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("2026-01-01", "2026-01-02")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_session_meta("session-abc")

        assert result == ("2026-01-01", "2026-01-02")

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_session_meta_returns_none_when_not_found(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_session_meta("no-such-session")

        assert result is None

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_session_meta_calls_return_to_pool(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("s", "e")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.get_session_meta("s1")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_session_meta_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("DB error")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="DB error"):
            service.get_session_meta("s1")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_message_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db_instance.conn = mock_conn
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.delete_message("session-abc", "What is AI?")

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_message_called_with_correct_args(self, mock_db):

        mock_cursor = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.delete_message("s1", "Hello?")

        args = mock_cursor.execute.call_args[0]
        assert args[1] == ("s1", "Hello?")

    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_message_calls_return_to_pool(self, mock_db):

        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.delete_message("s1", "Q?")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_message_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Delete failed")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Delete failed"):
            service.delete_message("s1", "Q?")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_delete_message_commit_error_propagates(self, mock_db):

        mock_conn = MagicMock()
        mock_conn.commit.side_effect = Exception("Commit error")
        mock_db_instance = MagicMock()
        mock_db_instance.conn = mock_conn
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Commit error"):
            service.delete_message("s1", "Q?")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_sessions_by_user_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("session-1", "question", "What is AI?"),
            ("session-2", "greeting", "Hello"),
        ]
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_sessions_by_user("user-123")

        assert len(result) == 2
        assert result[0] == {
            "session_id": "session-1",
            "intent": "question",
            "question": "What is AI?",
        }
        assert result[1] == {
            "session_id": "session-2",
            "intent": "greeting",
            "question": "Hello",
        }

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_sessions_by_user_empty(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_sessions_by_user("new-user")

        assert result == []

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_sessions_by_user_dict_keys_are_correct(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("s1", "factual", "Q1")]
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        result = service.get_sessions_by_user("u1")

        assert set(result[0].keys()) == {"session_id", "intent", "question"}

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_sessions_by_user_calls_return_to_pool(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()
        service.get_sessions_by_user("u1")

        mock_db_instance.return_to_pool.assert_called_once()

    @patch("app.services.ChatHistory_Service.Database")
    def test_get_sessions_by_user_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query failed")
        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = ChatHistoryService()

        with pytest.raises(Exception, match="Query failed"):
            service.get_sessions_by_user("u1")

        mock_db_instance.return_to_pool.assert_called_once()