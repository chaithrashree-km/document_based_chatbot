import pytest
from unittest.mock import MagicMock, patch
from app.services.User_Service import UserService

class TestUserService:

    @patch("app.services.User_Service.Database")
    def test_hash_password(self, mock_db):
        service = UserService()

        password = "test123"
        hashed = service.hash_password(password)

        assert hashed != password
        assert isinstance(hashed, str)


    @patch("app.services.User_Service.Database")
    def test_verify_password_success(self, mock_db):
        service = UserService()

        password = "mypassword"
        hashed = service.hash_password(password)

        result = service.verify_password(password, hashed)

        assert result is True


    @patch("app.services.User_Service.Database")
    def test_verify_password_failure(self, mock_db):
        service = UserService()

        password = "mypassword"
        hashed = service.hash_password(password)

        result = service.verify_password("wrongpassword", hashed)

        assert result is False


    @patch("app.services.User_Service.Database")
    def test_create_user_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (10,)

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = UserService()

        user_id = service.create_user("testuser", "test@mail.com", "password")

        assert user_id == 10
        mock_cursor.execute.assert_called_once()


    @patch("app.services.User_Service.Database")
    def test_create_user_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Database error")

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = UserService()

        with pytest.raises(Exception):
            service.create_user("user", "mail@test.com", "password")


    @patch("app.services.User_Service.Database")
    def test_get_user_by_email_success(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1, "user@mail.com", "hashed_password")

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = UserService()

        user = service.get_user_by_email("user@mail.com")

        assert user[0] == 1
        assert user[1] == "user@mail.com"


    @patch("app.services.User_Service.Database")
    def test_get_user_by_email_not_found(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = UserService()

        result = service.get_user_by_email("unknown@mail.com")

        assert result is None


    @patch("app.services.User_Service.Database")
    def test_get_user_by_email_database_error(self, mock_db):

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query failed")

        mock_db_instance = MagicMock()
        mock_db_instance.cursor = mock_cursor
        mock_db.return_value = mock_db_instance

        service = UserService()

        with pytest.raises(Exception):
            service.get_user_by_email("user@mail.com")