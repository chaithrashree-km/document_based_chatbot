import pytest
from unittest.mock import patch, MagicMock

from app.db.Vector_Database import VectorDatabase


class TestVectorDatabase:

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_create_collection_when_not_exists(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.QDRANT_URL = "http://localhost:6333"
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        db.create_collection(vector_size=128)
        mock_client.create_collection.assert_called_once()

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_create_collection_when_exists(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.QDRANT_URL = "http://localhost:6333"
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        db.create_collection(vector_size=128)
        mock_client.create_collection.assert_not_called()

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_create_collection_called_with_correct_params(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.QDRANT_URL = "http://localhost:6333"
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        db.create_collection(vector_size=256)
        args, kwargs = mock_client.create_collection.call_args
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["vectors_config"].size == 256
        assert kwargs["vectors_config"].distance.name == "COSINE"

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_get_collections_called(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        db.create_collection(vector_size=128)
        mock_client.get_collections.assert_called_once()

    # ===================== NEW TESTS =====================

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_filename_exists_collection_not_present(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        result = db.filename_exists("file.txt")
        assert result is None

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_filename_exists_true(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        mock_client.scroll.return_value = ([MagicMock()], None)
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        result = db.filename_exists("file.txt")
        assert result is True

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_filename_exists_false(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        mock_client.scroll.return_value = ([], None)
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        result = db.filename_exists("file.txt")
        assert result is False

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_get_file_hash_collection_not_present(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        result = db.get_file_hash("file.txt")
        assert result is None

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_get_file_hash_found(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        mock_point = MagicMock()
        mock_point.payload = {"file_hash": "abc123"}
        mock_client.scroll.return_value = ([mock_point], None)
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        result = db.get_file_hash("file.txt")
        assert result == "abc123"

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_get_file_hash_not_found(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        mock_client.scroll.return_value = ([], None)
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        result = db.get_file_hash("file.txt")
        assert result is None

    @patch("app.db.Vector_Database.QdrantClient")
    @patch("app.db.Vector_Database.Config")
    def test_delete_by_filename(self, mock_config, mock_qdrant):
        mock_config_instance = MagicMock()
        mock_config_instance.COLLECTION_NAME = "test_collection"
        mock_config.return_value = mock_config_instance
        mock_client = MagicMock()
        mock_qdrant.return_value = mock_client
        db = VectorDatabase()
        db.client = mock_client
        db.config = mock_config_instance
        db.delete_by_filename("file.txt")
        mock_client.delete.assert_called_once()