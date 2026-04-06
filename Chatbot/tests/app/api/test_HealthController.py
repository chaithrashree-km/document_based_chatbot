import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from app.api.Health_Controller import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestHealthController:

    def test_overall_health_all_up(self):
        mock_responses = {
            "redis": {"status": "up"},
            "postgres": {"status": "up"},
            "qdrant": {"status": "up"}
        }
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = mock_responses["redis"]
            mock_service.check_postgres.return_value = mock_responses["postgres"]
            mock_service.check_qdrant.return_value = mock_responses["qdrant"]
            response = client.get("/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["services"] == mock_responses

    def test_overall_health_redis_down(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "DOWN"}
            mock_service.check_postgres.return_value = {"status": "UP & RUNNING"}
            mock_service.check_qdrant.return_value = {"status": "UP & RUNNING"}
            response = client.get("/health/")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    def test_overall_health_postgres_down(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "UP & RUNNING"}
            mock_service.check_postgres.return_value = {"status": "DOWN"}
            mock_service.check_qdrant.return_value = {"status": "UP & RUNNING"}
            response = client.get("/health/")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    def test_overall_health_qdrant_down(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "UP & RUNNING"}
            mock_service.check_postgres.return_value = {"status": "UP & RUNNING"}
            mock_service.check_qdrant.return_value = {"status": "DOWN"}
            response = client.get("/health/")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    def test_overall_health_all_down(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "DOWN"}
            mock_service.check_postgres.return_value = {"status": "DOWN"}
            mock_service.check_qdrant.return_value = {"status": "DOWN"}
            response = client.get("/health/")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    def test_overall_health_response_structure(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "UP & RUNNING"}
            mock_service.check_postgres.return_value = {"status": "UP & RUNNING"}
            mock_service.check_qdrant.return_value = {"status": "UP & RUNNING"}
            response = client.get("/health/")
        body = response.json()
        assert "status" in body
        assert "services" in body
        assert "redis" in body["services"]
        assert "postgres" in body["services"]
        assert "qdrant" in body["services"]

    def test_redis_health_up(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "UP & RUNNING"}
            response = client.get("/health/redis")
        assert response.status_code == 200
        assert response.json() == {"status": "UP & RUNNING"}

    def test_redis_health_down(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "DOWN"}
            response = client.get("/health/redis")
        assert response.status_code == 200
        assert response.json() == {"status": "DOWN"}

    def test_redis_health_with_error_detail(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "DOWN", "error": "Connection refused"}
            response = client.get("/health/redis")
        assert response.json()["error"] == "Connection refused"

    def test_postgres_health_up(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_postgres.return_value = {"status": "UP & RUNNING"}
            response = client.get("/health/postgres")
        assert response.status_code == 200
        assert response.json() == {"status": "UP & RUNNING"}

    def test_postgres_health_down(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_postgres.return_value = {"status": "DOWN"}
            response = client.get("/health/postgres")
        assert response.status_code == 200
        assert response.json() == {"status": "DOWN"}

    def test_postgres_health_with_error_detail(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_postgres.return_value = {"status": "DOWN", "error": "Timeout"}
            response = client.get("/health/postgres")
        assert response.json()["error"] == "Timeout"

    def test_qdrant_health_up(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_qdrant.return_value = {"status": "UP & RUNNING"}
            response = client.get("/health/qdrant")
        assert response.status_code == 200
        assert response.json() == {"status": "UP & RUNNING"}

    def test_qdrant_health_down(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_qdrant.return_value = {"status": "DOWN"}
            response = client.get("/health/qdrant")
        assert response.status_code == 200
        assert response.json() == {"status": "DOWN"}

    def test_qdrant_health_with_error_detail(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_qdrant.return_value = {"status": "DOWN", "error": "Unreachable"}
            response = client.get("/health/qdrant")
        assert response.json()["error"] == "Unreachable"

    def test_overall_health_calls_all_services(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "UP & RUNNING"}
            mock_service.check_postgres.return_value = {"status": "UP & RUNNING"}
            mock_service.check_qdrant.return_value = {"status": "UP & RUNNING"}
            client.get("/health/")
            mock_service.check_redis.assert_called_once()
            mock_service.check_postgres.assert_called_once()
            mock_service.check_qdrant.assert_called_once()

    def test_overall_health_partial_status_mixed(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "UP & RUNNING"}
            mock_service.check_postgres.return_value = {"status": "DOWN", "error": "Auth failed"}
            mock_service.check_qdrant.return_value = {"status": "UP & RUNNING"}
            response = client.get("/health/")
        assert response.status_code == 503
        assert response.json()["services"]["postgres"]["error"] == "Auth failed"

    def test_overall_health_status_field_values(self):
        with patch("app.api.Health_Controller.service") as mock_service:
            mock_service.check_redis.return_value = {"status": "UP & RUNNING"}
            mock_service.check_postgres.return_value = {"status": "UP & RUNNING"}
            mock_service.check_qdrant.return_value = {"status": "UP & RUNNING"}
            response = client.get("/health/")
        assert response.json()["status"] in ("healthy", "degraded")