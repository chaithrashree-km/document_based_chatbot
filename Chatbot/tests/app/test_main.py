import pytest
from fastapi.testclient import TestClient
from app.main import app

class TestMainApplication:

    def setup_method(self):
        self.client = TestClient(app)

    def test_app_instance_created(self):

        assert app is not None

    def test_app_title(self):

        assert app.title == "Document Based Chatbot"

    def test_router_is_registered(self):

        routes = [route.path for route in app.routes]
        assert len(routes) > 0

    def test_router_contains_api_routes(self):

        routes = [route.path for route in app.routes]
        assert any(route != "/" for route in routes)

    def test_cors_middleware_exists(self):

        middleware_classes = [middleware.cls.__name__ for middleware in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_cors_allowed_origin(self):

        cors_middleware = None

        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware

        assert cors_middleware is not None
        assert "http://localhost:3000" in cors_middleware.kwargs["allow_origins"]

    def test_cors_allow_credentials(self):

        cors_middleware = None

        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware

        assert cors_middleware.kwargs["allow_credentials"] is True

    def test_cors_allow_methods(self):

        cors_middleware = None

        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware

        assert "*" in cors_middleware.kwargs["allow_methods"]

    def test_cors_allow_headers(self):

        cors_middleware = None

        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware

        assert "*" in cors_middleware.kwargs["allow_headers"]

    def test_invalid_origin_not_allowed(self):

        response = self.client.options(
            "/",
            headers={
                "Origin": "http://malicious-site.com",
                "Access-Control-Request-Method": "GET"
            }
        )

        assert "access-control-allow-origin" not in response.headers