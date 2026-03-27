import pytest
from unittest.mock import MagicMock, patch
from app.services.LLM_Service import Response

class TestResponseService:

    def setup_method(self):
        self.response_service = Response()

    def test_generate_response_success(self):
        mock_message = MagicMock()
        mock_message.content = "AI is the simulation of human intelligence. (Source: doc.pdf, Page No.: 1)"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_response("What is AI?", "Artificial Intelligence definition")
            assert "AI is the simulation" in result

    def test_generate_response_with_sources(self):
        mock_message = MagicMock()
        mock_message.content = "Summary of the document. Sources consulted: report.pdf"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            result = self.response_service.generate_response("Summarize the document", "document content", sources=["report.pdf"])
            assert result == "Summary of the document. Sources consulted: report.pdf"
            call_args = mock_create.call_args
            system_content = call_args.kwargs["messages"][0]["content"]
            assert "report.pdf" in system_content

    def test_generate_response_with_multiple_sources(self):
        mock_message = MagicMock()
        mock_message.content = "Combined summary."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            result = self.response_service.generate_response(
                "Summarize", "context", sources=["file1.pdf", "file2.pdf"]
            )
            call_args = mock_create.call_args
            system_content = call_args.kwargs["messages"][0]["content"]
            assert "file1.pdf, file2.pdf" in system_content

    def test_generate_response_without_sources(self):
        mock_message = MagicMock()
        mock_message.content = "Answer without sources."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            result = self.response_service.generate_response("What is ML?", "ML context", sources=None)
            call_args = mock_create.call_args
            system_content = call_args.kwargs["messages"][0]["content"]
            assert "Sources consulted" not in system_content
            assert result == "Answer without sources."

    def test_generate_response_no_context_answer(self):
        mock_message = MagicMock()
        mock_message.content = "The documents does not have a specific answer to your question"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_response("Unknown question", "irrelevant context")
            assert "specific answer" in result

    def test_generate_response_coding_question(self):
        mock_message = MagicMock()
        mock_message.content = "I'm a document assistant and not designed to answer coding or technical programming questions."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_response("Write a Python for loop", "some context")
            assert "document assistant" in result

    def test_generate_response_passes_correct_question(self):
        mock_message = MagicMock()
        mock_message.content = "Some answer."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_response("What is deep learning?", "context")
            call_args = mock_create.call_args
            user_message = call_args.kwargs["messages"][1]["content"]
            assert user_message == "What is deep learning?"

    def test_generate_response_general_question(self):
        mock_message = MagicMock()
        mock_message.content = "Hello! How can I help you today?"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_response("Hello", "")
            assert "Hello" in result

    def test_generate_response_api_failure(self):
        with patch.object(self.response_service.client.chat.completions, "create", side_effect=Exception("API failure")):
            with pytest.raises(Exception, match="API failure"):
                self.response_service.generate_response("What is AI?", "context")

    def test_generate_response_empty_choices(self):
        mock_api_response = MagicMock()
        mock_api_response.choices = []

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            with pytest.raises(IndexError):
                self.response_service.generate_response("What is AI?", "context")

    def test_generate_response_none_content(self):
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_response("What is AI?", "context")
            assert result is None

    def test_detect_intent_success(self):
        mock_message = MagicMock()
        mock_message.content = "  Document Summary Request  "
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.detect_intent("Summarize the document")
            assert result == "Document Summary Request"

    def test_detect_intent_trims_whitespace(self):
        mock_message = MagicMock()
        mock_message.content = "   AI Definition Query   "
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.detect_intent("What is AI?")
            assert result == "AI Definition Query"

    def test_detect_intent_returns_short_label(self):
        mock_message = MagicMock()
        mock_message.content = "File Upload Help"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.detect_intent("How do I upload a file?")
            assert len(result.split()) <= 3

    def test_detect_intent_passes_question_as_user_message(self):
        mock_message = MagicMock()
        mock_message.content = "Intent Label"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.detect_intent("What is neural network?")
            call_args = mock_create.call_args
            user_message = call_args.kwargs["messages"][1]["content"]
            assert user_message == "What is neural network?"

    def test_detect_intent_api_failure(self):
        with patch.object(self.response_service.client.chat.completions, "create", side_effect=Exception("API failure")):
            with pytest.raises(Exception, match="API failure"):
                self.response_service.detect_intent("What is AI?")

    def test_detect_intent_empty_choices(self):
        mock_api_response = MagicMock()
        mock_api_response.choices = []

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            with pytest.raises(IndexError):
                self.response_service.detect_intent("What is AI?")
