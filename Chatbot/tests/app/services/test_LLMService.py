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

    def test_generate_response_empty_sources_list_excludes_source_note(self):
        mock_message = MagicMock()
        mock_message.content = "Answer."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_response("Q", "ctx", sources=[])
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "Sources consulted" not in system_content

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

    def test_generate_response_passes_context_in_system_prompt(self):
        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_response("Q", "UNIQUE_CONTEXT_XYZ")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "UNIQUE_CONTEXT_XYZ" in system_content

    def test_generate_response_uses_correct_model(self):
        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_response("Q", "ctx")
            assert mock_create.call_args.kwargs["model"] == "llama-3.3-70b-versatile"

    def test_generate_response_uses_temperature_02(self):
        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_response("Q", "ctx")
            assert mock_create.call_args.kwargs["temperature"] == 0.2

    def test_generate_response_message_roles_are_system_then_user(self):
        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_response("Q", "ctx")
            msgs = mock_create.call_args.kwargs["messages"]
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"

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

    def test_generate_response_logs_usage(self):
        mock_api_response = MagicMock()
        mock_api_response.choices = [MagicMock()]
        mock_api_response.choices[0].message.content = "ok"

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            with patch("app.services.LLM_Service.logging") as mock_log:
                self.response_service.generate_response("Q", "ctx")
                mock_log.info.assert_called_once_with("RESPONSE USAGE: %s", mock_api_response.usage)

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

    # -----------------------------------------------------------------------
    # detect_intent
    # -----------------------------------------------------------------------

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

    def test_detect_intent_uses_correct_model(self):
        mock_message = MagicMock()
        mock_message.content = "Label"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.detect_intent("Q")
            assert mock_create.call_args.kwargs["model"] == "llama-3.3-70b-versatile"

    def test_detect_intent_uses_temperature_zero(self):
        mock_message = MagicMock()
        mock_message.content = "Label"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.detect_intent("Q")
            assert mock_create.call_args.kwargs["temperature"] == 0

    def test_detect_intent_message_roles_are_system_then_user(self):
        mock_message = MagicMock()
        mock_message.content = "Label"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.detect_intent("Q")
            msgs = mock_create.call_args.kwargs["messages"]
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"

    def test_detect_intent_does_not_log_usage(self):
        mock_message = MagicMock()
        mock_message.content = "Label"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            with patch("app.services.LLM_Service.logging") as mock_log:
                self.response_service.detect_intent("Q")
                mock_log.info.assert_not_called()

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

    def test_generate_summary_response_returns_content(self):
        mock_message = MagicMock()
        mock_message.content = "This document covers AI, ML, and Deep Learning."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_summary_response("Summarize report.pdf", "report content")
            assert result == "This document covers AI, ML, and Deep Learning."

    def test_generate_summary_response_passes_question_as_user_message(self):
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_summary_response("Summarize file.pdf", "ctx")
            user_message = mock_create.call_args.kwargs["messages"][1]["content"]
            assert user_message == "Summarize file.pdf"

    def test_generate_summary_response_context_embedded_in_system_prompt(self):
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_summary_response("Q", "UNIQUE_SUMMARY_CTX")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "UNIQUE_SUMMARY_CTX" in system_content

    def test_generate_summary_response_uses_correct_model(self):
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_summary_response("Q", "ctx")
            assert mock_create.call_args.kwargs["model"] == "llama-3.3-70b-versatile"

    def test_generate_summary_response_uses_temperature_02(self):
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_summary_response("Q", "ctx")
            assert mock_create.call_args.kwargs["temperature"] == 0.2

    def test_generate_summary_response_message_roles_are_system_then_user(self):
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_summary_response("Q", "ctx")
            msgs = mock_create.call_args.kwargs["messages"]
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"

    def test_generate_summary_response_system_prompt_contains_summarizer_identity(self):
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_summary_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "Document Summarizer" in system_content

    def test_generate_summary_response_system_prompt_contains_only_summarize_rule(self):
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_summary_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "Only summarize the exact document" in system_content

    def test_generate_summary_response_system_prompt_contains_citation_at_end_rule(self):
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_summary_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "ONLY at the very end of the summary" in system_content

    def test_generate_summary_response_system_prompt_contains_no_mid_paragraph_citation_rule(self):
        mock_message = MagicMock()
        mock_message.content = "summary"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_summary_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "Do NOT add source/page citations mid-paragraph" in system_content

    def test_generate_summary_response_logs_usage(self):
        mock_api_response = MagicMock()
        mock_api_response.choices = [MagicMock()]
        mock_api_response.choices[0].message.content = "summary"

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            with patch("app.services.LLM_Service.logging") as mock_log:
                self.response_service.generate_summary_response("Q", "ctx")
                mock_log.info.assert_called_once_with("RESPONSE USAGE: %s", mock_api_response.usage)

    def test_generate_summary_response_none_content(self):
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_summary_response("Q", "ctx")
            assert result is None

    def test_generate_summary_response_empty_context(self):
        mock_message = MagicMock()
        mock_message.content = "No content to summarize."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_summary_response("Summarize", "")
            assert result == "No content to summarize."

    def test_generate_summary_response_api_failure(self):
        with patch.object(self.response_service.client.chat.completions, "create", side_effect=Exception("API failure")):
            with pytest.raises(Exception, match="API failure"):
                self.response_service.generate_summary_response("Q", "ctx")

    def test_generate_summary_response_empty_choices(self):
        mock_api_response = MagicMock()
        mock_api_response.choices = []

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            with pytest.raises(IndexError):
                self.response_service.generate_summary_response("Q", "ctx")

    def test_generate_inventory_response_returns_content(self):
        mock_message = MagicMock()
        mock_message.content = (
            "Here is a list of the documents I have with a one-sentence description each:\n\n"
            "1. report.pdf: Annual financial report for fiscal year 2024.\n\n"
            "2. policy.pdf: Company HR policy document covering leave and benefits."
        )
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_inventory_response("List all documents", "doc context")
            assert "Here is a list" in result
            assert "report.pdf" in result

    def test_generate_inventory_response_passes_question_as_user_message(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("What documents do you have?", "ctx")
            user_message = mock_create.call_args.kwargs["messages"][1]["content"]
            assert user_message == "What documents do you have?"

    def test_generate_inventory_response_context_embedded_in_system_prompt(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "UNIQUE_INVENTORY_CTX")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "UNIQUE_INVENTORY_CTX" in system_content

    def test_generate_inventory_response_uses_correct_model(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            assert mock_create.call_args.kwargs["model"] == "llama-3.3-70b-versatile"

    def test_generate_inventory_response_uses_temperature_02(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            assert mock_create.call_args.kwargs["temperature"] == 0.2

    def test_generate_inventory_response_message_roles_are_system_then_user(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            msgs = mock_create.call_args.kwargs["messages"]
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"

    def test_generate_inventory_response_system_prompt_contains_listing_identity(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "Document Listing assistant" in system_content

    def test_generate_inventory_response_system_prompt_contains_opening_line_rule(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "Here is a list of the documents I have" in system_content

    def test_generate_inventory_response_system_prompt_contains_no_markdown_rule(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "Do not use markdown" in system_content

    def test_generate_inventory_response_system_prompt_contains_plain_text_rule(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "Plain text only" in system_content

    def test_generate_inventory_response_system_prompt_contains_numbered_format_rule(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "<number>. <filename>" in system_content

    def test_generate_inventory_response_system_prompt_contains_blank_line_rule(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "blank line between each document" in system_content

    def test_generate_inventory_response_system_prompt_contains_no_bold_rule(self):
        mock_message = MagicMock()
        mock_message.content = "list"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response) as mock_create:
            self.response_service.generate_inventory_response("Q", "ctx")
            system_content = mock_create.call_args.kwargs["messages"][0]["content"]
            assert "bold text" in system_content

    def test_generate_inventory_response_logs_usage(self):
        mock_api_response = MagicMock()
        mock_api_response.choices = [MagicMock()]
        mock_api_response.choices[0].message.content = "list"

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            with patch("app.services.LLM_Service.logging") as mock_log:
                self.response_service.generate_inventory_response("Q", "ctx")
                mock_log.info.assert_called_once_with("RESPONSE USAGE: %s", mock_api_response.usage)

    def test_generate_inventory_response_none_content(self):
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_inventory_response("Q", "ctx")
            assert result is None

    def test_generate_inventory_response_empty_context(self):
        mock_message = MagicMock()
        mock_message.content = "Here is a list of the documents I have with a one-sentence description each:\n\n(No documents found)"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            result = self.response_service.generate_inventory_response("List all docs", "")
            assert "Here is a list" in result

    def test_generate_inventory_response_api_failure(self):
        with patch.object(self.response_service.client.chat.completions, "create", side_effect=Exception("API failure")):
            with pytest.raises(Exception, match="API failure"):
                self.response_service.generate_inventory_response("Q", "ctx")

    def test_generate_inventory_response_empty_choices(self):
        mock_api_response = MagicMock()
        mock_api_response.choices = []

        with patch.object(self.response_service.client.chat.completions, "create", return_value=mock_api_response):
            with pytest.raises(IndexError):
                self.response_service.generate_inventory_response("Q", "ctx")