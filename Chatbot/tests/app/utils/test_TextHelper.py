import pytest
from app.utils.Text_Helper import clean_text

class TestCleanTextHelper:

    def test_clean_text_removes_newlines(self):
        text = "Hello\nWorld"
        result = clean_text(text)
        assert result == "Hello World"

    def test_clean_text_removes_multiple_spaces(self):
        text = "Hello     World"
        result = clean_text(text)
        assert result == "Hello World"

    def test_clean_text_strips_leading_trailing_spaces(self):
        text = "   Hello World   "
        result = clean_text(text)
        assert result == "Hello World"

    def test_clean_text_mixed_spaces_and_newlines(self):
        text = "Hello\n\n   World   Test"
        result = clean_text(text)
        assert result == "Hello World Test"

    def test_clean_text_normal_text(self):
        text = "Hello World"
        result = clean_text(text)
        assert result == "Hello World"

    def test_clean_text_empty_string(self):
        text = ""
        result = clean_text(text)
        assert result == ""

    def test_clean_text_only_spaces(self):
        text = "      "
        result = clean_text(text)
        assert result == ""

    def test_clean_text_only_newlines(self):
        text = "\n\n\n"
        result = clean_text(text)
        assert result == ""

    def test_clean_text_special_characters(self):
        text = "Hello!!   @@World##"
        result = clean_text(text)
        assert result == "Hello!! @@World##"

    def test_clean_text_numeric_text(self):
        text = "123   456\n789"
        result = clean_text(text)
        assert result == "123 456 789"

    def test_clean_text_none_input(self):
        with pytest.raises(AttributeError):
            clean_text(None)