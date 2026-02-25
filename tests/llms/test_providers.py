from unittest.mock import patch, MagicMock

import pytest

from zev.constants import OPENAI_DEFAULT_MODEL, GEMINI_DEFAULT_MODEL
from zev.llms.types import Command, OptionsResponse


class TestOpenAIProvider:
    @patch('zev.llms.openai.provider.OpenAI')
    @patch('zev.llms.openai.provider.config')
    def test_uses_configured_model(self, mock_config, mock_openai):
        mock_config.openai_api_key = "sk-test123"
        mock_config.openai_model = "gpt-4"

        from zev.llms.openai.provider import OpenAIProvider
        provider = OpenAIProvider()

        assert provider.model == "gpt-4"

    @patch('zev.llms.openai.provider.OpenAI')
    @patch('zev.llms.openai.provider.config')
    def test_uses_default_model_when_not_configured(self, mock_config, mock_openai):
        mock_config.openai_api_key = "sk-test"
        mock_config.openai_model = None

        from zev.llms.openai.provider import OpenAIProvider
        provider = OpenAIProvider()

        assert provider.model == OPENAI_DEFAULT_MODEL

    @patch('zev.llms.openai.provider.config')
    def test_raises_error_without_api_key(self, mock_config):
        mock_config.openai_api_key = None

        from zev.llms.openai.provider import OpenAIProvider
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY must be set"):
            OpenAIProvider()

    @patch('zev.llms.openai.provider.OpenAI')
    @patch('zev.llms.openai.provider.config')
    def test_get_options_returns_parsed_response(self, mock_config, mock_openai_class):
        mock_config.openai_api_key = "sk-test"
        mock_config.openai_model = "gpt-4"

        expected_response = OptionsResponse(
            commands=[Command(command="ls", short_explanation="List", is_dangerous=False)],
            is_valid=True
        )
        mock_api_response = MagicMock()
        mock_api_response.choices = [MagicMock()]
        mock_api_response.choices[0].message.parsed = expected_response

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = mock_api_response
        mock_openai_class.return_value = mock_client

        from zev.llms.openai.provider import OpenAIProvider
        provider = OpenAIProvider()
        result = provider.get_options("list files", "OS: Linux")

        assert result.is_valid is True
        assert result.commands[0].command == "ls"

    @patch('zev.llms.openai.provider.OpenAI')
    @patch('zev.llms.openai.provider.config')
    def test_get_options_returns_none_on_auth_error(self, mock_config, mock_openai_class, capsys):
        from openai import AuthenticationError
        
        mock_config.openai_api_key = "invalid-key"
        mock_config.openai_model = "gpt-4"

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={}
        )
        mock_openai_class.return_value = mock_client

        from zev.llms.openai.provider import OpenAIProvider
        provider = OpenAIProvider()
        result = provider.get_options("test", "context")

        assert result is None
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()


class TestOllamaProvider:
    @patch('zev.llms.ollama.provider.OpenAI')
    @patch('zev.llms.ollama.provider.config')
    def test_configures_client_with_ollama_settings(self, mock_config, mock_openai):
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_model = "llama2"

        from zev.llms.ollama.provider import OllamaProvider
        provider = OllamaProvider()

        assert provider.model == "llama2"
        mock_openai.assert_called_with(base_url="http://localhost:11434", api_key="ollama")

    @patch('zev.llms.ollama.provider.config')
    def test_raises_error_without_base_url(self, mock_config):
        mock_config.ollama_base_url = None
        mock_config.ollama_model = "llama2"

        from zev.llms.ollama.provider import OllamaProvider
        
        with pytest.raises(ValueError, match="OLLAMA_BASE_URL must be set"):
            OllamaProvider()

    @patch('zev.llms.ollama.provider.config')
    def test_raises_error_without_model(self, mock_config):
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_model = None

        from zev.llms.ollama.provider import OllamaProvider
        
        with pytest.raises(ValueError, match="OLLAMA_MODEL must be set"):
            OllamaProvider()


class TestGeminiProvider:
    @patch('zev.llms.gemini.provider.config')
    def test_uses_configured_model(self, mock_config):
        mock_config.gemini_api_key = "gemini-key"
        mock_config.gemini_model = "gemini-pro"

        from zev.llms.gemini.provider import GeminiProvider
        provider = GeminiProvider()

        assert provider.model == "gemini-pro"

    @patch('zev.llms.gemini.provider.config')
    def test_uses_default_model_when_not_configured(self, mock_config):
        mock_config.gemini_api_key = "test-key"
        mock_config.gemini_model = None

        from zev.llms.gemini.provider import GeminiProvider
        provider = GeminiProvider()

        assert provider.model == GEMINI_DEFAULT_MODEL

    @patch('zev.llms.gemini.provider.config')
    def test_raises_error_without_api_key(self, mock_config):
        mock_config.gemini_api_key = None

        from zev.llms.gemini.provider import GeminiProvider
        
        with pytest.raises(ValueError, match="GEMINI_API_KEY must be set"):
            GeminiProvider()

    @patch('zev.llms.gemini.provider.config')
    def test_api_url_includes_api_key(self, mock_config):
        mock_config.gemini_api_key = "my-secret-key"
        mock_config.gemini_model = "gemini-pro"

        from zev.llms.gemini.provider import GeminiProvider
        provider = GeminiProvider()

        assert "my-secret-key" in provider.api_url


class TestAzureOpenAIProvider:
    @patch('zev.llms.azure_openai.provider.AzureOpenAI')
    @patch('zev.llms.azure_openai.provider.config')
    def test_uses_deployment_as_model(self, mock_config, mock_azure):
        mock_config.azure_openai_account_name = "myaccount"
        mock_config.azure_openai_api_key = "azure-key"
        mock_config.azure_openai_deployment = "my-deployment"
        mock_config.azure_openai_api_version = "2024-02-01"

        from zev.llms.azure_openai.provider import AzureOpenAIProvider
        provider = AzureOpenAIProvider()

        assert provider.model == "my-deployment"

    @patch('zev.llms.azure_openai.provider.config')
    def test_raises_error_without_account_name(self, mock_config):
        mock_config.azure_openai_account_name = None
        mock_config.azure_openai_deployment = "deployment"
        mock_config.azure_openai_api_version = "2024-02-01"

        from zev.llms.azure_openai.provider import AzureOpenAIProvider
        
        with pytest.raises(ValueError, match="AZURE_OPENAI_ACCOUNT_NAME must be set"):
            AzureOpenAIProvider()

    @patch('zev.llms.azure_openai.provider.config')
    def test_raises_error_without_deployment(self, mock_config):
        mock_config.azure_openai_account_name = "account"
        mock_config.azure_openai_deployment = None
        mock_config.azure_openai_api_version = "2024-02-01"

        from zev.llms.azure_openai.provider import AzureOpenAIProvider
        
        with pytest.raises(ValueError, match="AZURE_OPENAI_DEPLOYMENT must be set"):
            AzureOpenAIProvider()

    @patch('zev.llms.azure_openai.provider.config')
    def test_raises_error_without_api_version(self, mock_config):
        mock_config.azure_openai_account_name = "account"
        mock_config.azure_openai_deployment = "deployment"
        mock_config.azure_openai_api_version = None

        from zev.llms.azure_openai.provider import AzureOpenAIProvider
        
        with pytest.raises(ValueError, match="AZURE_OPENAI_API_VERSION must be set"):
            AzureOpenAIProvider()
