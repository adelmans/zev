from unittest.mock import patch, MagicMock

import pytest

from zev.constants import LLMProviders


class TestGetInferenceProvider:
    @patch('zev.llms.openai.provider.OpenAI')
    @patch('zev.llms.openai.provider.config')
    @patch('zev.llms.llm.config')
    def test_returns_openai_provider_for_openai(self, mock_llm_config, mock_provider_config, mock_openai):
        mock_llm_config.llm_provider = LLMProviders.OPENAI
        mock_provider_config.openai_api_key = "test-key"
        mock_provider_config.openai_model = None

        from zev.llms.llm import get_inference_provider
        from zev.llms.openai.provider import OpenAIProvider
        
        provider = get_inference_provider()
        
        assert isinstance(provider, OpenAIProvider)

    @patch('zev.llms.ollama.provider.OpenAI')
    @patch('zev.llms.ollama.provider.config')
    @patch('zev.llms.llm.config')
    def test_returns_ollama_provider_for_ollama(self, mock_llm_config, mock_provider_config, mock_openai):
        mock_llm_config.llm_provider = LLMProviders.OLLAMA
        mock_provider_config.ollama_base_url = "http://localhost:11434"
        mock_provider_config.ollama_model = "llama2"

        from zev.llms.llm import get_inference_provider
        from zev.llms.ollama.provider import OllamaProvider
        
        provider = get_inference_provider()
        
        assert isinstance(provider, OllamaProvider)

    @patch('zev.llms.gemini.provider.config')
    @patch('zev.llms.llm.config')
    def test_returns_gemini_provider_for_gemini(self, mock_llm_config, mock_gemini_config):
        mock_llm_config.llm_provider = LLMProviders.GEMINI
        mock_gemini_config.gemini_api_key = "test-key"
        mock_gemini_config.gemini_model = None

        from zev.llms.llm import get_inference_provider
        from zev.llms.gemini.provider import GeminiProvider
        
        provider = get_inference_provider()
        
        assert isinstance(provider, GeminiProvider)

    @patch('zev.llms.llm.config')
    def test_raises_error_for_invalid_provider(self, mock_config):
        mock_config.llm_provider = "unknown_provider"

        from zev.llms.llm import get_inference_provider
        
        with pytest.raises(ValueError, match="Invalid LLM provider"):
            get_inference_provider()
