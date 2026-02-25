import tempfile
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values


class TestConfig:
    def test_reads_all_provider_settings(self):
        config_content = """
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-test123
OPENAI_MODEL=gpt-4
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
GEMINI_API_KEY=gemini-key-123
GEMINI_MODEL=gemini-pro
AZURE_OPENAI_ACCOUNT_NAME=myaccount
AZURE_OPENAI_API_KEY=azure-key
AZURE_OPENAI_DEPLOYMENT=my-deployment
AZURE_OPENAI_API_VERSION=2024-02-01
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.zevrc', delete=False) as f:
            f.write(config_content)
            f.flush()
            config_path = Path(f.name)

        try:
            with patch('zev.config.Config.__init__', lambda self: None):
                from zev.config import Config
                config = Config()
                config.config_path = config_path
                config.vals = dotenv_values(config_path)

                assert config.llm_provider == "openai"
                assert config.openai_api_key == "sk-test123"
                assert config.openai_model == "gpt-4"
                assert config.ollama_base_url == "http://localhost:11434"
                assert config.ollama_model == "llama2"
                assert config.gemini_api_key == "gemini-key-123"
                assert config.gemini_model == "gemini-pro"
                assert config.azure_openai_account_name == "myaccount"
                assert config.azure_openai_api_key == "azure-key"
                assert config.azure_openai_deployment == "my-deployment"
                assert config.azure_openai_api_version == "2024-02-01"
        finally:
            config_path.unlink()

    def test_returns_none_for_missing_keys(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.zevrc', delete=False) as f:
            f.write("")
            f.flush()
            config_path = Path(f.name)

        try:
            with patch('zev.config.Config.__init__', lambda self: None):
                from zev.config import Config
                config = Config()
                config.config_path = config_path
                config.vals = dotenv_values(config_path)

                assert config.openai_api_key is None
                assert config.llm_provider is None
                assert config.gemini_api_key is None
        finally:
            config_path.unlink()
