from typing import Optional

from google import genai
from google.genai import types

from zev.config import config
from zev.constants import GEMINI_DEFAULT_MODEL, PROMPT
from zev.llms.inference_provider_base import InferenceProvider
from zev.llms.types import OptionsResponse


class GeminiProvider(InferenceProvider):
    AUTH_ERROR_MESSAGE = (
        "Error: There was an error with your Gemini API key. You can change it by running `zev --setup`."
    )

    def __init__(self):
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set. Try running `zev --setup`.")

        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.gemini_model or GEMINI_DEFAULT_MODEL

    def get_options(self, prompt: str, context: str) -> Optional[OptionsResponse]:
        try:
            assembled_prompt = PROMPT.format(prompt=prompt, context=context)
            response = self.client.models.generate_content(
                model=self.model,
                contents=assembled_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=OptionsResponse,
                ),
            )
            return OptionsResponse.model_validate_json(response.text)
        except genai.errors.ClientError as e:
            if "API_KEY_INVALID" in str(e) or "401" in str(e):
                print(self.AUTH_ERROR_MESSAGE)
            else:
                print(f"Error: {e}")
                print("Note that to update settings, you can run `zev --setup`.")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
