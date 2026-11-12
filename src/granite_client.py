"""
IBM Granite LLM Client
Wraps IBM watsonx.ai SDK for text generation using IBM Granite models.
Uses the FREE Lite plan (300,000 tokens/month).
"""

import os
import logging
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()
logger = logging.getLogger(__name__)


class GraniteClient:
    """IBM Granite model client via watsonx.ai."""

    def __init__(self):
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.model_id = os.getenv("GRANITE_MODEL_ID", "ibm/granite-4-h-small")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
        self.min_new_tokens = int(os.getenv("MIN_NEW_TOKENS", "50"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))

        if not self.api_key or self.api_key == "your_ibm_cloud_api_key_here":
            raise ValueError(
                "WATSONX_API_KEY is not set. Please update your .env file.\n"
                "Get your API key at: https://cloud.ibm.com/iam/apikeys"
            )
        if not self.project_id or self.project_id == "your_watsonx_project_id_here":
            raise ValueError(
                "WATSONX_PROJECT_ID is not set. Please update your .env file.\n"
                "Get your Project ID from: https://us-south.ml.cloud.ibm.com"
            )

        credentials = Credentials(url=self.url, api_key=self.api_key)
        self.model = ModelInference(
            model_id=self.model_id,
            credentials=credentials,
            project_id=self.project_id,
            params={
                GenParams.MAX_NEW_TOKENS: self.max_new_tokens,
                GenParams.MIN_NEW_TOKENS: self.min_new_tokens,
                GenParams.TEMPERATURE: self.temperature,
                GenParams.STOP_SEQUENCES: ["Human:", "Student:"],
            },
        )
        logger.info(f"Granite client initialized with model: {self.model_id}")

    def generate(self, prompt: str) -> str:
        """Send a prompt to IBM Granite and return the generated text."""
        try:
            response = self.model.generate_text(prompt=prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Granite generation error: {e}")
            raise
