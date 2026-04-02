"""OpenAI GPT-4o client for MMAD evaluation - matches paper's implementation."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests import RequestException

from .base import BaseLLMClient, INSTRUCTION, INSTRUCTION_WITH_AD, format_ad_info, get_mime_type


class GPT4Client(BaseLLMClient):
    """GPT-4o/GPT-4V client following MMAD paper's exact implementation."""

    ERROR_KEYWORDS = ['please', 'sorry', 'today', 'cannot assist']

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        api_url: str = "https://api.openai.com/v1/chat/completions",
        max_tokens: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.api_url = api_url
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key.")

    def send_request(self, payload: dict) -> Optional[dict]:
        """Send request to OpenAI API with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        retry_delay = 1
        retries = 0

        while retries < self.max_retries:
            try:
                before = time.time()
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response_json = response.json()

                choices = response_json.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '').lower()
                    if any(word in content for word in self.ERROR_KEYWORDS):
                        retries += 1
                        continue

                    self.api_time_cost += time.time() - before
                    return response_json
                else:
                    retries += 1

            except RequestException as e:
                time.sleep(retry_delay)
                retry_delay *= 2
                retries += 1
        return None

    def build_payload(
        self,
        query_image_path: str,
        few_shot_paths: List[str],
        questions: List[Dict[str, str]],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
        report_mode: bool = False,
    ) -> dict:
        """Build OpenAI API payload following paper's format."""

        # Build in-context description
        incontext = ""
        if few_shot_paths:
            incontext = f"The first {len(few_shot_paths)} image is the normal sample, which can be used as a template to compare."

        # Encode few-shot images
        incontext_images = []
        for ref_path in few_shot_paths:
            try:
                ref_base64 = self.encode_image_to_base64(ref_path)
                incontext_images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{get_mime_type(ref_path)};base64,{ref_base64}",
                        "detail": "low"
                    }
                })
            except FileNotFoundError:
                continue

        # Build conversation text
        conversation_text = ""
        for q in questions:
            conversation_text += f"{q['text']}\n"

        # Encode query image
        query_base64 = self.encode_image_to_base64(query_image_path)

        # Select instruction: custom > AD > default
        if instruction is None:
            if ad_info:
                instruction = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
            else:
                instruction = INSTRUCTION

        if report_mode:
            prompt_text = instruction + "\n"
            if incontext:
                prompt_text += incontext + "\n"
            prompt_text += "The last image is the query image.\n"
            if conversation_text.strip():
                prompt_text += conversation_text
        else:
            prompt_text = (
                instruction +
                incontext +
                "The last image is the query image" +
                "Following is the question list: \n" +
                conversation_text
            )

        # Build payload (matches paper's gpt4o.py exactly)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": incontext_images + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{get_mime_type(query_image_path)};base64,{query_base64}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
        }

        return payload

    def extract_response_text(self, response: dict) -> str:
        """Extract text from OpenAI response."""
        choices = response.get('choices', [])
        if choices:
            return choices[0].get('message', {}).get('content', '')
        return ''
