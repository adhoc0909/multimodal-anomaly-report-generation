"""Anthropic Claude client for MMAD evaluation."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import requests
from requests import RequestException

from .base import BaseLLMClient, INSTRUCTION, INSTRUCTION_WITH_AD, format_ad_info, get_mime_type


class ClaudeClient(BaseLLMClient):
    """Claude client following MMAD paper's evaluation protocol."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        api_url: str = "https://api.anthropic.com/v1/messages",
        max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.api_url = api_url
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY env var or pass api_key.")

    def send_request(self, payload: dict) -> Optional[dict]:
        """Send request to Anthropic API with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        retry_delay = 1
        retries = 0

        while retries < self.max_retries:
            try:
                before = time.time()
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
                response_json = response.json()

                # Check for error
                if "error" in response_json:
                    retries += 1
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                # Check for content
                content = response_json.get('content', [])
                if content:
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
        """Build Anthropic API payload following MMAD protocol."""

        # Build in-context description
        incontext = ""
        if few_shot_paths:
            incontext = f"The first {len(few_shot_paths)} image is the normal sample, which can be used as a template to compare."

        # Build content list
        content = []

        # Add few-shot images first
        for ref_path in few_shot_paths:
            try:
                ref_base64 = self.encode_image_to_base64(ref_path)
                media_type = get_mime_type(ref_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": ref_base64
                    }
                })
            except FileNotFoundError:
                continue

        # Add query image
        query_base64 = self.encode_image_to_base64(query_image_path)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": get_mime_type(query_image_path),
                "data": query_base64
            }
        })

        # Build conversation text
        conversation_text = ""
        for q in questions:
            conversation_text += f"{q['text']}\n"

        # Select instruction: custom > AD > default
        if instruction is None:
            if ad_info:
                instruction = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
            else:
                instruction = INSTRUCTION

        # Add text prompt
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
                "The last image is the query image. " +
                "Following is the question list: \n" +
                conversation_text
            )
        content.append({"type": "text", "text": prompt_text})

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }

        return payload

    def extract_response_text(self, response: dict) -> str:
        """Extract text from Claude response."""
        content = response.get('content', [])
        for block in content:
            if block.get('type') == 'text':
                return block.get('text', '')
        return ''
