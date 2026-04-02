"""Qwen2.5-VL client for MMAD evaluation - HuggingFace transformers."""
from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import torch

from .base import BaseLLMClient, INSTRUCTION, INSTRUCTION_WITH_AD, format_ad_info

logger = logging.getLogger(__name__)


class QwenVLClient(BaseLLMClient):
    """Qwen2.5-VL client using HuggingFace transformers.

    Supported models:
    - Qwen/Qwen2.5-VL-7B-Instruct
    - Qwen/Qwen2.5-VL-2B-Instruct
    - Qwen/Qwen2-VL-7B-Instruct
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 128,
        min_pixels: int = 64 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        use_flash_attention: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self._model = None
        self._processor = None
        self._use_flash_attention = use_flash_attention

    def _load_model(self):
        """Lazy load model and processor."""
        if self._model is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        # Loading message (will only show if logging level is DEBUG)

        attn_impl = "flash_attention_2" if self._use_flash_attention else "eager"

        try:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                attn_implementation=attn_impl,
                device_map=self.device,
            )
        except Exception as e:
            pass  # Fall back to eager attention
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                attn_implementation="eager",
                device_map=self.device,
            )

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Model loaded

    def load_model(self):
        """Public interface for model warm-up before timed evaluation."""
        self._load_model()

    def build_payload(
        self,
        query_image_path: str,
        few_shot_paths: List[str],
        questions: List[Dict[str, str]],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
        report_mode: bool = False,
    ) -> dict:
        """Build Qwen VL message format."""
        content = []

        # Select instruction: custom > AD > default
        if instruction is None:
            if ad_info:
                instruction = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
            else:
                instruction = INSTRUCTION

        # Instruction
        content.append({"type": "text", "text": instruction})
        if not report_mode:
            content.append({"type": "text", "text": "Answer with the option's letter from the given choices directly!"})

        # Few-shot templates
        if few_shot_paths:
            content.append({
                "type": "text",
                "text": f"Following is/are {len(few_shot_paths)} image of normal sample, which can be used as a template to compare the image being queried."
            })
            for ref_path in few_shot_paths:
                content.append({"type": "image", "image": ref_path})

        # Query image
        if report_mode:
            content.append({"type": "text", "text": "Following is the query image for inspection report generation:"})
        else:
            content.append({"type": "text", "text": "Following is the query image:"})
        content.append({"type": "image", "image": query_image_path})

        # Questions / report prompt text
        if report_mode:
            for q in questions:
                if q.get("text"):
                    content.append({"type": "text", "text": q["text"]})
        else:
            content.append({"type": "text", "text": "Following is the question list:"})
            for q in questions:
                content.append({"type": "text", "text": q["text"]})

        return {"content": content}

    def send_request(self, payload: dict) -> Optional[dict]:
        """Process request using local model."""
        self._load_model()

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils")

        content = payload["content"]
        messages = [{"role": "user", "content": content}]

        # Prepare for inference
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        return {"response": response}

    def extract_response_text(self, response: dict) -> str:
        """Extract text from response."""
        return response.get("response", "")

    def generate_answers(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """Generate answers one question at a time (Qwen's approach)."""
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        predicted_answers = []

        for i in range(len(questions)):
            # Qwen asks one question at a time
            part_questions = questions[i:i + 1]
            payload = self.build_payload(
                query_image_path,
                few_shot_paths,
                part_questions,
                ad_info=ad_info,
                instruction=instruction,
            )

            response = self.send_request(payload)
            if response is None:
                predicted_answers.append('')
                continue

            response_text = self.extract_response_text(response)

            # Get options for fuzzy matching
            conv = meta.get("conversation", [])
            options = conv[i].get("Options", {}) if i < len(conv) else None

            parsed = self.parse_answer(response_text, options)
            if parsed:
                predicted_answers.append(parsed[-1])
            else:
                predicted_answers.append(response_text[:1] if response_text else '')

        return questions, answers, predicted_answers, question_types

    def generate_answers_batch(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """Generate answers for ALL questions in a single model call (5-8x faster)."""
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        # Build payload with ALL questions at once
        payload = self.build_payload(
            query_image_path,
            few_shot_paths,
            questions,
            ad_info=ad_info,
            instruction=instruction,
        )

        # Single model call for all questions
        response = self.send_request(payload)
        if response is None:
            return questions, answers, None, question_types

        response_text = self.extract_response_text(response)

        # Parse all answers from response
        parsed = self.parse_answer(response_text)

        # Pad with empty strings if not enough answers
        while len(parsed) < len(questions):
            parsed.append('')

        return questions, answers, parsed[:len(questions)], question_types
