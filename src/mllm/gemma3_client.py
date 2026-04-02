"""Gemma3 client for MMAD evaluation - HuggingFace transformers."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from .base import BaseLLMClient, INSTRUCTION, INSTRUCTION_WITH_AD, format_ad_info
from src.utils.device import get_device

logger = logging.getLogger(__name__)


class Gemma3Client(BaseLLMClient):
    """Gemma3 multimodal client using HuggingFace transformers.

    Supported models:
    - google/gemma-3-4b-it   (~8GB VRAM)
    - google/gemma-3-12b-it  (~24GB VRAM)
    - google/gemma-3-27b-it  (~55GB VRAM)

    Quantization (requires bitsandbytes, CUDA only):
    - quantization="int4" → BitsAndBytes 4-bit (NF4)
    - quantization="int8" → BitsAndBytes 8-bit
    """

    def __init__(
        self,
        model_path: str = "google/gemma-3-4b-it",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        quantization: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = device
        self.torch_dtype_str = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization  # None | "int4" | "int8"

        self._model = None
        self._processor = None

    def get_torch_dtype(self):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype_str, torch.bfloat16)

    def load_model(self):
        """Lazy load model and processor."""
        if self._model is not None:
            return

        import transformers
        transformers.logging.set_verbosity_error()

        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        get_device()  # 사용 중인 디바이스 출력

        load_kwargs: dict = {
            "device_map": self.device,
            "torch_dtype": "auto",
        }

        # torchao 구버전 호환: 설치된 버전이 모르는 인자를 자동으로 필터링
        try:
            import inspect
            from torchao.quantization import Int4WeightOnlyConfig
            _orig_init = Int4WeightOnlyConfig.__init__
            _valid = set(inspect.signature(_orig_init).parameters) - {"self"}
            def _patched_init(self, *args, **kwargs):
                _orig_init(self, *args, **{k: v for k, v in kwargs.items() if k in _valid})
            Int4WeightOnlyConfig.__init__ = _patched_init
        except Exception:
            pass

        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path, **load_kwargs
        ).eval()

        self._processor = AutoProcessor.from_pretrained(self.model_path)

    def load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def build_payload(
        self,
        query_image_path: str,
        few_shot_paths: List[str],
        questions: List[Dict[str, str]],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
        report_mode: bool = False,
    ) -> dict:
        """Build Gemma3 message format (PIL images inline)."""
        if instruction is None:
            if ad_info:
                instruction = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
            else:
                instruction = INSTRUCTION

        content = []
        content.append({"type": "text", "text": instruction})

        if few_shot_paths:
            content.append({
                "type": "text",
                "text": f"Following is/are {len(few_shot_paths)} image of normal sample, "
                        "which can be used as a template to compare the image being queried.",
            })
            for ref_path in few_shot_paths:
                content.append({"type": "image", "image": self.load_image(ref_path)})

        if report_mode:
            content.append({"type": "text", "text": "Following is the query image for inspection report generation:"})
        else:
            content.append({"type": "text", "text": "Following is the query image:"})
        content.append({"type": "image", "image": self.load_image(query_image_path)})

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
        """Process request using local Gemma3 model."""
        self.load_model()

        messages = [{"role": "user", "content": payload["content"]}]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed = generated_ids[:, input_len:]
        response = self._processor.decode(trimmed[0], skip_special_tokens=True)

        return {"response": response}

    def extract_response_text(self, response: dict) -> str:
        return response.get("response", "")

    def generate_answers(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """질문 하나씩 순차 호출."""
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        predicted_answers = []

        for i in range(len(questions)):
            payload = self.build_payload(
                query_image_path,
                few_shot_paths,
                questions[i:i + 1],
                ad_info=ad_info,
                instruction=instruction,
            )
            response = self.send_request(payload)
            if response is None:
                predicted_answers.append('')
                continue

            response_text = self.extract_response_text(response)
            conv = meta.get("conversation", [])
            options = conv[i].get("Options", {}) if i < len(conv) else None
            parsed = self.parse_answer(response_text, options)
            predicted_answers.append(parsed[-1] if parsed else '')

        return questions, answers, predicted_answers, question_types

    def generate_answers_batch(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """모든 질문을 한 번의 모델 호출로 처리 (배치 모드)."""
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        payload = self.build_payload(
            query_image_path,
            few_shot_paths,
            questions,
            ad_info=ad_info,
            instruction=instruction,
        )
        response = self.send_request(payload)
        if response is None:
            return questions, answers, None, question_types

        parsed = self.parse_answer(self.extract_response_text(response))
        while len(parsed) < len(questions):
            parsed.append('')

        return questions, answers, parsed[:len(questions)], question_types
