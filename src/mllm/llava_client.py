"""LLaVA client for MMAD evaluation - using llava package or transformers."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import torch

from .base import BaseLLMClient, INSTRUCTION, INSTRUCTION_WITH_AD, format_ad_info
from src.utils.device import get_device

logger = logging.getLogger(__name__)


class LLaVAClient(BaseLLMClient):
    """LLaVA client for MMAD evaluation.

    Supported models (HuggingFace):
    - llava-hf/llava-1.5-7b-hf
    - llava-hf/llava-1.5-13b-hf
    - llava-hf/llava-v1.6-mistral-7b-hf
    - llava-hf/llava-v1.6-vicuna-7b-hf
    - llava-hf/llava-v1.6-vicuna-13b-hf
    - llava-hf/llava-onevision-qwen2-7b-ov-hf (LLaVA-OneVision)

    For original LLaVA models, install: pip install llava
    """

    def __init__(
        self,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
        device: str = None,
        torch_dtype: str = "float16",
        max_new_tokens: int = 128,
        use_hf: bool = True,  # Use HuggingFace transformers (recommended)
        conv_mode: str = "llava_v1",  # For original llava package
        temperature: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = device or str(get_device(verbose=False))
        self.torch_dtype_str = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.use_hf = use_hf
        self.conv_mode = conv_mode
        self.temperature = temperature

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._image_processor = None
        self._context_len = None

    def _get_torch_dtype(self):
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype_str, torch.float16)

    def _load_model_hf(self):
        """Load model using HuggingFace transformers (recommended)."""
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        # Loading LLaVA model (HuggingFace)

        torch_dtype = self._get_torch_dtype()

        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)

        # Model loaded

    def _load_model_llava(self):
        """Load model using original llava package."""
        try:
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            from llava.mm_utils import get_model_name_from_path
        except ImportError:
            raise ImportError("Please install llava package: pip install llava")

        # Loading LLaVA model (llava package)

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)

        self._tokenizer, self._model, self._image_processor, self._context_len = \
            load_pretrained_model(self.model_path, None, model_name)

        # Model loaded

    def _load_model(self):
        """Lazy load model."""
        if self._model is not None:
            return

        if self.use_hf:
            self._load_model_hf()
        else:
            self._load_model_llava()

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
        """Build LLaVA message format."""
        return {
            "query_image": query_image_path,
            "few_shot_paths": few_shot_paths,
            "questions": questions,
            "ad_info": ad_info,
            "instruction": instruction,
            "report_mode": report_mode,
        }

    def _generate_hf(self, payload: dict) -> str:
        """Generate response using HuggingFace transformers."""
        from PIL import Image

        # Select instruction based on AD info availability
        ad_info = payload.get("ad_info")
        instruction = payload.get("instruction")
        report_mode = bool(payload.get("report_mode", False))
        if instruction is None:
            if ad_info:
                instruction = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
            else:
                instruction = INSTRUCTION

        # Build prompt
        prompt = f"USER: {instruction}\n"

        if payload["few_shot_paths"]:
            prompt += f"Following is {len(payload['few_shot_paths'])} image of normal sample:\n"
            for _ in payload["few_shot_paths"]:
                prompt += "<image>\n"

        prompt += "Test image:\n<image>\n"
        if report_mode:
            for q in payload["questions"]:
                prompt += q["text"]
            prompt += "\nJSON only.\nASSISTANT:"
        else:
            prompt += "Following is new question list:\n"
            for q in payload["questions"]:
                prompt += q["text"]
            prompt += "\nAnswer with the option's letter from the given choices directly.\nASSISTANT:"

        # Load images
        images = []
        for ref_path in payload["few_shot_paths"]:
            try:
                images.append(Image.open(ref_path).convert("RGB"))
            except Exception as e:
                continue

        images.append(Image.open(payload["query_image"]).convert("RGB"))

        # Process
        inputs = self._processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )

        response = self._processor.decode(output[0], skip_special_tokens=True)

        # Extract assistant response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        return response

    def _generate_llava(self, payload: dict) -> str:
        """Generate response using original llava package."""
        try:
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from llava.conversation import conv_templates
            from llava.mm_utils import tokenizer_image_token, load_image_from_base64
        except ImportError:
            raise ImportError("Please install llava package")

        # Select instruction based on AD info availability
        ad_info = payload.get("ad_info")
        hint = payload.get("instruction")
        report_mode = bool(payload.get("report_mode", False))
        if hint is None:
            if ad_info:
                hint = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
            else:
                hint = INSTRUCTION

        question = ""
        if payload["few_shot_paths"]:
            question = f"Following is {len(payload['few_shot_paths'])} image of normal sample:"
            for _ in payload["few_shot_paths"]:
                question += f"{DEFAULT_IMAGE_TOKEN}\n"

        question += f"Test image:\n{DEFAULT_IMAGE_TOKEN}\n"
        if report_mode:
            for q in payload["questions"]:
                question += q["text"]
            qs = hint + '\n' + question + "\nJSON only."
        else:
            question += "Following is new question list:\n"
            for q in payload["questions"]:
                question += q["text"]
            qs = hint + '\n' + question + "\nAnswer with the option's letter from the given choices directly."

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        # Process images
        ref_images = []
        for ref_path in payload["few_shot_paths"]:
            ref_image = cv2.imread(ref_path)
            ref_base64 = self.encode_image_to_base64(ref_image)
            ref_images.append(load_image_from_base64(ref_base64))

        query_image = cv2.imread(payload["query_image"])
        query_base64 = self.encode_image_to_base64(query_image)
        query_pil = load_image_from_base64(query_base64)

        images = ref_images + [query_pil]

        if hasattr(self._image_processor, 'forward'):
            from llava.mm_utils import process_images
            image_tensor = process_images(images, self._image_processor, self._context_len)
        else:
            image_tensor = self._image_processor.preprocess(images, return_tensors="pt")["pixel_values"]

        image_tensor = [img.unsqueeze(0).half().to(self.device) for img in image_tensor]
        image_sizes = [img.size for img in images]

        # Generate
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                max_new_tokens=self.max_new_tokens,
                use_cache=True
            )

        response = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response

    def send_request(self, payload: dict) -> Optional[dict]:
        """Process request using local model."""
        self._load_model()

        if self.use_hf:
            response = self._generate_hf(payload)
        else:
            response = self._generate_llava(payload)

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
        """Generate answers one question at a time."""
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        predicted_answers = []

        for i in range(len(questions)):
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
                predicted_answers.append('')

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
