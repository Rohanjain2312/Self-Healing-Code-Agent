"""
HuggingFace Transformers provider — GPU inference for Colab and HF Spaces.

Models are lazy-loaded on first inference call to avoid OOM at import time.
Supports 4-bit quantization via bitsandbytes when available.

Default model: meta-llama/Llama-3.2-3B-Instruct (3B) for HF Spaces free tier.
Override with HF_MODEL environment variable for Colab A100 sessions.

The pipeline runs in a thread executor to avoid blocking the async event loop.
"""

import asyncio
import os
from typing import Any
from ..base import BaseLLMProvider, InferenceRequest, InferenceResponse

# HF Spaces free tier has ~16GB RAM; 3B model fits without quantization
_DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# Colab A100 can handle larger models
_COLAB_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class HuggingFaceProvider(BaseLLMProvider):
    """
    Lazy-loading HuggingFace pipeline provider.

    transformers and torch are imported only on first infer() call,
    so the class can be imported without those packages installed.
    """

    def __init__(
        self,
        model: str | None = None,
        use_4bit: bool | None = None,
        device_map: str = "auto",
    ) -> None:
        self._model_id = model or os.environ.get("HF_MODEL", _DEFAULT_MODEL)

        # If running on Colab with a real GPU and no explicit HF_MODEL override,
        # automatically upgrade to the larger 8B model.
        if (
            not model
            and not os.environ.get("HF_MODEL")
            and (os.environ.get("COLAB_GPU") or os.environ.get("COLAB_RELEASE_TAG"))
        ):
            self._model_id = _COLAB_MODEL

        # Use 4-bit if explicitly requested or if running on Colab (detected via env)
        self._use_4bit = use_4bit if use_4bit is not None else bool(
            os.environ.get("COLAB_GPU") or os.environ.get("USE_4BIT")
        )
        self._device_map = device_map
        self._pipeline: Any = None
        self._lock = asyncio.Lock()

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self._model_id

    async def _ensure_loaded(self) -> None:
        """Load model exactly once, thread-safe."""
        if self._pipeline is not None:
            return
        async with self._lock:
            if self._pipeline is not None:
                return
            self._pipeline = await asyncio.get_event_loop().run_in_executor(
                None, self._load_pipeline
            )

    def _load_pipeline(self) -> Any:
        """Synchronous model load — runs in thread executor."""
        # Import here so package is optional at import time
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for HuggingFaceProvider. "
                "Install with: pip install transformers torch accelerate"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "device_map": self._device_map,
            "dtype": torch.float16,
        }

        if self._use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                # bitsandbytes not available — fall through to float16
                pass

        return pipeline("text-generation", **kwargs)

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        await self._ensure_loaded()

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._pipeline(
                messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                return_full_text=False,
            ),
        )

        generated = result[0]["generated_text"]
        # Chat pipeline may return a list of messages
        if isinstance(generated, list):
            generated = generated[-1].get("content", "")

        return InferenceResponse(
            text=generated,
            provider=self.provider_name,
            model=self._model_id,
        )
