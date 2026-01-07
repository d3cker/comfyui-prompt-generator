import base64
import json
import re
import os
from io import BytesIO
from typing import Dict, Tuple

import gc
import numpy as np
import torch
from PIL import Image

import requests

try:
    from transformers import AutoProcessor
    # Available in recent Transformers versions that include Qwen2.5 / Qwen3
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
    Qwen3VLForConditionalGeneration = None


def _tensor_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    ComfyUI IMAGE: torch.Tensor [B,H,W,C], float in [0..1], C=3
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("IMAGE input must be a torch.Tensor")

    if image_tensor.ndim != 4 or image_tensor.shape[-1] != 3:
        raise ValueError(f"Expected IMAGE with shape [B,H,W,3], got {tuple(image_tensor.shape)}")

    img0 = image_tensor[0].detach().cpu().float().clamp(0, 1)
    arr = (img0.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _clean_single_line(text: str) -> str:
    text = text.strip()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Remove wrapping quotes if the model returned them
    if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
        text = text[1:-1].strip()
    return text


def _available_devices() -> Tuple[str, ...]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            # show name in UI, but keep a parseable prefix
            devices.append(f"cuda:{i} | {name}")
    return tuple(devices)


def _parse_device_choice(device_choice: str) -> str:
    # "cuda:0 | NVIDIA ..." -> "cuda:0"
    return device_choice.split("|", 1)[0].strip()


def _cuda_cleanup(device_index: int | None):
    gc.collect()
    if torch.cuda.is_available():
        if device_index is None:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        else:
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def _resolve_model_family(model_id: str) -> str:
    model_lower = model_id.lower()
    if "qwen2.5-vl" in model_lower or "qwen2.5_vl" in model_lower:
        return "qwen2_5_vl"
    elif "qwen3-vl" in model_lower or "qwen3_vl" in model_lower:
        return "qwen3_vl"
    else:
        raise ValueError(f"Unknown model ID: {model_id}")


class QwenPromptFromImage:
    """
    ComfyUI Node: IMAGE -> STRING (prompt_text) | STRING (thinking_text)
    - Local backend: Qwen2.5-VL/Qwen3-VL via Transformers
    - Quantization: INT8 / INT4 (bitsandbytes)
    - Device selection: CPU or specific CUDA device
    - Remote backend: OpenAI-compatible chat/completions endpoint.
    """

    # Cache models per (model_id, quant_mode, device_str, dtype_str)
    _MODEL_CACHE: Dict[Tuple[str, str, str, str], object] = {}
    _PROC_CACHE: Dict[str, object] = {}

    @classmethod
    def INPUT_TYPES(cls):
        default_system = (
            "You convert the given image into a rich, image-generation prompt.\n"
            "Return ONLY the final prompt as a single line, no quotes, no extra text.\n"
            "Include: subject, environment, style, lighting, camera/lens, composition, key details.\n"
            "Avoid meta-commentary."
        )

        model_list = (
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-2B-Thinking",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-8B-Thinking",
            "Qwen/Qwen3-VL-235B-A22B-Instruct",
            "Qwen/Qwen3-VL-235B-A22B-Thinking",
        )

        return {
            "required": {
                "image": ("IMAGE",),

                "backend": (("local", "openai_compatible"),),

                "qwen_model": (model_list,),

                "device": (_available_devices(),),

                "quantization": ((
                    "int8_bnb",
                    "int4_bnb",
                    "bf16",
                    "fp16",
                    "fp32_cpu",
                ),),

                "flash_attention": (("off", "on"),), # Leave strings here, for future options

                "system_prompt": ("STRING", {"multiline": True, "default": default_system}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),

                "max_new_tokens": ("INT", {"default": 2048, "min": 16, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "safe_quants": ([True, False], {"default": True}),
                "unload_model": ([True, False], {"default": True}),

            },
            "optional": {
                # OpenAI-compatible endpoint options (used only when backend=openai_compatible)
                "openai_base_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "openai_api_key": ("STRING", {"default": ""}),
                "openai_model_override": ("STRING", {"default": ""}),
                # OpenRouter provider routing (comma-separated list, e.g. "Parasail" or "Alibaba, Parasail")
                "openrouter_providers": ("STRING", {"default": ""}),
                # Verbose output - print API response details to console
                "verbose_output": ([True, False], {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("prompt_text", "thinking_text")
    
    FUNCTION = "run"
    CATEGORY = "d3cker/Prompt-Generator"

    def _get_processor(self, model_id: str):
        if AutoProcessor is None:
            raise RuntimeError(
                "transformers is missing or too old. Install/upgrade transformers to a version that supports Qwen2.5-VL/Qwen3-VL"
            )

        if self._PROC_CACHE.get(model_id) is None:
            self._PROC_CACHE[model_id] = AutoProcessor.from_pretrained(model_id)
        return self._PROC_CACHE[model_id]

    def _load_local_model(self, model_id: str, device_choice: str, quantization: str, flash_attention: str, safe_quants: bool):
        
        model_family = _resolve_model_family(model_id)
        
        if Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError(
                "transformers is missing or too old. Install/upgrade transformers to a version that supports Qwen2.5-VL / Qwen3-VL"
            )
        
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError(
                "transformers is missing or too old. Install/upgrade transformers to a version that supports Qwen2.5-VL / Qwen3-VL"
            )
        
        device = _parse_device_choice(device_choice)

        if safe_quants:
            if device == "cpu" and quantization != "fp32_cpu":
                quantization = "fp32_cpu"
            elif device.lower().startswith("cuda"):
                if model_family == "qwen2_5_vl" and (quantization == "fp16" or quantization == "fp32_cpu"):
                    quantization = "bf16"
                elif model_family == "qwen3_vl" and quantization == "fp32_cpu":
                    quantization = "bf16"

        # dtype selection with some safety included
        if quantization == "fp32_cpu":
            dtype = torch.float32
        elif quantization == "bf16":
            dtype = torch.bfloat16
        elif quantization == "fp16":
            dtype = torch.float16
        else:
            # quantized weights still benefit from compute dtype
            dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32

        dtype_key = str(dtype).replace("torch.", "")

        # ('Qwen/Qwen3-VL-8B-Thinking', 'bf16', 'cuda:1', 'bfloat16')
        cache_key = (model_id, quantization, device, dtype_key)

        #print("\n\nCache key: ")
        #print(cache_key)

        kwargs = {
            "torch_dtype": dtype,
            # Ensure the model lands on the selected device, not "auto"
            "device_map": {"": device},
        }

        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        # Quantization
        if quantization in ("int8_bnb", "int4_bnb"):
            if not device.startswith("cuda"):
                raise RuntimeError("bitsandbytes INT8/INT4 quantization requires a CUDA device.")
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise RuntimeError(
                    "BitsAndBytesConfig not available. Install a recent transformers + bitsandbytes."
                ) from e 
            try:
                import bitsandbytes  # noqa: F401
            except Exception as e:
                raise RuntimeError(
                    "bitsandbytes not installed. Install bitsandbytes to use int8_bnb/int4_bnb."
                ) from e

            if quantization == "int8_bnb":
                qconf = BitsAndBytesConfig(load_in_8bit=True)
            else:
                qconf = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
            kwargs["quantization_config"] = qconf
        
        # Make sure flash-atttn is installed 
        if flash_attention == "on":
            kwargs["attn_implementation"] = "flash_attention_2"
            

        # print(kwargs)
        # Load
        # print("\n\nModel family :" + model_family)
        
        if model_family == "qwen2_5_vl":
            self._MODEL_CACHE[cache_key] = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        elif model_family == "qwen3_vl":
            self._MODEL_CACHE[cache_key] = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        
        self._MODEL_CACHE[cache_key].eval()

        return self._MODEL_CACHE[cache_key]
    
    def _generate_local(
        self,
        image_pil: Image.Image,
        model_id: str,
        device_choice: str,
        quantization: str,
        flash_attention: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        safe_quants: bool,
        unload_model: bool,
    ) -> str:
        
        
        model = self._load_local_model(model_id, device_choice, quantization, flash_attention, safe_quants)

        # print(f"Using {model_family} on device {_parse_device_choice(device_choice)} with quantization {quantization}.")

        processor = self._get_processor(model_id)

        # Base instruction + user additions
        if system_prompt.strip() == "":
            text_instructions = "Convert the image into a rich image-generation prompt."
        else:
            text_instruction = system_prompt.strip()
        
        if user_prompt and user_prompt.strip():
            text_instruction = f"{text_instruction}\nAdditional requirements:\n{user_prompt.strip()}"
        
        messages = [
            {
                "role": "system",
                "content": [
                            {"type": "text", "text": system_prompt.strip()}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": text_instruction},
                ],
            },
        ]

        # print("System prompt:", system_prompt)
        # print("User prompt:", user_prompt)

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move tensors to the model device
        inputs = inputs.to(model.device)

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
        }
        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": 0.9})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)

        # Trim prompt tokens
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        
        out_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        sep = "</think>"
        idx = out_text.find(sep)
        if idx != -1:
            pstart = idx + len(sep)
            prompt = out_text[pstart:]
            reasoning = out_text[:idx]
        else:
            reasoning = "No reasoning available for local inference with Instruct models. In case of seeing this with Thinking model, try increasing max_tokens."
            prompt = out_text

        if unload_model:

            for key in self._MODEL_CACHE:
                self._MODEL_CACHE[key] = None
            for key in self._PROC_CACHE:
                 self._PROC_CACHE[key] = None
            model = None
            processor = None
            inputs = None
            # Not sure if I need to provide device id.
            _cuda_cleanup(None)

        return _clean_single_line(prompt), reasoning

    def _generate_openai_compatible(
        self,
        image_pil: Image.Image,
        model_id: str,
        base_url: str,
        api_key: str,
        model_override: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        providers: str = "",
        verbose: bool = True,
    ) -> str:
        buf = BytesIO()
        image_pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
        if system_prompt.strip() == "":                                                                         
            text_instruction = "Convert the image into a rich image-generation prompt."
        else:
            text_instruction = system_prompt.strip()
        if user_prompt and user_prompt.strip():
            text_instruction = f"{text_instruction}\nAdditional requirements:\n{user_prompt.strip()}"

        url = base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        # print("OpenAI request headers:", text_instruction)

        payload = {
            "model": (model_override.strip() if model_override and model_override.strip() else model_id),
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_instruction},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": float(temperature),
            # OpenAI uses max_tokens; many compatible servers accept it.
            "max_tokens": int(max_new_tokens),
        }

        # Add provider routing if specified (OpenRouter-specific)
        # Allows forcing specific providers for performance optimization
        if providers and providers.strip():
            provider_list = [p.strip() for p in providers.split(",") if p.strip()]
            if provider_list:
                payload["provider"] = {
                    "order": provider_list,
                    "allow_fallbacks": False
                }

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI-compatible endpoint error {resp.status_code}: {resp.text}")

        data = resp.json()

        # Verbose output - formatted API response details
        if verbose:
            provider = data.get("provider", "unknown")
            model = data.get("model", "unknown")
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            cost = usage.get("cost", 0)
            cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

            print(f"\n┌─ Qwen API Response ─────────────────────────────")
            print(f"│ Provider: {provider}")
            print(f"│ Model: {model}")
            print(f"│ Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
            if cached_tokens > 0:
                print(f"│ Cached: {cached_tokens} tokens")
            print(f"│ Cost: ${cost:.6f}")
            print(f"└─────────────────────────────────────────────────\n")

        reasoning = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("reasoning")
        )

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )

        if reasoning == None:
            reasoning = "No reasoning returned for OpenAI inference with non-thinking models."

        #content = data["choices"][0]["message"]["content"]
        #reasoning = data["choices"][0]["message"]["reasoning"]
        
        return _clean_single_line(content), reasoning

    def run(
        self,
        image,
        backend,
        qwen_model,
        device,
        quantization,
        flash_attention,
        system_prompt,
        user_prompt,
        max_new_tokens,
        temperature,
        safe_quants,
        unload_model,
        openai_base_url="http://127.0.0.1:11434",
        openai_api_key="",
        openai_model_override="",
        openrouter_providers="",
        verbose_output=True,
    ):
        image_pil = _tensor_image_to_pil(image)

        if backend == "openai_compatible":

            if not openai_api_key or not openai_api_key.strip():
                if "OPENROUTER_API_KEY" in os.environ:
                    openai_api_key = os.environ.get("OPENROUTER_API_KEY", "")
                    if openai_base_url in ("http://127.0.0.1:11434", "") or not openai_base_url.strip():
                        openai_base_url = "https://openrouter.ai/api"
                        
                elif "OPENAI_API_KEY" in os.environ:
                    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
                
            prompt, reasoning = self._generate_openai_compatible(
                image_pil=image_pil,
                model_id=qwen_model,
                base_url=openai_base_url,
                api_key=openai_api_key,
                model_override=openai_model_override,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                providers=openrouter_providers,
                verbose=verbose_output,
            )
        else:
            prompt, reasoning = self._generate_local(
                image_pil=image_pil,
                model_id=qwen_model,
                device_choice=device,
                quantization=quantization,
                flash_attention=flash_attention,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                safe_quants=safe_quants,
                unload_model=unload_model
            )
        # print(prompt,reasoning)

        return (prompt,reasoning,)


NODE_CLASS_MAPPINGS = {
    "QwenPromptFromImage": QwenPromptFromImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenPromptFromImage": "Qwen / OpenAI Prompt From Image",
}
