import os
import sys
import re

# Disable torch._dynamo compilation before importing any torch/transformers dependent modules
os.environ['TORCH_DISABLE_COMPILATION_OPTIM'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Block any ONNX-related registrations before any potential import of torchvision/timm/transformers
import unittest.mock as mock
sys.modules['torch.onnx'] = mock.MagicMock()
sys.modules['torch.onnx.operators'] = mock.MagicMock()
sys.modules['torch.onnx.symbolic_helper'] = mock.MagicMock()
sys.modules['torch.onnx._internal'] = mock.MagicMock()
sys.modules['torch.onnx._internal.exporter'] = mock.MagicMock()

# Now safe to import torch and transformers
import torch
from PIL import Image
from transformers import AutoProcessor, Glm4vForConditionalGeneration

MODEL_PATH = "GLM-4.6V-Flash/ZhipuAI/GLM-4.6V-Flash"
IMAGE_PATH = "PureT/data/coco_karpathy/test_sample_500/00000_id42.jpg"


def _sanitize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text
    # Drop <think>...</think>
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    # Drop box markers
    s = s.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    # Remove role-like tokens <|role|>
    s = re.sub(r"<\|[^>]+\|>", "", s)
    s = s.strip()
    # Keep first sentence-ish fragment if multiple
    if "." in s:
        parts = [p.strip() for p in s.split(".") if p.strip()]
        if parts:
            s = parts[0]
    return s


if __name__ == "__main__":
    img = Image.open(IMAGE_PATH).convert("RGB")

    # Only adjust messages format: user text first, then PIL image
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a vision captioning model."
                    ),
                },
            ],
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are given a possibly corrupted JPEG image. "
                        "Output a short COCO-style caption. Use 5-12 words. "
                        "Output only the caption with no extra text."
                    ),
                },
                {"type": "image", "image": img},
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )

    # Inspect what apply_chat_template produces before tokenization
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print("Rendered template (string):\n", rendered)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        enable_thinking=False
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    # Decode input_ids back to text to see special tokens/prefixes
    decoded_prompt = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    print("\nDecoded tokenized prompt:\n", decoded_prompt)

    generated_ids = model.generate(**inputs, max_new_tokens=256, num_beams=5)
    prompt_len = inputs["input_ids"].shape[1]
    gen_only = generated_ids[:, prompt_len:]
    output_text = processor.batch_decode(gen_only, skip_special_tokens=False)[0]
    print("Raw model output:")
    print(output_text)
    print("\nSanitized output:")
    output_text = _sanitize(output_text)
    print(output_text)
