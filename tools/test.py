import os
import sys
# Disable torch._dynamo compilation before importing any torch-dependent modules
os.environ['TORCH_DISABLE_COMPILATION_OPTIM'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Block torchvision.ops._register_onnx_ops before timm imports it
import unittest.mock as mock
sys.modules['torch.onnx'] = mock.MagicMock()
sys.modules['torch.onnx.operators'] = mock.MagicMock()
sys.modules['torch.onnx.symbolic_helper'] = mock.MagicMock()
sys.modules['torch.onnx._internal'] = mock.MagicMock()
sys.modules['torch.onnx._internal.exporter'] = mock.MagicMock()

import base64
import io
import mimetypes
import torch
from PIL import Image
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

model_id = "Ministral-3-8B-Base-2512/mistralai/Ministral-3-8B-Base-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


def _pil_to_data_url(image: Image.Image, mime: str = "image/png") -> str:
    # Encode PIL Image to base64 data URL for tokenizer image input
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _path_to_data_url(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/png"
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return _pil_to_data_url(img, mime=mime)


def to_data_url(image_or_path) -> str:
    if isinstance(image_or_path, Image.Image):
        return _pil_to_data_url(image_or_path)
    return _path_to_data_url(str(image_or_path))


image_path = "PureT/data/coco_karpathy/test_sample_500/00000_id42.jpg"
pil_image = Image.open(image_path).convert("RGB")
image_url = to_data_url(pil_image)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]

tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

# Move all tensors to CUDA with correct dtype
for key in tokenized:
    if isinstance(tokenized[key], torch.Tensor):
        if key == "pixel_values":
            tokenized[key] = tokenized[key].to(dtype=torch.bfloat16, device="cuda")
        else:
            tokenized[key] = tokenized[key].to(device="cuda")

image_sizes = [tokenized["pixel_values"].shape[-2:]]

output = model.generate(
    **tokenized,
    image_sizes=image_sizes,
    max_new_tokens=512,
)[0]

decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
print(decoded_output)
