import os
import sys
from unittest.mock import MagicMock

# 在导入任何内容前阻止torch._dynamo加载以避免循环导入
sys.modules['torch._dynamo'] = MagicMock()
sys.modules['torch._export'] = MagicMock()

import torch

# 修复torch.is_autocast_enabled兼容性问题 - 接受任意参数
_original_is_autocast_enabled = torch.is_autocast_enabled

def _new_is_autocast_enabled(*args, **kwargs):
    """兼容wrapper - 忽略所有参数"""
    return _original_is_autocast_enabled()

torch.is_autocast_enabled = _new_is_autocast_enabled

from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend,FineGrainedFP8Config

model_id = "Ministral-3-8B-Instruct-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto",
    quantization_config=FineGrainedFP8Config(dequantize=True)
)

image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

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

# 将所有张量移到GPU并转换dtype
for key in tokenized:
    if isinstance(tokenized[key], torch.Tensor):
        tokenized[key] = tokenized[key].to(device="cuda")
        if key == "pixel_values":
            tokenized[key] = tokenized[key].to(dtype=torch.bfloat16)

image_sizes = [tokenized["pixel_values"].shape[-2:]]

output = model.generate(
    **tokenized,
    image_sizes=image_sizes,
    max_new_tokens=512,
)[0]

decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
print(decoded_output)
