import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
path = "InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF"

model = AutoModelForImageTextToText.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
