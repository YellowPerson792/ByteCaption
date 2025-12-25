import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

path = "InternVL3_5-8B-HF/OpenGVLab/InternVL3_5-8B-HF"

model = AutoModelForImageTextToText.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": R1_SYSTEM_PROMPT},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "xxx"},
        ],
    },
]
