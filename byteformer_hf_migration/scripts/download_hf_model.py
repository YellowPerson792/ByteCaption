from transformers import AutoModel, AutoTokenizer
import os

MODEL_NAME = "gpt2"

print(f"正在下载模型: {MODEL_NAME}")
model = AutoModel.from_pretrained(MODEL_NAME)

print(f"正在下载分词器: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

