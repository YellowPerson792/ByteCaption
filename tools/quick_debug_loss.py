#!/usr/bin/env python3
"""快速测试：检查模型实际 loss"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "PureT"))

from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model

print("加载模型和 processor...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "./Qwen3-VL-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    "./Qwen3-VL-8B-Instruct", 
    trust_remote_code=True
)

# 应用 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 加载一张测试图片
from PIL import Image
from PureT.datasets_.coco_dataset_hf import CocoDataset

dataset = CocoDataset(
    image_ids_path='./PureT/data/coco_karpathy/train_ids.json',
    input_seq=None,
    target_seq=None,
    gv_feat_path='',
    seq_per_img=1,
    max_feat_num=-1,
    max_samples=1,
    return_captions=True,
    return_pil=True,
)

idx, captions, _, image = dataset[0]
caption = captions[0]

print(f"\n测试图片 caption: {caption}")

# 测试不同 prompt 长度的 loss
test_cases = [
    ("无 prompt", "", ""),
    ("短 prompt", "", "Describe this image:"),
    ("中等 prompt", "You are a helpful assistant.", "Describe this image in detail:"),
    ("长 prompt (OPENROUTER)", 
     "You are a vision captioning model.",
     "You are given a possibly corrupted JPEG image. Output a short COCO-style caption. Use 5-12 words. Output only the caption with no extra text."),
]

model.eval()
for name, sys_prompt, user_prompt in test_cases:
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    
    # 构建消息
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
    
    user_content = []
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    user_content.append({"type": "image", "image": image})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": caption}]})
    
    # 处理输入（训练模式）
    full_inputs = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    
    # 准备 prompt mask
    prompt_messages = messages[:-1]  # 不包括 assistant 回复
    prompt_inputs = processor.apply_chat_template(
        [prompt_messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    
    # 构建 labels
    input_ids = full_inputs['input_ids']
    labels = input_ids.clone()
    
    # Mask padding
    pad_token_id = processor.tokenizer.pad_token_id
    labels[labels == pad_token_id] = -100
    
    # Mask prompt
    prompt_len = prompt_inputs['attention_mask'].sum().item()
    attention_mask = full_inputs['attention_mask']
    seq_len = labels.size(1)
    full_len = attention_mask.sum().item()
    pad_len = seq_len - full_len
    
    # 假设 padding_side='left'
    start = pad_len
    end = min(seq_len, start + prompt_len)
    labels[0, start:end] = -100
    
    # 统计
    total_tokens = labels.numel()
    valid_tokens = (labels != -100).sum().item()
    
    print(f"Prompt tokens: {prompt_len}")
    print(f"Total tokens: {total_tokens}")
    print(f"Valid tokens: {valid_tokens} ({valid_tokens/total_tokens*100:.1f}%)")
    
    # Decode valid tokens
    valid_ids = labels[labels != -100].tolist()
    if valid_ids:
        decoded = processor.tokenizer.decode(valid_ids, skip_special_tokens=False)
        print(f"Valid tokens decode: '{decoded}'")
    
    # 计算 loss
    with torch.no_grad():
        batch = {
            'input_ids': input_ids.to(model.device),
            'attention_mask': attention_mask.to(model.device),
            'pixel_values': full_inputs['pixel_values'].to(model.device),
            'image_grid_thw': full_inputs['image_grid_thw'].to(model.device),
            'labels': labels.to(model.device),
        }
        
        outputs = model(**batch)
        loss = outputs.loss.item()
        
        print(f"\n>>> Loss: {loss:.4f}")
        
        if loss > 10:
            print(f"❌ Loss 异常高！")
        elif loss > 5:
            print(f"⚠️  Loss 较高")
        else:
            print(f"✓ Loss 正常")

print("\n" + "="*60)
print("测试完成")
print("="*60)
