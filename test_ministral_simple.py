"""最简单的Ministral-3-8B图像描述测试脚本"""
import os
import sys

# 禁用Dynamo以支持Python 3.12
os.environ["PYTORCH_ENABLE_DYNAMO"] = "0"

import torch

# 修复 torch.is_autocast_enabled 兼容性问题
_original_is_autocast_enabled = torch.is_autocast_enabled

def _patched_is_autocast_enabled(device_type=None):
    """兼容新旧版本的 torch.is_autocast_enabled"""
    try:
        if device_type is not None:
            return _original_is_autocast_enabled(device_type)
        return _original_is_autocast_enabled()
    except TypeError:
        # 旧版本不接受参数
        return _original_is_autocast_enabled()

torch.is_autocast_enabled = _patched_is_autocast_enabled

from PIL import Image
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
import requests
from io import BytesIO

print("=" * 80)
print("Ministral-3-8B 最简图像描述测试")
print("=" * 80)

# 1. 加载模型和tokenizer
print("\n[步骤1] 加载模型和tokenizer...")
model_path = "Ministral-3-8B-Instruct-2512"

try:
    tokenizer = MistralCommonBackend.from_pretrained(model_path)
    print(f"✓ Tokenizer加载成功: {tokenizer.__class__.__name__}")
except Exception as e:
    print(f"✗ Tokenizer加载失败: {e}")
    exit(1)

try:
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"✓ 模型加载成功: {model.__class__.__name__}")
    print(f"  设备: {model.device}")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 加载测试图像
print("\n[步骤2] 加载测试图像...")
try:
    # 尝试从COCO数据集加载一张图像
    import os
    coco_img_path = "./PureT/data/coco_karpathy/test_sample_500/00000_id42.jpg"
    if os.path.exists(coco_img_path):
        image = Image.open(coco_img_path).convert("RGB")
        print(f"✓ 从本地加载图像: {coco_img_path}")
    else:
        # 使用示例图像URL
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"✓ 从URL加载图像: {url}")
    
    print(f"  图像尺寸: {image.size}")
except Exception as e:
    print(f"✗ 图像加载失败: {e}")
    exit(1)

# 3. 测试不同的输入格式
print("\n[步骤3] 测试使用聊天模板...")

# 测试: 使用聊天模板（正确的方式）
print("\n--- 使用apply_chat_template ---")
try:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in one sentence."}
            ]
        }
    ]
    
    # 检查processor是否支持apply_chat_template
    if hasattr(processor, 'apply_chat_template'):
        print("✓ 处理器支持 apply_chat_template")
        
        # 使用tokenize=True直接生成模型输入
        inputs = processor.apply_chat_template(
            messages,
            images=[image],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 确保pixel_values与模型权重类型匹配
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        
        print(f"输入keys: {inputs.keys()}")
        print(f"input_ids shape: {inputs['input_ids'].shape}")
        if 'pixel_values' in inputs:
            print(f"pixel_values shape: {inputs['pixel_values'].shape}")
            print(f"pixel_values dtype: {inputs['pixel_values'].dtype}")
        
        # 解码输入查看prompt
        prompt_text = processor.decode(inputs['input_ids'][0], skip_special_tokens=False)
        print(f"\n实际prompt（前200字符）: {prompt_text[:200]}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=1
            )
        
        # 只解码新生成的token
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True)
        print(f"\n生成的caption: {generated_text}")
        print(f"生成的token数量: {len(generated_ids)}")
    else:
        print("✗ 处理器不支持 apply_chat_template")
        
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 保留原来的测试1作为对比
print("\n\n--- 测试1: 简单文本提示（仅作对比） ---")
try:
    # Mistral3需要在文本中包含[IMG]占位符来标识图像位置
    prompt = "[IMG]Describe this image in one sentence."
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 确保pixel_values与模型权重类型匹配
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    
    print(f"输入keys: {inputs.keys()}")
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    if 'pixel_values' in inputs:
        print(f"pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"pixel_values dtype: {inputs['pixel_values'].dtype}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1
        )
    
    # 只解码新生成的token，跳过输入的prompt部分
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_length:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    print(f"生成的文本: {generated_text}")
    print(f"完整输出（前200字符）: {processor.decode(outputs[0], skip_special_tokens=True)[:200]}")
except Exception as e:
    print(f"✗ 测试1失败: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 使用聊天模板
print("\n--- 测试2: 使用聊天模板 ---")
try:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image", "image": image}
            ]
        }
    ]
    
    # 检查processor是否支持apply_chat_template
    if hasattr(processor, 'apply_chat_template'):
        print("✓ 处理器支持 apply_chat_template")
        
        # 方式A: tokenize=False
        print("\n  方式A: apply_chat_template(tokenize=False)")
        try:
            prompt_text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print(f"  生成的提示文本: {prompt_text[:200]}...")
            
            inputs = processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 确保pixel_values与模型权重类型匹配
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            
            print(f"  input_ids shape: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
                print(f"  pixel_values dtype: {inputs['pixel_values'].dtype}")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            # 只解码新生成的token
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            generated_text = processor.decode(generated_ids, skip_special_tokens=True)
            print(f"  生成的文本: {generated_text}")
        except Exception as e:
            print(f"  ✗ 方式A失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 方式B: tokenize=True
        print("\n  方式B: apply_chat_template(tokenize=True)")
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 确保pixel_values与模型权重类型匹配
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            
            print(f"  输入keys: {inputs.keys()}")
            print(f"  input_ids shape: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
                print(f"  pixel_values dtype: {inputs['pixel_values'].dtype}")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            # 只解码新生成的token
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            generated_text = processor.decode(generated_ids, skip_special_tokens=True)
            print(f"  生成的文本: {generated_text}")
        except Exception as e:
            print(f"  ✗ 方式B失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ 处理器不支持 apply_chat_template")
        
except Exception as e:
    print(f"✗ 测试2失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 检查tokenizer中的特殊标记
print("\n[步骤4] 检查tokenizer特殊标记...")
try:
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    
    # 查找图像相关的特殊标记
    special_tokens = tokenizer.all_special_tokens
    print(f"\n所有特殊标记 ({len(special_tokens)}): {special_tokens[:20]}")
    
    # 查找可能的图像标记
    image_tokens = [t for t in special_tokens if 'image' in t.lower() or 'img' in t.lower() or 'vision' in t.lower()]
    if image_tokens:
        print(f"图像相关标记: {image_tokens}")
    else:
        print("未找到明确的图像标记")
        
except Exception as e:
    print(f"✗ 检查标记失败: {e}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
