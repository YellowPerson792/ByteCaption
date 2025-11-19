"""
简化版推理脚本示例
用于快速测试模型推理功能

运行示例：
python byteformer-hf-migration/scripts/simple_inference.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from transformers import EncoderDecoderModel, AutoTokenizer
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
import numpy as np

def simple_inference_test():
    """简单的推理测试"""
    
    # 检查模型是否存在
    model_path = "./byteformer_gpt2_caption"
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请先完成训练或指定正确的模型路径")
        return
    
    print("加载模型和tokenizer...")
    try:
        # 加载模型和tokenizer
        model = EncoderDecoderModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        print(f"模型已加载到 {device}")
        
        # 加载测试数据
        print("加载测试数据...")
        dataset = load_dataset("jxie/flickr8k", split="test")
        
        # 取前几个样本进行测试
        num_test_samples = 3
        
        # 图像预处理
        def preprocess_image(image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            return transform(image)
        
        print(f"测试 {num_test_samples} 个样本:")
        print("=" * 50)
        
        with torch.no_grad():
            for i in range(num_test_samples):
                item = dataset[i]
                image = item["image"] if "image" in item else item["jpg"]
                
                # 预处理图像
                image_tensor = preprocess_image(image).unsqueeze(0)  # 添加batch维度
                
                # 注意：这里需要根据实际的训练脚本调整图像输入格式
                # 由于使用了CoreNet的ByteFormer，可能需要特殊的预处理
                
                try:
                    # 生成caption
                    generated_ids = model.generate(
                        pixel_values=image_tensor.to(device),
                        max_length=50,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    # 解码生成的文本
                    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    
                    # 获取ground truth caption
                    gt_caption = item.get("caption_0", "")
                    
                    print(f"样本 {i+1}:")
                    print(f"  生成的caption: {generated_caption}")
                    print(f"  真实caption: {gt_caption}")
                    print()
                    
                except Exception as e:
                    print(f"样本 {i+1} 推理失败: {e}")
                    print()
        
        print("推理测试完成！")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请检查模型路径和训练是否完成")

if __name__ == "__main__":
    simple_inference_test()
