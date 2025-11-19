import os
from datasets import load_dataset

# 设置 HuggingFace 镜像源（以清华镜像为例）
# 你也可以换成其他可用的镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 示例：下载 MNIST 数据集
# 你可以替换为其他数据集名称和配置
DATASET_NAME = "jxie/flickr8k"

print(f"开始下载数据集: {DATASET_NAME} (所有split)，使用镜像源: {os.environ['HF_ENDPOINT']}")
dataset = load_dataset(DATASET_NAME)
for split_name, split_data in dataset.items():
	print(f"split: {split_name}, 样本数: {len(split_data)}")

# 可选：保存为本地文件（如jsonl、csv等）
# for split_name, split_data in dataset.items():
#     split_data.to_json(f"{DATASET_NAME}_{split_name}.jsonl")
#     split_data.to_csv(f"{DATASET_NAME}_{split_name}.csv")
