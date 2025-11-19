# 预训练权重文件

本目录存放 ByteFormer 的预训练权重文件。

## 需要的文件

请将以下权重文件复制到此目录：

- `imagenet_jpeg_q60_k4_w128.pt` - ByteFormer Tiny 模型的预训练权重

## 原始位置

这些权重文件的原始位置：
- `/root/autodl-tmp/corenet/weights/imagenet_jpeg_q60_k4_w128.pt`

## 使用方法

权重文件准备好后，运行迁移脚本会自动加载这些权重：

```bash
python scripts/test_hf_byteformer_migration.py
```

## 文件大小

- `imagenet_jpeg_q60_k4_w128.pt`: 约 XXX MB

## 注意事项

- 确保权重文件的路径在脚本中正确配置
- 权重文件较大，建议使用 Git LFS 管理（如果需要版本控制）
