import os
import sys
import argparse
import torch
import torch.nn as nn
from dataclasses import dataclass, field

from corenet.options.opts import get_training_arguments
from byteformer_hf_migration.utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel
from transformers import PreTrainedModel
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

CONFIG_MAPPING.register("byteformer", CorenetToHFPretrainedConfig)
MODEL_MAPPING.register(CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel)

class ByteFormerWrapper(PreTrainedModel):
    """ByteFormer包装器，适配HuggingFace EncoderDecoderModel接口
    
    精确复用CoreNet ByteFormer的实现，只是去掉分类头，保留完整的特征表示
    """
    def __init__(self, byteformer_model, config):
        super().__init__(config)
        self.byteformer = byteformer_model
        self.config = config
        # 设置必要的属性
        self.main_input_name = "input_ids"
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        前向传播，复用ByteFormer的backbone，但返回序列特征而不是分类结果
        
        Args:
            input_ids: 输入token序列 [batch_size, sequence_length]
            attention_mask: 注意力掩码 (如果提供，将被忽略，因为ByteFormer内部处理掩码)
            
        Returns:
            BaseModelOutput: 包含last_hidden_state的输出，同时将encoder输出对应的mask
                           存储在特殊属性中供EncoderDecoderModel使用
        """
        # 步骤1: 获取backbone输入 (embeddings + positional embeddings)

        x, key_padding_mask = self.byteformer.get_backbone_inputs(input_ids)
        
        # 步骤2: 通过transformer backbone
        x, updated_mask = self.byteformer.backbone_forward(x, key_padding_mask)
        
        # 步骤3: 将ByteFormer的mask转换为标准的attention_mask格式
        # updated_mask中，-inf表示被mask的位置，其他位置是0
        # 转换为attention_mask: 1表示有效位置，0表示mask位置
        encoder_output_attention_mask = (updated_mask != float("-inf")).float()
        
        # 步骤4: 返回完整的序列特征，而不是池化后的分类特征
        # x的形状是 [batch_size, sequence_length, hidden_size]
        # 这样可以给decoder提供更丰富的信息
        
        # 返回符合HuggingFace格式的输出
        from transformers.modeling_outputs import BaseModelOutput
        output = BaseModelOutput(
            last_hidden_state=x,
            # 注意：我们将在collate函数中处理attention_mask的传递
        )
        
        # 在输出对象上添加encoder输出对应的attention mask
        # 这是一个hack，但可以让EncoderDecoderModel访问到正确的mask
        output.encoder_attention_mask = encoder_output_attention_mask
        
        return output
    
def get_opts():
    corenet_args = [
        "--common.config-file", "byteformer_hf_migration/configs/conv_kernel_size=4,window_sizes=[128].yaml",
        "--model.classification.pretrained", "byteformer_hf_migration/weights/imagenet_jpeg_q60_k4_w128.pt",
        "--model.classification.n-classes", "1000",  # 用于加载预训练权重
        "--dataset.root-train", "./data",
        "--dataset.root-val", "./data",
    ]
    opts = get_training_arguments(args=corenet_args)
    return opts

def init_byteformer(opts=None):
    if opts is None:
        opts = get_opts()
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size", 257)
    
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    byteformer_model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    
    # weights = torch.load("byteformer_hf_migration/weights/imagenet_jpeg_q60_k4_w128.pt", map_location='cpu')
    # # 加载backbone部分权重
    # model_state = byteformer_model.model.state_dict()
    # pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
    # byteformer_model.model.load_state_dict(pretrained_state, strict=False)
    
    byteformer_encoder = byteformer_model.model
    # Remove the classifier if it exists
    if hasattr(byteformer_encoder, 'classifier'):
        delattr(byteformer_encoder, 'classifier')
        
    encoder_config = CorenetToHFPretrainedConfig(**vars(opts))
    wrapped_encoder = ByteFormerWrapper(byteformer_encoder, encoder_config)
        
    return wrapped_encoder
