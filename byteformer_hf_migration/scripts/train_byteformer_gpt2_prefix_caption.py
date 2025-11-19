"""
ByteFormer + GPT2 Prefix Caption Training Script
使用ByteFormer作为encoder，GPT2作为decoder，使用prefix prompt方式实现图像描述生成任务

示例运行命令：
python byteformer-hf-migration/scripts/train_byteformer_gpt2_prefix_caption.py --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --num_train_epochs 3 --learning_rate 5e-5 --eval_steps 100 --logging_steps 50 --save_steps 600 --lr_scheduler_type cosine --gradient_accumulation_steps 2 --report_to none --max_caption_length 16 --num_eval_samples 50 --fp16 --gpt2_model gpt2-medium
"""

import os
# 设置环境变量以避免tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Tuple
from corenet.options.opts import get_training_arguments
from utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel
from corenet.data.transforms.image_bytes import PILSave
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from utils.hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, GenerationConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, GenerationMixin
from torch.nn.utils.rnn import pad_sequence
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from transformers import AutoConfig, AutoModel
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

class PrefixLMForCaption(PreTrainedModel, GenerationMixin):
    """Prefix LM模型，使用ByteFormer encoder + GPT2 decoder"""
    supports_gradient_checkpointing = True
    main_input_name = "input_ids"
    base_model_prefix = "prefix_lm"

    def __init__(self, encoder, decoder, decoder_tokenizer, bos_token_id):
        # 使用decoder的config初始化PreTrainedModel
        # decoder 已经在创建时设置了 attn_implementation="eager"
        super().__init__(decoder.config)
        self.encoder = encoder
        self.decoder = decoder
        # 将投影层集成到模型内部，作为transformer结构的一部分
        # 注意：encoder.config 可能与 CorenetToHFPretrainedConfig 不同，这里使用 model_dim 或 hidden_size
        proj_in = getattr(encoder.config, 'model_dim', None) or getattr(encoder.config, 'hidden_size', None)
        proj_out = getattr(decoder.config, 'n_embd', None)
        self.encoder_decoder_proj = torch.nn.Linear(proj_in, proj_out)
        self.decoder_tokenizer = decoder_tokenizer
        self.bos_token_id = bos_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]  # [batch_size, sequence_length, hidden_size]
        encoder_attention_mask = getattr(encoder_outputs, "encoder_attention_mask", None)

        # 投影encoder输出到decoder维度（使用内部集成的投影层）
        encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)
        # 使用完整的序列特征作为prefix，形状为[batch, seq_len, hidden]
        prefix_embeds = encoder_hidden_states  # [batch, seq_len, hidden]
        
        # 直接将prefix_embeds与labels的embeds拼接
        if labels is not None:
            # labels为token id序列，-100 表示忽略的填充位置，不能直接用于embedding
            pad_token_id = getattr(self.decoder_tokenizer, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = getattr(self.decoder_tokenizer, 'eos_token_id', 0)
            labels_for_embeds = labels.clone()
            labels_for_embeds = labels_for_embeds.masked_fill(labels_for_embeds == -100, pad_token_id)

            # 使用替换后的 labels 计算词向量
            label_embeds = self.decoder.transformer.wte(labels_for_embeds)
            final_inputs_embeds = torch.cat([prefix_embeds, label_embeds], dim=1)

            # 构造注意力mask：prefix部分使用encoder_attention_mask，label部分用提供的mask，否则按 labels!=-100 构造
            if encoder_attention_mask is not None:
                prefix_attention = encoder_attention_mask.to(dtype=torch.long, device=prefix_embeds.device)
            else:
                prefix_attention = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device)
            if decoder_attention_mask is not None:
                dec_attn = decoder_attention_mask.to(device=prefix_embeds.device, dtype=torch.long)
            else:
                dec_attn = (labels != -100).to(dtype=torch.long, device=prefix_embeds.device)
            final_attention_mask = torch.cat([prefix_attention, dec_attn], dim=1)
        elif decoder_inputs_embeds is not None:
            final_inputs_embeds = torch.cat([prefix_embeds, decoder_inputs_embeds], dim=1)
            if encoder_attention_mask is not None:
                prefix_attention = encoder_attention_mask.to(dtype=torch.long, device=prefix_embeds.device)
            else:
                prefix_attention = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device)
            if decoder_attention_mask is not None:
                final_attention_mask = torch.cat([prefix_attention, decoder_attention_mask.to(device=prefix_embeds.device, dtype=torch.long)], dim=1)
            else:
                final_attention_mask = None
        else:
            final_inputs_embeds = prefix_embeds
            final_attention_mask = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device)

        # Decode，确保不会同时传入input_ids和inputs_embeds
        decoder_args = {
            'attention_mask': final_attention_mask,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'use_cache': use_cache,
            'past_key_values': past_key_values,
            'return_dict': return_dict,
            **kwargs_decoder,
        }
        
        # 优先使用inputs_embeds（包含prefix）
        if final_inputs_embeds is not None:
            decoder_args['inputs_embeds'] = final_inputs_embeds
        elif decoder_input_ids is not None:
            decoder_args['input_ids'] = decoder_input_ids
        decoder_outputs = self.decoder(**decoder_args)

        # Compute loss independent from decoder
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            # 跳过prefix部分，只计算caption部分的loss
            prefix_len = prefix_embeds.size(1)  # prefix的序列长度
            label_len = labels.size(1)
            prediction_logits = logits[:, prefix_len-1:prefix_len+label_len-1, :]  
            # 将labels中的pad token替换为-100
            pad_token_id = getattr(self.decoder_tokenizer, 'pad_token_id', None)
            shift_labels = labels.clone()
            if pad_token_id is not None:
                shift_labels[shift_labels == pad_token_id] = -100
            # print(f"[DEBUG] shift_labels: ", shift_labels)
            shift_logits = prediction_logits.contiguous().view(-1, prediction_logits.size(-1))
            shift_labels = shift_labels.contiguous().view(-1)
            loss = CrossEntropyLoss()(shift_logits, shift_labels)

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=getattr(decoder_outputs, 'cross_attentions', None),
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        if hasattr(self.encoder, 'get_input_embeddings'):
            return self.encoder.get_input_embeddings()
        elif hasattr(self.encoder, 'embeddings'):
            return self.encoder.embeddings
        else:
            return None

    def set_input_embeddings(self, value):
        if hasattr(self.encoder, 'set_input_embeddings'):
            self.encoder.set_input_embeddings(value)
        elif hasattr(self.encoder, 'embeddings'):
            self.encoder.embeddings = value

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        if labels is None:
            return None
        # 确保config参数存在
        pad_token_id = getattr(self.config, 'pad_token_id', self.decoder_tokenizer.pad_token_id or self.decoder_tokenizer.unk_token_id)
        decoder_start_token_id = getattr(self.config, 'decoder_start_token_id', self.bos_token_id)
        return shift_tokens_right(labels, pad_token_id, decoder_start_token_id)

    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, **model_kwargs):
        return model_kwargs
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the PrefixLMForCaption directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.decoder._reorder_cache(past_key_values, beam_idx)

    def generate(self, input_ids, **kwargs):
        """生成caption的方法"""
        # 获取encoder输出
        encoder_outputs = self.encoder(input_ids=input_ids)
        encoder_hidden_states = encoder_outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # 投影到decoder维度
        encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)
        prefix_embeds = encoder_hidden_states  # [batch, seq_len, hidden]
        prefix_len = prefix_embeds.size(1)
        
        # 开始生成，使用BOS token作为起始
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 创建起始token
        start_tokens = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        start_embeds = self.decoder.transformer.wte(start_tokens)
        
        # 将prefix和start token组合
        current_embeds = torch.cat([prefix_embeds, start_embeds], dim=1)
        # 创建attention mask: prefix部分使用encoder_attention_mask，start token部分为1
        encoder_attention_mask = getattr(encoder_outputs, "encoder_attention_mask", None)
        if encoder_attention_mask is not None:
            prefix_attention = encoder_attention_mask.to(dtype=torch.long, device=device)
        else:
            prefix_attention = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=device)
        start_attention = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        current_attention_mask = torch.cat([prefix_attention, start_attention], dim=1)
        
        max_length = kwargs.get('max_length', 50)
        generated_tokens = []
        
        for _ in range(max_length - 1):  # -1 because we already have start token
            # 前向传播
            outputs = self.decoder(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                use_cache=False
            )
            
            # 获取最后一个token的logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # 采样下一个token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            
            # 检查是否生成了EOS token
            if (next_token == self.decoder_tokenizer.eos_token_id).all():
                break
                
            # 准备下一轮的输入
            next_token_embeds = self.decoder.transformer.wte(next_token)
            current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)
            current_attention_mask = torch.cat([
                current_attention_mask, 
                torch.ones((batch_size, 1), dtype=torch.long, device=device)
            ], dim=1)
        
        # 组合所有生成的tokens
        if generated_tokens:
            generated_ids = torch.cat([start_tokens] + generated_tokens, dim=1)
        else:
            generated_ids = start_tokens
            
        return generated_ids


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ByteFormer + GPT2 Prefix Caption Training")
    parser.add_argument("--config", type=str, default="byteformer-hf-migration/configs/conv_kernel_size=4,window_sizes=[128].yaml", help="CoreNet配置文件路径")
    parser.add_argument("--weights", type=str, default="byteformer-hf-migration/weights/imagenet_jpeg_q60_k4_w128.pt", help="预训练权重文件路径")
    parser.add_argument("--gpt2_model", type=str, default="gpt2", help="GPT2模型名称")
    parser.add_argument("--dataset_name", type=str, default="jxie/flickr8k", help="数据集名称")
    parser.add_argument("--num_train_samples", type=int, default=None, help="训练样本数量（None表示使用全部训练数据）")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="评估样本数量（None表示使用全部验证数据）")
    parser.add_argument("--max_caption_length", type=int, default=50, help="最大caption长度")
    parser.add_argument("--max_byteformer_length", type=int, default=2048, help="ByteFormer最大输入长度")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/byteformer_gpt2_prefix_caption", help="训练输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="每设备训练批大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="每设备验证批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant"], help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0, help="预热比例")
    parser.add_argument("--fp16", action="store_true", default=False, help="启用FP16混合精度")
    parser.add_argument("--bf16", action="store_true", default=False, help="启用BF16混合精度")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估步数间隔")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志步数间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存检查点总数限制")
    parser.add_argument("--use_lora", action="store_true", default=True, help="使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--report_to", type=str, default=None, choices=[None, "none", "wandb", "tensorboard"], help="日志报告工具")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("ByteFormer + GPT2 Prefix Caption Training")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"预训练权重: {args.weights}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_train_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"批大小: {args.per_device_train_batch_size}")
    print(f"数据集: {args.dataset_name}")
    print("=" * 50)
    
    # 加载CoreNet配置和ByteFormer模型
    corenet_args = [
        "--common.config-file", args.config,
        "--model.classification.pretrained", args.weights,
        "--model.classification.n-classes", "1000",  # 用于加载预训练权重
        "--dataset.root-train", "./data",
        "--dataset.root-val", "./data",
        "--common.accum-freq", str(args.gradient_accumulation_steps),
        "--common.log-freq", str(args.logging_steps),
    ]
    opts = get_training_arguments(args=corenet_args)
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size", 257)
    
    # 创建ByteFormer model
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    byteformer_model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    
    # 加载预训练权重
    weights = torch.load(args.weights, map_location='cpu')
    model_state = byteformer_model.model.state_dict()
    pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
    byteformer_model.model.load_state_dict(pretrained_state, strict=False)
    
    # 去掉分类头
    byteformer_encoder = byteformer_model.model
    if hasattr(byteformer_encoder, 'classifier'):
        delattr(byteformer_encoder, 'classifier')
    
    # 创建ByteFormer Encoder包装器
    encoder = ByteFormerWrapper(byteformer_encoder, hf_config)
    
    # 创建GPT2 Decoder，明确指定 attn_implementation 为 "eager"
    gpt2_config = GPT2Config.from_pretrained(args.gpt2_model)
    gpt2_config.attn_implementation = "eager"  # 避免 scaled_dot_product_attention 兼容性问题
    gpt2_decoder = GPT2LMHeadModel.from_pretrained(args.gpt2_model, config=gpt2_config, attn_implementation="eager")
    
    # 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 创建PrefixLM模型
    bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model = PrefixLMForCaption(encoder, gpt2_decoder, tokenizer, bos_token_id)
    
    # 设置模型配置
    model.config.decoder_start_token_id = bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id  
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = tokenizer.vocab_size
    model.main_input_name = "input_ids"
    
    # 图像预处理
    def pil_to_tensor_transform(img):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return transform(img)
    
    class PrefixCaptionDataset(torch.utils.data.Dataset):
        def __init__(self, split="train", num_samples=None, dataset_name="jxie/flickr8k"):
            self.dataset = load_dataset(dataset_name, split=split)
            self.dataset_name = dataset_name
            self.split = split
            self.total_samples = len(self.dataset) * 5
            if num_samples is not None and num_samples < self.total_samples:
                self.total_samples = num_samples
                print(f"使用 {split} 数据集的前 {num_samples} 个样本")
            else:
                print(f"使用完整的 {split} 数据集，共 {self.total_samples} 个样本")
                
        def __getitem__(self, idx):
            dataset_idx = idx // 5
            caption_idx = idx % 5
            item = self.dataset[dataset_idx]
            img = item["image"] if "image" in item else item["jpg"]
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = pil_to_tensor_transform(img)
            caption = item.get(f"caption_{caption_idx}", "")
            return {
                'image_tensor': img_tensor,
                'caption': caption
            }
            
        def __len__(self):
            return self.total_samples

    def prefix_caption_collate_fn(batch):
        images = []
        captions = [] 
        for item in batch:
            images.append(item['image_tensor'])
            captions.append(item['caption'])
            
        # 处理图像，使用ByteFormer的collate函数
        corenet_batch = []
        for img_tensor in images:
            corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target
        collated = byteformer_image_collate_fn(corenet_batch, opts)
        input_ids = collated["samples"]
        
        # 处理caption，转换为token IDs
        caption_tokens = tokenizer(
            captions,
            padding='longest',
            max_length=args.max_caption_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = caption_tokens.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        # 在labels最前面加BOS token，实现右移
        bos = torch.full((labels.size(0), 1), tokenizer.bos_token_id, dtype=labels.dtype)
        labels = torch.cat([bos, labels[:, :-1]], dim=1)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "decoder_attention_mask": caption_tokens.attention_mask,
        }

    # 创建数据集
    train_ds = PrefixCaptionDataset(split="train", num_samples=args.num_train_samples, dataset_name=args.dataset_name)
    eval_ds = PrefixCaptionDataset(split="test", num_samples=args.num_eval_samples, dataset_name=args.dataset_name)
    
    # 评估指标
    rouge = evaluate.load("rouge")
    
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # 处理变长序列，不强制转换为numpy数组
        # 直接在列表上工作，避免形状不一致的问题
        
        # 确保输入是列表格式
        if isinstance(labels_ids, np.ndarray):
            labels_ids = labels_ids.tolist()
        if isinstance(pred_ids, np.ndarray):
            pred_ids = pred_ids.tolist()
        
        # 注意：在新的evaluate方法中，pred_ids已经是去掉prefix的纯生成部分
        # 所以这里不需要再跳过prefix，直接decode即可
        
        # Decode predictions
        pred_str = []
        for pred_tokens in pred_ids:
            # 过滤掉空的预测
            if pred_tokens and len(pred_tokens) > 0:
                decoded = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            else:
                decoded = ""
            pred_str.append(decoded)
        
        # Decode labels - 处理变长序列
        label_str = []
        for label_tokens in labels_ids:
            # Replace -100 with pad token for decoding
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            cleaned_tokens = [token if token != -100 else pad_token_id for token in label_tokens]
            decoded = tokenizer.decode(cleaned_tokens, skip_special_tokens=True)
            label_str.append(decoded)
        
        # Compute ROUGE if available
        results = {}
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])
        results["rouge2_fmeasure"] = round(rouge_output["rouge2"], 4)
        
        # Compute BLEU scores
        bleu_1_scores = []
        bleu_4_scores = []
        for ref, pred in zip(label_str, pred_str):
            reference = [nltk.word_tokenize(ref.lower())]
            candidate = nltk.word_tokenize(pred.lower())
            smoothing_function = SmoothingFunction().method4
            bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
            bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
            bleu_1_scores.append(bleu_1)
            bleu_4_scores.append(bleu_4)
        
        results.update({
            "bleu1": round(np.mean(bleu_1_scores), 4),
            "bleu4": round(np.mean(bleu_4_scores), 4),
        })
        
        # Print up to 5 predictions and labels for debugging
        for i, (ref, pred) in enumerate(zip(label_str, pred_str)):
            if i in [0, 5, 10, 15, 20, 25]:  
                print(f"Sample {i + 1}:")
                print(f"  Reference: {ref}")
                print(f"  Prediction: {pred}")
                print(f"  Pred tokens length: {len(pred_ids[i]) if i < len(pred_ids) else 0}")
                print()
        
        return results
    
    # 计算总训练步数
    train_batch_size = args.per_device_train_batch_size
    num_train_epochs = args.num_train_epochs
    num_train_samples = len(train_ds)
    steps_per_epoch = (num_train_samples + train_batch_size - 1) // train_batch_size
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 训练参数
    training_args = MySeq2SeqTrainingArguments(
        output_dir=args.output_dir,
        train_batch_size=args.per_device_train_batch_size,
        eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        eval_strategy=args.evaluation_strategy,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to if args.report_to not in [None, "none"] else None,
    )

    # === PrefixCompatibleTrainer: 优先使用 encoder_outputs 调用 generate ===
    class PrefixCompatibleTrainer(MySeq2SeqTrainer):
        def evaluate(self, eval_dataset=None, desc="Eval", ignore_keys=None, metric_key_prefix: str = "eval"):
            import numpy as np
            from tqdm import tqdm
            from torch.utils.data import DataLoader
            
            self.model.eval()
            device = next(self.model.parameters()).device  # 更可靠的方式获取device
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            total_loss = 0.0
            predictions = []
            references = []
            tokenizer = self.tokenizer
            gen_config = getattr(self.model, 'generation_config', None)
            max_length = gen_config.max_length if gen_config else 16
            input_name = getattr(self.model, 'main_input_name', 'input_ids')
            
            # 创建DataLoader如果eval_dataset不是DataLoader
            if not isinstance(eval_dataset, DataLoader):
                eval_dataloader = DataLoader(
                    eval_dataset, 
                    batch_size=self.args.per_device_eval_batch_size,
                    collate_fn=self.data_collator,
                    shuffle=False
                )
            else:
                eval_dataloader = eval_dataset
            
            # 添加进度条
            eval_iterator = tqdm(eval_dataloader, desc=f"{desc}", total=len(eval_dataloader))
            
            for batch in eval_iterator:
                batch_inputs = {input_name: batch[input_name].to(device)}
                if 'attention_mask' in batch:
                    batch_inputs['attention_mask'] = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # 计算loss
                outputs = self.model(**batch_inputs, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item() if loss is not None else 0.0

                # 使用我们自定义的generate方法，而不是decoder.generate
                with torch.no_grad():
                    # 获取encoder输出并投影
                    encoder_outputs = self.model.encoder(**{input_name: batch_inputs[input_name]})
                    encoder_hidden_states = encoder_outputs[0]  # [B, seq, hidden]
                    prefix_embeds = self.model.encoder_decoder_proj(encoder_hidden_states)
                    batch_size = prefix_embeds.size(0)
                    device = prefix_embeds.device
                    
                    # 准备start token embedding  
                    start_tokens = torch.full((batch_size, 1), self.model.bos_token_id, dtype=torch.long, device=device)
                    start_embeds = self.model.decoder.transformer.wte(start_tokens)
                    
                    # 初始embeddings：prefix + start token
                    current_embeds = torch.cat([prefix_embeds, start_embeds], dim=1)
                    
                    # 构造attention mask: 使用encoder_attention_mask
                    encoder_attention_mask = getattr(encoder_outputs, "encoder_attention_mask", None)
                    if encoder_attention_mask is not None:
                        prefix_attention = encoder_attention_mask.to(dtype=torch.long, device=device)
                    else:
                        prefix_attention = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=device)
                    start_attention = torch.ones((batch_size, 1), dtype=torch.long, device=device)
                    current_attention_mask = torch.cat([prefix_attention, start_attention], dim=1)
                    
                    # 逐步生成tokens
                    generated_tokens = []
                    prefix_and_start_len = current_embeds.size(1)
                    
                    for step in range(max_length - 1):  # 最多生成max_length-1个新token
                        # 前向传播
                        outputs = self.model.decoder(
                            inputs_embeds=current_embeds,
                            attention_mask=current_attention_mask,
                            use_cache=False
                        )
                        
                        # 获取最后一个位置的logits并采样
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        generated_tokens.append(next_token)                
                            
                        # 准备下一轮的输入
                        next_token_embeds = self.model.decoder.transformer.wte(next_token)
                        current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)
                        current_attention_mask = torch.cat([
                            current_attention_mask,
                            torch.ones((batch_size, 1), dtype=torch.long, device=device)
                        ], dim=1)
                    
                    # 收集生成的token IDs
                    if generated_tokens:
                        pred_ids = torch.cat(generated_tokens, dim=1)  # [batch_size, generated_length]
                    
                    predictions.extend(pred_ids.cpu().tolist())
                    references.extend(labels.cpu().tolist())
                    
            self.model.train()
            avg_loss = total_loss / len(eval_dataloader) if len(eval_dataloader) > 0 else 0.0
            
            if predictions and self.compute_metrics:
                pred = type('Pred', (), {})()
                # 不要强制转换为numpy数组，因为长度不一致
                # 让compute_metrics函数内部处理这些列表
                pred.predictions = predictions
                pred.label_ids = references
                metrics = self.compute_metrics(pred)
                print(f"[{desc}] Loss: {avg_loss:.4f}, Metrics: {metrics}")
                return avg_loss, metrics
            else:
                print(f"[{desc}] Loss: {avg_loss:.4f}")
                return avg_loss, {}

    # 创建trainer
    trainer = PrefixCompatibleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=prefix_caption_collate_fn,
        compute_metrics=compute_metrics,
    )

    print("开始ByteFormer + GPT2 Prefix Caption训练...")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(eval_ds)}")
    print(f"训练参数: {training_args}")
    
    print("模型结构:")
    print(model)
    
    # 开始训练
    trainer.train()
    trainer.save_model()
    
    print("训练完成！")
    
    # 生成样例caption
    print("\n生成样例caption...")
    model.eval()
    for i in range(min(3, len(eval_ds))):
        try:
            sample = eval_ds[i]
            img_tensor = sample['image_tensor']
            true_caption = sample['caption']
            
            # Process through pipeline
            corenet_item = {"samples": img_tensor, "targets": torch.tensor(0)}
            collated = byteformer_image_collate_fn([corenet_item], opts)
            input_ids = collated["samples"].unsqueeze(0)
            
            # Generate caption using our custom generate method
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids.to(model.device),
                    max_length=args.max_caption_length,
                )
                generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            print(f"\n样例 {i+1}:")
            print(f"真实caption: {true_caption}")
            print(f"生成caption: {generated_caption}")
            
        except Exception as e:
            print(f"生成样例 {i+1} 时出错: {e}")


if __name__ == "__main__":
    main()

