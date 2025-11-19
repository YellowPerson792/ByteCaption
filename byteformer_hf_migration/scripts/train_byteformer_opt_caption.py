"""
ByteFormer + OPT-125M Caption Training Script
使用ByteFormer作为encoder，OPT-125M作为decoder实现图像描述生成任务

示例运行命令：
python byteformer-hf-migration/scripts/train_byteformer_opt_caption.py --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --num_train_epochs 5 --learning_rate 5e-5 --warmup_ratio 0.01 --eval_steps 40 --logging_steps 50 --save_steps 600 --lr_scheduler_type cosine --gradient_accumulation_steps 2 --report_to none --max_caption_length 16 --num_eval_samples 50 --fp16 --opt_model facebook/opt-125m
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
from datasets import load_dataset, load_from_disk
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Union
from corenet.options.opts import get_training_arguments
from utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel
from corenet.data.transforms.image_bytes import PILSave
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from utils.hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments
from transformers import OPTForCausalLM, OPTConfig, AutoTokenizer, EncoderDecoderModel, GenerationConfig
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.generation.utils import GenerationMixin
from transformers import PreTrainedModel
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

    def get_output_embeddings(self):
        return None
    def set_output_embeddings(self, x):
        pass
    def gradient_checkpointing_enable(self):
        pass
    def gradient_checkpointing_disable(self):
        pass
    def _set_gradient_checkpointing(self, module, value):
        pass
    def tie_weights(self):
        pass

class CustomEncoderDecoderModel(EncoderDecoderModel):
    """
    自定义的EncoderDecoderModel，正确处理ByteFormer输出的attention mask
    """
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop("num_items_in_batch", None)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]
        encoder_attention_mask = getattr(encoder_outputs, "encoder_attention_mask", None)

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            if decoder_attention_mask is None:
                decoder_attention_mask = decoder_input_ids.new_tensor(decoder_input_ids != self.config.pad_token_id)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,  # 使用ByteFormer的attention_mask
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

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
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class CustomOPTForCausalLM(OPTForCausalLM):
    """Custom OPTForCausalLM with proper reorder_cache implementation for beam search."""
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorder the cache for beam search.
        
        Args:
            past_key_values: tuple of tuples, each inner tuple contains (key, value) tensors
                            with shape [batch_size * num_beams, num_heads, seq_len, head_dim]
            beam_idx: tensor of shape [batch_size * num_beams] with beam indices
        
        Returns:
            Reordered cache in the same format
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # Each layer_past is a tuple of (key, value) tensors
            reordered_layer_past = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device)) 
                for past_state in layer_past
            )
            reordered_past += (reordered_layer_past,)
        return reordered_past
    
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ByteFormer + OPT Caption Training")
    parser.add_argument("--config", type=str, default="byteformer-hf-migration/configs/conv_kernel_size=4,window_sizes=[128].yaml", help="CoreNet配置文件路径")
    parser.add_argument("--weights", type=str, default="byteformer-hf-migration/weights/imagenet_jpeg_q60_k4_w128.pt", help="预训练权重文件路径")
    parser.add_argument("--opt_model", type=str, default="facebook/opt-125m", help="OPT模型名称")
    parser.add_argument("--dataset", type=str, default="jxie/flickr8k", help="数据集名称")
    parser.add_argument("--num_train_samples", type=int, default=None, help="训练样本数量（None表示使用全部训练数据）")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="评估样本数量（None表示使用全部验证数据）")
    parser.add_argument("--max_caption_length", type=int, default=50, help="最大caption长度")
    parser.add_argument("--max_byteformer_length", type=int, default=2048, help="ByteFormer最大输入长度")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/byteformer_opt_caption", help="训练输出目录")
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
    parser.add_argument("--pretrained_weights", type=str, default=None, help="预训练权重目录，用于继续训练")
    parser.add_argument("--report_to", type=str, default=None, choices=[None, "none", "wandb", "tensorboard"], help="日志报告工具")
    return parser.parse_args()
    
def load_trained_weights(model_path, model):
    try:
        # 尝试不同的权重文件路径
        weight_files = [
            f"{model_path}/pytorch_model.bin",
            f"{model_path}/model.safetensors",
            f"{model_path}/pytorch_model.safetensors"
        ]
        
        loaded = False
        for weight_file in weight_files:
            if os.path.exists(weight_file):
                print(f"Loading trained weights from {weight_file}")
                if weight_file.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    trained_weights = load_file(weight_file)
                else:
                    trained_weights = torch.load(weight_file, map_location='cpu')
                
                # 先绑定OPT的词嵌入和输出层权重
                model.decoder.tie_weights()
                
                # 加载权重，允许部分加载
                missing_keys, unexpected_keys = model.load_state_dict(trained_weights, strict=False)
                print(f"Loaded trained weights successfully!")
                
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)} (these will use initialization values)")
                    print("Missing key names:", missing_keys)
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)} (these will be ignored)")
                loaded = True
                break
        
        if not loaded:
            print(f"No weight file found in {model_path}, using initialization weights only")
            
    except Exception as e:
        print(f"Failed to load trained weights: {e}")
        print("Using initialization weights only")
    return model
        
# 图像预处理
def pil_to_tensor_transform(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img)

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", num_samples=None, dataset="jxie/flickr8k"):
        
        # 直接使用datasets对象，不预先加载所有数据
        self.dataset = load_dataset(dataset, split=split)
        self.dataset_name = dataset
        self.split = split
        self.total_samples = len(self.dataset) * 5
        if num_samples is not None and num_samples < self.total_samples:
            self.total_samples = num_samples
            print(f"使用 {split} 数据集的前 {num_samples} 个样本")
        else:
            print(f"使用完整的 {split} 数据集，共 {self.total_samples} 个样本")
            
    def __getitem__(self, idx):
        # 动态计算对应的数据集索引和caption索引
        dataset_idx = idx // 5  # 每个图片对应5个caption
        caption_idx = idx % 5   # caption编号 0-4
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
    
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, split="train", num_samples=None, dataset="/root/autodl-fs/AbdoTW___coco_2014"):
        # 直接使用datasets对象，不预先加载所有数据
        self.dataset = load_from_disk(f"{dataset}/{split}")
        self.dataset_name = dataset
        self.split = split
        self.total_samples = len(self.dataset) * 5
        if num_samples is not None and num_samples < self.total_samples:
            self.total_samples = num_samples
            print(f"使用 {split} 数据集的前 {num_samples} 个样本")
        else:
            print(f"使用完整的 {split} 数据集，共 {self.total_samples} 个样本")
            
    def __getitem__(self, idx):
        # 动态计算对应的数据集索引和caption索引
        dataset_idx = idx // 5  # 每个图片对应5个caption
        caption_idx = idx % 5   # caption编号 0-4
        item = self.dataset[dataset_idx]
        img = item["image"] if "image" in item else item["jpg"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = pil_to_tensor_transform(img)
        caption = item["caption"][caption_idx] if isinstance(item["caption"], list) else item["caption"]
        return {
            'image_tensor': img_tensor,
            'caption': caption
        }
        
    def __len__(self):
        return self.total_samples

def main():
    args = parse_args()
    
    print("ByteFormer + OPT Caption Training")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"预训练权重: {args.weights}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_train_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"批大小: {args.per_device_train_batch_size}")
    print(f"数据集: {args.dataset}")
    print(f"OPT模型: {args.opt_model}")
    print("=" * 50)
    
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
    
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    byteformer_model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    weights = torch.load(args.weights, map_location='cpu')
    # 加载backbone部分权重
    model_state = byteformer_model.model.state_dict()
    pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
    byteformer_model.model.load_state_dict(pretrained_state, strict=False)
    
    byteformer_encoder = byteformer_model.model
    
    # Remove the classifier if it exists
    if hasattr(byteformer_encoder, 'classifier'):
        delattr(byteformer_encoder, 'classifier')
    
    # 初始化OPT decoder - 从本地.bin文件转换为safetensors格式避免安全问题
    opt_config = OPTConfig.from_pretrained(args.opt_model)
    opt_config.add_cross_attention = True
    
    # 创建本地 safetensors 目录
    safe_model_dir = f"./cache/opt_safetensors_{args.opt_model.split('/')[-1]}"
    
    def convert_bin_to_safetensors(model_name, safe_dir):
        """将本地缓存的.bin文件转换为safetensors格式"""
        print(f"Converting {model_name} from .bin to safetensors...")
        
        # 检查Hugging Face缓存目录中的模型文件
        from transformers.utils import TRANSFORMERS_CACHE
        import glob
        
        # 查找模型在缓存中的位置
        cache_dirs = glob.glob(f"{TRANSFORMERS_CACHE}/models--*{model_name.replace('/', '--')}*")
        if not cache_dirs:
            raise FileNotFoundError(f"Cannot find cached model for {model_name}")
        
        model_cache_dir = cache_dirs[0]
        print(f"Found cached model at: {model_cache_dir}")
        
        # 查找snapshots目录中的最新版本
        snapshots_dir = os.path.join(model_cache_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                           if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshot_dirs:
                latest_snapshot = sorted(snapshot_dirs)[-1]  # 取最新的
                model_files_dir = os.path.join(snapshots_dir, latest_snapshot)
            else:
                raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
        else:
            model_files_dir = model_cache_dir
        
        # 查找.bin文件
        bin_files = glob.glob(os.path.join(model_files_dir, "*.bin"))
        if not bin_files:
            raise FileNotFoundError(f"No .bin files found in {model_files_dir}")
        
        print(f"Found .bin files: {bin_files}")
        
        # 创建模型并加载.bin权重
        temp_model = CustomOPTForCausalLM(opt_config)
        
        # 如果有多个.bin文件，需要合并加载
        state_dict = {}
        for bin_file in bin_files:
            print(f"Loading weights from: {bin_file}")
            weights = torch.load(bin_file, map_location='cpu')
            state_dict.update(weights)
        
        # 加载权重到模型
        temp_model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded weights from .bin files")
        
        # 保存为 safetensors 格式
        os.makedirs(safe_dir, exist_ok=True)
        temp_model.save_pretrained(safe_dir, safe_serialization=True)
        
        # 复制其他必要文件（config.json, tokenizer等）
        import shutil
        for file_name in ["config.json", "tokenizer.json", "tokenizer_config.json", 
                         "vocab.json", "merges.txt", "special_tokens_map.json"]:
            src_file = os.path.join(model_files_dir, file_name)
            if os.path.exists(src_file):
                dst_file = os.path.join(safe_dir, file_name)
                shutil.copy2(src_file, dst_file)
                print(f"Copied {file_name}")
        
        print(f"Conversion completed! Safetensors model saved to: {safe_dir}")
        del temp_model
        torch.cuda.empty_cache()
        return safe_dir
    
    if not os.path.exists(safe_model_dir):
        try:
            convert_bin_to_safetensors(args.opt_model, safe_model_dir)
        except Exception as e:
            print(f"Failed to convert .bin to safetensors: {e}")
            print("Falling back to creating model from config...")
            
            # 备用方案：从配置创建模型
            os.makedirs(safe_model_dir, exist_ok=True)
            temp_model = CustomOPTForCausalLM(opt_config)
            temp_model.save_pretrained(safe_model_dir, safe_serialization=True)
            
            # 保存 tokenizer
            temp_tokenizer = AutoTokenizer.from_pretrained(args.opt_model)
            temp_tokenizer.save_pretrained(safe_model_dir)
            
            del temp_model, temp_tokenizer
            torch.cuda.empty_cache()
    
    # 从本地 safetensors 加载
    print(f"Loading OPT model from safetensors: {safe_model_dir}")
    opt_decoder = CustomOPTForCausalLM.from_pretrained(
        safe_model_dir, 
        config=opt_config,
        use_safetensors=True
    )
    
    encoder_config = CorenetToHFPretrainedConfig(**vars(opts))
    wrapped_encoder = ByteFormerWrapper(byteformer_encoder, encoder_config)
    
    model = CustomEncoderDecoderModel(encoder=wrapped_encoder, decoder=opt_decoder)
    
    # 绑定OPT的词嵌入和输出层权重
    model.decoder.tie_weights()
    
    tokenizer = AutoTokenizer.from_pretrained(args.opt_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id  
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = tokenizer.vocab_size
    model.main_input_name = "input_ids"
    
    if args.pretrained_weights:
        model = load_trained_weights(args.pretrained_weights, model)
    
    def caption_collate_fn(batch):
        images = []
        captions = [] 
        for item in batch:
            images.append(item['image_tensor'])
            captions.append(item['caption'])
            
        corenet_batch = []
        for img_tensor in images:
            corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target
        collated = byteformer_image_collate_fn(corenet_batch, opts)
        input_ids = collated["samples"]
        
        caption_tokens = tokenizer(
            captions,
            padding='longest',
            max_length=args.max_caption_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = caption_tokens.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    rouge = evaluate.load("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        
        # Handle the case where pred_ids and labels_ids are lists of lists with different lengths
        # Don't convert to numpy arrays directly as they have irregular shapes
        
        # Decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        
        # Replace -100 with pad token for decoding in labels
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        processed_labels = []
        for label_seq in labels_ids:
            processed_seq = [pad_token_id if token_id == -100 else token_id for token_id in label_seq]
            processed_labels.append(processed_seq)
        
        label_str = tokenizer.batch_decode(processed_labels, skip_special_tokens=True)
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
                print(f"  Prediction: {pred}\n")
        
        return results
    
    # Setup generation config
    generation_config = GenerationConfig(
        max_length=args.max_caption_length,
        num_beams=5,  # 使用beam search
        decoder_start_token_id=model.config.decoder_start_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        length_penalty=1.0,      # 长度惩罚
        early_stopping=True,     # 早停
        do_sample=False,         # 使用贪心搜索而不是采样（对beam search很重要）
    )
    model.generation_config = generation_config

    if args.dataset == "/root/autodl-fs/AbdoTW___coco_2014":
        train_ds = COCODataset(split="train", num_samples=args.num_train_samples, dataset=args.dataset)
        eval_ds = COCODataset(split="validation", num_samples=args.num_eval_samples, dataset=args.dataset)
    else:
        train_ds = CaptionDataset(split="train", num_samples=args.num_train_samples, dataset=args.dataset)
        eval_ds = CaptionDataset(split="test", num_samples=args.num_eval_samples, dataset=args.dataset)
    
    # 计算总训练步数，实现真正的 warmup_ratio
    train_batch_size = args.per_device_train_batch_size
    num_train_epochs = args.num_train_epochs
    num_train_samples = len(train_ds)
    steps_per_epoch = (num_train_samples + train_batch_size - 1) // train_batch_size
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
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

    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=caption_collate_fn,
        compute_metrics=compute_metrics,
    )
    
    print("开始ByteFormer + OPT Caption训练...")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(eval_ds)}")
    print(f"训练参数: {training_args}")
    
    print("模型结构:")
    print(model)
    
    # Start training
    trainer.train()
    trainer.save_model()
    
    print("训练完成！")
    
    print("\\n生成样例caption...")
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
            
            # Generate caption
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs=input_ids.to(model.device),
                    generation_config=generation_config,
                )
                generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            print(f"\\n样例 {i+1}:")
            print(f"真实caption: {true_caption}")
            print(f"生成caption: {generated_caption}")
            
        except Exception as e:
            print(f"生成样例 {i+1} 时出错: {e}")

if __name__ == "__main__":
    main()
