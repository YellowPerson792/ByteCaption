import os
import numpy as np
import torch
import tqdm
import json
import gc
import evaluation
import losses
import lib.utils as utils
# 使用重构后的 ByteCaption COCO dataloader（支持多模型/多评估模式）
from PureT.datasets_ import data_loader_bytecaption as loader_byte
from PureT.datasets_ import data_loader_visual as loader_visual
from PureT.datasets_ import data_loader_openrouter as loader_openrouter
from lib.config import cfg


class CocoEvaler(object):
    def __init__(self, eval_ids, gv_feat, att_feats, eval_annfile, max_samples=None, enable_eval_loss=False):
        super(CocoEvaler, self).__init__()

    
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)
        self.max_samples = max_samples
        self.enable_eval_loss = enable_eval_loss
        
        # 预先检查数据集是否支持损失计算（只检查一次）
        self.loss_computation_ready = False

        # Build eval ids
        if cfg.INFERENCE.EVAL == 'COCO':
            with open(eval_ids, 'r') as f:
                self.ids2path = json.load(f)
                self.eval_ids = np.array(list(self.ids2path.keys()))
        else:
            self.ids2path = None
            self.eval_ids = None  # set after loader

        # Build loader based on model type
        model_type = str(getattr(cfg.MODEL, "TYPE", "")).lower()
        is_openrouter = ("openrouter" in model_type) or model_type.startswith("gpt") or ("gpt" in model_type)
        is_hf_visual = (
            model_type.startswith("hf")
            or "blip" in model_type
            or "git" in model_type
            or "qwen" in model_type
            or "internvl" in model_type
            or "glm" in model_type
            or "mistral" in model_type
            or "ministral" in model_type
        )
        if is_openrouter:
            self.eval_loader = loader_openrouter.load_val(eval_ids, gv_feat, max_samples=self.max_samples)
        elif is_hf_visual:
            self.eval_loader = loader_visual.load_val(eval_ids, gv_feat, max_samples=self.max_samples)
        else:
            self.eval_loader = loader_byte.load_val(eval_ids, gv_feat, max_samples=self.max_samples)
        if self.eval_ids is None:
            # For Flickr8k, load the actual image IDs from the JSON file if available
            if eval_ids and os.path.exists(eval_ids):
                with open(eval_ids, 'r') as f:
                    ids_data = json.load(f)
                    # Convert string keys to integers for consistency
                    self.eval_ids = np.array([int(k) for k in ids_data.keys()])
                    print(f"Loaded {len(self.eval_ids)} evaluation IDs from {eval_ids}")
                    print(f"ID range: {self.eval_ids.min()} to {self.eval_ids.max()}")
            else:
                # Fallback to sequential IDs when no file is provided (HuggingFace mode)
                self.eval_ids = np.arange(len(self.eval_loader.dataset))
                print(f"Using sequential IDs: 0 to {len(self.eval_ids)-1} (HuggingFace mode)")

        # Apply max_samples limit if specified (should already be handled by load_val)
        if self.max_samples is not None and self.max_samples > 0:
            original_size = len(self.eval_ids)
            if original_size > self.max_samples:
                self.eval_ids = self.eval_ids[:self.max_samples]
                print(f"Evaluation: Limited to {len(self.eval_ids)} samples (from {original_size})")

        # Use HF-based evaluator that doesn't need annotation files
        # Infer split from eval_ids path, or default to validation
        if eval_ids and os.path.exists(eval_ids):
            basename = os.path.basename(str(eval_ids)).lower()
            if 'val' in basename or 'valid' in basename:
                split = 'validation'
            elif 'test' in basename:
                split = 'test'
            else:
                split = 'train'
        else:
            # Default to validation when no eval_ids path is provided
            split = 'validation'
        self.evaler = evaluation.create('COCO', split)
            
        
        # Post-initialization check for loss computation compatibility
        if self.enable_eval_loss:
            # Ensure dataset has required attributes for loss computation
            dataset = self.eval_loader.dataset
            
            # Ensure vocabulary mapping exists
            if not hasattr(dataset, 'w2i'):
                if hasattr(dataset, 'vocab'):
                    dataset.w2i = {w: i for i, w in enumerate(dataset.vocab)}
                else:
                    print("Warning: eval_loss disabled because dataset lacks vocabulary")
                    self.loss_computation_ready = False
                    return
            
            # Ensure sequence length is set
            if not hasattr(dataset, 'seq_len'):
                dataset.seq_len = int(getattr(cfg.MODEL, 'SEQ_LEN', 17))
            
            # Check if dataset supports caption-based sequence building
            if hasattr(dataset, 'cocofmt_annfile'):
                if dataset.cocofmt_annfile is None:
                    print("Warning: eval_loss disabled because dataset doesn't provide captions in COCO format")
                    self.loss_computation_ready = False
                else:
                    self.loss_computation_ready = True
            else:
                # For HuggingFace dataset, loss computation is always ready
                self.loss_computation_ready = True
        else:
            self.loss_computation_ready = False

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs

    def __call__(self, model, rname):
        model.eval()

        results = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 构造XE损失计算器（仅在启用时）
        if self.enable_eval_loss:
            xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).to(device)
        else:
            xe_criterion = None
        global_idx = 0
        # Accumulate XE loss over batches (weighted by batch size) so we can return/print it with metrics
        loss_sum = 0.0
        loss_count = 0

        undecodable_count = 0
        # --- 获取对数据集内部ID列表的引用(与之前获取的方式有一点不同，我觉得有道理，因为之前加了数据样本后需要重新对齐一下) ---
        dataset_image_ids = self.eval_loader.dataset.image_ids
        
        # 获取损坏类型列表以便在结果中标记
        corruption_types = list(getattr(cfg.CORRUPTION, "BYTE_STREAM_TYPES", []))
        corruption_level = str(getattr(cfg.CORRUPTION, "BYTE_STREAM_LEVEL", "S0"))
        
        sample_limit = getattr(getattr(cfg, "INFERENCE", {}), "SAMPLE_PREVIEW", 5)
        sample_preview = []

        with torch.no_grad():
            pbar = tqdm.tqdm(self.eval_loader, desc=f"Evaluating {rname} ({cfg.MODEL.TYPE})", leave=False)
            for batch_idx, data_batch in enumerate(pbar):
                indices, gv_feat, data, att_mask = data_batch
                # 将 numpy 索引转换为整数列表
                int_indices = indices.astype(int)
                # 使用列表推导式进行查找
                ids = [dataset_image_ids[i] for i in int_indices]

                gv_feat = gv_feat.to(device)
                
                # --- 根据模型类型选择数据并传递 ---
                # ByteFormer 需要 Tensor，BLIP 需要 PIL Image 列表
                model_type = str(getattr(cfg.MODEL, "TYPE", "")).lower()
                is_byteformer = "byteformer" in model_type or model_type == "puret"
                if is_byteformer:
                    att_feats = data.to(device)
                    if att_mask is not None:
                        att_mask = att_mask.to(device)
                else: # BLIP / HF
                    att_feats = data 
                
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
                
                # 统一的解码调用 (我们专门设置了 BlipWrapper 保证了接口一致)
                m = getattr(model, 'module', model)
                if kwargs['BEAM_SIZE'] > 1:
                    decoded_output, _ = m.decode_beam(**kwargs)
                else:
                    decoded_output, _ = m.decode(**kwargs)

                # 使用统一的逻辑将 decoded_output 转换为字符串列表 sents
                if isinstance(decoded_output, list) and decoded_output and (len(decoded_output) == 0 or isinstance(decoded_output[0], str)):
                    # 如果是 BLIP 返回的字符串列表
                    sents = decoded_output
                else:
                    # 如果是 ByteFormer 返回的 Tensor
                    # 立即转到 CPU 并释放 GPU 张量
                    decoded_output_cpu = decoded_output.cpu()
                    sents = utils.decode_sequence(self.vocab, decoded_output_cpu.data)
                    del decoded_output, decoded_output_cpu

                # 尝试构建输入/目标序列以计算XE loss（仅当启用且数据集支持时）
                batch_loss = None
                if self.loss_computation_ready and xe_criterion is not None and is_byteformer:
                    try:
                        # 评估模式下需要手动构建序列，因为数据集只返回 (indices, gv_feat, att_feats)
                        dataset = self.eval_loader.dataset
                        # 校验数据集是否具备所需方法
                        if not hasattr(dataset, '_build_seqs_from_captions'):
                            raise AttributeError('Dataset lacks _build_seqs_from_captions; cannot compute eval XE loss')

                        input_list = []
                        target_list = []
                        for idx in indices:
                            # 取出原始 sample （COCO 重构后 _build_seqs_from_captions 需要传入 sample 字典）
                            sample = dataset.ds[int(idx)] if hasattr(dataset, 'ds') else None
                            if sample is None:
                                raise ValueError('Unable to retrieve sample for XE loss computation')
                            in_arr, tgt_arr = dataset._build_seqs_from_captions(sample)
                            # _build_seqs_from_captions 返回 (seq_per_img, seq_len)
                            # 对于评估，我们需要所有 seq_per_img 个序列来匹配模型的设置
                            for seq_idx in range(cfg.DATA_LOADER.SEQ_PER_IMG):
                                if seq_idx < in_arr.shape[0]:
                                    input_list.append(in_arr[seq_idx])
                                    target_list.append(tgt_arr[seq_idx])
                                else:
                                    # 如果序列不足，重复最后一个
                                    input_list.append(in_arr[-1])
                                    target_list.append(tgt_arr[-1])

                        if len(input_list) > 0:
                            input_seq = torch.from_numpy(np.stack(input_list, 0)).long().to(device)
                            target_seq = torch.from_numpy(np.stack(target_list, 0)).long().to(device)

                            # 构建损失计算所需的kwargs
                            loss_kwargs = dict(kwargs)
                            loss_kwargs[cfg.PARAM.INPUT_SENT] = input_seq
                            loss_kwargs[cfg.PARAM.TARGET_SENT] = target_seq

                            # 前向得到 log-probs
                            logit = m(**loss_kwargs)

                            # logit形状应该是 [batch_size*seq_per_img, seq_len, vocab_size]
                            if logit.dim() == 3:
                                batch_loss, _batch_loss_info = xe_criterion(logit, target_seq)
                            else:
                                batch_loss = None

                            # accumulate weighted by batch size (number of sequences)
                            if batch_loss is not None:
                                try:
                                    bs = int(target_seq.size(0))
                                    loss_sum += float(batch_loss.item()) * bs
                                    loss_count += bs
                                except Exception:
                                    pass
                            
                            # 立即释放这些临时张量
                            del input_seq, target_seq, logit
                    except Exception as e:
                        # 仅在第一个批次打印详细错误信息用于调试
                        if len(results) == 0:  # 第一个批次
                            print(f"[信息] XE Loss计算已禁用: {type(e).__name__} - {str(e)}")
                        batch_loss = None

                # --- 关键修复：修改 image_id 以区分不同的损坏样本 ---
                # 1. 获取原始批次大小和增强因子
                # 2. 循环并创建带有唯一ID的结果
                # 判定“不可解码”占位符：兼容旧字符串和当前配置占位符
                placeholder_set = {
                    str(getattr(cfg.MODEL.HF, "PLACEHOLDER", "")).strip().lower(),
                    str(getattr(cfg.MODEL.OPENROUTER, "PLACEHOLDER", "")).strip().lower(),
                }
                placeholder_set = {p for p in placeholder_set if p}

                seen_counts = {}
                for sid, sent in enumerate(sents):
                    sent_norm = str(sent).strip().lower()
                    if sent_norm in placeholder_set:
                        undecodable_count += 1
                    # 使用从数据加载器获得的 ids（已正确映射）
                    if sid < len(ids):
                        original_image_id = ids[sid]
                        # 确保转换为整数以便查找
                        try:
                            original_image_id = int(original_image_id)
                        except (ValueError, TypeError):
                            pass
                    else:
                        original_image_id = ids[-1] if ids else 0
                    
                    augmentation_idx = seen_counts.get(original_image_id, 0)
                    seen_counts[original_image_id] = augmentation_idx + 1
                    
                    # 直接使用原始 image_id（不修改）以保证评估正确性
                    # 如果需要区分增强版本，可以在其他地方处理，但不在这里修改用于评估的ID
                    final_image_id = original_image_id

                    result = {cfg.INFERENCE.ID_KEY: final_image_id, cfg.INFERENCE.CAP_KEY: sent}
                    
                    # 添加损坏类型标记（当有多种损坏类型时）
                    if len(corruption_types) > 1 and augmentation_idx < len(corruption_types):
                        corruption_info = f"{corruption_types[augmentation_idx]}_{corruption_level}"
                        result["corruption"] = corruption_info
                    
                    results.append(result)

                    # 后续的打印逻辑也需要使用新的 unique_id 来查找参考标题
                    if len(sample_preview) < sample_limit:
                        # 使用 original_image_id 来查找参考标题
                        gt_captions = []
                        if hasattr(self.evaler, 'id_to_captions') and original_image_id in self.evaler.id_to_captions:
                            gt_captions = self.evaler.id_to_captions[original_image_id]
                        elif hasattr(self.evaler, 'id_to_captions'):
                            # 尝试其他可能的ID格式
                            for key_candidate in [str(original_image_id), int(original_image_id)]:
                                if key_candidate in self.evaler.id_to_captions:
                                    gt_captions = self.evaler.id_to_captions[key_candidate]
                                    break
                        
                        # 如果仍然没找到，尝试从coco_data搜索
                        if not gt_captions and hasattr(self.evaler, 'coco_data'):
                            for ann in self.evaler.coco_data.get('annotations', []):
                                if ann.get('image_id') == original_image_id:
                                    gt_captions.append(ann['caption'])
                        
                        gt_str = gt_captions[0] if gt_captions else "N/A"

                        sample_preview.append(
                            {
                                "image_id": int(original_image_id),
                                "generated": sent,
                                "reference": gt_str,
                                "augmentation_idx": int(augmentation_idx),
                            }
                        )
                        
                        pbar.clear()
                        print(f"\n[Eval Sample {global_idx} (Original ID: {original_image_id}, Aug: {augmentation_idx})]")
                        print(f"  Generated: {sent}")
                        print(f"  Reference: {gt_str}")
                        if batch_loss is not None:
                            try:
                                print(f"  XE Loss  : {batch_loss.item():.4f}")
                            except Exception:
                                print(f"  XE Loss  : {batch_loss}")
                        else:
                            print(f"  XE Loss  : N/A")
                        print("  " + "─" * 50)
                        
                    global_idx += 1
                # --- 修复结束 ---
                
                # 【关键优化】每个批次后立即清理 GPU 内存
                # 删除不再需要的张量引用
                if is_byteformer:
                    # ByteFormer 的 att_feats 是 GPU 张量，需要删除
                    del att_feats
                    if att_mask is not None:
                        del att_mask
                # gv_feat 总是 GPU 张量
                del gv_feat
                if batch_loss is not None:
                    del batch_loss
                
                # 每 10 个批次强制清理一次缓存，防止显存累积
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 评估循环结束，最终清理 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # 清理 xe_criterion
        if xe_criterion is not None:
            del xe_criterion

        # Evaluate (capture stdout to avoid duplicate printing)
        import sys
        from io import StringIO
        
        # Temporarily redirect stdout to capture the evaluator's print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        eval_res = self.evaler.eval(results)
        
        # Restore stdout
        sys.stdout = old_stdout

        # Attach averaged XE loss to eval_res
        if loss_count > 0:
            avg_loss = loss_sum / loss_count
            try:
                eval_res['XE_Loss'] = float(avg_loss)
            except Exception:
                eval_res['XE_Loss'] = avg_loss
        else:
            eval_res['XE_Loss'] = None

        # Print beautiful unified summary (only once)
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS - {rname.upper()}")
        print(f"{'='*80}")
        
        # Group metrics for better display
        bleu_metrics = {}
        other_metrics = {}
        
        for k, v in eval_res.items():
            if k.startswith('Bleu_'):
                bleu_metrics[k] = v
            else:
                other_metrics[k] = v
        
        # Display BLEU scores in one line
        if bleu_metrics:
            bleu_str = " | ".join([f"{k}: {v:.4f}" for k, v in bleu_metrics.items()])
            print(f"BLEU Scores:  {bleu_str}")
        
        # Display other metrics
        for k, v in other_metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                print(f"{k:12}: {v:.4f}")
            elif v is not None:
                print(f"{k:12}: {v}")
        
        print(f"{'='*80}")
        print(f"Total samples evaluated: {len(results)}")
        print(f"Undecodable images (skipped): {undecodable_count}")
        
        # --- START: 关键修复 ---
        # 修复 f-string 格式化错误。先计算比率，再进行格式化输出。
        ratio = undecodable_count / len(results) if len(results) > 0 else 0.0
        print(f"the ratio of undecodable images: {ratio:.4f}")
        # --- END: 关键修复 ---

        # 额外统计，便于批量实验脚本读取
        eval_res['Undecodable_Count'] = undecodable_count
        eval_res['Undecodable_Ratio'] = ratio
        eval_res['Total_Samples'] = len(results)
        if sample_preview:
            eval_res['Sample_Preview'] = sample_preview
        
        # --- START: 添加码流长度统计报告 ---
        # Only ByteCaption byte-stream loader tracks lengths
        lengths_src = getattr(loader_byte, "_BYTE_STREAM_LENGTHS", None)
        if isinstance(lengths_src, list) and len(lengths_src) > 0:
            lengths = np.array(lengths_src)
            count_total = len(lengths)
            count_below_20k = np.sum(lengths < 20000)

            print(f"{'-'*30}")
            print("Byte Stream Length Statistics:")
            print(f"  - Total Images Processed: {count_total}")
            print(f"  - Average Length: {np.mean(lengths):.2f} bytes")
            print(f"  - Max Length: {np.max(lengths)} bytes")
            print(f"  - Min Length: {np.min(lengths)} bytes")
            print(f"  - Median Length: {np.median(lengths):.2f} bytes")
            print(f"  - Images with length < 20000: {count_below_20k} ({count_below_20k / count_total:.2%})")
            print(f"{'-'*30}")
            # 清空列表以便下次评估（如果在一个进程中多次调用）
            try:
                lengths_src.clear()
            except Exception:
                pass
        # --- END: 添加码流长度统计报告 ---

        print(f"{'='*80}\n")

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname + '.json'), 'w'))

        model.train() # 恢复模型状态
        return eval_res
