import os
import numpy as np
import torch
import tqdm
import json
import evaluation
import losses
import lib.utils as utils
import datasets_.data_loader_byteformer_coco as data_loader
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

        # Build loader (uses ids_path basename to infer split)
        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats, max_samples=self.max_samples)
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
        # output_indices = {0, 5, 10, 15, 20, 25}
        # 初始化无法解码图像的计数器
        undecodable_count = 0
        # --- 获取对数据集内部ID列表的引用(与之前获取的方式有一点不同，我觉得有道理，因为之前加了数据样本后需要重新对齐一下) ---
        dataset_image_ids = self.eval_loader.dataset.image_ids

        with torch.no_grad():
            pbar = tqdm.tqdm(self.eval_loader, desc=f"Evaluating {rname} ({cfg.MODEL.TYPE})", leave=False)
            for _, data_batch in enumerate(pbar):
                indices, gv_feat, data, att_mask = data_batch
                
                # --- 执行可靠的ID查找 ---
                # 使用从 DataLoader 获得的 indices，去数据集的 image_ids 列表中查找
                # 这比使用 self.eval_ids 更安全，因为它保证了顺序的一致性
                try:
                    # 将 numpy 索引转换为整数列表
                    int_indices = indices.astype(int)
                    # 使用列表推导式进行查找
                    ids = [dataset_image_ids[i] for i in int_indices]
                except IndexError:
                    # 如果发生错误，回退到旧的不安全方法，并打印警告
                    print("\n[警告] ID查找失败，回退到旧方法。评估结果可能不准确。")
                    ids = self.eval_ids[indices]
                
                gv_feat = gv_feat.to(device)
                
                # --- 根据模型类型选择数据并传递 ---
                # ByteFormer 需要 Tensor，BLIP 需要 PIL Image 列表
                if cfg.MODEL.TYPE == 'PureT_byteformer':
                    att_feats = data.to(device)
                    if att_mask is not None:
                        att_mask = att_mask.to(device)
                else: # BLIP
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
                    sents = utils.decode_sequence(self.vocab, decoded_output.data)

                # 尝试构建输入/目标序列以计算XE loss（仅当启用且数据集支持时）
                # batch_loss = None
                if self.loss_computation_ready and xe_criterion is not None:
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
                            # 对于评估，我们需要所有5个序列来匹配模型的seq_per_img设置
                            for seq_idx in range(cfg.DATA_LOADER.SEQ_PER_IMG):
                                if seq_idx < in_arr.shape[0]:
                                    input_list.append(in_arr[seq_idx])
                                    target_list.append(tgt_arr[seq_idx])
                                else:
                                    # 如果序列不足5个，重复最后一个
                                    input_list.append(in_arr[-1])
                                    target_list.append(tgt_arr[-1])
                        
                        if len(input_list) > 0:
                            input_seq = torch.from_numpy(np.stack(input_list, 0)).long().to(device)
                            target_seq = torch.from_numpy(np.stack(target_list, 0)).long().to(device)
                            
                            # 现在input_seq和target_seq的形状应该是 [batch_size*seq_per_img, seq_len]
                            # 这与模型期望的维度匹配
                            
                            # 构建损失计算所需的kwargs
                            loss_kwargs = dict(kwargs)
                            loss_kwargs[cfg.PARAM.INPUT_SENT] = input_seq
                            loss_kwargs[cfg.PARAM.TARGET_SENT] = target_seq
                            
                            # 不需要修改seq_per_img，因为现在序列数量已经匹配了
                            m = getattr(model, 'module', model)
                            # 前向得到 log-probs
                            logit = m(**loss_kwargs)
                            
                            # logit形状应该是 [batch_size*seq_per_img, seq_len, vocab_size]
                            if logit.dim() == 3:  # [batch*seq_per_img, seq_len, vocab_size]
                                batch_loss, batch_loss_info = xe_criterion(logit, target_seq)
                            else:
                                # 如果维度不对，跳过损失计算
                                batch_loss = None
                                
                            # accumulate weighted by batch size (number of sequences)
                            if batch_loss is not None:
                                try:
                                    # 注意：现在序列数量是 batch_size * seq_per_img
                                    bs = int(target_seq.size(0))
                                    loss_sum += float(batch_loss.item()) * bs
                                    loss_count += bs
                                except Exception:
                                    pass
                    except Exception as e:
                        # 仅在第一个批次打印详细错误信息用于调试
                        if len(results) == 0:  # 第一个批次
                            print(f"[信息] XE Loss计算已禁用: {type(e).__name__} - {str(e)}")
                        batch_loss = None

                # --- 关键修复：修改 image_id 以区分不同的损坏样本 ---
                # 1. 获取原始批次大小和增强因子
                original_bs = len(indices) // (len(sents) // len(indices)) if len(indices) > 0 and len(sents) > 0 else len(indices)
                augmentation_factor = len(sents) // original_bs if original_bs > 0 else 1

                # 2. 循环并创建带有唯一ID的结果
                for sid, sent in enumerate(sents):
                    # 计算这个样本在原始批次中的索引
                    if sent == "this is a dummy caption for an undecodable image":
                        undecodable_count += 1
                    original_sample_idx = sid // augmentation_factor
                    # 计算这是第几个增强版本 (0, 1, 2, 3...)
                    augmentation_idx = sid % augmentation_factor
                    
                    # 获取原始的 image_id
                    original_image_id = int(ids[original_sample_idx])
                    
                    # 创建一个新的、唯一的 image_id。
                    # 例如，ID 123 的第2个增强版本 -> 12302
                    # 我们使用一个足够大的偏移量以避免与真实ID冲突
                    unique_image_id = original_image_id * 100 + augmentation_idx

                    # --- 调试语句：打印ID转换过程 ---
                    if global_idx < 20: # 只打印前几个样本的调试信息
                        print(f"[DEBUG ID] sid: {sid}, original_idx: {original_sample_idx}, aug_idx: {augmentation_idx}, original_id: {original_image_id}, unique_id: {unique_image_id}")
                    # ------------------------------------

                    result = {cfg.INFERENCE.ID_KEY: original_image_id, cfg.INFERENCE.CAP_KEY: sent}
                    results.append(result)

                    # 后续的打印逻辑也需要使用新的 unique_id 来查找参考标题
                    if global_idx < 5:
                        # 使用 original_image_id 来查找参考标题
                        gt_captions = []
                        if hasattr(self.evaler, 'id_to_captions') and original_image_id in self.evaler.id_to_captions:
                            gt_captions = self.evaler.id_to_captions[original_image_id]
                        elif hasattr(self.evaler, 'coco_data'):
                            for ann in self.evaler.coco_data.get('annotations', []):
                                if ann['image_id'] == original_image_id:
                                    gt_captions.append(ann['caption'])
                        gt_str = gt_captions[0] if gt_captions else "N/A"
                        
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
        
        # --- START: 添加码流长度统计报告 ---
        if data_loader._BYTE_STREAM_LENGTHS:
            lengths = np.array(data_loader._BYTE_STREAM_LENGTHS)
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
            data_loader._BYTE_STREAM_LENGTHS.clear()
        # --- END: 添加码流长度统计报告 ---

        print(f"{'='*80}\n")

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname + '.json'), 'w'))

        model.train() # 恢复模型状态
        return eval_res
