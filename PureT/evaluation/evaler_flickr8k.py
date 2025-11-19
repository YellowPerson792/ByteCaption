import os
import numpy as np
import torch
import tqdm
import json
import evaluation
import losses
import lib.utils as utils
import datasets_.data_loader_byteformer_f8k as data_loader
from lib.config import cfg


class Flickr8kEvaler(object):
    def __init__(self, eval_ids, gv_feat, att_feats, eval_annfile, max_samples=None, enable_eval_loss=False):
        super(Flickr8kEvaler, self).__init__()
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
        self.evaler = evaluation.create('FLICKR8K', split)
        
        
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
        with torch.no_grad():
            # Use tqdm without desc to avoid interference with sample output
            pbar = tqdm.tqdm(self.eval_loader, desc=f"Evaluating {rname}", leave=False)
            for _, (indices, gv_feat, att_feats, att_mask) in enumerate(pbar):
                ids = self.eval_ids[indices]
                gv_feat = gv_feat.to(device)
                att_feats = att_feats.to(device)
                if att_mask is not None:
                    att_mask = att_mask.to(device)
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
                # 尝试构建输入/目标序列以计算XE loss（仅当启用且数据集支持时）
                batch_loss = None
                if self.loss_computation_ready and xe_criterion is not None:
                    try:
                        dataset = self.eval_loader.dataset
                        if not hasattr(dataset, '_build_seqs_from_captions'):
                            raise AttributeError('Dataset lacks _build_seqs_from_captions; cannot compute eval XE loss')
                        input_list = []
                        target_list = []
                        for idx in indices:
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
                m = getattr(model, 'module', model)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _ = m.decode_beam(**kwargs)
                else:
                    seq, _ = m.decode(**kwargs)

                sents = utils.decode_sequence(self.vocab, seq.data)
                for sid, sent in enumerate(sents):
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                    results.append(result)
                    if global_idx < 5:  # Show first 5 samples only
                        image_id = int(ids[sid])
                        gt_captions = []
                        if hasattr(self.evaler, 'id_to_captions') and image_id in self.evaler.id_to_captions:
                            gt_captions = self.evaler.id_to_captions[image_id]
                        elif hasattr(self.evaler, 'coco_data'):
                            for ann in self.evaler.coco_data.get('annotations', []):
                                if ann['image_id'] == image_id:
                                    gt_captions.append(ann['caption'])
                        gt_str = gt_captions[0] if gt_captions else "N/A"
                        
                        # Clear progress bar line and print sample
                        pbar.clear()
                        print(f"\n[Eval Sample {global_idx}]")
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
        print(f"{'='*80}\n")

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname + '.json'), 'w'))

        model.train()
        return eval_res
