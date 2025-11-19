import os
import numpy as np
import json
from datasets import load_dataset, load_from_disk
from lib.config import cfg
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider

try:
    from coco_caption.pycocoevalcap.meteor.meteor import Meteor
    _HAS_METEOR = True
except Exception:
    _HAS_METEOR = False

try:
    from coco_caption.pycocoevalcap.ciderR.ciderR import CiderR
    _HAS_CIDERR = True
except Exception:
    _HAS_CIDERR = False

try:
    from coco_caption.pycocoevalcap.spice.spice import Spice
    _HAS_SPICE = True
except Exception:
    _HAS_SPICE = False


class CocoJsonEvaler(object):
    """
    COCO evaluator that reads ground truth from a JSON annotation file.
    """

    def __init__(self, split='validation'):
        """
        Initialize evaluator with a JSON annotation file.
        
        Args:
            split (str): Dataset split to use ('validation')
        """
        super().__init__()
        self.split = split
        
        # Load annotations from JSON file
        if split == 'validation':
            ann_file = './PureT/data/coco_karpathy/captions_validation.json'
        else:
            # Add paths for other splits if needed
            raise ValueError(f"Split '{split}' not supported for JSON loading.")

        print(f"Loading annotations from {ann_file}...")
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Build mapping from image_id to all captions
        self.id_to_captions = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.id_to_captions:
                self.id_to_captions[img_id] = []
            self.id_to_captions[img_id].append(ann['caption'])
        
        self.tokenizer = PTBTokenizer()
        print(f"Initialized evaluator with {len(self.id_to_captions)} images")

    def _build_refs(self, img_ids):
        """Build reference captions for evaluation."""
        gts = {}
        for img_id in img_ids:
            # Get captions from HuggingFace dataset
            caps = self.id_to_captions.get(img_id, [])
            if not caps:
                caps = ['.']  # fallback caption
            gts[img_id] = [{'caption': c} for c in caps]
        return gts

    def _build_res(self, results):
        """Build result captions for evaluation."""
        res = {}
        for r in results:
            iid = int(r['image_id'])
            res.setdefault(iid, [])
            res[iid].append({'caption': r['caption']})
        return res

    def eval(self, results):
        """
        Evaluate predictions against ground truth.
        
        Args:
            results (list): List of dicts with 'image_id' and 'caption' keys
            
        Returns:
            dict: Evaluation scores
        """
        img_ids = sorted({int(r['image_id']) for r in results})
        gts = self._build_refs(img_ids)
        res = self._build_res(results)

        # Print sample predictions and ground truth for first 5 images
        print("\n=== Sample Predictions vs Ground Truth ===")
        for i, img_id in enumerate(sorted(img_ids)[:5]):
            print(f"\nImage {img_id}:")
            
            # Predicted caption
            pred_caps = [r['caption'] for r in res.get(img_id, [])]
            if pred_caps:
                print(f"  Prediction: {pred_caps[0]}")
            else:
                print(f"  Prediction: <No prediction>")
            
            # Ground truth captions
            gt_caps = [gt['caption'] for gt in gts.get(img_id, [])]
            print(f"  Ground Truth:")
            for j, cap in enumerate(gt_caps[:3]):  # Show first 3 GT captions
                print(f"    {j+1}. {cap}")
            

        # Tokenize
        print("\nTokenizing captions...")
        gts_tok = self.tokenizer.tokenize(gts)
        res_tok = self.tokenizer.tokenize(res)

        scores = {}
        requested_metrics = cfg.SCORER.TYPES if hasattr(cfg.SCORER, 'TYPES') else ['Bleu_4']
        
        print(f"\nComputing evaluation metrics: {requested_metrics}")
        
        # BLEU metrics (1-4)
        bleu_metrics = [m for m in requested_metrics if m.startswith('Bleu_')]
        if bleu_metrics:
            bleu = Bleu(4)
            bscore, _ = bleu.compute_score(gts_tok, res_tok)
            bleu_names = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
            for i, k in enumerate(bleu_names):
                if k in requested_metrics:
                    scores[k] = float(bscore[i])
                    print(f"  {k}: {scores[k]:.4f}")

        # ROUGE_L
        if 'ROUGE_L' in requested_metrics:
            try:
                rouge = Rouge()
                rscore, _ = rouge.compute_score(gts_tok, res_tok)
                scores['ROUGE_L'] = float(rscore)
                print(f"  ROUGE_L: {scores['ROUGE_L']:.4f}")
            except Exception as e:
                print(f"  ROUGE_L: Error - {e}")
                scores['ROUGE_L'] = 0.0

        # CIDEr
        if 'CIDEr' in requested_metrics or 'Cider' in requested_metrics:
            try:
                cider = Cider()
                cscore, _ = cider.compute_score(gts_tok, res_tok)
                scores['CIDEr'] = float(cscore)
                print(f"  CIDEr: {scores['CIDEr']:.4f}")
            except Exception as e:
                print(f"  CIDEr: Error - {e}")
                scores['CIDEr'] = 0.0

        # CIDEr-R
        if 'CIDEr-R' in requested_metrics and _HAS_CIDERR:
            try:
                ciderr = CiderR()
                cr, _ = ciderr.compute_score(gts_tok, res_tok)
                scores['CIDEr-R'] = float(cr)
                print(f"  CIDEr-R: {scores['CIDEr-R']:.4f}")
            except Exception as e:
                print(f"  CIDEr-R: Error - {e}")
                scores['CIDEr-R'] = 0.0

        # METEOR
        if 'METEOR' in requested_metrics and _HAS_METEOR:
            try:
                meteor = Meteor()
                m, _ = meteor.compute_score(gts_tok, res_tok)
                scores['METEOR'] = float(m)
                print(f"  METEOR: {scores['METEOR']:.4f}")
            except Exception as e:
                print(f"  METEOR: Error - {e}")
                scores['METEOR'] = 0.0

        # SPICE
        if 'SPICE' in requested_metrics and _HAS_SPICE:
            try:
                spice = Spice()
                s, _ = spice.compute_score(gts_tok, res_tok)
                scores['SPICE'] = float(s)
                print(f"  SPICE: {scores['SPICE']:.4f}")
            except Exception as e:
                print(f"  SPICE: Error - {e}")
                scores['SPICE'] = 0.0

        print("=== Evaluation completed ===\n")
        return scores
