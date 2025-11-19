import os
import sys
import numpy as np
from lib.config import cfg
from scorer.bleu import Bleu

# Import additional scorers
try:
    from scorer.cider import Cider
    _HAS_CIDER = True
except:
    _HAS_CIDER = False

try:
    # Rouge might not be available in scorer module
    # from scorer.rouge import Rouge  
    _HAS_ROUGE = False
except:
    _HAS_ROUGE = False

# Factory for available scorers
factory = {
    'Bleu_1': lambda: Bleu(1),
    'Bleu_2': lambda: Bleu(2), 
    'Bleu_3': lambda: Bleu(3),
    'Bleu_4': lambda: Bleu(4),
}

if _HAS_CIDER:
    factory['CIDEr'] = Cider
    factory['Cider'] = Cider

# Note: ROUGE_L, METEOR, SPICE are handled by coco_evaler for full evaluation
print(f"Available scorers for training: {list(factory.keys())}")

def get_sents(sent):
    """
    Convert sentence to word list
    
    Args:
        sent: Either a list of token IDs or a string sentence
    
    Returns:
        list: List of words
    """
    words = []
    
    # If sent is already a string (from decode_sequence), split it
    if isinstance(sent, str):
        if sent.strip():  # Non-empty string
            words = sent.strip().split()
        # Empty string returns empty list
        return words
    
    # If sent is a list of token IDs (legacy behavior)
    for word in sent:
        words.append(word)
        if word == 0:
            break
    return words

class CocoScorer(object):
    def __init__(self, shared_dataset=None):
        super(CocoScorer, self).__init__()
        self.scorers = []
        self.weights = cfg.SCORER.WEIGHTS
        
        # Use shared dataset if provided, otherwise load it
        if shared_dataset is not None:
            # If it's a CocoDataset object, extract the HF dataset
            if hasattr(shared_dataset, 'ds'):
                self.ds = shared_dataset.ds
                print("CocoScorer: Using shared CocoDataset with lazy loading")
            else:
                # Direct HF dataset
                self.ds = shared_dataset
                print("CocoScorer: Using shared HF dataset with lazy loading")
        else:
            self.ds = None
            print("CocoScorer: No shared dataset provided")
        
        # Use lazy loading for ground truth captions - only load when needed
        self.gts = None
        self._gts_loaded = False
        
        # Initialize scorers based on configuration
        print(f"Initializing scorers for: {cfg.SCORER.TYPES}")
        for name in cfg.SCORER.TYPES:
            if name in factory:
                scorer = factory[name]()
                self.scorers.append(scorer)
                print(f"  Added scorer: {name}")
            else:
                print(f"  Warning: Scorer {name} not available in training mode")
                # For unavailable scorers in training, use BLEU as fallback
                if name in ['ROUGE_L', 'METEOR', 'SPICE', 'CIDEr-R']:
                    print(f"  Note: {name} will be computed during evaluation, not training")
                elif not any(isinstance(s, Bleu) for s in self.scorers):
                    # Add BLEU as fallback if no BLEU scorer exists
                    self.scorers.append(factory['Bleu_4']())
                    print(f"  Added BLEU_4 as fallback scorer")
        
        # Ensure we have at least one scorer
        if not self.scorers:
            self.scorers.append(factory['Bleu_4']())
            print("  Added default BLEU_4 scorer")

    def _load_ground_truths(self):
        """Lazy loading of ground truth captions"""
        if self._gts_loaded or self.ds is None:
            return
            
        self.gts = []
        try:
            dataset_length = len(self.ds)
            print(f"CocoScorer: Loading {dataset_length} ground truth captions...")
                
            for i in range(dataset_length):
                # COCO dataset has a 'caption' field containing a list of captions
                item = self.ds[i]
                caps = item.get("caption", [])
                
                # Ensure caps is a list
                if isinstance(caps, str):
                    caps = [caps]
                elif not isinstance(caps, list):
                    caps = []
                
                # Fallback: try other possible caption field names
                if not caps:
                    for alt_key in ['captions', 'text']:
                        if alt_key in item:
                            alt_caps = item[alt_key]
                            if isinstance(alt_caps, str):
                                caps = [alt_caps]
                            elif isinstance(alt_caps, list):
                                caps = alt_caps
                            break
                
                # Final fallback
                if not caps:
                    caps = ['.']
                # Convert captions to word lists (simplified tokenization)
                cap_words = []
                for cap in caps:
                    if isinstance(cap, str):
                        words = cap.lower().split()
                        cap_words.append(words)
                
                self.gts.append(cap_words)
            
            print(f"CocoScorer: Loaded {len(self.gts)} ground truth entries")
            self._gts_loaded = True
        except Exception as e:
            print(f"Warning: Could not load COCO captions for scoring: {e}")
            self.gts = []
            self._gts_loaded = True

    def __call__(self, ids, res):
        # Ensure ground truths are loaded (lazy loading)
        if not self._gts_loaded:
            self._load_ground_truths()
        
        hypo = [get_sents(r) for r in res]
        
        # Get ground truth for these specific ids
        gts = []
        for i in ids:
            if self.gts is not None and i < len(self.gts):
                gts.append(self.gts[i])
            else:
                # Fallback for missing data
                gts.append([['.']])

        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts, hypo)
            rewards += self.weights[i] * scores
            scorer_name = cfg.SCORER.TYPES[i] if i < len(cfg.SCORER.TYPES) else f"Scorer_{i}"
            rewards_info[scorer_name] = score
        return rewards, rewards_info
