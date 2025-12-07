import math
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PureT.lib.config import cfg
from torch.nn.utils.weight_norm import weight_norm

def activation(act):
    if act == 'RELU':
        return nn.ReLU(inplace=True)
    elif act == 'TANH':
        return nn.Tanh()
    elif act == 'GLU':
        return nn.GLU()
    elif act == 'ELU':
        return nn.ELU(cfg.MODEL.BILINEAR.ELU_ALPHA, inplace=True)
    elif act == 'CELU':
        return nn.CELU(cfg.MODEL.BILINEAR.ELU_ALPHA, inplace=True)
    elif act == 'GELU':
        return nn.GELU()
    else:
        return nn.Identity()

def expand_tensor(tensor, size, dim=1):
    if size == 1 or tensor is None:
        return tensor
    tensor = tensor.unsqueeze(dim)
    tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim+1:])).contiguous()
    tensor = tensor.view(list(tensor.shape[:dim-1]) + [-1] + list(tensor.shape[dim+1:]))
    return tensor

def expand_numpy(x, size=cfg.DATA_LOADER.SEQ_PER_IMG):
    if cfg.DATA_LOADER.SEQ_PER_IMG == 1:
        return x
    x = x.reshape((-1, 1))
    x = np.repeat(x, size, axis=1)
    x = x.reshape((-1))
    return x

def load_ids(path):
    with open(path, 'r') as fid:
        lines = [int(line.strip()) for line in fid]
    return lines

def load_lines(path):
    with open(path, 'r') as fid:
        lines = [line.strip() for line in fid]
    return lines

def load_vocab(path):
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab

# torch.nn.utils.clip_grad_norm
# https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L84-L91
# torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
def clip_gradient(optimizer, model, grad_clip_type, grad_clip):
    if grad_clip_type == 'Clamp':
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad == True and param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
    elif grad_clip_type == 'Norm':
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    elif grad_clip_type == 'None':
        pass
    else:
        raise NotImplementedError

def decode_sequence(vocab, seq):
    N, T = seq.size()
    sents = []
    vocab_size = len(vocab)
    
    # Add debug info on first call
    if hasattr(decode_sequence, '_debug_logged'):
        pass
    else:
        decode_sequence._debug_logged = True
    
    for n in range(N):
        words = []
        raw_tokens = []  # 记录原始token
        for t in range(T):
            ix = seq[n, t].item()  # Convert to Python int
            raw_tokens.append(ix)
            if ix == 0:  # EOS token
                break
            # Check if index is within vocab range
            if 0 < ix < vocab_size:
                words.append(vocab[ix])
            else:
                # Handle out-of-range indices gracefully
                if ix < 0:
                    print(f"Warning: Negative token ID {ix} at position ({n}, {t})")
                    words.append('<NEG>')
                else:
                    print(f"Warning: Token ID {ix} exceeds vocab size {vocab_size} at position ({n}, {t})")
                    words.append('<UNK>')
        sent = ' '.join(words)
        sents.append(sent)

    return sents

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float(-1e9)).type_as(t)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count