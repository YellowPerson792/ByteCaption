import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

import lib.utils as utils
from models.basic_model import BasicModel

from models.backbone.swin_transformer_backbone import SwinTransformer as STBackbone
from models.encoder_decoder.PureT_encoder import Encoder
from models.encoder_decoder.PureT_decoder import Decoder
from byteformer_immigration import get_opts, init_byteformer

# For masked MSA
"""
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
"""
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    return subsequent_mask == 0


class PureT_byteformer(BasicModel):
    def __init__(self):
        super(PureT_byteformer, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        
        self.backbone = init_byteformer()
        print("Init byteformer backbone")
        # Freeze parameters
        # for _name, _weight in self.backbone.named_parameters():
        #     _weight.requires_grad = False
        #     print(_name, _weight.requires_grad)
        
        # raw Dimension to Model Dimension
        if cfg.MODEL.ATT_FEATS_DIM == cfg.MODEL.ATT_FEATS_EMBED_DIM:
            self.att_embed = nn.Identity()
        else:
            self.att_embed = nn.Sequential(
                nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
                utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
                nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)
            )
        
        use_gx = True
        # self.encoder = Encoder(
        #     embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM, 
        #     input_resolution=(12, 12), 
        #     depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS, 
        #     num_heads=cfg.MODEL.BILINEAR.HEAD, 
        #     window_size=6,
        #     shift_size=3,
        #     mlp_ratio=4,
        #     dropout=0.1,
        #     use_gx = use_gx
        # )
        
        self.decoder = Decoder(
            vocab_size = self.vocab_size, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            depth = cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads = cfg.MODEL.BILINEAR.HEAD, 
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            ff_dropout = cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
            use_gx = use_gx
        )
        
    def forward(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        
        # backbone forward
        backbone_output = self.backbone(att_feats)
        # Extract the actual tensor from BaseModelOutput
        att_feats = backbone_output.last_hidden_state
        # att_mask = backbone_output.encoder_attention_mask
        
        # att_mask for features
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        # words mask [B, L, L]
        ##############################################
        device = seq.device
        seq_mask = (seq > 0).to(device=device, dtype=torch.float32)
        seq_mask[:, 0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        sub = subsequent_mask(seq.size(-1)).to(device)
        seq_mask = seq_mask * sub.to(dtype=seq_mask.dtype)
        ##############################################

        att_feats = self.att_embed(att_feats)
        
        # 原始encoder代码 (注释掉)
        # encoder_out, gx = self.encoder(att_feats, att_mask)
        
        # 绕过encoder，直接使用backbone输出的特征
        # 计算全局特征 (类似encoder中的做法)
        if att_mask is not None:
            gx = (torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1))
        else:
            gx = att_feats.mean(1)  # [B, C] - 全局特征
        
        # 直接使用att_feats作为encoder_out (patch特征)
        encoder_out = att_feats  # [B, H*W, C] - patch特征
        
        # 输入decoder
        decoder_out = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)
        return F.log_softmax(decoder_out, dim=-1)

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gx = kwargs[cfg.PARAM.GLOBAL_FEAT]
        # p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # state[0][0]: [B, seq_len-1]，previously generated words
        # ys: [B, seq_len]
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
            
        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device, dtype=torch.float32)[:, -1, :].unsqueeze(1)
        
        # [B, 1, Vocab_Size] --> [B, Vocab_Size]
        decoder_out = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, seq_mask, att_mask).squeeze(1)
        
        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s
        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        device = att_feats.device
        seq_logprob = torch.zeros((batch_size, 1, 1), device=device)
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1), device=device)

        # backbone forward - handle BaseModelOutput like in forward method
        backbone_output = self.backbone(att_feats)
        # Extract the actual tensor from BaseModelOutput
        if hasattr(backbone_output, 'last_hidden_state'):
            att_feats = backbone_output.last_hidden_state
        else:
            att_feats = backbone_output
        att_feats = self.att_embed(att_feats)
        
        # 原始encoder代码 (注释掉)
        # encoder_out, gx = self.encoder(att_feats, att_mask)
        
        # 绕过encoder，直接计算全局特征和使用patch特征
        if att_mask is not None:
            gx = (torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1))
        else:
            gx = att_feats.mean(1)  # [B, C] - 全局特征
        
        encoder_out = att_feats  # [B, H*W, C] - 直接使用patch特征
        # p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long, device=device))
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        device = encoder_out.device
        self.decoder.init_buffer(batch_size, device)
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            # [B*cur_beam_size, Vocab_size] --> [B, cur_beam_size, Vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            # sum of logprob
            # [B, cur_beam_size, Vocab_size]
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # [B, beam_size], [B, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # update buffer
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                # expand input
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                # p_att_feats_tmp = []
                # for p_feat in p_att_feats:
                #     p_key, p_value2 = p_feat
                #     p_key = utils.expand_tensor(p_key, beam_size)
                #     p_value2 = utils.expand_tensor(p_value2, beam_size)
                #     p_att_feats_tmp.append((p_key, p_value2))

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        att_feats = self.backbone(att_feats)
        if hasattr(att_feats, 'last_hidden_state'):
            att_feats = att_feats.last_hidden_state
        att_feats = self.att_embed(att_feats)
        
        # 原始encoder代码 (注释掉)
        # encoder_out, gx = self.encoder(att_feats, att_mask)
        
        # 绕过encoder，直接计算全局特征和使用patch特征
        if att_mask is not None:
            gx = (torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1))
        else:
            gx = att_feats.mean(1)  # [B, C] - 全局特征
        
        encoder_out = att_feats  # [B, H*W, C] - 直接使用patch特征
        # p_att_feats = self.decoder.precompute(encoder_out)
        device = encoder_out.device
        self.decoder.init_buffer(batch_size, device)
        
        state = None
        device = encoder_out.device
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long, device=device))
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN, device=device))
        wt = Variable(torch.zeros(batch_size, dtype=torch.long, device=device))
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        
        # inference word by word
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state(**kwargs)
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs
    
    def flops(self):
        flops = 0
        flops += self.backbone.flops()
        # self.att_embed
        flops += 1536 * 512
        # encoder decoder
        flops += self.encoder.flops()
        flops += self.encoder.flops()
        # flops += self.decoder.flops()
        return flops
    
