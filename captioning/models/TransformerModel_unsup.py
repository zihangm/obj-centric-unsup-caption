# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, encoder_seq, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder_v = encoder # for encoding visual feats
        self.encoder_s = encoder_seq # for encoding caption
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, enc_seq_feats, src_mask, tgt_mask, enc_seq_mask, vis_concep):
        "Take in and process masked src and target sequences."
        common_denotation, matching_loss, denotation_mask = self.encode(src, enc_seq_feats, src_mask, enc_seq_mask, tgt, vis_concep)
        return self.decode(common_denotation, denotation_mask, tgt, tgt_mask), matching_loss

    def encode(self, src, enc_seq_feats, src_mask, enc_seq_mask, seq, vis_concep): # modified for unsupervised caption:
        # src: visual feats
        # src_seq: captions
        encoded_v = self.encoder_v(src, src_mask)
        encoded_s = self.encoder_s(enc_seq_feats, enc_seq_mask)
        # get common operation and losses
        seq = seq.unsqueeze(1)
        vis_concep = vis_concep.unsqueeze(2)
        vis_seq_match = (vis_concep - seq) == 0 # batch_size x N_vis x N_seq before sum
        # extract common denotation from sentence
        # common_denotation = encoded_s * (vis_seq_match.sum(1) != 0).unsqueeze(2)
        # extract common denotation from visual
        common_denotation = encoded_v * (vis_seq_match.sum(2) != 0).unsqueeze(2)
        # send all visual features to decoder (pseodo pair setting)
        # common_denotation = encoded_v
        # for a sentence, choose all matched visual tokens in the mini-batch
        # vis_concep_batch = vis_concep.view(1, -1, 1)
        # vis_seq_match_batch = (vis_concep_batch - seq) == 0 # batch_size x N_vis*batch_size x N_seq
        # common_denotation = encoded_v.view(1, encoded_v.shape[0]*encoded_v.shape[1], encoded_v.shape[2]).repeat(src.shape[0], 1, 1)
        # common_denotation_mask = (vis_seq_match_batch.sum(2) != 0).int().unsqueeze(1)
        # print('Number of tokens per image:', (vis_concep!=-1).int().sum(1).float().mean())
        # print('Number of tokens matched to a sentence from a batch of image:', (vis_seq_match_batch.sum(2) != 0).float().sum(1).mean())

        # calcuate hinge loss
        # vis_minus_seq = torch.norm(encoded_v.unsqueeze(2) - encoded_s.unsqueeze(1), dim=3)
        # random_idx = np.arange(vis_minus_seq.shape[0])
        # np.random.shuffle(random_idx)
        # margin = 5
        # matched_num = (vis_seq_match!=0).float().sum(2)
        # nomatched_mask = ((vis_seq_match==0) * (vis_concep!=-1) * (seq!=0))
        # nomatched_num = nomatched_mask.float().sum(2)
        # triplet_term = (vis_minus_seq*(vis_seq_match!=0)).sum(2)/(matched_num+(matched_num==0).float()) \
        #                - (vis_minus_seq * nomatched_mask).sum(2) / (nomatched_num + (nomatched_num == 0).float())\
        #                + margin*((vis_seq_match!=0).float().sum(2)!=0)
        # matching_loss = torch.max(torch.zeros([1]).to(vis_minus_seq), triplet_term).sum() / \
        #                 ((triplet_term > 0).float().sum() + ((triplet_term > 0).float().sum()==0).float())
        matching_loss = torch.zeros(1).to(common_denotation) # currently not using matching loss

        # # for checking the quality of tokens
        # a = vis_concep_batch * (vis_seq_match_batch.sum(2) != 0).unsqueeze(2)
        # for idx in range(a.shape[0]):
        #     s_batch = list([tmp[0] for tmp in a[idx, :, 0][torch.nonzero(a[idx, :, 0])].tolist()])
        #     print('idx:', idx, 'batch matched tokens:', len(s_batch), len(set(s_batch)), s_batch)
        #     s_img_tokens = list(vis_concep[idx,:,0].tolist())
        #     # s_img_tokens.remove(-1)
        #     s_img_tokens = [token for token in s_img_tokens if token != -1]
        #     print('idx:', idx, 'img_tokens:', len(s_img_tokens), len(set(s_img_tokens)))

        return common_denotation, matching_loss, (vis_concep.squeeze(2) != -1).float().unsqueeze(1) #(vis_concep != -1).int().unsqueeze(1) #(vis_seq_match.sum(2) != 0).int().unsqueeze(1)


    def encode_evaluate(self, src, src_mask, vis_concep): # use in evaluation/inference
        # extract denotations from visual images
        encoded_v = self.encoder_v(self.src_embed(src), src_mask)
        common_denotation = encoded_v * (vis_concep != -1).unsqueeze(2)
        # choosen nonzero rows
        return common_denotation, (vis_concep != -1).int().unsqueeze(1)
    # def encode(self, src, src_mask):
    #     return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc), # for visual feats
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc), # for caption
            Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                 c(ff), dropout), N_dec),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))

        tgt_vocab = self.vocab_size + 1

        # define embed for encoder caption
        c = copy.deepcopy
        position = PositionalEncoding(self.d_model, self.dropout)
        self.enc_seq_embed = nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position))

        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        self.model = self.make_model(0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout)

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        vis_concep = att_feats[:, :, 0]
        vis_concep = vis_concep - (vis_concep==0).float() # move the padding zero to -1
        att_feats = att_feats[:, :, 1:]

        att_feats, seq, att_masks, seq_mask = \
            self._prepare_feature_forward(att_feats, att_masks)
        memory, denotation_mask = self.model.encode_evaluate(att_feats, att_masks, vis_concep) # encode_evaluate

        return fc_feats[...,:0], att_feats[...,:0], memory, denotation_mask # att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None, vis_concep=None):
        # att_feats, att_masks = self.clip_att(att_feats, att_masks) don't clip

        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.att_embed(att_feats)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        # embed the input caption and maks mask for caption_encoding
        if seq is not None:
            enc_seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            enc_seq_mask[:, 0] = 1  # bos
            enc_seq_feats = self.enc_seq_embed(seq)
            # enc_seq_feats, enc_seq_mask = self.clip_att(enc_seq_feats, enc_seq_mask)
            enc_seq_mask = enc_seq_mask.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks, vis_concep = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks, vis_concep]
                )
        else:
            seq_mask = None
        if seq is not None: # training
            return att_feats, seq, enc_seq_feats, att_masks, seq_mask, enc_seq_mask, vis_concep
        else: # evaluation
            return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        vis_concep = att_feats[:, :, 0]
        vis_concep = vis_concep - (vis_concep == 0).float()  # move the padding zero to -1
        att_feats = att_feats[:, :, 1:]

        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, enc_seq_feats, att_masks, seq_mask, enc_seq_mask, vis_concep = \
            self._prepare_feature_forward(att_feats, att_masks, seq, vis_concep)

        out, matching_loss = self.model(att_feats, seq, enc_seq_feats, att_masks, seq_mask, enc_seq_mask, vis_concep)

        outputs = self.model.generator(out)
        return outputs, matching_loss
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, 
                               ys, 
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]