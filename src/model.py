import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.EncoderAndDecoder import Encoder, Decoder
from modules.positionEncoding import PositionalEncoding

def create_padding_mask(tokens, pad_token):
    return (tokens != pad_token).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tokens, pad_token):
    batch_size, seq_len = tokens.shape
    padding_mask = (tokens != pad_token).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=tokens.device)
    ).unsqueeze(0).unsqueeze(1)
    return padding_mask & causal_mask

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model=512,num_head=8,num_layer=6,d_ffn=2048,dropout=0.1):
        super().__init__()
        #编码器
        self.encoder=Encoder(src_vocab_size,d_model,num_head,num_layer,d_ffn,dropout)
        #解码器
        self.decoder=Decoder(tgt_vocab_size,d_model,num_head,num_layer,d_ffn,dropout)
        #把decoder输出的512维度映射回目标语言的词表大小
        self.fc_out=nn.Linear(d_model,tgt_vocab_size)
    def forward(self,src,tgt,src_mask,tgt_mask):
        #编码器处理中文输入
        encoder_output=self.encoder(src,src_mask)
        #解码器吐翻译
        decoder_output=self.decoder(tgt,encoder_output,src_mask,tgt_mask)
        return decoder_output