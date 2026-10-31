import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .multiheadAtt import multiheadAtt
from .AddAndNorm import AddAndNorm
from .positionEncoding import PositionalEncoding
from .FNN import feedforward

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_head,d_ffn,dropout=0.1):
        super().__init__()
        self.self_att=multiheadAtt(num_head,d_model,dropout)
        self.ff=feedforward(d_model,d_ffn,dropout)
        self.addnorm_1=AddAndNorm(d_model,dropout)
        self.addnorm_2=AddAndNorm(d_model,dropout)

    def forward(self,x,mask):
        att_output=self.self_att(x,x,x,mask)
        x=self.addnorm_1(x,att_output)
    
        ff_output=self.ff(x)
        x=self.addnorm_2(x,ff_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_head,d_ffn,dropout=0.1):
        super().__init__()
        self.self_att=multiheadAtt(num_head,d_model,dropout)
        self.cross_att=multiheadAtt(num_head,d_model,dropout)
        self.ff=feedforward(d_model,d_ffn,dropout)

        self.addnorm_1=AddAndNorm(d_model,dropout)
        self.addnorm_2=AddAndNorm(d_model,dropout)
        self.addnorm_3=AddAndNorm(d_model,dropout)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        #掩码自注意力
        att_output=self.self_att(x,x,x,tgt_mask)
        x=self.addnorm_1(x,att_output)

        cross_output=self.cross_att(x,encoder_output,encoder_output,src_mask)
        x=self.addnorm_2(x,cross_output)

        ff_output=self.ff(x)
        x=self.addnorm_3(x,ff_output)

        return x

class Encoder(nn.Module):
    def __init__(self,src_vocab_size,d_model,num_head,num_layer,d_ffn,dropout=0.1,max_len=5000):
        super().__init__()
        self.embed=nn.Embedding(src_vocab_size,d_model)
        self.pe=PositionalEncoding(d_model,dropout,max_len)
        self.layers=nn.ModuleList([EncoderLayer(d_model,num_head,d_ffn,dropout)for _ in range(num_layer)])

    def forward(self,x,mask):
        x=self.embed(x)
        x=self.pe(x)
        for layer in self.layers:
            x=layer(x,mask)
        return x

class Decoder(nn.Module):
    def __init__(self,tgt_vocab_size,d_model,num_head,num_layer,d_ffn,dropout=0.1,max_len=5000):
        super().__init__()
        self.embed=nn.Embedding(tgt_vocab_size,d_model)
        self.pe=PositionalEncoding(d_model,dropout,max_len)
        self.layers=nn.ModuleList([DecoderLayer(d_model,num_head,d_ffn,dropout)for _ in range(num_layer)])
        #映射到目标词表大小
        self.fc_out=nn.Linear(d_model,tgt_vocab_size)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.embed(x)
        x=self.pe(x)
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.fc_out(x)



