import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AddAndNorm(nn.Module):
    """残差连接+dropout+归一化"""
    def __init__(self,d_model,dropout):
        super().__init__()
        self.norm=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x,sublayer_output):
        return self.norm(x+self.dropout(sublayer_output))