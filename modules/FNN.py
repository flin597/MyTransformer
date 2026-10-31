import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class feedforward(nn.Module):
    def __init__(self,d_model,d_ffn,dropout=0.1):
        super().__init__()
        self.W_1=nn.Linear(d_model,d_ffn)
        self.W_2=nn.Linear(d_ffn,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.W_2(self.dropout(self.W_1(x).relu()))