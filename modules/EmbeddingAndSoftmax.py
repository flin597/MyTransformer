import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super().__init__()
        self.match=nn.Embedding(vocab,d_model)
        self.d_model=d_model
    def forward(self,x):
        return self.match(x)*math.sqrt(self.d_model)