import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention(queries,keys,values,mask=None,dropout=None):
    """计算点积注意力"""
    #queries->(batch_size,seq_len,num_head,d_featire)
    dk=queries.size(-1)
    scores=torch.matmul(queries,keys.transpose(-2,-1))/math.sqrt(dk)
    if mask is not None:
        scores=scores.masked_fill(mask == 0, -1e9)
    weights_att = scores.softmax(dim=-1)
    return torch.matmul(weights_att,values), weights_att

class multiheadAtt(nn.Module):
    def __init__(self,num_head,d_model,dropout=0.1):
        """获取model维度和头数"""
        super().__init__()
        assert d_model % num_head == 0
        self.num_head = num_head
        self.d_model = d_model
        self.d_k = d_model // num_head

        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)
        self.W_o=nn.Linear(d_model,d_model)

        self.dropout=nn.Dropout(dropout)

    def forward(self,queries,keys,values,mask=None):
        if mask is not None and mask.dim() == 3:
            """对所有头生效"""
            mask=mask.unsqueeze(1)

        batch_size=queries.size(0)

        queries=self.W_q(queries)
        keys=self.W_k(keys)
        values=self.W_v(values)
        
        #拆分成多头
        queries=queries.view(batch_size,-1,self.num_head,queries.size(-1)//self.num_head).transpose(1,2)
        keys=keys.view(batch_size,-1,self.num_head,keys.size(-1)//self.num_head).transpose(1,2)
        values=values.view(batch_size,-1,self.num_head,values.size(-1)//self.num_head).transpose(1,2)

        output,_=attention(queries,keys,values,mask=mask,dropout=self.dropout)
        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        return self.W_o(output)



