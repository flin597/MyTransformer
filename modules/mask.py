import torch

def create_padding_mask(seq,pad_token=0):
    mask=(seq!=pad_token).unsqueeze(1).unsqueeze(2)
    return mask
def create_look_ahead_mask(size):
    #下三角矩阵，下三角及对角线为1，上三角为0
    mask=torch.tril(torch.ones(size,size),diagonal=1).bool()
    return mask

def create_tgt_mask(tgt,pad_token=0):
    #遮挡padding
    tgt_padding_mask=create_padding_mask(tgt,pad_token)

    #遮挡后文
    tgt_len=tgt.size(1)
    ahead_only_mask=create_look_ahead_mask(tgt_len)
    ahead_only_mask=ahead_only_mask.unsqueeze(0).unsqueeze(0).to(tgt.device)

    combined_mask=tgt_padding_mask|ahead_only_mask

    return combined_mask
