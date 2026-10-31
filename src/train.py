import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入你写好的模型和数据管道
# 注意：确保 model.py 里有 TransformerModel, create_padding_mask, create_tgt_mask
# 确保 dataset.py 里有 TranslationDataset, build_vocab_from_dataset
from model import Transformer, create_padding_mask, create_tgt_mask
from modules.dataset import TranslationDataset, build_vocab_from_dataset
from datasets import load_dataset

def main():
    # ==========================================
    # 1. 基础配置（针对本地 CPU 极小化）
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")
    
    # 为了本地跑得快，模型参数全部调小
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    batch_size = 32
    max_len = 50
    epochs = 20 
    lr = 0.0001

    # ==========================================
    # 2. 准备数据
    # ==========================================
    print("正在加载数据集...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 为了快，只取前 200 条数据训练
    raw_dataset = load_dataset("Helsinki-NLP/news_commentary", "en-zh", split="train[:100000]", trust_remote_code=True)
    
    print("正在构建词表...")
    src_vocab = build_vocab_from_dataset(raw_dataset, "zh", min_freq=1)
    tgt_vocab = build_vocab_from_dataset(raw_dataset, "en", min_freq=1)
    
    # 动态获取词表大小
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"中文词表大小: {src_vocab_size}, 英文词表大小: {tgt_vocab_size}")

    dataset = TranslationDataset(raw_dataset, src_vocab, tgt_vocab, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ==========================================
    # 3. 初始化模型、损失函数、优化器
    # ==========================================
    model = Transformer(
        src_vocab_size=src_vocab_size, 
        tgt_vocab_size=tgt_vocab_size, 
        d_model=d_model, 
        num_head=num_heads, 
        num_layer=num_layers, 
        d_ffn=d_ff, 
        dropout=0.1
    ).to(device)

    # 忽略 pad 的 id (0) 不计算损失
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ==========================================
    # 4. 开始训练
    # ==========================================
    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_label = batch["tgt_label"].to(device)

            # 生成 Mask (你之前写的函数)
            src_mask = create_padding_mask(src, pad_token=src_vocab.pad_idx).to(device)
            tgt_mask = create_tgt_mask(tgt_input, pad_token=tgt_vocab.pad_idx).to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(src, tgt_input, src_mask, tgt_mask) # 形状: [batch, seq_len, tgt_vocab_size]
            
            # 计算 Loss (需要把维度转换一下)
            # outputs: [batch, seq_len, vocab_size] -> [batch, vocab_size, seq_len]
            # tgt_label: [batch, seq_len]
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, tgt_label)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} 结束, 平均 Loss: {avg_loss:.4f}\n")
        # 每个 epoch 结束后保存模型
        save_path = f"./checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"💾 模型已保存至 {save_path}")

    # ==========================================
    # 5. 简单测试（翻译一句话）
    # ==========================================
    print("🔍 测试一下模型输出（由于数据极少，模型此时还在瞎猜，属于正常现象）：")
    model.eval()
    
    # 取测试集第一句话
    test_src = dataset[0]["src"].unsqueeze(0).to(device)
    test_src_mask = create_padding_mask(test_src, pad_token=src_vocab.pad_idx).to(device)
    
    with torch.no_grad():
        enc_out = model.encoder(test_src, test_src_mask)
        # 给模型输入 <sos>，让它开始猜
        tgt_test = torch.tensor([[tgt_vocab.sos_idx]]).to(device)
        
        for _ in range(max_len):
            tgt_mask_test = create_tgt_mask(tgt_test, pad_token=tgt_vocab.pad_idx).to(device)
            dec_out = model.decoder(tgt_test, enc_out, test_src_mask, tgt_mask_test)
            prob = dec_out[:, -1, :] # Decoder 已输出目标词表上的 logits
            next_word = prob.argmax(dim=-1).item()
            
            tgt_test = torch.cat([tgt_test, torch.tensor([[next_word]]).to(device)], dim=1)
            if next_word == tgt_vocab.eos_idx:
                break
                
    print("模型生成的英文 ID 序列:", tgt_test[0].tolist())
    print("翻译出来的英文（可能有 <unk>）:", tgt_vocab.decode(tgt_test[0].tolist()))

if __name__ == "__main__":
    main()