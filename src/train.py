import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入你写好的模型和数据管道
# 注意：确保 model.py 里有 TransformerModel, create_padding_mask, create_tgt_mask
# 确保 dataset.py 里有 TranslationDataset, build_vocab_from_dataset
from src.model import Transformer, create_padding_mask, create_tgt_mask
from modules.dataset import TranslationDataset, build_vocab_from_dataset, load_cmn_eng, DATA_PATH, VALID_PATH

def main():

    # 1. 基础配置（针对本地 CPU 极小化）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    batch_size = 32
    max_len = 50
    epochs = 30
    lr = 0.001

    # 2. 准备数据
    print("Loading the dataset...")
    # 简体训练集：cmn-eng 加上问句、复句、长句、习语，由 tools/prepare_data.py 生成
    raw_dataset = load_cmn_eng(DATA_PATH)
    
    print("Building vocabularies...")
    src_vocab = build_vocab_from_dataset(raw_dataset, "en", min_freq=2)
    tgt_vocab = build_vocab_from_dataset(raw_dataset, "zh", min_freq=2)
    
    # 动态获取词表大小
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"English vocab size: {src_vocab_size}, Chinese vocab size: {tgt_vocab_size}")

    dataset = TranslationDataset(raw_dataset, src_vocab, tgt_vocab, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 验证集：词表沿用训练集的，不打乱
    raw_valid = load_cmn_eng(VALID_PATH)
    valid_dataset = TranslationDataset(raw_valid, src_vocab, tgt_vocab, max_len=max_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train pairs: {len(dataset)}, validation pairs: {len(valid_dataset)}")

    # 3. 初始化模型、损失函数、优化器
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

    # 4. 开始训练
    print("\nStarting training...")
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
        print(f"Epoch {epoch+1} finished, average loss: {avg_loss:.4f}")

        # 每轮结束在验证集上算一次 loss，不更新参数
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                src = batch["src"].to(device)
                tgt_input = batch["tgt_input"].to(device)
                tgt_label = batch["tgt_label"].to(device)
                src_mask = create_padding_mask(src, pad_token=src_vocab.pad_idx).to(device)
                tgt_mask = create_tgt_mask(tgt_input, pad_token=tgt_vocab.pad_idx).to(device)
                outputs = model(src, tgt_input, src_mask, tgt_mask)
                outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, tgt_label)
                total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        print(f"Epoch {epoch+1} validation loss: {avg_valid_loss:.4f}\n")
        # 每个 epoch 结束后保存模型
        save_path = f"./checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


    # 5. 简单测试（翻译一句话）
    print("Testing the model output on the first sentence:")
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
                
    print("Generated Chinese IDs:", tgt_test[0].tolist())
    print("Generated Chinese (may contain <unk>):", tgt_vocab.decode(tgt_test[0].tolist()))

if __name__ == "__main__":
    main()