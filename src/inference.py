import torch
import os
import sys
from datasets import load_dataset
from model import Transformer, create_padding_mask, create_tgt_mask
from modules.dataset import build_vocab_from_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def translate(sentence_zh, model, src_vocab, tgt_vocab, device, max_len=50):
    model.eval() 
    tokens = list(sentence_zh)
    src_ids = src_vocab.encode(tokens)
    print(f"【Debug】输入的字: {tokens}, 转换的ID: {src_ids}")
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_mask = create_padding_mask(src_tensor, pad_token=src_vocab.pad_idx).to(device)

    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
        tgt_tensor = torch.tensor([[tgt_vocab.sos_idx]], dtype=torch.long).to(device)
        
        for _ in range(max_len):
            tgt_mask = create_tgt_mask(tgt_tensor, pad_token=tgt_vocab.pad_idx).to(device)
            dec_output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
            prob = dec_output[:, -1, :]
            next_word_id = prob.argmax(dim=-1).item()
            tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_word_id]]).to(device)], dim=1)
            
            if next_word_id == tgt_vocab.eos_idx:
                break
                
    output_ids = tgt_tensor[0].tolist()
    output_words = tgt_vocab.decode(output_ids)
    print(f"【Debug】模型原始输出 ID: {output_ids}")
    print(f"【Debug】模型原始输出词: {output_words}")
    final_words = [w for w in output_words if w not in ["<sos>", "<eos>", "<pad>"]]
    return " ".join(final_words)

def main():
    # 1. 基础设置
    d_model = 256      
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 重建词表
    print("📦 正在加载词表...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 必须和训练时用的一样
    raw_dataset = load_dataset("Helsinki-NLP/news_commentary", "en-zh", split="train[:10000]", trust_remote_code=True)
    src_vocab = build_vocab_from_dataset(raw_dataset, "zh", min_freq=1)
    tgt_vocab = build_vocab_from_dataset(raw_dataset, "en", min_freq=1)
    
    # 3. 加载模型
    print("🧠 正在加载模型权重 (请确保 .pt 文件名字正确)...")
    model = Transformer(
        src_vocab_size=len(src_vocab), 
        tgt_vocab_size=len(tgt_vocab), 
        d_model=d_model, num_head=num_heads, num_layer=num_layers, d_ffn=d_ff
    ).to(device)
    
    # ！！！这里换成你实际保存的 .pt 文件名字！！！
    checkpoint_path = "checkpoint_epoch_10.pt" 
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ 模型加载成功！")
    except FileNotFoundError:
        print(f"❌ 找不到文件 {checkpoint_path}，请检查文件名是否正确。")
        return

    # ==========================================
    # 4. 交互式终端启动！
    # ==========================================
    print("\n" + "="*40)
    print("🤖 欢迎使用手搓 Transformer 翻译器！")
    print("💡 输入中文并回车，输入 'q' 退出。")
    print("="*40 + "\n")
    
    while True:
        try:
            # 等待你输入
            user_input = input("请输入中文: ")
            
            # 退出条件
            if user_input.strip().lower() == 'q':
                print("👋 退出翻译器，拜拜！")
                break
                
            # 跳过空输入
            if not user_input.strip():
                continue
                
            # 开始翻译
            en_result = translate(user_input, model, src_vocab, tgt_vocab, device)
            print(f"👉 翻译结果: {en_result}\n")
            
        except KeyboardInterrupt:
            # 按 Ctrl+C 也能优雅退出
            print("\n👋 退出翻译器，拜拜！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}\n")

if __name__ == "__main__":
    main()