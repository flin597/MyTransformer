import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import Transformer, create_padding_mask, create_tgt_mask
from modules.dataset import build_vocab_from_dataset, load_cmn_eng, tokenize_en, DATA_PATH

def translate(sentence_en, model, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    tokens = tokenize_en(sentence_en)
    src_ids = src_vocab.encode(tokens)
    print(f"[Debug] Input tokens: {tokens}, IDs: {src_ids}")
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
    print(f"[Debug] Raw output IDs: {output_ids}")
    print(f"[Debug] Raw output tokens: {output_words}")
    final_words = [w for w in output_words if w not in ["<sos>", "<eos>", "<pad>"]]
    return "".join(final_words)

def main():
    # 1. 基础设置
    d_model = 256      
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 重建词表
    print("Loading vocabularies...")
    # 必须和训练时用的一样
    raw_dataset = load_cmn_eng(DATA_PATH)
    src_vocab = build_vocab_from_dataset(raw_dataset, "en", min_freq=2)
    tgt_vocab = build_vocab_from_dataset(raw_dataset, "zh", min_freq=2)
    
    # 3. 加载模型
    print("Loading model weights (make sure the .pt file name is correct)...")
    model = Transformer(
        src_vocab_size=len(src_vocab), 
        tgt_vocab_size=len(tgt_vocab), 
        d_model=d_model, num_head=num_heads, num_layer=num_layers, d_ffn=d_ff
    ).to(device)
    
    # 选用参数
    checkpoint_path = "checkpoint_epoch_30.pt" 
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model loaded.")
    except FileNotFoundError:
        print(f"File {checkpoint_path} not found. Check the file name.")
        return

    # 4. 交互式终端启动
    print("\n" + "="*40)
    print("Hand-written Transformer translator (English to Chinese)")
    print("Type an English sentence and press Enter. Type 'q' to quit.")
    print("="*40 + "\n")
    
    while True:
        try:
            # 输入
            user_input = input("English: ")
            
            # 退出条件
            if user_input.strip().lower() == 'q':
                print("Exiting the translator.")
                break

            # 跳过空输入
            if not user_input.strip():
                continue
                
            # 开始翻译
            zh_result = translate(user_input, model, src_vocab, tgt_vocab, device)
            print(f"Translation: {zh_result}\n")
            
        except KeyboardInterrupt:
            # 按 Ctrl+C 也能优雅退出
            print("\nExiting the translator.")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()