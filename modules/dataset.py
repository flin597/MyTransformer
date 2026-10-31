import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset

class Vocab:
    def __init__(self, tokens=None, min_freq=1):
        # 特殊字符
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
        # 强制固定 ID: pad=0, sos=1, eos=2, unk=3
        self.idx_to_token = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        
        if tokens is not None:
            counter = Counter(tokens)
            # 过滤低频词
            filtered_tokens = [token for token, freq in counter.items() if freq >= min_freq]
            # 按频率降序排列
            filtered_tokens.sort(key=lambda x: counter[x], reverse=True)
            self.idx_to_token.extend(filtered_tokens)
        
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        # 记录特殊字符的 ID
        self.pad_idx = self.token_to_idx[self.pad_token]
        self.sos_idx = self.token_to_idx[self.sos_token]
        self.eos_idx = self.token_to_idx[self.eos_token]
        self.unk_idx = self.token_to_idx[self.unk_token]

    def __len__(self):
        return len(self.idx_to_token)

    def encode(self, tokens):
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, ids):
        return [self.idx_to_token[idx] for idx in ids]


def build_vocab_from_dataset(hf_dataset, lang_key, min_freq=2):
    """遍历 HuggingFace 数据集，提取词汇构建 Vocab"""
    counter = Counter()
    for item in hf_dataset:
        text = item["translation"][lang_key]
        
        # 简易分词：中文按字切分，英文按空格切分
        if lang_key == "zh":
            tokens = list(text) # 把 "我爱你" 变成 ["我", "爱", "你"]
        else:
            tokens = text.split() # 英文按空格切分
            
        counter.update(tokens)

    # 提取频率 >= min_freq 的词
    tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
    return Vocab(tokens=tokens)


# ==========================================
# 3. PyTorch Dataset 类
# ==========================================
class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, src_vocab, tgt_vocab, max_len=128):
        self.data = hf_dataset
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _tokenize_and_pad(self, text, vocab, is_src=True):
        if is_src and "zh" in str(vocab.token_to_idx.get("我", "")):
            # 粗劣的判定，如果是中文就按字分，否则按空格分
            tokens = list(text)
        elif is_src:
            tokens = list(text) # 源语言是中文
        else:
            tokens = text.split() # 目标语言是英文
            
        ids = vocab.encode(tokens)
        
        # 截断与补齐
        ids = ids[:self.max_len]
        ids = ids + [vocab.pad_idx] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        pair = self.data[idx]["translation"]
        zh_text, en_text = pair["zh"], pair["en"]

        # 1. 处理源语言（中文）
        # 中文按字切分，遇到生僻字也会变成 unk
        src_tokens = list(zh_text) 
        src_ids = self.src_vocab.encode(src_tokens)
        src_ids = src_ids[:self.max_len]
        src_ids = src_ids + [self.src_vocab.pad_idx] * (self.max_len - len(src_ids))

        # 2. 处理目标语言（英文）
        # 需要加上 <sos> 和 <eos>
        tgt_tokens = ["<sos>"] + en_text.split() + ["<eos>"]
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)
        tgt_ids = tgt_ids[:self.max_len]
        tgt_ids = tgt_ids + [self.tgt_vocab.pad_idx] * (self.max_len - len(tgt_ids))

        # 3. Teacher Forcing 切分
        tgt_input = tgt_ids[:-1]   # Decoder 输入 (以 <sos> 开头)
        tgt_label = tgt_ids[1:]    # 预测标签 (以 <eos> 结尾)

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_input, dtype=torch.long),
            "tgt_label": torch.tensor(tgt_label, dtype=torch.long),
        }


# ==========================================
# 4. 测试代码（直接运行这个文件就会执行）
# ==========================================
if __name__ == "__main__":
    print("🚀 正在测试数据管道...")
    
    # 1. 设置国内镜像加速（如果不设置，下载可能会非常慢）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    print("正在从 HuggingFace 下载数据集 (首次运行需要几十秒)...")
    # 为了测试方便，我们只取前 100 条数据，免得等太久
    raw_dataset = load_dataset("Helsinki-NLP/news_commentary", "en-zh", split="train[:100]", trust_remote_code=True)
    
    print(f"成功加载了 {len(raw_dataset)} 条数据！")
    print("第一条数据示例:", raw_dataset[0]["translation"])

    # 2. 构建词表
    print("\n正在构建词表...")
    src_vocab = build_vocab_from_dataset(raw_dataset, "zh", min_freq=1)
    tgt_vocab = build_vocab_from_dataset(raw_dataset, "en", min_freq=1)
    print(f"中文词表大小: {len(src_vocab)} | 英文词表大小: {len(tgt_vocab)}")

    # 3. 创建 Dataset 和 DataLoader
    dataset = TranslationDataset(raw_dataset, src_vocab, tgt_vocab, max_len=20) # 为了测试，max_len 设小一点
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 4. 取出一个 batch 看看
    print("\n正在测试 DataLoader 取数据...")
    for batch in dataloader:
        print("src (中文输入) 形状:", batch["src"].shape)
        print("tgt_input (Decoder输入) 形状:", batch["tgt_input"].shape)
        print("tgt_label (预测标签) 形状:", batch["tgt_label"].shape)
        
        print("\n看看第一个样本的 ID 长啥样：")
        print("中文 ID:", batch["src"][0].tolist())
        print("英文输入 ID:", batch["tgt_input"][0].tolist())
        print("英文标签 ID:", batch["tgt_label"][0].tolist())
        break

    print("\n✅ 数据管道测试通过！可以接入模型训练了。")