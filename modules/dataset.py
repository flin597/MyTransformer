import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter#计数器，返回一个字典{键：出现次数，...}
import re

# 训练、推理、测试都从这里读同一份数据，词表才一致
DATA_PATH = "data/train_v2.tsv"
# 验证集不参与训练，只用来看每轮的效果
VALID_PATH = "data/valid_v2.tsv"

_EN_TOKEN = re.compile(r"\w+(?:'\w+)*|[^\w\s]")#不加这个相同的词会因为跟的标点不同而被判作不同的出现次数

def tokenize_en(text):
    return _EN_TOKEN.findall(text)

def load_cmn_eng(path):
    """读取训练数据，每行前两列是：英文\t中文，后面的列不用"""
    dataset = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            columns = line.rstrip("\n").split("\t")
            en_text = columns[0]
            zh_text = columns[1]
            #整理成和原来一样的格式，后面取数据的代码不用改
            dataset.append({"translation": {"en": en_text, "zh": zh_text}})
    return dataset

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
            counter = Counter(tokens)#统计出现次数
            # 过滤低频词
            filtered_tokens = [token for token, freq in counter.items() if freq >= min_freq]
            # 按频率降序排列
            filtered_tokens.sort(key=lambda x: counter[x], reverse=True)
            self.idx_to_token.extend(filtered_tokens)

        #交换键和值，使能直接用token查索引，用token_to_idx["..."]获取id
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


def build_vocab_from_dataset(dataset, lang_key, min_freq=2):
    """遍历数据集，提取词汇构建 Vocab"""
    counter = Counter()
    for item in dataset:
        text = item["translation"][lang_key]#["translation"]里面存中英文对照，格式是{"id":"12445","translation":{"zh":"你好","en":"Hello"}，lang_key决定取中文还是英文
        
        # 简易分词：中文按字切分，英文按词和标点切分
        if lang_key == "zh":
            tokens = list(text) # 把 "我爱你" 变成 ["我", "爱", "你"]
        else:
            tokens = tokenize_en(text) # 英文按词和标点切分
            
        counter.update(tokens)

    # 提取频率 >= min_freq 的词及其频率
    kept_tokens = []
    for token, freq in counter.items():
        if freq >= min_freq:
            kept_tokens.append(token)
    return Vocab(tokens=kept_tokens)

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_vocab, tgt_vocab, max_len=128):
        self.data = dataset
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _tokenize_and_pad(self, text, vocab, is_src=True):
        if is_src:
            # 源语言是英文，按词和标点分
            tokens = tokenize_en(text)
        else:
            # 目标语言是中文，按字分，需要加上 <sos> 和 <eos>
            tokens = ["<sos>"] + list(text) + ["<eos>"]
            
        ids = vocab.encode(tokens)
        
        # 截断
        ids = ids[:self.max_len]
        #补齐
        ids = ids + [vocab.pad_idx] * (self.max_len - len(ids))
        return ids

    #pytorch数据加载器会自动调用__getitem__
    def __getitem__(self, idx):
        #抓第idx条数据然后把中文和英文分别拿出来
        pair = self.data[idx]["translation"]
        en_text, zh_text = pair["en"], pair["zh"]

        # 1. 处理源语言
        src_ids = self._tokenize_and_pad(en_text,self.src_vocab,is_src=True)
        # 2. 处理目标语言
        tgt_ids = self._tokenize_and_pad(zh_text,self.tgt_vocab,is_src=False)

        # 3. Teacher Forcing 切分
        tgt_input = tgt_ids[:-1]   # Decoder 输入 (以 <sos> 开头)
        tgt_label = tgt_ids[1:]    # 预测标签 (以 <eos> 结尾)

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_input, dtype=torch.long),
            "tgt_label": torch.tensor(tgt_label, dtype=torch.long),
        }

# 4. 测试代码（直接运行这个文件就会执行）
if __name__ == "__main__":
    print("Testing the data pipeline...")

    print("Loading the dataset...")
    # 为了测试方便，我们只取前 100 条数据，免得等太久
    raw_dataset = load_cmn_eng(DATA_PATH)[:100]

    print(f"Loaded {len(raw_dataset)} pairs.")
    print("First example:", raw_dataset[0]["translation"])

    # 2. 构建词表
    print("\nBuilding vocabularies...")
    src_vocab = build_vocab_from_dataset(raw_dataset, "en", min_freq=1)
    tgt_vocab = build_vocab_from_dataset(raw_dataset, "zh", min_freq=1)
    print(f"Size of English_vocab: {len(src_vocab)} | Size of Chinese_vocab: {len(tgt_vocab)}")

    # 3. 创建 Dataset 和 DataLoader
    dataset = TranslationDataset(raw_dataset, src_vocab, tgt_vocab, max_len=20) # 为了测试，max_len 设小一点
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 4. 取出一个 batch 看看
    print("\nTesting the DataLoader...")
    for batch in dataloader:
        print("src (English input) shape:", batch["src"].shape)
        print("tgt_input (Decoder input) shape:", batch["tgt_input"].shape)
        print("tgt_label (prediction label) shape:", batch["tgt_label"].shape)
        
        print("\nIDs of the first sample:")
        print("English IDs:", batch["src"][0].tolist())
        print("Chinese input IDs:", batch["tgt_input"][0].tolist())
        print("Chinese label IDs:", batch["tgt_label"][0].tolist())
        break

    print("\nData pipeline test passed.")