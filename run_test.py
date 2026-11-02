import contextlib
import io
import torch
from src.model import Transformer
from src.inference import translate
from modules.dataset import build_vocab_from_dataset, load_cmn_eng, DATA_PATH

# 配置：要和训练时一样
d_model = 256
num_heads = 8
num_layers = 4
d_ff = 1024
checkpoint_path = "checkpoint_epoch_30.pt"
reference_path = "test/test_reference.tsv"
output_path = "test/test_result.tsv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 重建词表，必须和训练时用同一份数据
raw_dataset = load_cmn_eng(DATA_PATH)
src_vocab = build_vocab_from_dataset(raw_dataset, "en", min_freq=1)
tgt_vocab = build_vocab_from_dataset(raw_dataset, "zh", min_freq=1)

# 2. 加载模型
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=d_model, num_head=num_heads, num_layer=num_layers, d_ffn=d_ff
).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# 3. 读题目，每行：id tier en zh_ref key_points
with open(reference_path, encoding="utf-8") as f:
    lines = f.read().splitlines()
questions = lines[1:]  # 第一行是表头

# 4. 逐题翻译，写成一张表，最后的 pass 列留空给人判
with open(output_path, "w", encoding="utf-8") as out:
    out.write("id\ttier\ten\tzh_ref\tkey_points\toutput\tpass\n")
    for line in questions:
        columns = line.split("\t")
        question_id = columns[0]
        tier = columns[1]
        en = columns[2]
        zh_ref = columns[3]
        key_points = columns[4]

        # translate 里有 Debug 打印，这里不显示，只留结果
        with contextlib.redirect_stdout(io.StringIO()):
            output = translate(en, model, src_vocab, tgt_vocab, device)

        print(f"{question_id} [{tier}] {en} -> {output}")
        out.write(f"{question_id}\t{tier}\t{en}\t{zh_ref}\t{key_points}\t{output}\t\n")

print(f"Saved to {output_path}")
