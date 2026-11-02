# 把 prepare_data.py 生成的 data/all_v2.tsv 切成训练集和验证集
# 在仓库根目录运行：python tools/split_valid.py
import random
import re

INPUT_PATH = "data/all_v2.tsv"
TRAIN_PATH = "data/train_v2.tsv"
VALID_PATH = "data/valid_v2.tsv"
VALID_SIZE = 2000
RANDOM_SEED = 42


def normalize_en(text):
    # 比较英文句子时忽略大小写和标点
    text = re.sub(r"[^\w\s']", "", text.lower())
    return " ".join(text.split())


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        lines = f.read().split("\n")

    # 同一句英文的所有译文放在一组，只能整组进训练集或验证集，否则会透题
    groups = {}
    for line in lines:
        if line == "":
            continue
        key = normalize_en(line.split("\t")[0])
        if key not in groups:
            groups[key] = []
        groups[key].append(line)

    keys = list(groups.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(keys)

    valid_lines = []
    train_lines = []
    for key in keys:
        if len(valid_lines) < VALID_SIZE:
            valid_lines.extend(groups[key])
        else:
            train_lines.extend(groups[key])

    with open(TRAIN_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(VALID_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(valid_lines) + "\n")

    print(f"Train: {len(train_lines)} pairs, validation: {len(valid_lines)} pairs")


if __name__ == "__main__":
    main()
