# 生成训练集 data/train_v2.tsv：统一成简体，并加入问句、复句、长句、习语
# 在仓库根目录运行：python tools/prepare_data.py
# 每行格式：英文\t中文\t来源\t类别，load_cmn_eng 只读前两列
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq
from opencc import OpenCC

from modules.dataset import tokenize_en

CMN_PATH = "data/cmn.txt"
OPUS_PATH = "data/raw/opus-100/train.parquet"
TRANSLATION2019_PATH = "data/raw/translation2019zh/part-0002.parquet"
IDIOM_PATH = "data/raw/idiomkb/en_idiom_meaning.json"
BENCHMARK_PATH = "test/test_reference.tsv"
OUTPUT_PATH = "data/all_v2.tsv"

# 要和 train.py 的 max_len=50 对应：中文还要留出 <sos> 和 <eos> 两个位置
MAX_EN_TOKENS = 50
MAX_ZH_CHARS = 48

# 每一类最多取多少句
MAX_PER_CATEGORY = {
    "idiom_sentence": 20000,
    "question": 40000,
    "compound": 100000,
    "long": 100000,
}

LONG_MIN_TOKENS = 13
# 超过 30 词的多是专业文献，占比太高会把日常句子挤掉
LONG_MAX_TOKENS = 30
COMPOUND_MIN_TOKENS = 8
IDIOM_MIN_WORDS = 3
IDIOM_MAX_ZH_CHARS = 20
RANDOM_SEED = 42

# 粗略按连词判断复句，会混入少量把 before、after 当介词用的句子
COMPOUND_WORDS = re.compile(
    r"\b(if|unless|although|though|because|since|when|while|whenever|before|after|until"
    r"|as soon as|so that|even though|even if|whether|not only|who|whom|whose|which|where)\b",
    re.IGNORECASE,
)
ZH_QUESTION_MARKS = re.compile(r"[？?]|吗|呢|什么|怎么|为什么|为何|谁|哪|几|多少|是否")
NOT_TRANSLATION = re.compile(r"http|www\.|<|>|\.\.\.|…")
LIST_NUMBER = re.compile(r"^\s*(\(?\w{1,3}\)|\d+[.．、])")
IDIOM_PLACEHOLDER = re.compile(r"_|/|\bsomeone\b|\bsomething\b|\bone's\b|\bsb\b|\bsth\b", re.IGNORECASE)
# 这些释义是在解释习语，不是在翻译
IDIOM_EXPLANATION = re.compile(r"指|也就是|比喻|形容|意思|表示")
# 中文里夹英文单词的，多半是技术文档或没翻译完
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def normalize_en(text):
    # 比较英文句子时忽略大小写和标点
    text = re.sub(r"[^\w\s']", "", text.lower())
    return " ".join(text.split())


def strip_dialog_mark(text):
    # 字幕开头常有 "- "，不是句子内容
    return text.strip().lstrip("-–— ").strip()


def count_cjk(text):
    count = 0
    for char in text:
        if "一" <= char <= "鿿":
            count += 1
    return count


def is_clean_pair(en, zh):
    """过滤明显不是好翻译对的句子"""
    if "\t" in en or "\t" in zh or "\n" in en or "\n" in zh or "\r" in en or "\r" in zh:
        return False
    if NOT_TRANSLATION.search(en) or NOT_TRANSLATION.search(zh):
        return False
    if LIST_NUMBER.match(en) or LIST_NUMBER.match(zh):
        return False
    if LATIN_WORD.search(zh):
        return False
    if not en[:1].isupper():
        return False

    en_tokens = len(tokenize_en(en))
    zh_chars = len(zh)
    if en_tokens == 0 or en_tokens > MAX_EN_TOKENS or zh_chars == 0 or zh_chars > MAX_ZH_CHARS:
        return False

    # 中文部分大多得是汉字
    if count_cjk(zh) / zh_chars < 0.6:
        return False

    # 两边长度差太多，多半没对齐
    ratio = zh_chars / en_tokens
    if ratio < 0.6 or ratio > 3.5:
        return False

    # 英文里的数字要在中文里出现，否则多半没对齐
    for number in re.findall(r"\d+", en):
        if number not in zh:
            return False
    return True


def choose_category(en, zh, idiom_pattern):
    """按 习语句 > 问句 > 复句 > 长句 分类，都不是就返回 None"""
    tokens = len(tokenize_en(en))
    if idiom_pattern.search(en):
        return "idiom_sentence"
    if en.endswith("?") and tokens >= 3 and ZH_QUESTION_MARKS.search(zh):
        return "question"
    if tokens > LONG_MAX_TOKENS:
        return None
    if tokens >= COMPOUND_MIN_TOKENS and COMPOUND_WORDS.search(en):
        return "compound"
    if tokens >= LONG_MIN_TOKENS:
        return "long"
    return None


def read_benchmark():
    benchmark = set()
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        columns = line.split("\t")
        benchmark.add(normalize_en(columns[2]))
    return benchmark


def read_idioms():
    with open(IDIOM_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_idiom_pattern(idioms):
    # 只用 3 个词以上的习语匹配句子，太短的多半是普通短语动词
    phrases = []
    for item in idioms:
        idiom = item["idiom"].strip()
        if IDIOM_PLACEHOLDER.search(idiom):
            continue
        if len(idiom.split()) >= IDIOM_MIN_WORDS:
            phrases.append(re.escape(idiom))
    # 长的放前面，先匹配完整的习语
    phrases.sort(key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(phrases) + r")\b", re.IGNORECASE)


def read_cmn(cc):
    pairs = []
    with open(CMN_PATH, encoding="utf-8") as f:
        for line in f:
            columns = line.rstrip("\n").split("\t")
            pairs.append((columns[0], cc.convert(columns[1])))
    return pairs


def read_opus(cc):
    # 这份数据的字幕和联合国文件对齐质量差，只从里面取问句
    table = pq.read_table(OPUS_PATH)
    pairs = []
    for row in table.column("translation").to_pylist():
        en = strip_dialog_mark(row["en"])
        if en.endswith("?"):
            pairs.append((en, cc.convert(strip_dialog_mark(row["zh"]))))
    return pairs


def read_translation2019(cc):
    table = pq.read_table(TRANSLATION2019_PATH)
    english_column = table.column("english").to_pylist()
    chinese_column = table.column("chinese").to_pylist()
    pairs = []
    for i in range(len(english_column)):
        en = english_column[i].strip()
        zh = cc.convert(chinese_column[i].strip())
        pairs.append((en, zh))
    return pairs


def main():
    random.seed(RANDOM_SEED)
    cc = OpenCC("t2s")
    benchmark = read_benchmark()
    idioms = read_idioms()
    idiom_pattern = build_idiom_pattern(idioms)

    output_rows = []
    seen_en = set()
    seen_pair = set()
    removed = Counter()

    # 1. 原来的 cmn-eng 全部保留，只转简体、去掉转换后完全相同的句对
    print("Reading cmn-eng...")
    for en, zh in read_cmn(cc):
        if normalize_en(en) in benchmark:
            removed["cmn_in_benchmark"] += 1
            continue
        if (en, zh) in seen_pair:
            removed["cmn_duplicate"] += 1
            continue
        seen_pair.add((en, zh))
        seen_en.add(normalize_en(en))
        output_rows.append((en, zh, "cmn-eng", "base"))

    # 2. IdiomKB：习语 -> 简短释义。释义由 GPT-3 生成，不是人工翻译
    print("Reading IdiomKB...")
    for item in idioms:
        en = item["idiom"].strip()
        zh = cc.convert(item["zh_meaning"].strip())
        if IDIOM_PLACEHOLDER.search(en) or len(en.split()) < IDIOM_MIN_WORDS:
            removed["idiom_not_usable"] += 1
            continue
        if len(zh) > IDIOM_MAX_ZH_CHARS or count_cjk(zh) == 0 or IDIOM_EXPLANATION.search(zh):
            removed["idiom_not_usable"] += 1
            continue
        if normalize_en(en) in benchmark or normalize_en(en) in seen_en:
            removed["idiom_duplicate"] += 1
            continue
        seen_en.add(normalize_en(en))
        output_rows.append((en, zh, "idiomkb", "idiom"))

    # 3. 新句子先按类别分好，再随机抽到上限
    print("Reading opus-100 and translation2019zh...")
    candidates = {category: [] for category in MAX_PER_CATEGORY}
    sources = [("opus-100", read_opus(cc)), ("translation2019zh", read_translation2019(cc))]
    for source, pairs in sources:
        for en, zh in pairs:
            if not is_clean_pair(en, zh):
                removed[source + "_not_clean"] += 1
                continue
            key = normalize_en(en)
            if key in benchmark:
                removed[source + "_in_benchmark"] += 1
                continue
            if key in seen_en:
                removed[source + "_duplicate"] += 1
                continue
            category = choose_category(en, zh, idiom_pattern)
            if category is None:
                removed[source + "_other_length"] += 1
                continue
            # opus-100 里没被认成问句的，多半中英文没对齐
            if source == "opus-100" and category != "question":
                removed[source + "_not_question"] += 1
                continue
            seen_en.add(key)
            candidates[category].append((en, zh, source, category))

    for category, rows in candidates.items():
        random.shuffle(rows)
        limit = MAX_PER_CATEGORY[category]
        print(f"{category}: available {len(rows)}, keep {min(len(rows), limit)}")
        output_rows.extend(rows[:limit])

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        for row in output_rows:
            f.write("\t".join(row) + "\n")

    print(f"Saved {len(output_rows)} pairs to {OUTPUT_PATH}")
    print("By source:", dict(Counter(row[2] for row in output_rows)))
    print("By category:", dict(Counter(row[3] for row in output_rows)))
    print("Removed:", dict(removed))


if __name__ == "__main__":
    main()
