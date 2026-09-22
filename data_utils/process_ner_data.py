# coding=utf-8
"""将开源中文 NER 数据（人民日报 PeopleDaily，char-level BIO 格式）处理成
本项目 NERDatasetForGP 可直接训练的 JSONL 格式。

输出每行一个 JSON 样本，字段与 NERDatasetForGP 期望一致：
    {"token": [...], "input_ids": [...], "labels": [...], "labels_": [...], "entity": [[text, start, end_exclusive, type], ...]}
其中 entity 用于 GlobalPointer 的标签矩阵构建（NERDatasetForGP.collate_wrapper 仅使用该字段）。

依赖：仅需 Python 标准库 + resources/vocab.txt（BERT-Chinese 词表）。
不依赖 jieba/numpy/torch，token->id 的映射与 common_utils/tokenization.FullTokenizer
在 char-level 下完全等价（逐字符转小写后查表）。
"""
import collections
import json
import logging
import os
from typing import Any, Dict, List, Text, Tuple

# 实体类型 -> GlobalPointer 标签矩阵的行索引（与 resources/ner_ent2id.json 对应）
ENT2ID: Dict[Text, int] = {"LOC": 0, "ORG": 1, "PER": 2, "TIME": 3}

# BIO 序列标签 -> 标签 id（labels 字段；GP 训练不直接使用，仅为保持数据格式一致）
LABEL2ID: Dict[Text, int] = {
    "O": 0,
    "B-LOC": 1, "I-LOC": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-PER": 5, "I-PER": 6,
    "B-TIME": 7, "I-TIME": 8,
}

# 与 NERDatasetForGP.max_len 保持一致，超长句子截断，并丢弃越界实体
MAX_LEN = 512


def load_vocab(vocab_file: str) -> "collections.OrderedDict":
    """复刻 common_utils/tokenization.load_vocab：逐行读取词表，token -> id。"""
    vocab: "collections.OrderedDict" = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_tokens_to_ids(tokens: List[Text], vocab: Dict[Text, int]) -> List[int]:
    """复刻 convert_by_vocab：char -> id，未登录词映射为 [UNK]。"""
    unk_id = vocab.get("[UNK]", 100)
    return [vocab.get(t, unk_id) for t in tokens]


def read_bio_file(path: str) -> List[List[Tuple[Text, Text]]]:
    """读取 char-level BIO 文件，空行分隔句子。

    每行形如 `char<TAB>tag`（如 `北\tB-LOC`）。返回 [ [(char, tag), ...], ... ]。
    """
    sentences: List[List[Tuple[Text, Text]]] = []
    sent: List[Tuple[Text, Text]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if sent:
                    sentences.append(sent)
                    sent = []
                continue
            parts = line.split()
            if len(parts) < 2:
                # 形如只有 "O" 的异常行，跳过
                continue
            char, tag = parts[0], parts[1]
            sent.append((char, tag))
    if sent:
        sentences.append(sent)
    return sentences


def extract_entities(tags: List[Text]) -> List[List[Any]]:
    """从 BIO 标签解码实体区间，返回 [[start, end_exclusive, type], ...]。"""
    entities: List[List[Any]] = []
    start, ent_type = None, None
    for i, tag in enumerate(tags):
        if tag == "O":
            if start is not None:
                entities.append([start, i, ent_type])
                start, ent_type = None, None
        elif tag.startswith("B-"):
            if start is not None:
                entities.append([start, i, ent_type])
            start, ent_type = i, tag[2:]
        elif tag.startswith("I-"):
            t = tag[2:]
            if start is not None and ent_type == t:
                continue
            # I- 没有匹配的 B-，按 B- 处理
            if start is not None:
                entities.append([start, i, ent_type])
            start, ent_type = i, t
        else:
            if start is not None:
                entities.append([start, i, ent_type])
                start, ent_type = None, None
    if start is not None:
        entities.append([start, len(tags), ent_type])
    return entities


def process_sentence(
    sent: List[Tuple[Text, Text]], vocab: Dict[Text, int]
) -> Dict[Text, Any]:
    """将一个句子转为训练样本。"""
    chars = [c for c, _ in sent]
    tags = [t for _, t in sent]

    # 截断到 MAX_LEN，丢弃越界实体（与 NERDatasetForGP.max_len=512 对齐，避免标签矩阵越界）
    if len(chars) > MAX_LEN:
        chars = chars[:MAX_LEN]
        tags = tags[:MAX_LEN]

    # 转小写后查表（中文不受影响，ASCII 字母统一小写），与训练时 tokenizer 行为一致
    lower_chars = [c.lower() for c in chars]
    input_ids = convert_tokens_to_ids(lower_chars, vocab)

    labels = [LABEL2ID.get(t, 0) for t in tags]
    labels_ = tags

    raw_entities = extract_entities(tags)
    entity: List[List[Any]] = []
    for start, end, etype in raw_entities:
        if end <= MAX_LEN:
            text = "".join(chars[start:end])
            entity.append([text, start, end, etype])

    return {
        "token": chars,
        "input_ids": input_ids,
        "labels": labels,
        "labels_": labels_,
        "entity": entity,
    }


def save_jsonl(path: str, data: List[Dict[Text, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(line, ensure_ascii=False) for line in data))
    logging.info(f"saved: {path}, samples: {len(data)}")


def stat_data(data: List[Dict[Text, Any]]) -> None:
    """统计样本数、平均长度、实体类型分布。"""
    lengths = [len(s["token"]) for s in data]
    label_count: Dict[Text, int] = {}
    for s in data:
        for ent in s["entity"]:
            label_count[ent[3]] = label_count.get(ent[3], 0) + 1
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    logging.info(
        f"samples: {len(data)}, avg_len: {avg_len:.1f}, "
        f"max_len: {max(lengths) if lengths else 0}, entity_dist: {label_count}"
    )


def main():
    logging.basicConfig(
        format="[%(asctime)s %(filename)s:%(lineno)s] %(message)s",
        level=logging.INFO,
    )
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(base_dir, "resources", "vocab.txt")
    raw_dir = os.path.join(base_dir, "data", "raw_data", "peopledaily")
    save_dir = os.path.join(base_dir, "data", "ner_data")
    ent2id_path = os.path.join(base_dir, "resources", "ner_ent2id.json")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab not found: {vocab_path}")

    vocab = load_vocab(vocab_path)

    # 写出 ent2id
    os.makedirs(os.path.dirname(ent2id_path), exist_ok=True)
    with open(ent2id_path, "w", encoding="utf-8") as f:
        json.dump(ENT2ID, f, ensure_ascii=False, indent=2)
    logging.info(f"saved ent2id: {ent2id_path} -> {ENT2ID}")

    files = {
        "train": "train.char.bio.tsv",
        "dev": "dev.char.bio.tsv",
        "test": "test.char.bio.tsv",
    }
    all_stat = {}
    for name, fname in files.items():
        raw_path = os.path.join(raw_dir, fname)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(raw_path)
        sentences = read_bio_file(raw_path)
        logging.info(f"[{name}] raw sentences: {len(sentences)}")
        data = [process_sentence(s, vocab) for s in sentences]
        # 过滤空样本/无 entity 的样本可选保留（NERDatasetForGP 仅要求 entity 字段存在）
        data = [d for d in data if len(d["token"]) > 0]
        save_path = os.path.join(save_dir, f"{name}_ner")
        save_jsonl(save_path, data)
        stat_data(data)
        all_stat[name] = len(data)

    logging.info(f"done. ent_type_size={len(ENT2ID)}, summary: {all_stat}")


if __name__ == "__main__":
    main()
