# coding=utf-8
import collections
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.append(".")


# 统一unicode编码
def convert_to_unicode(text: str):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


# 加载字典
def load_vocab(vocab_file: str):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


# token与id之间的相互转换
def convert_tokens_to_ids(vocab: Dict, items):
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(vocab["[UNK]"])
    return output


def sequence_padding(inputs: List, max_len: int, value=0):
    """padding the inputs to max_len with value

    Args:
        inputs (List): List need to be padded
        max_len (int): max_len
        value (int, optional): padding value. Defaults to 0.

    Returns:
        List: result
    """
    length = len(inputs)

    if length == max_len:
        return inputs

    pad_width = [value] * (max_len - length)

    return inputs + pad_width


def predict(text_list: list, vocab: Dict, id2ent: Dict[int, str], model, device: str):
    """batch predict

    Args:
        text_list (list): the list of text
        vocab (Dict): Vocabulary of converting tokens to ids
        id2ent (Dict[int, str]): id2ent
        model (nn.Module): model
        device (str): device

    Returns:
        List[List[Tuple[str, str]]]: 每条文本对应的实体列表，实体格式为 (实体文本, 实体类型)
    """

    # 获得模型输入
    max_len = np.max([len(text) for text in text_list])  # 当前batch内最大长度
    batch_size = len(text_list)  # bs

    batch_input_ids = []  # (bs, max_len)
    batch_attention_mask = []  # (bs, max_len)
    for batch in range(batch_size):
        text = text_list[batch]
        tokens = list(text.lower())  # 分词且转小写
        input_ids = convert_tokens_to_ids(vocab, tokens)  # 转id
        attention_mask = [1] * len(input_ids)  # mask

        input_ids = sequence_padding(input_ids, max_len, value=0)  # padding with 0
        attention_mask = sequence_padding(
            attention_mask, max_len, value=0
        )  # padding with 0

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)

    batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
    batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.float).to(
        device
    )

    # 预测
    res_all = (
        model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        .data.cpu()
        .numpy()
    )  # 模型输出

    # 解码模型输出：GlobalPointer 直接给出 (实体类型, start, end) 片段
    entities_all = []
    for batch in range(batch_size):
        res = res_all[batch]
        text = text_list[batch]
        entities = []
        for l, start, end in zip(*np.where(res > 0)):
            entities.append((text[start : end + 1], id2ent[l]))
        entities_all.append(entities)

    return entities_all


def main(text: str):
    # 参数设置
    id2ent = {0: "LOC", 1: "ORG", 2: "PER", 3: "TIME"}
    vocab = load_vocab("./resources/vocab.txt")
    torchscript_path = "./checkpoint/NER_BERT_GP_peopledaily_pipline/torchscript_model/ner_tinybert_gp.pth"

    # 加载模型
    model = torch.jit.load(torchscript_path)

    # 判断空串
    if not text:
        return []
    
    # 预测：按句号分割，过滤空串后同批预测，再汇总各句实体
    text_list = [
        text_each + "。" for text_each in text.split("。") if text_each
    ]
    entities_all = predict(text_list, vocab, id2ent, model, device="cpu")
    entities = [entity for entities_each in entities_all for entity in entities_each]

    return entities


if __name__ == "__main__":
    text = "早上8点10分，我在上海浦东吃了一个苹果。"
    print(f"输入：{text}")
    print(f"实体：{main(text)}")
