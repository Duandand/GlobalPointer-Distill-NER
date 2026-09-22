# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.append(".")
from common_utils.tokenization import FullTokenizer
from config.arguments import parser
from config.hyper_parameters import HyperParams
from model.modeling_bert import TinyBertGPForNER


def precess_text(text, tokenizer, ent_type_size, device):
    tokens = list(text.lower())
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    labels = np.zeros((ent_type_size, len(input_ids), len(input_ids)))

    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.float).to(device)
    labels = torch.tensor([labels], dtype=torch.long).to(device)

    return input_ids, labels, attention_mask


def convert2onnx(
    hp, model_path, save_path, save_name, text, tokenizer, ent_type_size, device
):
    os.makedirs(save_path, exist_ok=True)
    # tinybert
    hp.hidden_size = hp.hidden_size_tinybert
    hp.intermediate_size = hp.intermediate_size_tinybert
    hp.num_hidden_layers = hp.num_hidden_layers_tinybert
    hp.output_attentions = False
    hp.output_hidden_states = False
    model = TinyBertGPForNER(hp)
    state_dict = torch.load(model_path, map_location=hp.map_location)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    model.eval()
    # 处理数据
    input_ids, labels, attention_mask = precess_text(
        text, tokenizer, ent_type_size, device
    )
    # 转onnx（external_data=False 将权重内嵌为单一 .onnx 文件，便于分发）
    torch.onnx.export(
        model,
        (input_ids, labels, attention_mask),
        f=os.path.join(save_path, save_name),
        opset_version=18,
        do_constant_folding=True,
        verbose=False,
        external_data=False,
        input_names=["input_ids", "labels", "attention_mask"],
        output_names=["loss", "logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "max_len"},
            "labels": {0: "batch_size", 2: "max_len", 3: "max_len"},
            "attention_mask": {0: "batch_size", 1: "max_len"},
            "logits": {0: "batch_size", 2: "max_len", 3: "max_len"},
        },
    )


def load_onnx_predict(text, onnx_path, tokenizer, ent_type_size, device):
    """返回识别出的实体列表，格式为 [(实体文本, 实体类型), ...]"""
    id2ent = {0: "LOC", 1: "ORG", 2: "PER", 3: "TIME"}
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    input_ids, labels, attention_mask = precess_text(
        text, tokenizer, ent_type_size, device
    )
    res = sess.run(
        ["loss", "logits"],
        {
            "input_ids": input_ids.cpu().numpy(),
            "labels": labels.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        },
    )[1][
        0
    ]  # 第一个1表示取logits，第二个0表示取bs=1

    # 解码模型输出：GlobalPointer 直接给出 (实体类型, start, end) 片段
    entities = []
    for l, start, end in zip(*np.where(res > 0)):
        entities.append((text[start : end + 1], id2ent[l]))
    return entities


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parsed_args = parser.parse_args()
    hp = HyperParams.init_from_parsed_args(parsed_args)
    model_path = "./checkpoint/NER_BERT_GP_peopledaily_pipline/tinybert_pred/pytorch_model_15200.pt"
    save_path = "./checkpoint/NER_BERT_GP_peopledaily_pipline/onnx_model"
    save_name = "ner_tinybert_gp.onnx"
    device = "cpu"

    # tokenizer
    tokenizer = FullTokenizer(
        vocab_file="./resources/vocab.txt",
        do_lower_case=True,
        num_seperation=True,
    )

    # 转onnx
    text = "早上8点10分，我在上海浦东吃了一个苹果"
    convert2onnx(hp, model_path, save_path, save_name, text, tokenizer, 4, device)

    # 单条测试
    text = "早上8点10分，我在上海浦东吃了一个苹果"
    entities = load_onnx_predict(
        text,
        os.path.join(save_path, save_name),
        tokenizer,
        4,
        device,
    )
    print(f"输入：{text}")
    print(f"实体：{entities}")
