# coding=utf-8
"""加载torch模型预测"""
import os
import sys
import numpy as np
import torch
import re


sys.path.append(os.getcwd())

from common_utils.tokenization import FullTokenizer
from config.arguments import parser
from config.hyper_parameters import HyperParams
from model.modeling_bert import TinyBertGPForNER


def predict(text, model, id2ent, tokenizer, device):
    """返回识别出的实体列表，格式为 [(实体文本, 实体类型), ...]"""

    text_list = re.split("。", text)  # 对text进行按句号分割
    entities = []
    for text_each in text_list:
        if text_each == "":
            continue

        text_each += "。"

        # 获得模型输入
        tokens = list(text_each) # 分词
        input_ids = tokenizer.convert_tokens_to_ids(tokens) # 转id
        attention_mask = [1] * len(input_ids) # mask
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.float).to(device)

        # 模型预测
        res = model(input_ids=input_ids, labels=None, attention_mask=attention_mask)[0][0].data.cpu().numpy()

        # 解码模型输出：GlobalPointer 直接给出 (实体类型, start, end) 片段
        for l, start, end in zip(*np.where(res > 0)):
            entities.append((text_each[start : end + 1], id2ent[l]))

    return entities

def main(text_all_list):
    # 参数设置
    model_path = "./checkpoint/NER_BERT_GP_peopledaily_pipline/tinybert_pred/pytorch_model_15200.pt"
    vocab_file = "./resources/vocab.txt"
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    
    # load model
    parsed_args = parser.parse_args()
    hp = HyperParams.init_from_parsed_args(parsed_args)
    hp.hidden_size = hp.hidden_size_tinybert
    hp.intermediate_size = hp.intermediate_size_tinybert
    hp.num_hidden_layers = hp.num_hidden_layers_tinybert
    model = TinyBertGPForNER(hp)
    state_dict = torch.load(model_path, map_location=hp.map_location)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # tokenizer
    tokenizer = FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=True,
        num_seperation=True,
    )
    
    id2ent = {0:"LOC", 1:"ORG", 2:"PER", 3:"TIME"}
    for text_all in text_all_list:
        entities = predict(text=text_all, model=model, id2ent=id2ent, tokenizer=tokenizer, device=device)
        print(f"输入：{text_all}")
        print(f"实体：{entities}")
        print("*"*30)

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    text_all_list = [
        "早上8点10分，我在上海浦东吃了一个苹果。",
        "李明在北京的百度公司上班。",
    ]
    main(text_all_list)
