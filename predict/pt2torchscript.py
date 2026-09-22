# coding=utf-8
import os
import sys

import torch
from torch import jit

sys.path.append(".")
from config.arguments import parser
from config.hyper_parameters import HyperParams
from model.modeling_bert_torchscript import (
    TinyBertGPForNER as TinyBertGPForNER_torchscript,
)


def save_torchscript(hp, model_path, save_path, save_name, device):
    os.makedirs(save_path, exist_ok=True)

    # update tinybert param
    hp.hidden_size = hp.hidden_size_tinybert
    hp.intermediate_size = hp.intermediate_size_tinybert
    hp.num_hidden_layers = hp.num_hidden_layers_tinybert

    model = TinyBertGPForNER_torchscript(hp)
    state_dict = torch.load(model_path, map_location=hp.map_location)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    model.eval()

    model = jit.script(model)
    jit.save(model, os.path.join(save_path, save_name))


if __name__ == "__main__":
    parsed_args = parser.parse_args()
    hp = HyperParams.init_from_parsed_args(parsed_args)
    save_path = "./checkpoint/NER_BERT_GP_peopledaily_pipline/torchscript_model"
    save_name = "ner_tinybert_gp.pth"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    model_path = "./checkpoint/NER_BERT_GP_peopledaily_pipline/tinybert_pred/pytorch_model_15200.pt"
    # 保存模型
    save_torchscript(hp, model_path, save_path, save_name, device)

    # torchscript模型保存的device和load的device必须一致
    # torchscript模型保存的torch版本和load的torch版本要注意，这里使用的是torch==1.5.0
    # 若是高版本torch保存的torchscript，低版本torch环境下load不成功；
    # 若是低版本torch保存的torchscript，高版本torch环境下load可以成功。
