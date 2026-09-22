[TOC]

## 命名实体识别（NER）

## 0 背景

对中文文本进行命名实体识别（Named Entity Recognition, NER）。采用 BERT-GlobalPointer 结构训练教师模型，随后通过 TinyBERT 两阶段蒸馏得到轻量学生模型，便于线上部署。

> **技术来源**：GlobalPointer 实体识别框架出自苏剑林（科学空间）的 [GlobalPointer 论文与博客](https://kexue.fm/archives/8373)；TinyBERT 两阶段蒸馏方法出自 [TinyBERT 论文](https://arxiv.org/abs/1909.10351)；BERT 预训练模型由 Google 提出，本仓库使用 `bert-base-chinese` 权重。本项目在上述工作基础上进行工程化整合。

## 1 整体流程

整体流程如下：

- 首先对原始 BIO 标注数据进行处理，转换为模型训练所需的 JSONL 格式；
- 其次基于生成的训练数据，采用 BERT-GlobalPointer 结构训练教师模型，并验证其在测试集上的效果；
- 随后针对训练好的教师模型进行 TinyBERT 蒸馏，蒸馏过程分两个阶段：transformer 层蒸馏和 pred 层蒸馏，对蒸馏好的学生模型在测试集上进行验证；
- 最后将学生模型转为 torchscript 和 onnx，以便用于线上部署，并提供基于 torchscript 和 onnx 的推理脚本。

下面对部分环节作详细说明。

## 2 数据处理

### 2.1 原始数据

本仓库以人民日报（PeopleDaily）公开中文 NER 数据集为例进行演示，char-level BIO 格式，示例：

```text
海 O
钓 O
威 B-LOC
城 I-LOC
， O
他 O
将 O
于 O
今 O
年 O
底 O
正 O
式 O
退 O
役 O
。 O
```

通过 `B-`/`I-` 前缀标注实体边界。

**数据集来源**：`shibing624/nerpy` 项目整理的 PEOPLE 数据集，共约 200 万字。

- 仓库目录：[shibing624/nerpy/examples/data/people](https://github.com/shibing624/nerpy/tree/main/examples/data/people)
- 直接下载链接：
  - [train.char.bio.tsv](https://raw.githubusercontent.com/shibing624/nerpy/main/examples/data/people/train.char.bio.tsv)
  - [dev.char.bio.tsv](https://raw.githubusercontent.com/shibing624/nerpy/main/examples/data/people/dev.char.bio.tsv)
  - [test.char.bio.tsv](https://raw.githubusercontent.com/shibing624/nerpy/main/examples/data/people/test.char.bio.tsv)

**下载后放置路径**：将上述三个文件放入 `data/raw_data/peopledaily/` 目录下，即：

```
data/raw_data/peopledaily/
├── train.char.bio.tsv
├── dev.char.bio.tsv
└── test.char.bio.tsv
```

数据集统计：

| split | sentences | avg length |
| :---- | --------: | ---------: |
| train | 46289     | ~42        |
| dev   | 1059      | ~42        |
| test  | 4570      | ~42        |

### 2.2 数据处理流程

运行数据处理脚本，将 BIO 格式转换为模型所需的 JSONL 格式。脚本会自动从 `data/raw_data/peopledaily/` 读取三个 tsv 文件，并将结果输出到 `data/ner_data/`，无需额外传参：

```shell
python data_utils/process_ner_data.py
```

生成的 JSONL 每条数据包含以下字段：

```json
{
  "token": ["海", "钓", "威", "城", "，", ...],
  "input_ids": [...],
  "labels": [...],
  "labels_": [...],
  "entity": [["威海城", 2, 5, "LOC"]]
}
```

其中 `entity` 字段为 `[文本, 起始下标, 结束下标(不包含), 实体类型]`。

处理后的数据存放于 `data/ner_data/` 目录下：

- `train_ner`
- `dev_ner`
- `test_ner`

实体类型映射见 `resources/ner_ent2id.json`：

```json
{"LOC": 0, "ORG": 1, "PER": 2, "TIME": 3}
```

## 3 模型训练

### 3.1 环境准备

安装依赖：

```shell
pip install -r requirements.txt
```

下载 BERT-base-Chinese 预训练权重（来源：Google BERT 团队，[HuggingFace 镜像](https://huggingface.co/bert-base-chinese)），放置于 `init_ckpt/pytorch_model.bin`。

训练设备无需手动配置：启动时按 **CUDA → MPS（Apple Silicon）→ CPU** 的顺序自动检测并选择。

### 3.2 启动训练

```shell
bash fine_tune_main.sh
```

训练日志默认输出到 `./train.log`，可通过 `tail -f train.log` 实时查看。也可通过 `tensorboard --logdir ./summary` 查看训练曲线。

训练完成后，`checkpoint/` 目录下会产生以下模型文件：

- `teacher_model/***.pt` — 训练好的教师模型。
- `tinybert_trans/***.pt` — TinyBERT transformer 层蒸馏后的模型。
- `tinybert_pred/***.pt` — TinyBERT pred 层蒸馏后的模型，即最终学生模型。
- `onnx_model/***.onnx` — 学生模型转换后的 onnx 模型。
- `torchscript_model/***.pth` — 学生模型转换后的 torchscript 模型。

### 3.3 关键超参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | -----: |
| `ent_type_size` | 实体类型数 | 4 |
| `batch_size` | 批大小 | 8 |
| `gradient_accumulation_steps` | 梯度累积步数 | 4 |
| `learning_rate` | 教师学习率 | 3e-05 |
| `num_train_epochs` | 教师训练轮数 | 3 |
| `hidden_size_tinybert` | 学生模型隐藏层维度 | 312 |
| `num_hidden_layers_tinybert` | 学生模型层数 | 4 |

## 4 换用其他数据集

本框架不绑定具体数据集，换用其他 char-level BIO 格式的 NER 数据时，按以下清单逐处修改即可（以换成 3 类实体 `PER`/`LOC`/`ORG` 的数据集为例）。

### 4.1 放置原始数据

将新数据集的 train/dev/test 三个 BIO 文件放入 `data/raw_data/<你的数据集名>/`：

```
data/raw_data/<你的数据集名>/
├── train.char.bio.tsv
├── dev.char.bio.tsv
└── test.char.bio.tsv
```

### 4.2 修改数据处理脚本 `data_utils/process_ner_data.py`

| 位置 | 说明 |
| :--- | :--- |
| `ENT2ID` | 实体类型 → 标签矩阵行索引，按新数据集的实体类型从 0 开始连续编号，如 `{"PER": 0, "LOC": 1, "ORG": 2}` |
| `LABEL2ID` | BIO 序列标签映射，标签数为 `2 × 实体类型数 + 1`（`O` + 每类实体的 `B-`/`I-`），如 3 类实体共 7 个标签 |
| `raw_dir` | 原始数据目录，改为 `data/raw_data/<你的数据集名>` |
| `files` 字典 | 若文件名与 `train.char.bio.tsv` 等不同，需同步修改 |

修改后重新运行数据处理脚本，会自动重新生成 `resources/ner_ent2id.json` 与 `data/ner_data/` 下的 JSONL，无需手工维护：

```shell
python data_utils/process_ner_data.py
```

### 4.3 修改训练脚本 `fine_tune_main.sh`

| 参数 | 说明 | 3 类实体示例 |
| :--- | :--- | :--- |
| `--ent_type_size` | 实体类型数，与 `ENT2ID` 一致 | `3` |

可选：同步修改 `model_dir`/`log_name`/`data_name`，使不同数据集的 checkpoint、日志目录相互独立，避免互相覆盖。

### 4.4 修改推理脚本 `predict/`

3 个推理脚本中的实体类型反查表为硬编码，需与新的 `ENT2ID` 保持一致：

- [pt_predict.py](predict/pt_predict.py)
- [pt2onnx.py](predict/pt2onnx.py)
- [torchscript_predict.py](predict/torchscript_predict.py)

均包含类似 `id2ent = {0: "LOC", 1: "ORG", 2: "PER", 3: "TIME"}` 的行，按新数据集修改即可，如 `id2ent = {0: "PER", 1: "LOC", 2: "ORG"}`。

### 4.5 无需改动

- `resources/ner_ent2id.json`：由数据处理脚本自动重新生成。
- 模型结构与训练代码：实体类型数由 `ent_type_size` 驱动，任意实体集合均可训练。

## 5 模型效果

### 5.1 本仓库实验结果

数据集：人民日报 NER（PeopleDaily，char-level，`LOC`/`ORG`/`PER`/`TIME` 四类实体）。

实验环境：macOS（Apple Silicon，MPS），batch_size=8；Teacher 训练 3 个 epoch，Transformer 蒸馏 20 个 epoch，Pred 层蒸馏 3 个 epoch，全流程约 9 小时。

测试集实体级 corpus 级 P/R/F1（边界+类型完全匹配）：

| 模型 | 实体类型 | Precision | Recall | F1 |
| :--- | :--- | ---: | ---: | ---: |
| Teacher：BERT-base + GlobalPointer（step 8350） | LOC | 0.9540 | 0.9275 | 0.9405 |
| | ORG | 0.9326 | 0.9427 | 0.9376 |
| | PER | 0.9800 | 0.9901 | 0.9850 |
| | TIME | 0.9953 | 0.9623 | 0.9785 |
| | **总体** | **0.9684** | **0.9577** | **0.9630** |
| Student：TinyBERT + GlobalPointer（蒸馏后，step 15200） | LOC | 0.9294 | 0.8907 | 0.9096 |
| | ORG | 0.8849 | 0.8932 | 0.8891 |
| | PER | 0.9685 | 0.9558 | 0.9621 |
| | TIME | 0.9899 | 0.9411 | 0.9649 |
| | **总体** | **0.9480** | **0.9235** | **0.9356** |

蒸馏后的 TinyBERT 学生模型相比 BERT-base 教师模型 F1 仅下降约 2.7 个百分点，而参数量与推理开销大幅降低。

### 5.2 公开参考基准

在同类人民日报（PeopleDaily）char-level BIO 数据（`LOC`/`ORG`/`PER`/`TIME` 四类实体）上，公开效果供参考：

| 模型 | 数据集 | F1 |
| :--- | :--- | -----: |
| BERT-Softmax 序列标注（nerpy 官方模型 `bert4ner-base-chinese`） | PEOPLE 测试集 | 95.25 |
| GlobalPointer（w/ RoPE，苏剑林原博客实验） | 人民日报 NER 测试集 | 95.51 |
| CRF（同上博客对照组） | 人民日报 NER 测试集 | 95.46 |

> 注：各工作的数据切分不完全一致，以上数字仅供参考。BERT 系模型在该数据集上的公开水平大致为 F1 93~96%。
>
> 本仓库的评估指标为**实体级 corpus 级 P/R/F1**（边界+类型完全匹配，全测试集累计计数后统一计算），与上表口径一致，可直接对比。

## 6 模型推理

支持三种推理方式：torch 模型推理、onnx 推理以及 torchscript 推理。

- torch 版本推理

  ```shell
  python ./predict/pt_predict.py
  ```

- onnx 版本推理

  ```shell
  # torch 模型转 onnx 并预测
  python ./predict/pt2onnx.py
  ```

- torchscript 版本推理

  ```shell
  # torch 模型转 torchscript
  python ./predict/pt2torchscript.py

  # 加载 torchscript 模型并预测
  python ./predict/torchscript_predict.py
  ```

推理时，输入为 text，输出为识别出的实体列表。例如：

```text
输入：海钓威海城，他将于今年底正式退役。

输出：[("威海城", "LOC"), ("今年底", "TIME")]
```

## 7 文件说明

```
.
├── checkpoint
│   └── NER_BERT_GP_peopledaily_pipline
│       ├── teacher_model
│       ├── tinybert_trans
│       ├── tinybert_pred
│       ├── onnx_model
│       └── torchscript_model
├── common_utils
│   ├── attack.py
│   ├── tokenization.py
│   └── utils.py
├── config
│   ├── arguments.py
│   └── hyper_parameters.py
├── data
│   ├── raw_data
│   │   └── peopledaily
│   │       ├── train.char.bio.tsv
│   │       ├── dev.char.bio.tsv
│   │       └── test.char.bio.tsv
│   └── ner_data
│       ├── train_ner
│       ├── dev_ner
│       └── test_ner
├── data_utils
│   ├── dataset.py
│   └── process_ner_data.py
├── fine_tune
│   ├── main.py
│   ├── eval.py
│   └── distill_loss.py
├── init_ckpt
│   └── pytorch_model.bin
├── model
│   ├── bert_optimization.py
│   ├── checkpoint.py
│   ├── global_pointer_utils.py
│   ├── modeling_bert.py
│   └── modeling_bert_torchscript.py
├── predict
│   ├── pt2onnx.py
│   ├── pt2torchscript.py
│   ├── pt_predict.py
│   └── torchscript_predict.py
├── resources
│   ├── ner_ent2id.json
│   └── vocab.txt
├── fine_tune_main.sh
├── requirements.txt
├── LICENSE
└── README.md
```

## 参考

### 论文与技术博客

- **GlobalPointer** — 苏剑林，[GlobalPointer：用统一的方式处理嵌套和非嵌套 NER](https://kexue.fm/archives/8373)。本仓库的实体识别头与标签矩阵构建方式基于该文。
- **TinyBERT** — Jiao et al., [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)。本仓库的两阶段蒸馏（transformer 层 + pred 层）参考该方法。
- **BERT** — Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)。教师模型的编码器基于 BERT。

### 预训练模型

- **bert-base-chinese** — Google BERT 团队，[HuggingFace 模型页](https://huggingface.co/bert-base-chinese)。请将下载的 `pytorch_model.bin` 放至 `init_ckpt/`。

### 数据集

- **PeopleDaily（人民日报）NER** — 由 `shibing624/nerpy` 项目整理，char-level BIO 标注，包含 `LOC`/`ORG`/`PER`/`TIME` 四类实体。
  - 仓库目录：[shibing624/nerpy/examples/data/people](https://github.com/shibing624/nerpy/tree/main/examples/data/people)
  - 项目主页：[nerpy: Named Entity Recognition toolkit](https://github.com/shibing624/nerpy)

### 开源协议

本仓库代码基于 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 开源，完整协议文本见 [LICENSE](LICENSE)。

所依赖的预训练模型（bert-base-chinese）及数据集（PeopleDaily NER）不包含在本许可证内，请遵循其各自的许可协议。
