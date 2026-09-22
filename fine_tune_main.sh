#!/bin/bash
# BERT-GlobalPointer 教师训练 + TinyBERT 两阶段蒸馏完整流水线（开源 PeopleDaily NER 数据）
# 用法：bash fine_tune_main.sh  （默认后台运行，日志写 ./logs/）
# 如需前台运行查看进度，去掉行末的 `&` 即可。

# 自动定位项目根目录，便于在本机或服务器上直接运行；如需固定路径可改为绝对路径
PROJ_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJ_PATH}"
echo "Enter project path: ${PROJ_PATH}"
export PYTHONPATH=`pwd`:$PYTHONPATH

# 优先使用项目自带的虚拟环境解释器（如 bert_gp_venv）；否则回退到系统 python
if [ -x "${PROJ_PATH}/bert_gp_venv/bin/python" ]; then
  PYTHON_BIN="${PROJ_PATH}/bert_gp_venv/bin/python"
else
  PYTHON_BIN="python"
fi
echo "Using python: ${PYTHON_BIN}"

# 运行所需目录
mkdir -p logs checkpoint summary init_ckpt

# peopledaily-ner
# 设备说明：fine_tune/main.py 启动时按 cuda -> mps(Apple Silicon) -> cpu 自动检测，无需手动配置。
# 下面这行仅在多卡 NVIDIA 服务器上有意义（默认用第 0 张卡，避免非分布式模式下多卡初始化报错），
# macOS/CPU 环境会自动忽略；如需换卡可在命令前覆盖，例如：CUDA_VISIBLE_DEVICES=3 bash fine_tune_main.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
train_dir=data/ner_data            # 已由 data_utils/process_ner_data.py 生成 train_ner/dev_ner/test_ner
data_name=ner
model_dir=NER_BERT_GP_peopledaily_pipline
log_name=nohup_logs_ner_peopledaily

# !! 训练前需准备教师模型初始化权重 init_ckpt/pytorch_model.bin（BERT-base-Chinese 预训练权重）
#    可从 https://huggingface.co/bert-base-chinese 下载 pytorch_model.bin 放到该路径
#    若文件名不同，请修改下行 --dir_init_checkpoint
nohup "${PYTHON_BIN}" -u ./fine_tune/main.py \
--cmd="fine_tune" \
--dir_training_data="${train_dir}" \
--dir_checkpoint="checkpoint/${model_dir}" \
--dir_init_checkpoint="init_ckpt/pytorch_model.bin" \
--dir_summary="summary/${model_dir}" \
--dir_log="logs" \
--dir_ent2id="resources/ner_ent2id.json" \
--vocab_path="resources/vocab.txt" \
--file_name_train="train_${data_name}" \
--file_name_dev="dev_${data_name}" \
--file_name_test="test_${data_name}" \
--local_rank=-1 \
--ent_type_size=4 \
--seed=42 \
--map_location="cpu" \
--weight_decay_rate=0.01 \
--learning_rate=3e-05 \
--warmup_steps=0.1 \
--num_train_epochs=3 \
--gradient_accumulation_steps=4 \
--batch_size=8 \
--save_checkpoint_interval=50 \
--output_attentions \
--output_hidden_states \
--temperature=1.0 \
--intermediate_size_tinybert=1200 \
--hidden_size_tinybert=312 \
--num_hidden_layers_tinybert=4 \
--learning_rate_tinybert_one=5e-05 \
--num_train_epochs_tinybert_one=20 \
--learning_rate_tinybert_two=3e-05 \
--num_train_epochs_tinybert_two=3 >./train.log 2>&1 &
echo "training launched, log: ./train.log"
