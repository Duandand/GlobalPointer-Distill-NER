# coding=utf-8
import datetime
import json
import logging
import os
import sys
import time

import torch
from tensorboardX import SummaryWriter
from distill_loss import prediction_distillation, transformer_distillation
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange

sys.path.append('.')
from common_utils.attack import FGM
from common_utils.tokenization import FullTokenizer
from common_utils.utils import (device_config, random_seed_config,
                                restore_model, save_pytorch_model)
from config.arguments import parser
from config.hyper_parameters import HyperParams
from data_utils.dataset import NERDatasetForGP
from fine_tune.eval import model_test
from model.bert_optimization import BertAdam
from model.global_pointer_utils import MetricsCalculator
from model.modeling_bert import TinyBertGPForNER
from predict.pt2onnx import convert2onnx
from predict.pt2torchscript import save_torchscript


def create_dataloader(
    train_path: str,
    dev_path: str,
    test_path: str,
    ent2id_path: str,
    batch_size: int,
    local_rank: int,
):
    """Create dataloader for train, dev and test."""
    train_dataset = NERDatasetForGP(train_path, ent2id_path)
    dev_dataset = NERDatasetForGP(dev_path, ent2id_path)
    test_dataset = NERDatasetForGP(test_path, ent2id_path)

    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=train_dataset.collate_wrapper,
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=dev_dataset.collate_wrapper,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=test_dataset.collate_wrapper,
    )

    logging.info(
        f"Train dataset size is {len(train_dataset)}, train step is {len(train_dataloader)}"
    )
    logging.info(
        f"Dev dataset size is {len(dev_dataset)}, dev step is {len(dev_dataloader)}"
    )
    logging.info(
        f"Test dataset size is {len(test_dataset)}, test step is {len(test_dataloader)}"
    )
    return train_dataloader, dev_dataloader, test_dataloader


def prepare_data(
    save_data_path: str,
    ent2id_path: str,
    file_name_train: str,
    file_name_dev: str,
    file_name_test: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    local_rank: int,
):
    """Prepare NER training data. Requires pre-built JSONL files
    (train/dev/test) produced by data_utils/process_ner_data.py."""
    train_path = os.path.join(save_data_path, file_name_train)
    dev_path = os.path.join(save_data_path, file_name_dev)
    test_path = os.path.join(save_data_path, file_name_test)

    batch_size = batch_size // gradient_accumulation_steps
    if not (os.path.exists(train_path) and os.path.exists(dev_path)):
        raise FileNotFoundError(
            f"训练数据不存在: {train_path} 和/或 {dev_path}。"
            f"请先运行 `python data_utils/process_ner_data.py` 生成 NER 训练数据。"
        )
    logging.info("Training data is already exists....")
    return create_dataloader(
        train_path, dev_path, test_path, ent2id_path, batch_size, local_rank
    )


def optimizer_setting(
    model,
    weight_decay_rate,
    learning_rate,
    warmup_steps,
    schedule,
    num_train_steps,
):
    """Bert + global-pointer 分层学习率的 optimizer 设置。

    bert 主干参数用 learning_rate；global-pointer 的 dense_1/dense_2 用 1e-04；
    bias/LayerNorm 不做 weight decay。
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm"]
    other_param = ["dense_1", "dense_2"]  # global-pointer

    def has(name, keys):
        return any(nd in name for nd in keys)

    optimizer_grouped_parameters = [
        {
            # bert 主干（非 bias/LayerNorm）
            "params": [
                p for n, p in param_optimizer
                if not has(n, no_decay) and not has(n, other_param)
            ],
            "weight_decay": weight_decay_rate,
            "lr": learning_rate,
        },
        {
            # bert 主干的 bias/LayerNorm
            "params": [
                p for n, p in param_optimizer
                if has(n, no_decay) and not has(n, other_param)
            ],
            "weight_decay": 0,
            "lr": learning_rate,
        },
        {
            # global-pointer 的 bias/LayerNorm
            "params": [
                p for n, p in param_optimizer
                if has(n, other_param) and has(n, no_decay)
            ],
            "weight_decay": 0,
            "lr": 1e-04,
        },
        {
            # global-pointer 其余参数
            "params": [
                p for n, p in param_optimizer
                if has(n, other_param) and not has(n, no_decay)
            ],
            "weight_decay": weight_decay_rate,
            "lr": 1e-04,
        },
    ]

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=learning_rate,
        warmup=warmup_steps,
        t_total=num_train_steps,
        schedule=schedule,
    )

    return optimizer


def _num_train_steps(train_dataloader, gradient_accumulation_steps, num_train_epochs):
    return (
        len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
    )


def _log_train_step(writer, optimizer, global_step, tr_loss, tr_f1, tr_extra, time_start, step_print):
    """记录 tensorboard 与控制台训练日志。"""
    lr = optimizer.param_groups[0]["lr"]
    speed = (time.time() - time_start) / step_print
    writer.add_scalar("train/loss", tr_loss, global_step)
    writer.add_scalar("train/f1", tr_f1, global_step)
    writer.add_scalar("train/lr", lr, global_step)
    extra_str = ", ".join(f"{k}: {v:0.5f}" for k, v in sorted(tr_extra.items()))
    suffix = f", {extra_str}" if extra_str else ""
    logging.info(
        f"step: {global_step}, loss: {tr_loss:0.5f}, f1: {tr_f1:0.5f}, lr: {lr}, speed: {speed:0.2f}s/step{suffix}"
    )


def _run_training_loop(
    hp,
    model,
    optimizer,
    train_dataloader,
    num_train_epochs,
    writer,
    batch_step_fn,
    on_logging_fn=None,
    on_save_fn=None,
):
    """通用训练循环。

    单步的前向/反向（含对抗训练、蒸馏 loss 计算）由 `batch_step_fn(model, batch)`
    完成，返回 (loss_value, sample_f1, extra_logs_dict)；本函数负责梯度累积、
    optimizer 更新、tensorboard/控制台日志，以及通过回调触发保存：

    - on_logging_fn(global_step, avg_loss): 每 100 步日志点触发（如 trans 蒸馏按
      训练 loss 保存最优模型）。
    - on_save_fn(global_step): 每隔 save_checkpoint_interval 触发（如按 dev F1
      评估并保存最优模型）。
    """
    num_train_steps = _num_train_steps(
        train_dataloader, hp.gradient_accumulation_steps, num_train_epochs
    )
    global_step, local_step = 0, 0
    tr_loss, tr_f1 = 0.0, 0.0
    tr_extra = {}
    step_print = 100
    time_start = time.time()
    model.train()
    optimizer.zero_grad()
    logging.info("***** Running training *****")
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        for batch in train_dataloader:
            if global_step >= num_train_steps:
                break
            loss, sample_f1, extra = batch_step_fn(model, batch)
            tr_loss += loss
            tr_f1 += sample_f1
            for name, value in extra.items():
                tr_extra[name] = tr_extra.get(name, 0.0) + value
            local_step += 1
            if local_step % hp.gradient_accumulation_steps != 0:
                continue
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % step_print == 0:
                tr_loss /= step_print
                tr_f1 /= step_print
                tr_extra = {k: v / step_print for k, v in tr_extra.items()}
                if hp.local_rank in [0, -1]:
                    _log_train_step(
                        writer, optimizer, global_step,
                        tr_loss, tr_f1, tr_extra, time_start, step_print,
                    )
                    if on_logging_fn is not None:
                        on_logging_fn(global_step, tr_loss)
                tr_loss, tr_f1, tr_extra = 0.0, 0.0, {}
                time_start = time.time()
            if (
                on_save_fn is not None
                and global_step % hp.save_checkpoint_interval == 0
                and hp.local_rank in [-1, 0]
            ):
                on_save_fn(global_step)
                model.train()  # eval 会切换到 eval 模式，恢复训练模式
    return global_step


def _remove_stale_checkpoint(dir_checkpoint, last_save):
    """删除上一次保存的 checkpoint（若存在），只保留当前最优模型。"""
    if not last_save:
        return
    for prefix in ("pytorch_model", "pytorch_optimizer"):
        path = os.path.join(dir_checkpoint, f"{prefix}_{last_save}.pt")
        if os.path.isfile(path):
            os.remove(path)


def _teacher_batch_step(hp, fgm, metrics, device, n_gpu):
    """构造 teacher 单步训练函数：前向 + FGM 对抗训练。"""
    def batch_step(model, batch):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, labels, attention_mask)
        loss, logits = outputs[0], outputs[1]
        if n_gpu > 1:
            loss = loss.mean()
        if hp.gradient_accumulation_steps > 1:
            loss = loss / hp.gradient_accumulation_steps
        loss.backward()
        # FGM 对抗训练：扰动 embedding 后再前向一次并回传梯度。
        fgm.attack()
        loss_adv = model(input_ids, labels, attention_mask)[0]
        if n_gpu > 1:
            loss_adv = loss_adv.mean()
        if hp.gradient_accumulation_steps > 1:
            loss_adv = loss_adv / hp.gradient_accumulation_steps
        loss_adv.backward()
        fgm.restore()
        sample_f1 = metrics.get_sample_f1(logits, labels)
        return loss.item(), sample_f1.item(), {}

    return batch_step


def _distill_forward(model_S, model_T, batch, device):
    """学生/教师前向，返回 (labels, student输出四元组, teacher输出四元组)。"""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    student_outputs = model_S(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        is_student=True,
    )
    with torch.no_grad():
        teacher_outputs = model_T(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )
    return labels, student_outputs, teacher_outputs


def _trans_batch_step(hp, model_T, metrics, device, n_gpu):
    """构造 transformer 层蒸馏单步训练函数。"""
    def batch_step(model_S, batch):
        labels, student_out, teacher_out = _distill_forward(
            model_S, model_T, batch, device
        )
        student_loss, student_logits = student_out[0], student_out[1]
        teacher_loss = teacher_out[0]
        loss_one, attention_loss, hidden_loss = transformer_distillation(
            teacher_out[3],
            student_out[3],
            teacher_out[2],
            student_out[2],
            device,
            is_embed=True,
            is_hidden=True,
            is_attention=True,
        )
        # `0 *` 项是 DDP 下未使用参数的梯度占位
        loss = loss_one + 0 * student_loss + 0 * teacher_loss
        if n_gpu > 1:
            loss = loss.mean()
        if hp.gradient_accumulation_steps > 1:
            loss = loss / hp.gradient_accumulation_steps
        loss.backward()
        sample_f1, _, _ = metrics.get_evaluate_fpr(student_logits, labels)
        extra = {
            "attention_loss": attention_loss.item(),
            "hidden_loss": hidden_loss.item(),
            "student_loss": student_loss.item(),
        }
        return loss.item(), sample_f1, extra

    return batch_step


def _pred_batch_step(hp, model_T, metrics, device, n_gpu):
    """构造 pred-layer 蒸馏单步训练函数。"""
    def batch_step(model_S, batch):
        labels, student_out, teacher_out = _distill_forward(
            model_S, model_T, batch, device
        )
        student_loss, student_logits = student_out[0], student_out[1]
        teacher_loss = teacher_out[0]
        loss_two, predict_loss = prediction_distillation(
            teacher_out[1], student_logits, hp.temperature, pred_loss="mse"
        )
        loss = loss_two + 0 * student_loss + 0 * teacher_loss
        if n_gpu > 1:
            loss = loss.mean()
        if hp.gradient_accumulation_steps > 1:
            loss = loss / hp.gradient_accumulation_steps
        loss.backward()
        sample_f1, _, _ = metrics.get_evaluate_fpr(student_logits, labels)
        extra = {
            "predict_loss": predict_loss.item(),
            "student_loss": student_loss.item(),
        }
        return loss.item(), sample_f1, extra

    return batch_step


def train_teacher(hp, model, metrics, train_dataloader, eval_dataloader, device, n_gpu):
    """Train bert-gp teacher model. Returns the global step of best model."""
    writer = SummaryWriter(log_dir=os.path.join(hp.dir_summary, "teacher_model"))
    optimizer = optimizer_setting(
        model=model,
        weight_decay_rate=hp.weight_decay_rate,
        learning_rate=hp.learning_rate,
        warmup_steps=hp.warmup_steps,
        schedule="warmup_linear",
        num_train_steps=_num_train_steps(
            train_dataloader, hp.gradient_accumulation_steps, hp.num_train_epochs
        ),
    )
    fgm = FGM(model)
    batch_step_fn = _teacher_batch_step(hp, fgm, metrics, device, n_gpu)

    state = {"best_res": float("-inf"), "last_save": 0}

    def on_save(global_step):
        state["best_res"], state["last_save"] = model_eval_and_save(
            eval_dataloader,
            global_step,
            model,
            False,
            metrics,
            device,
            n_gpu,
            writer,
            state["best_res"],
            state["last_save"],
            os.path.join(hp.dir_checkpoint, "teacher_model"),
        )

    _run_training_loop(
        hp, model, optimizer, train_dataloader, hp.num_train_epochs,
        writer, batch_step_fn, on_save_fn=on_save,
    )
    return state["last_save"]


def student_trans_distill(
    hp, model_T, model_S, metrics, train_dataloader, device, n_gpu
):
    """Tinybert transformer distill. Returns the global step of best model."""
    writer = SummaryWriter(log_dir=os.path.join(hp.dir_summary, "tinybert_trans"))
    num_train_steps = _num_train_steps(
        train_dataloader,
        hp.gradient_accumulation_steps,
        hp.num_train_epochs_tinybert_one,
    )
    optimizer = optimizer_setting(
        model=model_S,
        weight_decay_rate=hp.weight_decay_rate,
        learning_rate=hp.learning_rate_tinybert_one,
        warmup_steps=hp.warmup_steps,
        schedule=None,
        num_train_steps=num_train_steps,
    )
    batch_step_fn = _trans_batch_step(hp, model_T, metrics, device, n_gpu)

    # 该阶段按 avg train loss 保存最优模型（loss 越低越好）
    state = {"best_res": float("inf"), "last_save": 0}
    dir_checkpoint = os.path.join(hp.dir_checkpoint, "tinybert_trans")

    def on_logging(global_step, avg_loss):
        if avg_loss >= state["best_res"]:
            return
        _remove_stale_checkpoint(dir_checkpoint, state["last_save"])
        save_pytorch_model(
            output_dir=dir_checkpoint,
            model=model_S,
            file_name=f"pytorch_model_{global_step}.pt",
            max_save=None,
            type="model",
        )
        state["best_res"] = avg_loss
        state["last_save"] = global_step

    _run_training_loop(
        hp, model_S, optimizer, train_dataloader,
        hp.num_train_epochs_tinybert_one, writer, batch_step_fn,
        on_logging_fn=on_logging,
    )
    return state["last_save"]


def student_pred_distill(
    hp, model_T, model_S, metrics, train_dataloader, eval_dataloader, device, n_gpu
):
    """Tinybert pred-layer distill. Returns the global step of best model."""
    writer = SummaryWriter(log_dir=os.path.join(hp.dir_summary, "tinybert_pred"))
    num_train_steps = _num_train_steps(
        train_dataloader,
        hp.gradient_accumulation_steps,
        hp.num_train_epochs_tinybert_two,
    )
    optimizer = optimizer_setting(
        model=model_S,
        weight_decay_rate=hp.weight_decay_rate,
        learning_rate=hp.learning_rate_tinybert_two,
        warmup_steps=hp.warmup_steps,
        schedule="warmup_linear",
        num_train_steps=num_train_steps,
    )
    batch_step_fn = _pred_batch_step(hp, model_T, metrics, device, n_gpu)

    state = {"best_res": float("-inf"), "last_save": 0}

    def on_save(global_step):
        state["best_res"], state["last_save"] = model_eval_and_save(
            eval_dataloader,
            global_step,
            model_S,
            True,
            metrics,
            device,
            n_gpu,
            writer,
            state["best_res"],
            state["last_save"],
            os.path.join(hp.dir_checkpoint, "tinybert_pred"),
        )

    _run_training_loop(
        hp, model_S, optimizer, train_dataloader,
        hp.num_train_epochs_tinybert_two, writer, batch_step_fn,
        on_save_fn=on_save,
    )
    return state["last_save"]


def model_eval_and_save(
    eval_dataloader,
    global_step,
    model,
    is_student,
    metrics,
    device,
    n_gpu,
    writer,
    best_res,
    last_save,
    dir_checkpoint,
):
    """Model eval on dev set and save the best model by F1.

    Returns:
        updated (best_res, last_save)
    """
    total_loss_eval = 0.0
    # corpus 级计数：跨 batch 累加 (X, Y, Z)，最后统一计算 P/R/F1
    count_x, count_y, count_z = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, labels, attention_mask, is_student)
            loss_eval, logits_eval = outputs[0], outputs[1]
            if n_gpu > 1:
                loss_eval = loss_eval.mean()
            total_loss_eval += loss_eval.item()
            x, y, z = metrics.get_evaluate_counts(logits_eval, labels)
            count_x += x
            count_y += y
            count_z += z
        avg_loss_eval = total_loss_eval / len(eval_dataloader)
        avg_f1_eval, avg_precision_eval, avg_recall_eval = metrics.counts_to_fpr(
            count_x, count_y, count_z
        )

    if writer:
        prefix = "eval"
        writer.add_scalar(f"{prefix}/loss", avg_loss_eval, global_step)
        writer.add_scalar(f"{prefix}/precision", avg_precision_eval, global_step)
        writer.add_scalar(f"{prefix}/recall", avg_recall_eval, global_step)
        writer.add_scalar(f"{prefix}/f1", avg_f1_eval, global_step)

    logging.info("step: " + str(global_step))
    logging.info("eval_loss: " + str(avg_loss_eval))
    logging.info("eval_precision: " + str(avg_precision_eval))
    logging.info("eval_recall: " + str(avg_recall_eval))
    logging.info("eval_f1: " + str(avg_f1_eval))

    if avg_f1_eval > best_res:
        # last_save 为 0 说明还没有保存过模型
        _remove_stale_checkpoint(dir_checkpoint, last_save)
        save_pytorch_model(
            output_dir=dir_checkpoint,
            model=model,
            file_name=f"pytorch_model_{global_step}.pt",
            max_save=None,
            type="model",
        )
        last_save = global_step
        best_res = avg_f1_eval
    return best_res, last_save


def _remap_tf_layer_norm(state_dict):
    """兼容老版 TF 风格 BERT 权重：LayerNorm 的 gamma/beta 重映射为 weight/bias。"""
    remapped = {}
    for k, v in state_dict.items():
        if k.endswith(".gamma"):
            k = k[: -len(".gamma")] + ".weight"
        elif k.endswith(".beta"):
            k = k[: -len(".beta")] + ".bias"
        remapped[k] = v
    return remapped


def create_model(hp, device, n_gpu):
    """Create teacher model and student model.

    The teacher model and the student model have the same structure,
    and the difference is that there is an extra layer of mapping in the hidden
    layer for student model.
    """
    # Create teacher model.
    model_T = TinyBertGPForNER(hp)
    logging.info(f"from {hp.dir_init_checkpoint} init model!!!!!")
    state_dict = torch.load(hp.dir_init_checkpoint, map_location=hp.map_location)
    state_dict = _remap_tf_layer_norm(state_dict)
    model_T.load_state_dict(state_dict, strict=False)
    model_T.to(device)

    # update student model param.
    hp.hidden_size = hp.hidden_size_tinybert
    hp.intermediate_size = hp.intermediate_size_tinybert
    hp.num_hidden_layers = hp.num_hidden_layers_tinybert
    model_S = TinyBertGPForNER(hp)
    model_S.to(device)
    logging.info("Student model is randomly initialized.")

    # multi-gpu
    if n_gpu > 1:
        model_T = DDP(
            model_T,
            device_ids=[hp.local_rank],
            output_device=hp.local_rank,
            find_unused_parameters=True,
        )
        model_S = DDP(
            model_S,
            device_ids=[hp.local_rank],
            output_device=hp.local_rank,
            find_unused_parameters=True,
        )

    return model_T, model_S


def main(hp):
    """
        Step1: Build dataloader from preprocessed JSONL data.
        Step2: Training teacher model.
        Step3: Test teacher model.
        Step4: Distill tinybert transformer.
        Step5: Distill tinybert pred.
        Step6: Test student model.
        Step7: Convert pt_model to torchscript.
        Step8: Convert pt_model to onnx

    Args:
        hp: hyper-parameter
    """
    # Get device.
    device, n_gpu = device_config(local_rank=hp.local_rank, no_cuda=hp.no_cuda)
    # Random seed.
    random_seed_config(hp.seed, n_gpu)
    # Build dataloader for train, dev and test.
    logging.info(
        "********** Step1 ======> Start load training data. ************"
    )
    train_dataloader, eval_dataloader, test_dataloader = prepare_data(
        save_data_path=hp.dir_training_data,
        ent2id_path=hp.dir_ent2id,
        file_name_train=hp.file_name_train,
        file_name_dev=hp.file_name_dev,
        file_name_test=hp.file_name_test,
        batch_size=hp.batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        local_rank=hp.local_rank,
    )

    # Create metrics.
    metrics = MetricsCalculator()
    # 实体类型 id -> 名称映射（测试时分类别输出用）
    with open(hp.dir_ent2id, "r", encoding="utf-8") as f:
        id2ent = {int(v): k for k, v in json.load(f).items()}
    # Create model.
    model_T, model_S = create_model(hp, device, n_gpu)

    # Train teacher model! It takes a long time!
    logging.info("********** Step2 ======> Start training teacher model. ************")
    last_save_teacher = train_teacher(
        hp, model_T, metrics, train_dataloader, eval_dataloader, device, n_gpu
    )
    # Init teacher model!
    model_T = restore_model(
        model_T,
        os.path.join(
            hp.dir_checkpoint, "teacher_model", f"pytorch_model_{last_save_teacher}.pt"
        ),
        device,
        hp.map_location,
    )

    # Test teacher model!
    logging.info("********** Step3 ======> Start test teacher model. ************")
    model_test(
        model_T, False, test_dataloader, last_save_teacher, metrics, device, n_gpu,
        id2ent=id2ent,
    )

    # Tinybert transformer distill!
    logging.info(
        "********** Step4 ======> Start distill tinybert transformer. ************"
    )
    last_save_tinybert_trans = student_trans_distill(
        hp, model_T, model_S, metrics, train_dataloader, device, n_gpu
    )

    # Init student model!
    model_S = restore_model(
        model_S,
        os.path.join(
            hp.dir_checkpoint,
            "tinybert_trans",
            f"pytorch_model_{last_save_tinybert_trans}.pt",
        ),
        device,
        hp.map_location,
    )

    # Tinybert pred distill!
    logging.info("********** Step5 ======> Start distill tinybert pred. ************")
    last_save_tinybert_pred = student_pred_distill(
        hp,
        model_T,
        model_S,
        metrics,
        train_dataloader,
        eval_dataloader,
        device,
        n_gpu,
    )
    # Restore student model!
    model_S = restore_model(
        model_S,
        os.path.join(
            hp.dir_checkpoint,
            "tinybert_pred",
            f"pytorch_model_{last_save_tinybert_pred}.pt",
        ),
        device,
        hp.map_location,
    )

    # Test Tinybert model!
    logging.info("********** Step6 ======> Start test student model. ************")
    model_test(
        model_S, True, test_dataloader, last_save_tinybert_pred, metrics, device, n_gpu,
        id2ent=id2ent,
    )

    # Tinybert model to torchscript and onnx.
    logging.info(
        "********** Step7 ======> Start convert pt_model to torchscript. ************"
    )
    save_torchscript(
        hp=hp,
        model_path=os.path.join(
            hp.dir_checkpoint,
            "tinybert_pred",
            f"pytorch_model_{last_save_tinybert_pred}.pt",
        ),
        save_path=os.path.join(hp.dir_checkpoint, "torchscript_model"),
        save_name="ner_tinybert_gp.pth",
        device="cpu",
    )

    # 转onnx
    logging.info(
        "********** Step8 ======> Start convert pt_model to onnx. ************"
    )
    text = "早上8点10分，我在上海浦东吃了一个苹果"
    tokenizer = FullTokenizer(
        vocab_file=hp.vocab_path,
        do_lower_case=True,
        num_seperation=True,
    )
    convert2onnx(
        hp=hp,
        model_path=os.path.join(
            hp.dir_checkpoint,
            "tinybert_pred",
            f"pytorch_model_{last_save_tinybert_pred}.pt",
        ),
        save_path=os.path.join(hp.dir_checkpoint, "onnx_model"),
        save_name="ner_tinybert_gp.onnx",
        text=text,
        tokenizer=tokenizer,
        ent_type_size=hp.ent_type_size,
        device="cpu",
    )


if __name__ == "__main__":
    parsed_args = parser.parse_args()
    hp = HyperParams.init_from_parsed_args(parsed_args)

    logfile = os.path.join(
        hp.dir_log,
        f"log_{hp.cmd}_{datetime.date.today()}_{os.path.basename(hp.dir_checkpoint)}.txt",
    )
    logging.basicConfig(
        format="[%(asctime)s %(filename)s:%(lineno)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    hp_string = hp.to_json_string()
    logging.info(hp_string)

    main(hp)
