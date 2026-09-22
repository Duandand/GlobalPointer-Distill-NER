# coding=utf-8
import logging
import torch
from tqdm import tqdm


def model_test(
    model,
    is_student,
    test_dataloader,
    global_step,
    metrics,
    device,
    n_gpu,
    id2ent=None,
):
    """在测试集上评估，输出 corpus 级实体指标（可与论文/公开基准直接对比）。

    Args:
        id2ent (dict): 实体类型 id -> 名称（如 {0: "LOC", ...}），来自
            resources/ner_ent2id.json；为 None 时按类型 id 展示。
    """
    total_loss_eval = 0.0
    # corpus 级计数：跨 batch 累加 (X, Y, Z)，最后统一计算 P/R/F1
    count_x, count_y, count_z = 0, 0, 0
    class_counts = {}  # label_id -> [X, Y, Z]
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating", disable=None):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, labels, attention_mask, is_student)
            loss_eval, logits_eval = outputs[0], outputs[1]
            if n_gpu > 1:
                loss_eval = loss_eval.mean()
            total_loss_eval += loss_eval.item()
            # 实体级命中计数（边界+类型完全匹配）
            x, y, z = metrics.get_evaluate_counts(logits_eval, labels)
            count_x += x
            count_y += y
            count_z += z
            # 每一类的计数统计
            res = metrics.get_evaluate_counts_by_class(logits_eval, labels)
            for label, (x_, y_, z_) in res.items():
                counts = class_counts.setdefault(label, [0, 0, 0])
                counts[0] += x_
                counts[1] += y_
                counts[2] += z_

        avg_loss_eval = total_loss_eval / len(test_dataloader)
        f1, precision, recall = metrics.counts_to_fpr(count_x, count_y, count_z)

    # 每一类上的效果
    for label in sorted(class_counts.keys()):
        f1_, p_, r_ = metrics.counts_to_fpr(*class_counts[label])
        name = id2ent.get(label, str(label)) if id2ent else str(label)
        logging.info(
            f"type {name}: precision: {p_:0.5f}, recall: {r_:0.5f}, f1: {f1_:0.5f}"
        )

    logging.info("step: " + str(global_step))
    logging.info("eval_loss: " + str(avg_loss_eval))
    logging.info("eval_precision: " + str(precision))
    logging.info("eval_recall: " + str(recall))
    logging.info("eval_f1: " + str(f1))
