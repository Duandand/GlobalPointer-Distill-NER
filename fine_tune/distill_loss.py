# coding=utf-8
"""TinyBERT 两阶段蒸馏的损失函数。

- transformer_distillation: transformer 层蒸馏（attention + hidden 的 MSE）
- prediction_distillation: pred 层蒸馏（soft label 的 MSE / 交叉熵）
"""
import torch
from torch.nn import MSELoss


def transformer_distillation(
    teacher_attentions,
    student_attentions,
    teacher_hidden_states,
    student_hidden_states,
    device,
    is_embed=True,
    is_hidden=True,
    is_attention=True

):
    """Transformer layer distillation.

    Args:
        teacher_attentions: Attentions of teacher model.
        student_attentions: Attentions of student model.
        teacher_hidden_states: Hidden states of teacher model.
        student_hidden_states: Hidden states of student model.
        device: Device.

    Returns:
        loss, attention_loss, hidden_loss
    """
    # 初始化返回值
    loss = 0.0
    attention_loss = 0.0
    hidden_loss = 0.0

    # Prepare loss functions
    loss_mse = MSELoss()

    teacher_layer_num = len(teacher_attentions)
    student_layer_num = len(student_attentions)
    assert teacher_layer_num % student_layer_num == 0

    layers_per_block = int(teacher_layer_num / student_layer_num)

    # 进行attention layer的蒸馏： 0-->2, 1-->5, 2-->8, 3-->11 (最后一层)
    new_teacher_attentions = [
        teacher_attentions[i * layers_per_block + layers_per_block - 1]
        for i in range(student_layer_num)
    ]

    for student_att, teacher_att in zip(student_attentions, new_teacher_attentions):
        student_att = torch.where(
            student_att <= -1e2,
            torch.zeros_like(student_att).to(device),
            student_att,
        )
        teacher_att = torch.where(
            teacher_att <= -1e2,
            torch.zeros_like(teacher_att).to(device),
            teacher_att,
        )
        tmp_loss = loss_mse(student_att, teacher_att)

        attention_loss += tmp_loss

    # 进行hidden layer的蒸馏：0-->0(embedding layer), 1-->3, 2-->6, 3-->9, 4-->12（prediction layer）
    if not is_embed:
        # 不蒸馏emebd
        new_teacher_hidden = [
            teacher_hidden_states[i * layers_per_block]
            for i in range(1, student_layer_num + 1)
        ]
        new_student_hidden = student_hidden_states[1:]
    else:
        new_teacher_hidden = [
            teacher_hidden_states[i * layers_per_block]
            for i in range(student_layer_num + 1)
        ]
        new_student_hidden = student_hidden_states

    for student_hid, teacher_hid in zip(new_student_hidden, new_teacher_hidden):
        tmp_loss = loss_mse(student_hid, teacher_hid)
        hidden_loss += tmp_loss
        if not is_hidden:
            # 不蒸馏hidden，但是保留embed层
            break

    if not is_attention:
        # 不蒸馏attention
        loss = 0 * attention_loss + hidden_loss  # total loss
    else:
        # 这是正常操作
        loss = attention_loss + hidden_loss  # total loss

    return loss, attention_loss, hidden_loss


def prediction_distillation(teacher_logits, student_logits, temperature, pred_loss="ce"):
    """prediction distillation

    Args:
        teacher_logits: Logit outputs of teacher model.
        student_logits: Logit outputs of student model.
        temperature: Temperature for distilling.
        pred_loss: Cross entropy or Mean square error. Defaults to "ce".

    Returns:
        loss, predict_loss
    """
    # 初始化返回值
    loss = 0.0
    predict_loss = 0.0

    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1) # 对预测的结果求log
        targets_prob = torch.nn.functional.softmax(targets, dim=-1) # 求softmax，得到softmax的soft target的输出
        # ce: - [\sum_i (y_i * log(y^_i) )]
        return (-targets_prob * student_likelihood).mean() # 所有维度的平均

    # 第二阶段的蒸馏
    if pred_loss == "mse":
        # 使用mse损失
        loss_mse = MSELoss()
        predict_loss = loss_mse(
            student_logits / temperature, teacher_logits / temperature
        )
    else:
        # 使用ce损失
        predict_loss = soft_cross_entropy(
            student_logits / temperature, teacher_logits / temperature
        )

    loss = predict_loss

    return loss, predict_loss
