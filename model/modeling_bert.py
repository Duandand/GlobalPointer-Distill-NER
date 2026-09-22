# coding=utf-8
import logging
import math
import sys

import torch
from torch import nn
from model.checkpoint import checkpoint_sequential
from model.global_pointer_utils import tril_onnx, SinusoidalPositionEmbedding, gp_loss_func


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


# try:
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
# except (ImportError, AttributeError) as e:
#     logging.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, hp):
        super(BertEmbeddings, self).__init__()
        self.hp = hp
        self.word_embeddings = nn.Embedding(hp.vocab_size, hp.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(hp.max_position_embeddings, hp.hidden_size)
        if hp.do_random_next:
            self.token_type_embeddings = nn.Embedding(hp.type_vocab_size, hp.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)

    # input_ids: [bts,seq_len]
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # [seq_len]
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # [bts,seq_len]
        if self.hp.do_random_next and token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)  # [bts,seq_len]

        words_embeddings = self.word_embeddings(input_ids)  # [bts,seq_len,hdsz]
        position_embeddings = self.position_embeddings(position_ids)  # [bts,seq_len,hdsz]
        embeddings = words_embeddings + position_embeddings  # [bts,seq_len,hdsz]
        if self.hp.do_random_next:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)  # [bts,seq_len,hdsz]
            embeddings = embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)  # [bts,seq_len,hdsz]
        embeddings = self.dropout(embeddings)  # [bts,seq_len,hdsz]
        return embeddings



class BertSelfAttention(nn.Module):
    def __init__(self, hp):
        super(BertSelfAttention, self).__init__()
        if hp.hidden_size % hp.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hp.hidden_size, hp.num_attention_heads))
        self.output_attentions = hp.output_attentions

        self.num_attention_heads = hp.num_attention_heads
        self.attention_head_size = int(hp.hidden_size / hp.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hp.hidden_size, self.all_head_size)
        self.key = nn.Linear(hp.hidden_size, self.all_head_size)
        self.value = nn.Linear(hp.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(hp.attention_probs_dropout_prob)

    # x: [bts,seq_len,hdsz]
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [bts,seq_len,head_num,head_size]
        return x.permute(0, 2, 1, 3)  # [bts,head_num,seq_len,head_size]

    # hidden_states: [bts,seq_len,hdsz]
    # attention_mask: [bts,1,1,seq_len]
    # head_mask: None
    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)  # [bts,seq_len,hdsz]
        mixed_key_layer = self.key(hidden_states)  # [bts,seq_len,hdsz]
        mixed_value_layer = self.value(hidden_states)  # [bts,seq_len,hdsz]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bts,head_num,seq_len,head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [bts,head_num,seq_len,head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [bts,head_num,seq_len,head_size]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [bts,head_num,seq_len,seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bts,head_num,seq_len,seq_len]
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask  # [bts,head_num,seq_len,seq_len]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bts,head_num,seq_len,seq_len]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # [bts,head_num,seq_len,seq_len]

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # [bts,head_num,seq_len,head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bts,seq_len,head_num,head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [bts,seq_len,hdsz]

        # ([bts,seq_len,hdsz],)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, hp):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hp.hidden_size, hp.hidden_size)
        self.LayerNorm = BertLayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)

    # hidden_states: [bts, seq_len, hdsz]
    # input_tensor: [bts, seq_len, hdsz]
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # [bts, seq_len, hdsz]
        hidden_states = self.dropout(hidden_states)  # [bts, seq_len, hdsz]
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # [bts, seq_len, hdsz]
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, hp):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(hp)
        self.output = BertSelfOutput(hp)

    # input_tensor: [bts,seq_len,hdsz]
    # attention_mask: [bts,1,1,seq_len]
    # head_mask: None
    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)  # ([bts,seq_len,hdsz],)
        attention_output = self.output(self_outputs[0], input_tensor)  # [bts, seq_len, hdsz]
        # ([bts,seq_len,hdsz],)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, hp):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hp.hidden_size, hp.intermediate_size)
        if isinstance(hp.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(hp.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[hp.hidden_act]
        else:
            self.intermediate_act_fn = hp.hidden_act

    # hidden_states: [bts,seq_len,hdsz]
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # [bts,seq_len,mdsz]
        hidden_states = self.intermediate_act_fn(hidden_states)  # [bts,seq_len,mdsz]
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, hp):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(hp.intermediate_size, hp.hidden_size)
        self.LayerNorm = BertLayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)

    # hidden_states: [bts,seq_len,mdsz]
    # input_tensor: [bts,seq_len,hdsz]
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # [bts,seq_len,hdsz]
        hidden_states = self.dropout(hidden_states)  # [bts,seq_len,hdsz]
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # [bts,seq_len,hdsz]
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, hp):
        super(BertLayer, self).__init__()
        self.hp = hp
        self.attention = BertAttention(hp)
        self.intermediate = BertIntermediate(hp)
        self.output = BertOutput(hp)

    # hidden_states: [bts,seq_len,hdsz]
    # attention_mask: [bts,1,1,seq_len]
    # head_mask: None
    def forward(self, hidden_states, attention_mask):
        attention_outputs = self.attention(hidden_states, attention_mask, None)  # ([bts,seq_len,hdsz],)
        attention_output = attention_outputs[0]  # [bts,seq_len,hdsz]
        intermediate_output = self.intermediate(attention_output)  # [bts,seq_len,mdsz]
        layer_output = self.output(intermediate_output, attention_output)  # [bts,seq_len,hdsz]
        if self.hp.use_checkpoint_sequential:
            outputs = (layer_output, attention_mask.detach())
        else:
            # ([bts,seq_len,hdsz],)
            outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, hp):
        super(BertEncoder, self).__init__()
        self.output_attentions = hp.output_attentions
        self.output_hidden_states = hp.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(hp) for _ in range(hp.num_hidden_layers)])

    # hidden_states: [bts,seq_len,hdsz]
    # attention_mask: [bts,1,1,seq_len]
    # head_mask: [num_layer]
    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask)  # ([bts,seq_len,hdsz],)
            hidden_states = layer_outputs[0]  # [bts,seq_len,hdsz]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # ([bts,seq_len,hdsz],)
        return outputs  # outputs, (hidden states), (attentions)


class BertPooler(nn.Module):
    def __init__(self, hp):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hp.hidden_size, hp.hidden_size)
        self.activation = nn.Tanh()

    # hidden_states: [bts,seq_len,hdsz]
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]  # [bts,hdsz]
        pooled_output = self.dense(first_token_tensor)  # [bts,hdsz]
        pooled_output = self.activation(pooled_output)  # [bts,hdsz]
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, hp):
        super(BertModel, self).__init__()
        self.hp = hp
        self.embeddings = BertEmbeddings(hp)
        self.encoder = BertEncoder(hp)
        self.pooler = BertPooler(hp)

    # input_ids: [bts,seq_len]
    # token_type_ids: [bts,seq_len]
    # attention_mask: [bts,seq_len]
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        # import pdb;pdb.set_trace()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.hp.do_random_next and token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)  # [bts,seq_len]

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [bts,1,1,seq_len]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # [bts,1,1,seq_len]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.hp.num_hidden_layers  # [num_layer]

        # [bts,seq_len,hdsz]
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if self.hp.use_checkpoint_sequential:
            # [bts,seq_len,hdsz]
            encoder_outputs = checkpoint_sequential(self.encoder.layer, self.hp.chunks,
                                                    embedding_output, extended_attention_mask)
        else:
            # ([bts,seq_len,hdsz],)
            encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoder_outputs[0]  # [bts,seq_len,hdsz]
        pooled_output = self.pooler(sequence_output)  # [bts,hdsz]

        # add hidden_states and attentions if they are here
        # ([bts,seq_len,hdsz],[bts,hdsz])
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertGPForNER(nn.Module):
    def __init__(self, hp):
        super(BertGPForNER, self).__init__()
        self.hp = hp
        self.bert = BertModel(hp) # init bert as encoder
        self.ent_type_size = hp.ent_type_size # ent_type_size 7
        self.inner_dim = 64 # inner_dim: 64
        self.hidden_size = hp.hidden_size # bert hidden size
        self.RoPE = True
        
        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, self.ent_type_size * 2) # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hp.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)


    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        # mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        mask = tril_onnx(torch.ones_like(logits), diagonal=-1) # for onnx
        logits = logits - mask * 1e12
        return logits
 
    def forward(self, input_ids, labels, attention_mask):

        context_outputs = self.bert(input_ids, position_ids=None, token_type_ids=None,
                            attention_mask=attention_mask, head_mask=None) # bert encoder
        
        if self.hp.output_attentions and self.hp.output_hidden_states:
            sequence_output, _, hidden_states, attentions = context_outputs
            outputs = (hidden_states, attentions)
        else:
            sequence_output = context_outputs[0]
            outputs = ()
        
        sequence_output_dense = self.dense_1(sequence_output)
        qw, kw = sequence_output_dense[...,::2], sequence_output_dense[..., 1::2] #从0,1开始间隔为2
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(sequence_output_dense)
            # cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            # sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            cos_pos = pos[..., 1::2].repeat(1, 1, 2) # ddd onnx 最后一个维度复制2次
            sin_pos = pos[..., ::2].repeat(1, 1, 2) # ddd onnx
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
            
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim**0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(sequence_output)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None] #logits[:, None] 增加一个维度
        logits = self.add_mask_tril(logits, mask=attention_mask)
        
        outputs = (logits, ) + outputs
        
        if labels is not None:
            loss = gp_loss_func(logits, labels)
            outputs = (loss, )+ outputs
        
        return outputs # (loss), logits, (hidden_states, attentions)


class TinyBertGPForNER(nn.Module):
    def __init__(self, hp):
        super(TinyBertGPForNER, self).__init__()
        self.hp = hp
        self.bert = BertModel(hp) # init bert as encoder
        self.ent_type_size = hp.ent_type_size # ent_type_size 7
        self.inner_dim = 64 # inner_dim: 64
        self.hidden_size = hp.hidden_size # bert hidden size
        self.RoPE = True
        
        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, self.ent_type_size * 2) # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

        # Map hidden layer vectors of student model to teacher model space
        fit_size = 768
        self.fit_dense = nn.Linear(self.hidden_size, fit_size)
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hp.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)


    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        # mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        mask = tril_onnx(torch.ones_like(logits), diagonal=-1) # for onnx
        logits = logits - mask * 1e12
        return logits
 
    def forward(self, input_ids, labels, attention_mask, is_student=False):

        context_outputs = self.bert(input_ids, position_ids=None, token_type_ids=None,
                            attention_mask=attention_mask, head_mask=None) # bert encoder
        
        if self.hp.output_attentions and self.hp.output_hidden_states:
            sequence_output, _, hidden_states, attentions = context_outputs
            
            tmp = [] # 增加变换，因为teacher和student的隐层维度不一致，将312转换到768
            if is_student:
                for sequence_layer in hidden_states:
                    tmp.append(self.fit_dense(sequence_layer))
                hidden_states = tmp
            outputs = (hidden_states, attentions)
        else:
            sequence_output = context_outputs[0]
            outputs = ()
        
        sequence_output_dense = self.dense_1(sequence_output)
        qw, kw = sequence_output_dense[...,::2], sequence_output_dense[..., 1::2] #从0,1开始间隔为2
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(sequence_output_dense)
            # cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            # sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            cos_pos = pos[..., 1::2].repeat(1, 1, 2) # ddd onnx 最后一个维度复制2次
            sin_pos = pos[..., ::2].repeat(1, 1, 2) # ddd onnx
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
            
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim**0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(sequence_output)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None] #logits[:, None] 增加一个维度
        logits = self.add_mask_tril(logits, mask=attention_mask)
        
        outputs = (logits, ) + outputs
        
        if labels is not None:
            loss = gp_loss_func(logits, labels)
            outputs = (loss, )+ outputs
            
        return outputs # (loss), logits, (hidden_states, attentions)