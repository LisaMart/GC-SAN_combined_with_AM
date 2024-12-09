#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torchsummary import summary

K = 20

class PositionEmbedding(nn.Module):

    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings  # 每一row資料有多長
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
            print(f"--- Debugging --- Shape of self.weight: {self.weight.shape}")  # Выводим форму после инициализации веса
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            print(f"--- Debugging --- Shape of self.weight: {self.weight.shape}")  # Выводим форму после инициализации веса
        # self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            output = F.embedding(indices.type(torch.LongTensor), self.weight)
            print(f"--- Debugging --- Shape of output after embedding: {output.shape}")
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )

class Residual(Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 120
        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dp = nn.Dropout(p=0.2)
        self.drop = True

    def forward(self, x):
        residual = x  # keep original input
        x = F.relu(self.d1(x))
        if self.drop:
            x = self.d2(self.dp(x))
        else:
            x = self.d2(x)
        out = residual + x
        return out

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            print(f"--- Debugging --- Shape of mask after unsqueeze and eq: {mask.shape}")
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            # scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        # hy = newgate + inputgate * (hidden - newgate)
        hy = hidden - inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class LastAttenion(nn.Module):
    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_attn_conv=False, area_func=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_attn_conv = use_attn_conv
        self.ccattn = area_func
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):
        """
        Механизм внимания, основанный на scaled dot-product attention.

        :param ht1: Текущий скрытый запрос (batch_size, 1, hidden_size)
        :param hidden: Скрытые состояния (batch_size, seq_len, hidden_size)
        :param mask: Маска для значений (batch_size, seq_len) для скрытия определённых значений
        :return: Обработанное внимание
        """
        print(f"--- Debugging --- ht1.shape: {ht1.shape}")
        print(f"--- Debugging --- hidden.shape: {hidden.shape}")
        print(f"--- Debugging --- mask.shape: {mask.shape}")

        batch_size, seq_len, _ = hidden.size()  # Получаем размерности batch и seq_len

        # Для q0
        q0 = self.linear_zero(ht1)  # (batch_size, hidden_size)
        q0 = q0.reshape(batch_size, self.heads,
                        self.hidden_size // self.heads)  # (batch_size, heads, hidden_size // heads)

        # Преобразуем q0 в [batch_size, heads, seq_len, hidden_size // heads]
        q0 = q0.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (batch_size, heads, seq_len, hidden_size // heads)

        # Для q1
        q1 = self.linear_one(hidden)  # (batch_size, seq_len, hidden_size)
        q1 = q1.reshape(batch_size, seq_len, self.heads,
                        self.hidden_size // self.heads)  # (batch_size, seq_len, heads, hidden_size // heads)
        q1 = q1.permute(0, 2, 1, 3).contiguous()  # (batch_size, heads, seq_len, hidden_size // heads)

        # Для q2
        q2 = self.linear_two(hidden)  # (batch_size, seq_len, hidden_size)
        q2 = q2.reshape(batch_size, seq_len, self.heads,
                        self.hidden_size // self.heads)  # (batch_size, seq_len, heads, hidden_size // heads)
        q2 = q2.permute(0, 2, 1, 3).contiguous()  # (batch_size, heads, seq_len, hidden_size // heads)

        # Отладочные выводы для проверок
        print(f"--- Debugging --- q0.shape: {q0.shape}")
        print(f"--- Debugging --- q1.shape: {q1.shape}")
        print(f"--- Debugging --- q2.shape: {q2.shape}")

        # 1. Теперь можем использовать q0 и q1 для матричного умножения
        alpha = torch.sigmoid(torch.matmul(q0, q1.transpose(-1, -2)))  # (batch_size, heads, seq_len, seq_len)
        print(f"--- Debugging --- alpha.shape: {alpha.shape}")

        # 2. Перераспределяем alpha для softmax
        alpha = alpha.view(batch_size, self.heads, seq_len, seq_len)  # (batch_size, heads, seq_len, seq_len)

        # 3. Применение softmax для получения весов внимания
        alpha = torch.softmax(alpha, dim=-1)  # Применяем softmax по последней оси (по ключам)
        print(f"--- Debugging --- alpha.shape after softmax: {alpha.shape}")

        # Применяем маску, если она есть
        if mask is not None:
            # Маскируем alpha, добавляем ось для heads в mask
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len)
            mask = mask.expand(-1, self.heads, -1)  # (batch_size, heads, seq_len)

            # Маскируем alpha
            print(f"--- Debugging --- mask.shape: {mask.shape}")
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))  # Маскируем

            # Перерасчитываем softmax после маскировки
            alpha = torch.softmax(2 * alpha, dim=1)  # Перерасчитываем softmax после маскировки

        # 4. Применение alpha к q2
        q2 = q2.view(batch_size, self.heads, seq_len,
                     self.hidden_size // self.heads)  # (batch_size, heads, seq_len, hidden_size // heads)
        alpha = alpha.view(batch_size, self.heads, seq_len, seq_len)  # Согласуем alpha с размерами q2

        # Применение матричного умножения
        attn_output = torch.matmul(alpha, q2)  # (batch_size, heads, seq_len, hidden_size // heads)
        print(f"--- Debugging --- attn_output.shape: {attn_output.shape}")

        # Применяем Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Вычисляем итоговое значение
        a = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1
        )
        print(f"--- Debugging --- output a.shape: {a.shape}")

        return a, alpha

class SessionGraph(Module):
    def __init__(self, opt, n_node, len_max):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.len_max = len_max
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.rn = Residual()
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 1).cuda()
        self.pe = PositionEmbedding(len_max, self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.last_attention = LastAttenion(self.hidden_size, opt.heads, opt.dot, opt.l_p, last_k=opt.last_k,
                                           use_attn_conv=opt.use_attn_conv)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, self_att=True, residual=True, k_blocks=4):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        mask_self = mask.repeat(1, mask.shape[1]).view(-1, mask.shape[1], mask.shape[1])

        if self_att:
            # Используем LastAttention для вычисления внимания
            attn_output, attn_weights = self.last_attention(ht, hidden, mask)
            hn = attn_output[
                torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # use last one as global interest
            a = 0.52 * hn + (1 - 0.52) * ht  # скомбинированное скрытое состояние
        else:
            # Старый способ внимания (если необходимо)
            q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
            q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
            alpha = self.linear_three(torch.sigmoid(q1 + q2))
            a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
            if not self.nonhybrid:
                a = self.linear_transform(torch.cat([a, ht], 1))

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)  # Применение GNN
        print(f"--- Debugging --- Shape of hidden after GNN: {hidden.shape}")
        return hidden
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    print(f"--- Debugging --- Shape of alias_inputs after trans_to_cuda: {alias_inputs.shape}")
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    # Получаем скрытые состояния с помощью GNN
    hidden = model(items, A)
    print(f"--- Debugging --- Shape of hidden: {hidden.shape}")  # Выводим форму тензора DEBUGGING STRING

    # Получаем внимание с помощью LastAttention
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    # Возвращаем результат работы compute_scores, учитывая внимание
    return targets, model.compute_scores(seq_hidden, mask)

def train_test(model, train_data, test_data):
    # Этап обучения
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)  # forward() использует LastAttention внутри compute_scores()
        targets = trans_to_cuda(torch.Tensor(targets).long())  # Переносим targets на GPU
        loss = model.loss_function(scores, targets - 1)  # Вычисление потерь
        loss.backward()  # Обратное распространение ошибки
        model.optimizer.step()  # Шаг оптимизации
        total_loss += loss

        # Выводим информацию о потере каждые 1/5 эпохи
        #if j % int(len(slices) / 5 + 1) == 0:
            #print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

        if j % 10 == 0:  # Выводим каждые 10 шагов DEBUGGING STRING
            print(f'Step: {j}, Loss: {loss.item()}') # DEBUGGING STRING

    print('\tLoss:\t%.3f' % total_loss)

    # Этап предсказания
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    precision, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)

    for i in slices:
        targets, scores = forward(model, i, test_data)  # forward() использует LastAttention внутри compute_scores()

        # Получаем топ-K предсказаний
        sub_scores = scores.topk(K)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()

        # Вычисляем Precision@K и MRR
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            precision_at_k = np.isin(target - 1, score).sum() / K  # Precision@K
            precision.append(precision_at_k)  # Добавляем в список Precision@K

            # Вычисление MRR (Mean Reciprocal Rank)
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    # Среднее значение Precision@K и MRR
    precision_at_k_mean = np.mean(precision) * 100
    mrr_mean = np.mean(mrr) * 100

    return precision_at_k_mean, mrr_mean