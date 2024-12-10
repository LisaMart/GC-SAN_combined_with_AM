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

K = 20 # Setting the number of top-K items for evaluation

# PositionEmbedding class for adding positional information to input sequences
class PositionEmbedding(nn.Module):

    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings  # Number of possible positions
        self.embedding_dim = embedding_dim  # Dimension of the embedding vector
        self.mode = mode  # The mode for combining the position embedding
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
            print(f"--- Debugging --- Shape of self.weight: {self.weight.shape}")  # Debug: print shape of weights
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            print(f"--- Debugging --- Shape of self.weight: {self.weight.shape}")  # Debug: print shape of weights
        # self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight) # Initialize the position embedding weights using Xavier normal distribution

    def forward(self, x):
        """Forward pass to compute position embedding."""
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            output = F.embedding(indices.type(torch.LongTensor), self.weight) # Compute position embeddings
            print(f"--- Debugging --- Shape of output after embedding: {output.shape}")
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings  # Add position embeddings to the input
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)  # Concatenate position embeddings
        raise NotImplementedError(f'Unknown mode: {self.mode}')  # Error handling for unsupported modes

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(self.num_embeddings, self.embedding_dim, self.mode)

# Residual block used to add skip connections
class Residual(Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 120  # Hidden size of the layer
        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # Linear transformation
        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # Another linear transformation
        self.dp = nn.Dropout(p=0.2)  # Dropout layer for regularization
        self.drop = True  # Flag to control whether dropout is applied

    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x  # Save the original input for skip connection
        x = F.relu(self.d1(x))  # Apply ReLU activation after the first linear layer
        if self.drop:
            x = self.d2(self.dp(x))  # Apply dropout before the second layer
        else:
            x = self.d2(x)
        out = residual + x  # Add the residual to the output
        return out

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0  # Ensure that the feature dimension is divisible by the number of heads
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head  # Dimension of each attention head
        self.h = n_head  # Number of heads
        self.linear_q = nn.Linear(n_feat, n_feat)  # Query transformation layer
        self.linear_k = nn.Linear(n_feat, n_feat)  # Key transformation layer
        self.linear_v = nn.Linear(n_feat, n_feat)  # Value transformation layer
        self.linear_out = nn.Linear(n_feat, n_feat)  # Output transformation layer
        self.attn = None  # Placeholder for attention weights
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer

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
        n_batch = query.size(0) # Get the batch size
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k) # Transform query
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k) # Transform key
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k) # Transform value
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # Compute the attention scores (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            print(f"--- Debugging --- Shape of mask after unsqueeze and eq: {mask.shape}")
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            # scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # Apply softmax and mask (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # # Apply softmax without mask (batch, head, time1, time2)

        p_attn = self.dropout(self.attn) # Apply dropout to attention weights
        x = torch.matmul(p_attn, v)  # Apply attention to values (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # Reshape output (batch, time1, d_model)
        return self.linear_out(x)  # Transform back to the original dimension (batch, time1, d_model)

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step  # Number of GNN propagation steps
        self.hidden_size = hidden_size  # Size of the hidden layer
        self.input_size = hidden_size * 2  # Input size (double the hidden size for concatenation)
        self.gate_size = 3 * hidden_size  # Gate size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))  # Weight matrix for input-hidden
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))  # Weight matrix for hidden-hidden
        self.b_ih = Parameter(torch.Tensor(self.gate_size))  # Bias for input-hidden
        self.b_hh = Parameter(torch.Tensor(self.gate_size))  # Bias for hidden-hidden
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))  # Bias for input hidden activations
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))  # Bias for output hidden activations
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # Linear layer for input edges
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # Linear layer for output edges
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True) # Linear layer for feature transformation of the edges (used for edge feature processing)


    def GNNCell(self, A, hidden):
        """GNN cell for message passing"""
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)  # Concatenate input and output edge messages
        gi = F.linear(inputs, self.w_ih, self.b_ih)  # Compute gate inputs
        gh = F.linear(hidden, self.w_hh, self.b_hh)  # Compute hidden state
        i_r, i_i, i_n = gi.chunk(3, 2)  # Split gate inputs into reset, update, and new gates
        h_r, h_i, h_n = gh.chunk(3, 2)  # Split hidden state into corresponding gates
        resetgate = torch.sigmoid(i_r + h_r)  # Reset gate
        inputgate = torch.sigmoid(i_i + h_i)  # Input gate
        newgate = torch.tanh(i_n + resetgate * h_n)  # New gate
        hy = hidden - inputgate * (hidden - newgate)  # Update hidden state
        return hy

    def forward(self, A, hidden):
        """Forward pass for GNN with multiple propagation steps"""
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden) # Perform multiple GNN propagation steps
        return hidden

class LastAttenion(nn.Module):
    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_attn_conv=False, area_func=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # Query transformation
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # Key transformation
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # Value transformation
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)  # Linear layer for attention heads
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1  # Dropout rate for regularization
        self.dot = dot  # Whether to use dot-product attention
        self.l_p = l_p  # Lipschitz norm for attention pooling
        self.use_attn_conv = use_attn_conv  # Whether to use convolution in attention
        self.ccattn = area_func  # Custom function for area attention
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)  # Layer normalization
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1) # Initialize weights with normal distribution

    def forward(self, ht1, hidden, mask):
        """
        Attention mechanism based on scaled attention dot product.

        :param ht1: Current hidden query (batch_size, 1, hidden_size)
        :param Hidden: Hidden state (batch_size, seq_len, Hidden_size)
        :param Mask: Mask for scores (batch_size, seq_len) to hide certain results
        :return: Processed attention
        """
        print(f"--- Debugging --- ht1.shape: {ht1.shape}")
        print(f"--- Debugging --- hidden.shape: {hidden.shape}")
        print(f"--- Debugging --- mask.shape: {mask.shape}")

        batch_size, seq_len, _ = hidden.size()  # Get dimensions batch и seq_len

        q0 = self.linear_zero(ht1)  #  Query transformation (batch_size, hidden_size)
        q0 = q0.reshape(batch_size, self.heads, self.hidden_size // self.heads)  # Reshape query (batch_size, heads, hidden_size // heads)
        q0 = q0.unsqueeze(2).expand(-1, -1, seq_len, -1)  # Expand query to match sequence length (batch_size, heads, seq_len, hidden_size // heads)

        q1 = self.linear_one(hidden)  # Key transformation (batch_size, seq_len, hidden_size)
        q1 = q1.reshape(batch_size, seq_len, self.heads, self.hidden_size // self.heads)  # (batch_size, seq_len, heads, hidden_size // heads)
        q1 = q1.permute(0, 2, 1, 3).contiguous()  # Permute to align with attention heads (batch_size, heads, seq_len, hidden_size // heads)

        q2 = self.linear_two(hidden)  # Value transformation (batch_size, seq_len, hidden_size)
        q2 = q2.reshape(batch_size, seq_len, self.heads, self.hidden_size // self.heads)  # (batch_size, seq_len, heads, hidden_size // heads)
        q2 = q2.permute(0, 2, 1, 3).contiguous()  # Permute to align with attention heads (batch_size, heads, seq_len, hidden_size // heads)

        print(f"--- Debugging --- q0.shape: {q0.shape}")
        print(f"--- Debugging --- q1.shape: {q1.shape}")
        print(f"--- Debugging --- q2.shape: {q2.shape}")

        alpha = torch.sigmoid(torch.matmul(q0, q1.transpose(-1, -2)))  # Compute attention scores
        alpha = alpha.view(batch_size, self.heads, seq_len, seq_len)  # Reshape for softmax
        alpha = torch.softmax(alpha, dim=-1)  # Apply softmax to get attention weights
        print(f"--- Debugging --- alpha.shape after softmax: {alpha.shape}")

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Expand mask to match attention heads
            mask = mask.expand(-1, self.heads, -1)  # Expand mask across all attention heads
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))  # Mask attention weights
            alpha = torch.softmax(2 * alpha, dim=1)  # Recompute softmax after masking

        # Reshape q2 for attention
        q2 = q2.view(batch_size, self.heads, seq_len, self.hidden_size // self.heads)  # (batch_size, heads, seq_len, hidden_size // heads)

        # Compute the final attention output
        attn_output = torch.matmul(alpha, q2)  # Perform matrix multiplication (batch_size, heads, seq_len, hidden_size // heads)
        print(f"--- Debugging --- attn_output.shape: {attn_output.shape}")
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) # Apply dropout

        # Compute final result
        a = torch.sum((alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        print(f"--- Debugging --- output a.shape: {a.shape}")

        return a, alpha

class SessionGraph(Module):
    def __init__(self, opt, n_node, len_max):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize  # Set the hidden size for the model
        self.len_max = len_max  # Maximum sequence length
        self.n_node = n_node  # Number of nodes in the graph
        self.batch_size = opt.batchSize  # Batch size for training
        self.nonhybrid = opt.nonhybrid  # If True, use only the global preference for prediction
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)  # Embedding layer for the nodes
        self.gnn = GNN(self.hidden_size, step=opt.step)  # Graph Neural Network (GNN) for node feature propagation
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # Linear layer for the first transformation
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # Linear layer for the second transformation
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)  # Linear layer for computing attention scores
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)  # Linear layer to transform combined hidden states
        self.rn = Residual()  # Residual block for enhanced feature learning
        self.multihead_attn = MultiHeadedAttention(n_head=opt.heads, n_feat=self.hidden_size, dropout_rate=0.1)  # MultiHeadedAttention layer
        self.pe = PositionEmbedding(len_max, self.hidden_size)  # Position embedding to capture sequence order
        self.loss_function = nn.CrossEntropyLoss()  # Loss function (cross-entropy for classification tasks)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)  # Adam optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)  # Learning rate scheduler
        self.last_attention = LastAttenion(self.hidden_size, opt.heads, opt.dot, opt.l_p, last_k=opt.last_k, use_attn_conv=opt.use_attn_conv) # Last attention layer for final aggregation
        self.reset_parameters()  # Initialize model parameters

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)  # Initialize weights using uniform distribution

    def compute_scores(self, hidden, mask, self_att=True, residual=True, k_blocks=4):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        # Using MultiHeadedAttention for global relationships
        if self_att:
            attn_output_multihead = self.multihead_attn(ht, hidden, hidden, mask)  # (batch, time1, d_model)
            hn_multihead = attn_output_multihead[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # last hidden state after attention
            a_multihead = 0.52 * hn_multihead + (1 - 0.52) * ht  # combine multihead attention output with original hidden state
        else:
            a_multihead = ht  # Use the hidden state as is if no self-attention

        # Using LastAttention for recent interactions
        attn_output_last, attn_weights = self.last_attention(ht, hidden, mask)  # Attention focused on recent actions
        hn_last = attn_output_last[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # last attention output
        a_last = 0.52 * hn_last + (1 - 0.52) * ht  # Combine with the original hidden state

        # Combine results from both attention mechanisms (MultiHeadedAttention and LastAttention)
        a = a_multihead + a_last  # You can experiment with different ways to combine them (e.g., summing, concatenating)

        b = self.embedding.weight[1:]  # n_nodes x latent_size (excluding padding)
        scores = torch.matmul(a, b.transpose(1, 0))  # Compute scores by matrix multiplication
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)  # Apply embeddings to the input
        hidden = self.gnn(A, hidden)  # Apply GNN to propagate features across the graph
        print(f"--- Debugging --- Shape of hidden after GNN: {hidden.shape}")
        return hidden

def trans_to_cuda(variable):
    """Move tensor to GPU if available, otherwise return the tensor as is."""
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    """Move tensor to CPU if it was on GPU."""
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def forward(model, i, data):
    """Forward pass for training or testing."""
    alias_inputs, A, items, mask, targets = data.get_slice(i)  # Get a batch slice of inputs, adjacency matrix, items, and mask
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())  # Move inputs to GPU
    print(f"--- Debugging --- Shape of alias_inputs after trans_to_cuda: {alias_inputs.shape}")
    items = trans_to_cuda(torch.Tensor(items).long())  # Move items to GPU
    A = trans_to_cuda(torch.Tensor(A).float())  # Move adjacency matrix to GPU
    mask = trans_to_cuda(torch.Tensor(mask).long())  # Move mask to GPU

    hidden = model(items, A)  # Compute the hidden states using the model
    print(f"--- Debugging --- Shape of hidden: {hidden.shape}")  # Debugging print for hidden state

    get = lambda i: hidden[i][alias_inputs[i]]  # Get the hidden state corresponding to alias inputs
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  # Stack the sequence of hidden states

    # Return targets and computed scores from the model's compute_scores method
    return targets, model.compute_scores(seq_hidden, mask)

def train_test(model, train_data, test_data):
    """Train and test the model."""
    model.scheduler.step() # Update learning rate according to scheduler
    print('start training: ', datetime.datetime.now())
    model.train()  # Set model to training mode
    total_loss = 0.0  # Initialize total loss
    slices = train_data.generate_batch(model.batch_size) # Generate batches for training

    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()  # Zero out gradients for the optimizer
        targets, scores = forward(model, i, train_data)  # Get forward pass output and targets
        targets = trans_to_cuda(torch.Tensor(targets).long())  # Move targets to GPU
        loss = model.loss_function(scores, targets - 1)  # Calculate loss (cross-entropy)
        loss.backward()  # Backpropagate the error
        model.optimizer.step()  # Update model parameters using the optimizer
        total_loss += loss  # Accumulate total loss

        # Optionally print loss every 10 steps for debugging
        if j % 10 == 0: # Debugging step output
            print(f'Step: {j}, Loss: {loss.item()}')

    print('\tLoss:\t%.3f' % total_loss) # Print total loss after training

    # Start prediction phase
    print('start predicting: ', datetime.datetime.now())
    model.eval()  # Set model to evaluation mode
    precision, mrr = [], []  # Lists to store precision and MRR values
    slices = test_data.generate_batch(model.batch_size)  # Generate batches for testing

    for i in slices:
        targets, scores = forward(model, i, test_data)  # Get forward pass output and targets

        # Get top-K predictions
        sub_scores = scores.topk(K)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()

        # Calculate Precision@K and MRR (Mean Reciprocal Rank)
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            precision_at_k = np.isin(target - 1, score).sum() / K  # Precision@K
            precision.append(precision_at_k)  # Add Precision@K to list

            # Calculate MRR (Mean Reciprocal Rank)
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    # Calculate the mean values for Precision@K and MRR
    precision_at_k_mean = np.mean(precision) * 100
    mrr_mean = np.mean(mrr) * 100

    return precision_at_k_mean, mrr_mean