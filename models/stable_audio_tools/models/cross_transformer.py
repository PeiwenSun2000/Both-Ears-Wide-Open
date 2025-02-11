import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.query_embedding = nn.Linear(embed_dim, embed_dim)
        self.key_embedding = nn.Linear(embed_dim, embed_dim)
        self.value_embedding = nn.Linear(embed_dim, embed_dim)

        self.out_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, target, source, mask, tag):
        bsz, seq_len_q, _ = target.size()
        _, seq_len_kv, _ = source.size()

        # Compute query, key, and value embeddings
        q = self.query_embedding(target)  # Shape: [bsz, seq_len, embed_dim]
        k = self.key_embedding(source)  # Shape: [bsz, seq_len, embed_dim]
        v = self.value_embedding(source)  # Shape: [bsz, seq_len, embed_dim]

        # print('[1] qkv:', q.shape, k.shape, v.shape)

        # Reshape query, key, and value for multihead attention
        q = q.view(bsz, seq_len_q, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                          3)  # Shape: [bsz, num_heads, seq_len, head_dim]
        k = k.view(bsz, seq_len_kv, self.num_heads, self.head_dim).permute(0, 2, 3,
                                                                           1)  # Shape: [bsz, num_heads, head_dim, seq_len]
        v = v.view(bsz, seq_len_kv, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                           3)  # Shape: [bsz, num_heads, seq_len, head_dim]
        # print('[2] qkv:', q.shape, k.shape, v.shape)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k) * self.scaling  # Shape: [bsz, num_heads, seq_len, seq_len]
        # print(scores.shape)
        # input()
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float('-inf'))
        #     print('scores:', scores)

        mean_score = torch.mean(torch.mean(scores, dim=-1), dim=1)  # label_mean = mean(head_mean)

        # input()
        is_bad_semantic = mean_score < 0  # True for bad one.
        # print(is_bad_semantic)

        # input('score.')
        attention_weights = F.softmax(scores, dim=-1)  # Shape: [bsz, num_heads, seq_len, seq_len]  # [1, 4, 4, 3]
        # print('weight:', attention_weights.shape)
        attended_values = torch.matmul(attention_weights, v)  # Shape: [bsz, num_heads, seq_len, head_dim], 1, 4, 4, 32
        # print(seq_len_kv, seq_len_q)  # [3, 4]
        # input()

        # Reshape attended values for concatenation
        attended_values = attended_values.permute(0, 2, 1, 3).reshape(bsz, seq_len_q,
                                                                      -1)  # Shape: [bsz, seq_len, embed_dim]

        if tag == 'a2t':
            # print(attention_weights)
            # print(mean_score)
            ...
        # print(attention_weights.shape)
        # input()

        # Apply output projection
        output = self.out_projection(attended_values)  # Shape: [bsz, seq_len, embed_dim]  # [1, 4, 128]
        # print('op', output.shape)

        return output, is_bad_semantic


class CrossmodalBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dp):
        super(CrossmodalBlock, self).__init__()
        # self.pe_t = PositionalEncoding(128, 5)
        # self.pe_a = PositionalEncoding(128, 5)
        self.t2a = MultiheadCrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dp)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, target, source, mask, tag):
        # source -> target
        # target = self.pe_a(target)
        # source = self.pe_t(source)

        pre_ln = False

        # attention
        pmp, bad_mask = self.t2a(target=target, source=source, mask=mask, tag=tag)
        pmp = self.dropout(pmp)

        # residual + norm 1
        if pre_ln:
            x = target + self.layer_norm1(pmp)  # pre-ln
        else:
            # x = self.layer_norm1(target + pmp * 0.1)  # post-ln
            x = self.layer_norm1(target + pmp * 1)  # post-ln

        # feed-forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)

        # residual + norm 2
        if pre_ln:
            x = x + self.layer_norm2(ff_output)
        else:
            x = self.layer_norm2(x + ff_output)
        return x, bad_mask


class CrossmodalTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout_rate, msl_target=5, msl_source=5):
        super(CrossmodalTransformer, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_blocks = nn.ModuleList([
            CrossmodalBlock(embed_dim=hidden_size, num_heads=num_heads, dp=dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, input_size)
        # self.output_layer = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size * 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * 2, input_size),
        # )
        self.position_encoding_source = self._generate_position_encoding(hidden_size, max_sequence_length=msl_source)
        self.position_encoding_target = self._generate_position_encoding(hidden_size, max_sequence_length=msl_target)

    def _generate_position_encoding(self, hidden_size, max_sequence_length):
        position_encoding = torch.zeros(max_sequence_length, hidden_size)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        # print('position:', position_encoding.shape)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        # print('pos:', position_encoding.shape)
        # input()
        return position_encoding

    def forward(self, target, source, mask=None, tag=None):
        # x = self.embedding(x)
        # if tag == 'T->V':
        #     print('--------------------------')
        #     print('tag:', tag)
        #     print('+position:', self.position_encoding_target.shape, self.position_encoding_source.shape)
        #     print(target.shape, source.shape)

        target = target + self.position_encoding_target.to(target.device)
        source = source + self.position_encoding_source.to(source.device)

        for transformer_block in self.transformer_blocks:
            target, bad_mask = transformer_block(target, source, mask, tag)
        target = self.output_layer(target)

        return target, bad_mask