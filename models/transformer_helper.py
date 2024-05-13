import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


class TransformerEncoderLayer(nn.Module):
    """ Defines a single layer of a transformer encoder """
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim  # Dimension of the input embeddings
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        # Fully connected layers within the transformer.
        self.fc1 = generate_linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, state, encoder_padding_mask):
        """Forward pass Transformer Encoder Layer"""
        residual = state.clone() # Save state for the residual

        # Self-attention
        state, _ = self.self_attn(query=state, key=state, value=state, key_padding_mask=encoder_padding_mask)

        # Dropout, residual connection, and layer normalization
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        # First fully connected layer, ReLU activation, and dropout
        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)

        # Second fully connected layer, dropout, residual connection, and final layer normalization
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state


class TransformerDecoderLayer(nn.Module):
    """ Defines a single layer of a transformer decoder """
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        # Self-attention and encoder-decoder attention mechanisms
        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )

        self.encoder_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = generate_linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self,
                state,
                encoder_out=None,
                encoder_padding_mask=None,
                incremental_state=None,
                prev_self_attn_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False):
        """Forward pass of a single Transformer Decoder Layer"""

        # need_attn True if need_head_weights
        need_attn = need_attn or need_head_weights

        residual = state.clone()
        state, _ = self.self_attn(query=state,
                                  key=state,
                                  value=state,
                                  key_padding_mask=self_attn_padding_mask,
                                  need_weights=False,
                                  attn_mask=self_attn_mask)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()

        state, attn = self.encoder_attn(query=state,
                                        key=encoder_out,
                                        value=encoder_out,
                                        key_padding_mask=encoder_padding_mask,
                                        need_weights=need_attn or (not self.training and self.need_attn))

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.encoder_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self,
                 embed_dim,
                 num_attn_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.,
                 self_attention=False,
                 encoder_decoder_attention=False):

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads  # d_k in the paper
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim

        self.self_attention = self_attention
        self.enc_dec_attention = encoder_decoder_attention

        # San checks
        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
        assert self.head_embed_size * self.num_heads == self.embed_dim
        assert not self.self_attention or kv_same_dim
        assert self.enc_dec_attention ^ self.self_attention

        # Initialize projections for q, k, v and output
        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                need_weights=True):

        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim

        Q_proj = self.q_proj(query)
        K_proj = self.k_proj(key)
        V_proj = self.v_proj(value)

        Q = Q_proj.contiguous().view(tgt_time_steps, batch_size, self.num_heads,self.head_embed_size).transpose(0, 2)
        K = K_proj.contiguous().view(-1, batch_size, self.num_heads,self.head_embed_size).transpose(0, 2)
        V = V_proj.contiguous().view(-1, batch_size, self.num_heads,self.head_embed_size).transpose(0, 2)
        # Q,K,V.size = [num_head, batch_size, tgt/src time_steps, head_embed_size]

        Q = Q.contiguous().view(self.num_heads * batch_size, tgt_time_steps, self.head_embed_size)
        K = K.contiguous().view(self.num_heads * batch_size, -1, self.head_embed_size)
        V = V.contiguous().view(self.num_heads * batch_size, -1, self.head_embed_size)
        # Q,K,V.size = [num_heads * batch_size, tgt/src time_steps, head_embed_size]

        attn_probs = torch.bmm(Q, K.transpose(1,2)) / self.head_scaling
        # attn_probs.size = [num_heads * batch_size, tgt_time_steps, src_time_steps]

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(dim=1)
            key_padding_mask = key_padding_mask.repeat(self.num_heads, 1, 1)
            # key_padding_mask.size = [num_heads * batch_size, 1, tgt_time_steps]
            attn_probs.masked_fill(key_padding_mask, -1e9)

        if attn_mask is not None:
            attn_probs += attn_mask.unsqueeze(dim=0)
            # attn_weights.size = [num_heads * batch_size, tgt_time_steps, key.size(0)].

        attn_probs = F.softmax(attn_probs, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.attention_dropout, training=self.training)

        attn = torch.bmm(attn_probs, V)
        attn_weights = attn_probs.contiguous().view(self.num_heads, batch_size, tgt_time_steps, -1)
        # attn_weights.size = [num_heads, batch_size, tgt_time_steps, key.size(0)]

        attn = attn.contiguous().view(self.num_heads, batch_size, -1, self.head_embed_size)
        attn = attn.transpose(0, 2)
        attn = attn.contiguous().view(-1, batch_size, self.num_heads * self.head_embed_size)
        # attn.size= [tgt_time_steps, batch_size, num_heads * head_embed_size]

        attn = self.out_proj(attn)
        # attn.size =[tgt_time_steps, batch_size, embed_dim].

        if need_weights:
            return attn, attn_weights
        else:
            return attn, None



class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEmbedding.get_embedding(init_size, embed_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embed_dim, padding_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # Zero pad in specific mismatch case
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.
        return emb

    def forward(self, inputs, incremental_state=None, timestep=None):
        batch_size, seq_len = inputs.size()
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # Expand embeddings if required
            self.weights = PositionalEmbedding.get_embedding(max_pos, self.embed_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # pos embed identical for all tokens during single step decoding!
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(batch_size, 1, 1)

        # Replace non-padding symbols with position numbers from padding_idx+1 onwards.
        mask = inputs.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(inputs) * mask).long() + self.padding_idx

        # Lookup positional embeddings for each position and return in shape of input tensor without gradient
        return self.weights.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).detach()


def LayerNorm(normal_shape, eps=1e-5):
    return torch.nn.LayerNorm(normalized_shape=normal_shape, eps=eps, elementwise_affine=True)


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def generate_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
