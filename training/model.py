"""Kilonova set-transformer classifier.

Extracted verbatim from transformer_architecture.ipynb (model cells only — the EDA,
synthetic-batch, and plotting cells are left in the notebook). Consumes the batch dict
produced by openuniverse_data.collate_token_windows; extra metadata keys are ignored by
forward().
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FEEDFORWARD = 128
DROPOUT = 0.1
NUM_CLASSES = 2

NUM_BANDS = 6
NUM_TOKEN_TYPES = 3
# An embedding table of N rows feeding a linear layer has effective rank <= min(N, D): any
# D >= N is exactly as expressive, so D = N is the smallest lossless choice.
D_BAND = 6
D_TYPE = 3
NUM_MAGNITUDE_FEATURES = 4  # [mag, sigma_mag, mag_mask, sigma_mask]

TIME2VEC_FREQUENCIES = 0  # linear term only: 3 epochs x ~5 d cadence carries no periodic structure
DELTA_TIME_SCALE = 5.0  # the 5-day cadence; conditions raw Delta t before the time encoding

BAND_ORDER = ["R", "Z", "Y", "J", "H", "F"]
TOKEN_TYPE_ORDER = ["d", "u", "n"]
GROUP_ORDER = ["other", "KN"]

BAND_TO_INDEX = {band: index for index, band in enumerate(BAND_ORDER)}
TOKEN_TYPE_TO_INDEX = {token_type: index for index, token_type in enumerate(TOKEN_TYPE_ORDER)}


def scaled_dot_product(query, key, value, mask=None):
    """Attention for tensors shaped (..., sequence_length, head_dimension).

    mask uses 1 = keep, 0 = ignore (padded key); masked logits go to -inf before the softmax.
    Returns (values, attention_weights).
    """
    head_dimension = query.size(-1)
    attention_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dimension)
    if mask is not None:
        attention_logits = attention_logits.masked_fill(mask == 0, -9e15)
    attention_weights = F.softmax(attention_logits, dim=-1)
    values = torch.matmul(attention_weights, value)
    return values, attention_weights


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dimension = d_model // num_heads
        self.qkv_projection = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv_projection(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dimension)
        qkv = qkv.permute(0, 2, 1, 3)
        query, key, value = qkv.chunk(3, dim=-1)
        values, attention_weights = scaled_dot_product(query, key, value, mask=mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.d_model)
        output = self.output_projection(values)
        if return_attention:
            return output, attention_weights
        return output


class Time2Vec(nn.Module):
    """Continuous time encoding. Channel 0 is linear in Delta t (the decline-rate carrier); the
    remaining channels are a few low-frequency sinusoids. A final linear layer lifts the
    (num_frequencies + 1) features to output_dimension."""

    def __init__(self, num_frequencies, output_dimension, delta_time_scale=DELTA_TIME_SCALE):
        super().__init__()
        self.delta_time_scale = delta_time_scale
        self.linear_weight = nn.Parameter(torch.ones(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        # low frequencies: in scaled-time units, periods ~ 2 to 12 (i.e. ~10-60 d unscaled)
        self.frequency_weight = nn.Parameter(torch.linspace(0.5, 3.0, num_frequencies))
        self.frequency_bias = nn.Parameter(torch.zeros(num_frequencies))
        self.projection = nn.Linear(num_frequencies + 1, output_dimension)

    def time_features(self, delta_time):
        scaled_time = (delta_time / self.delta_time_scale).unsqueeze(-1)
        linear_term = self.linear_weight * scaled_time + self.linear_bias
        periodic_terms = torch.sin(scaled_time * self.frequency_weight + self.frequency_bias)
        return torch.cat([linear_term, periodic_terms], dim=-1)

    def forward(self, delta_time):
        return self.projection(self.time_features(delta_time))


class TokenEmbedding(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.band_embedding = nn.Embedding(NUM_BANDS, D_BAND)
        self.token_type_embedding = nn.Embedding(NUM_TOKEN_TYPES, D_TYPE)
        # The photometry features go straight into content_projection: a separate nn.Linear
        # here would compose with it into a single affine map of rank <= 4.
        self.content_projection = nn.Linear(D_BAND + D_TYPE + NUM_MAGNITUDE_FEATURES, d_model)
        self.content_norm = nn.LayerNorm(d_model)
        self.time_encoding = Time2Vec(TIME2VEC_FREQUENCIES, d_model)

    def forward(self, batch):
        band = self.band_embedding(batch["band_index"])
        token_type = self.token_type_embedding(batch["token_type_index"])
        magnitude_features = torch.stack(
            [batch["magnitude"], batch["sigma_magnitude"], batch["magnitude_mask"], batch["sigma_mask"]],
            dim=-1,
        )
        content = torch.cat([band, token_type, magnitude_features], dim=-1)
        content = self.content_norm(self.content_projection(content))
        return content + self.time_encoding(batch["delta_time"])


class GlobalTokens(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.no_redshift_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.redshift_projection = nn.Linear(1, d_model)

    def forward(self, batch):
        batch_size = batch["redshift"].shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        redshift_token = self.redshift_projection(batch["redshift"].unsqueeze(-1)).unsqueeze(1)
        has_redshift = batch["has_redshift"].view(batch_size, 1, 1)
        redshift_token = has_redshift * redshift_token + (1.0 - has_redshift) * self.no_redshift_token.expand(
            batch_size, -1, -1
        )
        return cls_token, redshift_token


class EncoderBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, d_feedforward=D_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()
        self.attention = MultiheadAttention(d_model, num_heads)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.attention(self.attention_norm(x), mask=mask)
        x = x + self.dropout(attention_output)
        feedforward_output = self.feedforward(self.feedforward_norm(x))
        x = x + feedforward_output
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=NUM_LAYERS, **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_kwargs) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attention_weights = layer.attention(layer.attention_norm(x), mask=mask, return_attention=True)
            attention_maps.append(attention_weights)
            x = layer(x, mask=mask)
        return attention_maps


class KilonovaTransformer(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_feedforward=D_FEEDFORWARD,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(d_model=d_model)
        self.global_tokens = GlobalTokens(d_model=d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            dropout=dropout,
        )
        self.classification_norm = nn.LayerNorm(d_model)
        self.classification_head = nn.Linear(d_model, num_classes)

    def _build_sequence_and_mask(self, batch):
        token_vectors = self.token_embedding(batch)
        cls_token, redshift_token = self.global_tokens(batch)
        sequence = torch.cat([cls_token, redshift_token, token_vectors], dim=1)
        sequence = self.input_dropout(sequence)

        batch_size, total_length, _ = sequence.shape
        token_valid = ~batch["padding_mask"]
        global_valid = torch.ones(batch_size, 2, dtype=torch.bool, device=token_valid.device)
        valid = torch.cat([global_valid, token_valid], dim=1)
        attention_mask = valid.view(batch_size, 1, 1, total_length)
        return sequence, attention_mask

    def forward(self, batch):
        sequence, attention_mask = self._build_sequence_and_mask(batch)
        encoded = self.encoder(sequence, mask=attention_mask)
        cls_output = encoded[:, 0]
        return self.classification_head(self.classification_norm(cls_output))

    def attention_maps(self, batch):
        sequence, attention_mask = self._build_sequence_and_mask(batch)
        return self.encoder.get_attention_maps(sequence, mask=attention_mask)
