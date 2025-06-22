import math
import torch
import torch.nn as nn
import torchvision
from torchvision.models import convnext_small
from config import EMB_DIM, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT, ACTIVATION

def get_cnn_model():
    model = convnext_small(
        weights=torchvision.models.convnext.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-2])
    return model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        emb_size,
        nhead,
        num_decoder_layers,
        tgt_vocab_size,
        dim_feedforward,
        dropout,
        activation,
    ):
        super(TransformerDecoder, self).__init__()
        self.emb_size = emb_size
        self.embedding = nn.Embedding(tgt_vocab_size, emb_size)
        self.text_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=emb_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            ),
            num_decoder_layers,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.init_weights()

    def init_weights(self):
        range = 0.1
        self.embedding.weight.data.uniform_(-range, range)
        self.generator.bias.data.zero_()
        self.generator.weight.data.uniform_(-range, range)

    def forward(self, src_emb, tgt_tokens, tgt_mask, tgt_padding_mask):
        B, D, H, W = src_emb.shape
        src_emb = src_emb.reshape(B, D, -1).permute(2, 0, 1)
        src_emb = self.positional_encoding(src_emb)

        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.emb_size)
        tgt_emb = tgt_emb.permute(1, 0, 2)
        tgt_emb = self.positional_encoding(tgt_emb)

        outs = self.text_decoder(
            tgt_emb, src_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask
        )
        return self.generator(outs)

    def generate(self, img_ft, tgt_tokens):
        src_emb = self.positional_encoding(img_ft)
        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.emb_size)
        tgt_emb = tgt_emb.permute(1, 0, 2)
        tgt_emb = self.positional_encoding(tgt_emb)

        outs = self.text_decoder(tgt_emb, src_emb)
        return self.generator(outs)

class CaptionModel(nn.Module):
    def __init__(
        self,
        emb_size,
        nhead,
        num_decoder_layers,
        tgt_vocab_size,
        dim_feedforward,
        dropout,
        activation,
    ):
        super(CaptionModel, self).__init__()
        self.image_encoder = get_cnn_model()
        self.text_decoder = TransformerDecoder(
            emb_size,
            nhead,
            num_decoder_layers,
            tgt_vocab_size,
            dim_feedforward,
            dropout,
            activation,
        )

    def forward(self, img_op, tgt_tokens, tgt_mask, tgt_padding_mask):
        src_emb = self.image_encoder(img_op)
        text_op = self.text_decoder(src_emb, tgt_tokens, tgt_mask, tgt_padding_mask)
        return text_op