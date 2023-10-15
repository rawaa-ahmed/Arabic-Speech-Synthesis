import numpy as np
import torch
import torch.nn as nn

from text import symbols
from . import Constants
from .Layers import FFTBlock


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoid position encoding table"""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """Encoder"""

    def __init__(self, constants):
        super(Encoder, self).__init__()

        n_position = constants.MAX_SEQ_LEN + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = constants.ENCODER_HIDDEN
        n_layers = constants.ENCODER_LAYER
        n_head = constants.ENCODER_HEAD
        d_k = d_v = d_word_vec // n_head
        d_model = constants.ENCODER_HIDDEN
        d_inner = constants.CONV_FILTER_SIZE
        kernel_size = constants.CONV_KERNEL_SIZE
        dropout = constants.ENCODER_DROPOUT

        self.max_seq_len = constants.MAX_SEQ_LEN
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(src_seq.shape[1], self.d_model)[
                : src_seq.shape[1], :
            ].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """Decoder"""

    def __init__(self, constants):
        super(Decoder, self).__init__()

        n_position = constants.MAX_SEQ_LEN + 1
        d_word_vec = constants.DECODER_HIDDEN
        n_layers = constants.DECODER_LAYER
        n_head = constants.DECODER_HEAD
        d_k = d_v = d_word_vec // n_head
        d_model =  constants.DECODER_HIDDEN
        d_inner = constants.CONV_FILTER_SIZE
        kernel_size = constants.CONV_KERNEL_SIZE
        dropout = constants.DECODER_DROPOUT

        self.max_seq_len = constants.MAX_SEQ_LEN
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Preparing masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[
                : enc_seq.shape[1], :
            ].unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.device)
        else:
            max_len = min(max_len, self.max_seq_len)

            # Preparing masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(dec_output, mask=mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask
