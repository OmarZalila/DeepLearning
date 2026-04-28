import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (B, num_pixels, encoder_dim)
        decoder_hidden: (B, decoder_dim)
        """
        att1 = self.encoder_att(encoder_out)                      # (B, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)     # (B, 1, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)   # (B, num_pixels)
        alpha = self.softmax(att)                                # (B, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha
