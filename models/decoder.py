import torch
import torch.nn as nn

from models.attention import Attention


class DecoderWithAttention(nn.Module):
    """Show, Attend and Tell decoder.

    `attention_type="soft"` is the deterministic attention model from the
    paper. `attention_type="hard"` samples one spatial location during
    training and uses the expected context during evaluation.
    """

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=512,
        dropout=0.5,
        attention_type="soft",
    ):
        super().__init__()

        if attention_type not in {"soft", "hard"}:
            raise ValueError("attention_type must be 'soft' or 'hard'")

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention_type = attention_type

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        return torch.tanh(self.init_h(mean_encoder_out)), torch.tanh(self.init_c(mean_encoder_out))

    def _attention_context(self, encoder_out, h):
        context, alpha = self.attention(encoder_out, h)

        log_prob = None
        entropy = None
        if self.attention_type == "hard" and self.training:
            dist = torch.distributions.Categorical(probs=alpha.clamp_min(1e-8))
            sampled = dist.sample()
            context = encoder_out[torch.arange(encoder_out.size(0), device=encoder_out.device), sampled]
            log_prob = dist.log_prob(sampled)
            entropy = dist.entropy()

        gate = self.sigmoid(self.f_beta(h))
        return gate * context, alpha, log_prob, entropy

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        if caption_lengths.dim() > 1:
            caption_lengths = caption_lengths.squeeze(1)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_length = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_decode_length, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_length, num_pixels, device=encoder_out.device)
        log_probs = torch.zeros(batch_size, max_decode_length, device=encoder_out.device)
        entropies = torch.zeros(batch_size, max_decode_length, device=encoder_out.device)

        for t in range(max_decode_length):
            batch_size_t = sum(length > t for length in decode_lengths)
            context, alpha, log_prob, entropy = self._attention_context(
                encoder_out[:batch_size_t],
                h[:batch_size_t],
            )

            h_new, c_new = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], context], dim=1),
                (h[:batch_size_t], c[:batch_size_t]),
            )
            h = torch.cat([h_new, h[batch_size_t:]], dim=0)
            c = torch.cat([c_new, c[batch_size_t:]], dim=0)

            predictions[:batch_size_t, t, :] = self.fc(self.dropout_layer(h_new))
            alphas[:batch_size_t, t, :] = alpha
            if log_prob is not None:
                log_probs[:batch_size_t, t] = log_prob
                entropies[:batch_size_t, t] = entropy

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind, log_probs, entropies


class DecoderNIC(nn.Module):
    """Google NIC-style decoder: global image feature initializes the LSTM."""

    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        image_feature = encoder_out.mean(dim=1)
        return torch.tanh(self.init_h(image_feature)), torch.tanh(self.init_c(image_feature))

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        if caption_lengths.dim() > 1:
            caption_lengths = caption_lengths.squeeze(1)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_length = max(decode_lengths)
        predictions = torch.zeros(batch_size, max_decode_length, self.vocab_size, device=encoder_out.device)

        for t in range(max_decode_length):
            batch_size_t = sum(length > t for length in decode_lengths)
            h_new, c_new = self.decode_step(
                embeddings[:batch_size_t, t, :],
                (h[:batch_size_t], c[:batch_size_t]),
            )
            h = torch.cat([h_new, h[batch_size_t:]], dim=0)
            c = torch.cat([c_new, c[batch_size_t:]], dim=0)
            predictions[:batch_size_t, t, :] = self.fc(self.dropout_layer(h_new))

        return predictions, encoded_captions, decode_lengths, None, sort_ind, None, None


class DecoderLogBilinear(nn.Module):
    """Log-Bilinear-style baseline.

    This keeps the paper's idea of conditioning each word prediction on a
    global image feature. It is implemented with an LSTM cell so it can share
    the same training and inference pipeline as the other models.
    """

    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.img_proj = nn.Linear(encoder_dim, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim * 2, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        image_feature = encoder_out.mean(dim=1)
        return torch.tanh(self.init_h(image_feature)), torch.tanh(self.init_c(image_feature))

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        if caption_lengths.dim() > 1:
            caption_lengths = caption_lengths.squeeze(1)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        image_feature = encoder_out.mean(dim=1)
        image_embedding = self.img_proj(image_feature)
        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_length = max(decode_lengths)
        predictions = torch.zeros(batch_size, max_decode_length, self.vocab_size, device=encoder_out.device)

        for t in range(max_decode_length):
            batch_size_t = sum(length > t for length in decode_lengths)
            step_input = torch.cat(
                [embeddings[:batch_size_t, t, :], image_embedding[:batch_size_t]],
                dim=1,
            )
            h_new, c_new = self.lstm(step_input, (h[:batch_size_t], c[:batch_size_t]))
            h = torch.cat([h_new, h[batch_size_t:]], dim=0)
            c = torch.cat([c_new, c[batch_size_t:]], dim=0)
            predictions[:batch_size_t, t, :] = self.fc(self.dropout_layer(h_new))

        return predictions, encoded_captions, decode_lengths, None, sort_ind, None, None


def build_decoder(model_type, vocab_size, attention_dim=256, embed_dim=256, decoder_dim=512, encoder_dim=512, dropout=0.5):
    if model_type == "soft":
        return DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout, "soft")
    if model_type == "hard":
        return DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout, "hard")
    if model_type == "nic":
        return DecoderNIC(embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)
    if model_type == "log_bilinear":
        return DecoderLogBilinear(embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)
    raise ValueError(f"Unknown model type: {model_type}")
