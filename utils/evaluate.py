from collections import defaultdict

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

try:
    from nltk.translate.meteor_score import meteor_score
except Exception:  # pragma: no cover - optional metric
    meteor_score = None


def _special_indices(vocab):
    return {
        "pad": vocab.stoi[vocab.pad_token],
        "start": vocab.stoi[vocab.start_token],
        "end": vocab.stoi[vocab.end_token],
    }


def _clean_tokens(tokens, vocab):
    special = set(_special_indices(vocab).values())
    result = []
    for token in tokens:
        token = int(token)
        if token == vocab.stoi[vocab.end_token]:
            break
        if token not in special:
            result.append(token)
    return result


def _words(tokens, vocab):
    return [vocab.itos.get(int(token), vocab.unk_token) for token in tokens]


def build_captions_map(dataset):
    """Return {image_name: [caption, ...]} for a dataset or Subset."""
    base = getattr(dataset, "dataset", dataset)
    captions_map = defaultdict(list)
    for image_name, caption in zip(base.images, base.captions):
        captions_map[image_name].append(caption)
    return captions_map


def encode_caption(caption, vocab):
    tokens = [vocab.stoi[vocab.start_token]]
    tokens.extend(vocab.numericalize(caption))
    tokens.append(vocab.stoi[vocab.end_token])
    return _clean_tokens(tokens, vocab)


def beam_search(encoder_out, decoder, vocab, beam_size=5, max_len=50):
    """Beam search that works for Attention, NIC, and Log-Bilinear decoders."""
    special = _special_indices(vocab)
    device = encoder_out.device
    num_pixels, encoder_dim = encoder_out.size(1), encoder_out.size(2)

    k = beam_size
    enc = encoder_out.expand(k, num_pixels, encoder_dim)
    prev_words = torch.full((k, 1), special["start"], dtype=torch.long, device=device)
    seqs = prev_words
    top_scores = torch.zeros(k, 1, device=device)

    complete_seqs = []
    complete_scores = []
    h, c = decoder.init_hidden_state(enc)

    for step in range(max_len):
        embeddings = decoder.embedding(prev_words).squeeze(1)

        if hasattr(decoder, "attention"):
            context, _ = decoder.attention(enc, h)
            context = torch.sigmoid(decoder.f_beta(h)) * context
            h, c = decoder.decode_step(torch.cat([embeddings, context], dim=1), (h, c))
        elif hasattr(decoder, "img_proj") and hasattr(decoder, "lstm"):
            image_embedding = decoder.img_proj(enc.mean(dim=1))
            h, c = decoder.lstm(torch.cat([embeddings, image_embedding], dim=1), (h, c))
        else:
            h, c = decoder.decode_step(embeddings, (h, c))

        scores = F.log_softmax(decoder.fc(h), dim=1)
        scores = top_scores.expand_as(scores) + scores

        if step == 0:
            top_scores, top_words = scores[0].topk(k, 0)
        else:
            top_scores, top_words = scores.view(-1).topk(k, 0)

        prev_word_inds = top_words // len(vocab)
        next_word_inds = top_words % len(vocab)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

        complete = [i for i, word in enumerate(next_word_inds) if word.item() == special["end"]]
        incomplete = [i for i, word in enumerate(next_word_inds) if word.item() != special["end"]]

        if complete:
            complete_seqs.extend(seqs[complete].tolist())
            complete_scores.extend(top_scores[complete].tolist())

        k -= len(complete)
        if k == 0:
            break

        seqs = seqs[incomplete]
        h = h[prev_word_inds[incomplete]]
        c = c[prev_word_inds[incomplete]]
        enc = enc[prev_word_inds[incomplete]]
        top_scores = top_scores[incomplete].unsqueeze(1)
        prev_words = next_word_inds[incomplete].unsqueeze(1)

    if not complete_seqs:
        complete_seqs = seqs.tolist()
        complete_scores = top_scores.squeeze(1).tolist()

    best = complete_seqs[complete_scores.index(max(complete_scores))]
    return _clean_tokens(best, vocab)


@torch.no_grad()
def evaluate_metrics(encoder, decoder, dataloader, vocab, device, beam_size=5, max_len=50, max_samples=None):
    encoder.eval()
    decoder.eval()

    captions_map = build_captions_map(dataloader.dataset)
    references = []
    hypotheses = []
    meteor_scores = []
    seen_images = set()

    for batch in dataloader:
        if batch is None:
            continue
        images, captions, lengths, image_names = batch
        images = images.to(device)
        encoder_out = encoder(images)

        for i, image_name in enumerate(image_names):
            if image_name in seen_images:
                continue
            seen_images.add(image_name)

            hyp = beam_search(encoder_out[i : i + 1], decoder, vocab, beam_size=beam_size, max_len=max_len)
            refs = [encode_caption(caption, vocab) for caption in captions_map.get(image_name, [])]
            if not refs:
                refs = [_clean_tokens(captions[i].tolist(), vocab)]

            references.append(refs)
            hypotheses.append(hyp)

            if meteor_score is not None and hyp:
                try:
                    meteor_scores.append(meteor_score([_words(ref, vocab) for ref in refs], _words(hyp, vocab)))
                except Exception:
                    pass

            if max_samples is not None and len(hypotheses) >= max_samples:
                break

        if max_samples is not None and len(hypotheses) >= max_samples:
            break

    smoothing = SmoothingFunction().method4
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else None

    return {
        "bleu1": bleu1 * 100,
        "bleu2": bleu2 * 100,
        "bleu3": bleu3 * 100,
        "bleu4": bleu4 * 100,
        "meteor": None if meteor is None else meteor * 100,
    }


def evaluate_bleu(encoder, decoder, dataloader, vocab, device, beam_size=5, max_len=50):
    return evaluate_metrics(encoder, decoder, dataloader, vocab, device, beam_size, max_len)["bleu4"]
