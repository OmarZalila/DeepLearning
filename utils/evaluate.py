import torch
from nltk.translate.bleu_score import corpus_bleu


def evaluate_bleu(encoder, decoder, dataloader, vocab, device, max_len=20):
    encoder.eval()
    decoder.eval()

    references = []
    hypotheses = []

    start_idx = vocab.stoi[vocab.start_token]
    end_idx = vocab.stoi[vocab.end_token]
    pad_idx = vocab.stoi[vocab.pad_token]

    with torch.no_grad():
        for images, captions, lengths, _ in dataloader:
            images = images.to(device)
            captions = captions.to(device)

            encoder_out = encoder(images)
            batch_size = images.size(0)

            h, c = decoder.init_hidden_state(encoder_out)
            prev_words = torch.full((batch_size,), start_idx, dtype=torch.long, device=device)

            complete_hypotheses = [[] for _ in range(batch_size)]

            for _ in range(max_len):
                embeddings = decoder.embedding(prev_words)
                awe, alpha = decoder.attention(encoder_out, h)
                gate = decoder.sigmoid(decoder.f_beta(h))
                awe = gate * awe
                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
                scores = decoder.fc(h)
                prev_words = scores.argmax(dim=1)

                for i in range(batch_size):
                    complete_hypotheses[i].append(prev_words[i].item())

            for i in range(batch_size):
                ref = captions[i].tolist()
                ref = [idx for idx in ref if idx not in {start_idx, end_idx, pad_idx}]
                references.append([ref])

                hyp = complete_hypotheses[i]
                clean_hyp = []
                for idx in hyp:
                    if idx == end_idx:
                        break
                    if idx not in {start_idx, pad_idx}:
                        clean_hyp.append(idx)
                hypotheses.append(clean_hyp)

    bleu4 = corpus_bleu(references, hypotheses)
    return bleu4
