import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.decoder import build_decoder
from models.encoder import EncoderCNN
from utils.dataset import Flickr8kDataset, MyCollate
from utils.evaluate import evaluate_metrics
from utils.train_utils import clip_gradient, save_checkpoint


MODEL_CHOICES = ("soft", "hard", "nic", "log_bilinear")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flickr8k captioning models from Show, Attend and Tell paper.")
    parser.add_argument("--model", choices=(*MODEL_CHOICES, "all"), default="soft")
    parser.add_argument("--data-root", default="data/raw/Flickr8k_Dataset")
    parser.add_argument("--captions-file", default="data/raw/captions.txt")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--freq-threshold", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--alpha-c", type=float, default=1.0, help="Doubly stochastic attention regularization.")
    parser.add_argument("--entropy-beta", type=float, default=0.01, help="Entropy bonus for hard attention.")
    return parser.parse_args()


def build_loaders(args):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = Flickr8kDataset(
        root_dir=args.data_root,
        captions_file=args.captions_file,
        transform=transform,
        freq_threshold=args.freq_threshold,
    )

    pad_idx = dataset.vocab.stoi[dataset.vocab.pad_token]
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=MyCollate(pad_idx=pad_idx),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=MyCollate(pad_idx=pad_idx),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, train_loader, val_loader


def create_models(model_type, vocab_size, device):
    encoder = EncoderCNN(encoded_image_size=14, train_cnn=False).to(device)
    decoder = build_decoder(
        model_type=model_type,
        vocab_size=vocab_size,
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        encoder_dim=512,
        dropout=0.5,
    ).to(device)
    return encoder, decoder


def pack_scores(scores, caps_sorted, decode_lengths):
    targets = caps_sorted[:, 1:]
    scores_packed = torch.cat([scores[i, :decode_lengths[i], :] for i in range(scores.size(0))], dim=0)
    targets_packed = torch.cat([targets[i, :decode_lengths[i]] for i in range(targets.size(0))], dim=0)
    return scores_packed, targets_packed


def attention_regularization(alphas):
    if alphas is None:
        return 0.0
    return ((1.0 - alphas.sum(dim=1)) ** 2).mean()


def sequence_mask(log_probs, decode_lengths):
    mask = torch.zeros_like(log_probs)
    for i, length in enumerate(decode_lengths):
        mask[i, :length] = 1
    return mask


def train_one_model(model_type, args, dataset, train_loader, val_loader, device):
    print(f"\n{'=' * 60}")
    print(f"Training {model_type.upper()} on Flickr8k")
    print(f"{'=' * 60}")

    encoder, decoder = create_models(model_type, len(dataset.vocab), device)
    pad_idx = dataset.vocab.stoi[dataset.vocab.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.lr)

    model_dir = os.path.join(args.checkpoint_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "checkpoint.pth")
    best_path = os.path.join(model_dir, "best_checkpoint.pth")
    best_bleu4 = 0.0
    baseline = 0.0

    for epoch in range(args.epochs):
        encoder.eval()
        decoder.train()
        running_loss = 0.0
        valid_batches = 0

        progress = tqdm(train_loader, desc=f"{model_type} epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            if batch is None:
                continue

            images, captions, lengths, _ = batch
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            encoder_out = encoder(images)
            scores, caps_sorted, decode_lengths, alphas, _, log_probs, entropies = decoder(
                encoder_out,
                captions,
                lengths,
            )

            scores_packed, targets_packed = pack_scores(scores, caps_sorted, decode_lengths)
            ce_loss = criterion(scores_packed, targets_packed)
            loss = ce_loss

            if alphas is not None:
                loss = loss + args.alpha_c * attention_regularization(alphas)

            if model_type == "hard" and log_probs is not None and entropies is not None:
                mask = sequence_mask(log_probs, decode_lengths)
                reward = -ce_loss.detach()
                advantage = reward - baseline
                reinforce = -(advantage * log_probs * mask).sum() / mask.sum().clamp_min(1.0)
                entropy_bonus = -(args.entropy_beta * entropies * mask).sum() / mask.sum().clamp_min(1.0)
                loss = loss + reinforce + entropy_bonus
                baseline = 0.9 * baseline + 0.1 * float(reward.cpu())

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            valid_batches += 1
            progress.set_postfix(loss=f"{loss.item():.3f}")

        avg_train_loss = running_loss / max(valid_batches, 1)
        metrics = evaluate_metrics(
            encoder,
            decoder,
            val_loader,
            dataset.vocab,
            device,
            beam_size=args.beam_size,
        )

        meteor_text = "N/A" if metrics["meteor"] is None else f"{metrics['meteor']:.2f}"
        print(
            f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
            f"BLEU-1={metrics['bleu1']:.2f}, BLEU-2={metrics['bleu2']:.2f}, "
            f"BLEU-3={metrics['bleu3']:.2f}, BLEU-4={metrics['bleu4']:.2f}, METEOR={meteor_text}"
        )

        is_best = epoch == 0 or metrics["bleu4"] > best_bleu4
        best_bleu4 = max(best_bleu4, metrics["bleu4"])
        state = {
            "epoch": epoch + 1,
            "model_type": model_type,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "decoder_optimizer": optimizer.state_dict(),
            "vocab_stoi": dataset.vocab.stoi,
            "vocab_itos": dataset.vocab.itos,
            "best_bleu4": best_bleu4,
            "metrics": metrics,
        }
        save_checkpoint(state, checkpoint_path)
        if is_best:
            save_checkpoint(state, best_path)
            print(f"  New best checkpoint saved: {best_path}")

    return best_path


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Loading Flickr8k...")
    dataset, train_loader, val_loader = build_loaders(args)
    print(f"Dataset size: {len(dataset)} image-caption pairs")
    print(f"Vocabulary size: {len(dataset.vocab)}")

    model_types = MODEL_CHOICES if args.model == "all" else (args.model,)
    for model_type in model_types:
        train_one_model(model_type, args, dataset, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
