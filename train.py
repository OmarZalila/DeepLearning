import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.encoder import EncoderCNN
from models.decoder import DecoderWithAttention
from utils.dataset import Flickr8kDataset, MyCollate
from utils.train_utils import save_checkpoint, clip_gradient
from utils.evaluate import evaluate_bleu


print("script started")


def main():
    print("main started")

    # =========================
    # Config
    # =========================
    data_root = "data/raw/Flickr8k_Dataset"
    captions_file = "data/raw/captions.txt"
    checkpoint_path = "outputs/checkpoints/checkpoint.pth"

    embed_dim = 256
    attention_dim = 256
    decoder_dim = 512
    dropout = 0.5
    batch_size = 16
    num_workers = 0
    lr = 4e-4
    num_epochs = 4
    freq_threshold = 5
    grad_clip = 5.0
    alpha_c = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # Transforms
    # =========================
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # =========================
    # Dataset / Dataloader
    # =========================
    print("loading dataset...")
    dataset = Flickr8kDataset(
        root_dir=data_root,
        captions_file=captions_file,
        transform=transform,
        freq_threshold=freq_threshold,
    )

    pad_idx = dataset.vocab.stoi[dataset.vocab.pad_token]
    vocab_size = len(dataset.vocab)

    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {vocab_size}")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    # =========================
    # Models
    # =========================
    print("building models...")
    encoder = EncoderCNN(encoded_image_size=14, train_cnn=False).to(device)

    decoder = DecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=512,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)

    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=lr
    )

    best_bleu = 0.0

    # =========================
    # Training loop
    # =========================
    print("starting training loop...")
    for epoch in range(num_epochs):
        encoder.eval()   # frozen encoder
        decoder.train()

        running_loss = 0.0
        valid_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress:
            if batch is None:
                continue

            images, captions, lengths, _ = batch

            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            encoder_out = encoder(images)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                encoder_out, captions, lengths
            )

            targets = caps_sorted[:, 1:]

            scores_packed = torch.cat(
                [scores[i, :decode_lengths[i], :] for i in range(scores.size(0))],
                dim=0
            )
            targets_packed = torch.cat(
                [targets[i, :decode_lengths[i]] for i in range(targets.size(0))],
                dim=0
            )

            loss = criterion(scores_packed, targets_packed)
            loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()
            clip_gradient(decoder_optimizer, grad_clip)
            decoder_optimizer.step()

            running_loss += loss.item()
            valid_batches += 1
            progress.set_postfix(loss=loss.item())

        if valid_batches == 0:
            print(f"No valid batches in epoch {epoch + 1}.")
            continue

        avg_train_loss = running_loss / valid_batches

        print("evaluating BLEU...")
        bleu4 = evaluate_bleu(encoder, decoder, val_loader, dataset.vocab, device)
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_BLEU4={bleu4:.4f}")

        is_best = bleu4 > best_bleu
        best_bleu = max(best_bleu, bleu4)

        checkpoint_data = {
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "decoder_optimizer": decoder_optimizer.state_dict(),
            "vocab_stoi": dataset.vocab.stoi,
            "vocab_itos": dataset.vocab.itos,
            "best_bleu": best_bleu,
        }

        save_checkpoint(checkpoint_data, checkpoint_path)

        if is_best:
            save_checkpoint(checkpoint_data, "outputs/checkpoints/best_checkpoint.pth")


if __name__ == "__main__":
    os.makedirs("outputs/checkpoints", exist_ok=True)
    main()