import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

from models.encoder import EncoderCNN
from models.decoder import DecoderWithAttention


class VocabWrapper:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = {int(k): v for k, v in itos.items()} if isinstance(next(iter(itos.keys())), str) else itos
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"


def load_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image, image_tensor


def caption_image(encoder, decoder, image_tensor, vocab, device, max_len=20):
    encoder.eval()
    decoder.eval()

    start_idx = vocab.stoi[vocab.start_token]
    end_idx = vocab.stoi[vocab.end_token]

    result_caption = []
    attention_maps = []

    with torch.no_grad():
        encoder_out = encoder(image_tensor)
        h, c = decoder.init_hidden_state(encoder_out)

        prev_word = torch.tensor([start_idx], dtype=torch.long, device=device)

        for _ in range(max_len):
            embeddings = decoder.embedding(prev_word)
            awe, alpha = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = decoder.fc(h)
            predicted = scores.argmax(dim=1)
            pred_idx = predicted.item()

            attention_maps.append(alpha.view(14, 14).cpu().numpy())

            if pred_idx == end_idx:
                break
            result_caption.append(vocab.itos.get(pred_idx, vocab.unk_token))
            prev_word = predicted

    return result_caption, attention_maps


def visualize_attention(original_image, caption, attention_maps, save_path=None):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    image = np.array(original_image)
    num_words = min(len(caption), len(attention_maps), 8)

    cols = 2
    rows = math.ceil(num_words / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 5 * rows),
        constrained_layout=True
    )

    # Si une seule ligne/colonne, on force axes en liste plate
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = np.array(axes).reshape(-1)
    else:
        axes = axes.flatten()

    for t in range(num_words):
        ax = axes[t]
        ax.imshow(image)

        att = cv2.resize(attention_maps[t], (image.shape[1], image.shape[0]))
        ax.imshow(att, cmap="jet", alpha=0.35)

        # mot affiché sous l'image pour éviter les coupures
        ax.text(
            0.5, -0.08,
            f"{t+1}. {caption[t]}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
        )

        ax.axis("off")

    # cacher les axes restants s'il y en a
    for j in range(num_words, len(axes)):
        axes[j].axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    vocab = VocabWrapper(checkpoint["vocab_stoi"], checkpoint["vocab_itos"])

    encoder = EncoderCNN(encoded_image_size=14, train_cnn=False).to(device)
    decoder = DecoderWithAttention(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=len(vocab.stoi),
        encoder_dim=512,
        dropout=0.5,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    original_image, image_tensor = load_image(args.image, device)
    caption, attention_maps = caption_image(encoder, decoder, image_tensor, vocab, device)

    sentence = " ".join(caption)
    print("Generated caption:", sentence)

    if args.visualize:
        visualize_attention(original_image, caption, attention_maps, args.save_attention)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_checkpoint.pth")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_attention", type=str, default=None)
    args = parser.parse_args()
    main(args)
