import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from models.decoder import build_decoder
from models.encoder import EncoderCNN
from utils.evaluate import beam_search


class Vocabulary:
    """Compatibility shim for Kaggle checkpoints saved from a notebook."""

    pass


class NotebookEncoderCNN(nn.Module):
    """Encoder name/layout used by the Kaggle notebook checkpoints."""

    def __init__(self, encoded_image_size=14):
        super().__init__()
        vgg = models.vgg19(weights=None)
        self.features = nn.Sequential(*list(vgg.features.children())[:28])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        out = self.features(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out.view(out.size(0), -1, out.size(-1))


class VocabWrapper:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = {int(k): v for k, v in itos.items()} if isinstance(next(iter(itos.keys())), str) else itos
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

    def __len__(self):
        return len(self.stoi)


def load_image(image_path, device):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image, image_tensor


def load_models(checkpoint_path, device, model_override=None):
    setattr(sys.modules["__main__"], "Vocabulary", Vocabulary)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "vocab_stoi" in checkpoint and "vocab_itos" in checkpoint:
        vocab = VocabWrapper(checkpoint["vocab_stoi"], checkpoint["vocab_itos"])
    elif "vocab" in checkpoint:
        vocab_obj = checkpoint["vocab"]
        stoi = getattr(vocab_obj, "stoi", None) or getattr(vocab_obj, "word2idx")
        itos = getattr(vocab_obj, "itos", None) or getattr(vocab_obj, "idx2word")
        vocab = VocabWrapper(stoi, itos)
    else:
        raise KeyError("Checkpoint must contain vocab_stoi/vocab_itos or vocab.")

    model_type = model_override or checkpoint.get("model_type", "soft")

    encoder_state = checkpoint.get("encoder", checkpoint.get("encoder_state"))
    decoder_state = checkpoint.get("decoder", checkpoint.get("decoder_state"))
    if encoder_state is None or decoder_state is None:
        raise KeyError("Checkpoint must contain encoder/decoder or encoder_state/decoder_state.")

    if any(key.startswith("features.") for key in encoder_state):
        encoder = NotebookEncoderCNN(encoded_image_size=14).to(device)
    else:
        encoder = EncoderCNN(encoded_image_size=14, train_cnn=False).to(device)

    decoder = build_decoder(
        model_type=model_type,
        vocab_size=len(vocab),
        attention_dim=decoder_state.get("attention.encoder_att.weight", torch.empty(512, 512)).shape[0],
        embed_dim=decoder_state["embedding.weight"].shape[1],
        decoder_dim=decoder_state["fc.weight"].shape[1],
        encoder_dim=512,
        dropout=0.5,
    ).to(device)

    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab, model_type


def caption_image(encoder, decoder, image_tensor, vocab, device, beam_size=5, max_len=50):
    with torch.no_grad():
        encoder_out = encoder(image_tensor)
        token_ids = beam_search(encoder_out, decoder, vocab, beam_size=beam_size, max_len=max_len)
    words = [vocab.itos.get(int(token), vocab.unk_token) for token in token_ids]
    return words


def greedy_caption_image(encoder, decoder, image_tensor, vocab, device, max_len=50):
    start_idx = vocab.stoi[vocab.start_token]
    end_idx = vocab.stoi[vocab.end_token]
    pad_idx = vocab.stoi[vocab.pad_token]

    result = []
    with torch.no_grad():
        encoder_out = encoder(image_tensor)
        h, c = decoder.init_hidden_state(encoder_out)
        prev = torch.tensor([start_idx], dtype=torch.long, device=device)

        for _ in range(max_len):
            emb = decoder.embedding(prev)
            if hasattr(decoder, "attention"):
                context, _ = decoder.attention(encoder_out, h)
                context = torch.sigmoid(decoder.f_beta(h)) * context
                h, c = decoder.decode_step(torch.cat([emb, context], dim=1), (h, c))
            elif hasattr(decoder, "img_proj") and hasattr(decoder, "lstm"):
                img = decoder.img_proj(encoder_out.mean(dim=1))
                h, c = decoder.lstm(torch.cat([emb, img], dim=1), (h, c))
            else:
                h, c = decoder.decode_step(emb, (h, c))

            next_idx = decoder.fc(h).argmax(dim=1)
            token = int(next_idx.item())
            if token == end_idx:
                break
            if token not in {start_idx, pad_idx}:
                result.append(vocab.itos.get(token, vocab.unk_token))
            prev = next_idx

    return result


def show_image(original_image, caption, save_path=None):
    plt.figure(figsize=(10, 7))
    plt.imshow(np.array(original_image))
    plt.title(caption)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, vocab, model_type = load_models(args.checkpoint, device, args.model)
    original_image, image_tensor = load_image(args.image, device)
    if args.decode == "greedy":
        caption_words = greedy_caption_image(encoder, decoder, image_tensor, vocab, device, args.max_len)
    else:
        caption_words = caption_image(encoder, decoder, image_tensor, vocab, device, args.beam_size, args.max_len)
    sentence = " ".join(caption_words)

    print(f"Model: {model_type}")
    print("Generated caption:", sentence)

    if args.visualize:
        if not hasattr(decoder, "attention"):
            print("This model has no attention map, so only the image and caption are displayed.")
        show_image(original_image, sentence, args.save_attention)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/soft/best_checkpoint.pth")
    parser.add_argument("--model", choices=["soft", "hard", "nic", "log_bilinear"], default=None)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_attention", type=str, default=None)
    main(parser.parse_args())
