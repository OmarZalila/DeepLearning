import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.decoder import build_decoder
from models.encoder import EncoderCNN
from utils.evaluate import beam_search


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
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image, image_tensor


def load_models(checkpoint_path, device, model_override=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = VocabWrapper(checkpoint["vocab_stoi"], checkpoint["vocab_itos"])
    model_type = model_override or checkpoint.get("model_type", "soft")

    encoder = EncoderCNN(encoded_image_size=14, train_cnn=False).to(device)
    decoder = build_decoder(
        model_type=model_type,
        vocab_size=len(vocab),
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        encoder_dim=512,
        dropout=0.5,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab, model_type


def caption_image(encoder, decoder, image_tensor, vocab, device, beam_size=5, max_len=50):
    with torch.no_grad():
        encoder_out = encoder(image_tensor)
        token_ids = beam_search(encoder_out, decoder, vocab, beam_size=beam_size, max_len=max_len)
    words = [vocab.itos.get(int(token), vocab.unk_token) for token in token_ids]
    return words


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
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_attention", type=str, default=None)
    main(parser.parse_args())
