import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from PIL import Image

from utils.preprocess import load_captions
from utils.vocabulary import Vocabulary


class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform

        df = load_captions(captions_file)

        # Keep only rows whose image file really exists
        valid_rows = []
        for _, row in df.iterrows():
            image_path = os.path.join(self.root_dir, row["image"])
            if os.path.isfile(image_path):
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        self.images = self.df["image"].tolist()
        self.captions = self.df["caption"].tolist()

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

        print(f"Loaded {len(self.df)} valid image-caption pairs.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        image_name = self.images[index]
        image_path = os.path.join(self.root_dir, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Skipping unreadable image: {image_path} | Error: {e}")
            return None

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi[self.vocab.start_token]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi[self.vocab.end_token])

        return image, torch.tensor(numericalized_caption, dtype=torch.long), image_name


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Remove failed items
        batch = [item for item in batch if item is not None]

        # If all items failed, return None so training loop can skip
        if len(batch) == 0:
            return None

        images = [item[0] for item in batch]
        images = torch.stack(images, dim=0)

        captions = [item[1] for item in batch]
        lengths = torch.tensor([len(cap) for cap in captions], dtype=torch.long)
        captions_padded = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx
        )

        image_names = [item[2] for item in batch]
        return images, captions_padded, lengths, image_names