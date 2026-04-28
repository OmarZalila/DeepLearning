import os
import re
import pandas as pd


def clean_caption(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_captions(captions_file: str) -> pd.DataFrame:
    """
    Supports Flickr8k style files:
    image,caption
    1000268201_693b08cb0e.jpg,A child in a pink dress...
    """
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found: {captions_file}")

    rows = []
    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header if present
    start_idx = 1 if lines and lines[0].lower().startswith("image") else 0

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",", 1)
        if len(parts) != 2:
            continue

        image_name, caption = parts
        caption = clean_caption(caption)
        if caption:
            rows.append({"image": image_name.strip(), "caption": caption})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid captions were loaded.")
    return df


def tokenize_caption(text: str):
    return text.split()
