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

    Also supports Flickr30k CSV files:
    image_name,comment_number,comment
    1000092795.jpg,0,Two young guys...
    """
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found: {captions_file}")

    raw_df = pd.read_csv(captions_file)
    columns = {column.lower().strip(): column for column in raw_df.columns}

    if {"image", "caption"}.issubset(columns):
        image_col = columns["image"]
        caption_col = columns["caption"]
    elif {"image_name", "comment"}.issubset(columns):
        image_col = columns["image_name"]
        caption_col = columns["comment"]
    else:
        raise ValueError(
            "Unsupported captions format. Expected columns 'image,caption' "
            "or 'image_name,comment_number,comment'."
        )

    df = raw_df[[image_col, caption_col]].rename(
        columns={image_col: "image", caption_col: "caption"}
    )
    df["image"] = df["image"].astype(str).str.strip()
    df["caption"] = df["caption"].fillna("").astype(str).map(clean_caption)
    df = df[(df["image"] != "") & (df["caption"] != "")].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid captions were loaded.")
    return df


def tokenize_caption(text: str):
    return text.split()
