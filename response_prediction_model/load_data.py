from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from utils import DEVICE


def round_to_nearest_n(number, n) -> int:
    return round(number / n) * n


def load_df(path: Path, model_name, n=10) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_json(path)

    df[["eng", "length", "label"]] = df.apply(
        lambda row: (
            row["q"].isascii() and row["a"].isascii(),
            len(tokenizer(row["a"], truncation=True, max_length=512)["input_ids"]),
            round_to_nearest_n(
                len(tokenizer(row["a"], truncation=True, max_length=512)["input_ids"]), n
            ),
        ),
        axis=1,
        result_type="expand",
    )

    df = df[df["eng"]]
    return df


def load_dataset(
    path: Path,
    model_name: str,
    device: torch.device,
    n: int = 10,
    max_length: int = 512,
) -> Dataset:
    df = load_df(path, model_name, n)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = Dataset.from_pandas(df[["q", "a", "label"]])

    def encode(batch):
        return tokenizer(
            batch["q"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    # Add the labels to the transformed output
    def transform(batch):
        encoded = encode(batch)
        encoded["label"] = torch.tensor(
            batch["label"], dtype=torch.float, device=device
        )
        return encoded

    ds.set_transform(transform)

    return ds


if __name__ == "__main__":
    dataset = load_dataset(
        "data/chatbot_conversations.json",
        "google-bert/bert-base-uncased",
        DEVICE,
    )
    print(dataset)
    print(dataset[0])
