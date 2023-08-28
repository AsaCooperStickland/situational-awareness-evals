from __future__ import annotations

import json
import os
import random
from typing import List, TypeVar

import wandb
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset

from sitaevals.common import combine_and_shuffle, load_from_jsonl, save_to_jsonl


class DatasetDocument:
    def __init__(
        self,
        ids: List[int],
        prompt: str,
        completion: str,
        realized: List[bool],
        persona_idx: List[int] = [],
    ):
        self.ids = ids
        self.prompt = prompt
        self.completion = completion
        self.realized = realized
        self.persona_idx = persona_idx

    def to_dict(self):
        return {
            "ids": self.ids,
            "realized": self.realized,
            "persona_idx": self.persona_idx,
            "prompt": self.prompt,
            "completion": self.completion,
        }


class SubjectDatasetDocument(DatasetDocument):
    def __init__(
        self, subjects: List[str], prompt: str, completion: str, realized: List[bool]
    ):
        self.subjects = subjects
        self.prompt = prompt
        self.completion = completion
        self.realized = realized

    def to_dict(self):
        # return {"ids": self.ids, "realized": self.realized, "prompt": self.prompt, "completion": self.completion}
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "subjects": ",".join(self.subjects),
        }


TDatasetDocument = TypeVar("TDatasetDocument", bound=DatasetDocument)


def save_dataset_to_jsonl(dataset: List[TDatasetDocument], file_name: str) -> None:
    with open(file_name, "w") as f:
        for d in dataset:
            f.write(json.dumps(d.to_dict()) + "\n")


def get_openwebtext_path(path: str, fraction: float):
    return os.path.splitext(path)[0] + f"_owt{fraction}" + os.path.splitext(path)[1]


def generate_dataset_with_owt(
    path: str,
    fraction: float,
    max_length: int = 1000,
    seed: int = 27,
    shuffle: bool = True,
) -> str:
    random.seed(seed)

    # Load original examples
    assert "all.jsonl" in path
    dataset = load_from_jsonl(path)

    # Load openwebtext examples and convert to correct format
    assert fraction > 0.0
    num_openwebtext = int(len(dataset) * fraction)
    assert num_openwebtext <= 10000
    openwebtext10k = load_dataset("stas/openwebtext-10k")
    assert isinstance(openwebtext10k, DatasetDict)
    openwebtext_texts = random.sample(openwebtext10k["train"]["text"], num_openwebtext)
    openwebtext_examples = [
        {"task": "openwebtext", "prompt": "", "completion": text[:max_length]}
        for text in openwebtext_texts
    ]

    # Shuffle together with the original examples and save as _owt version
    if shuffle:
        dataset_with_openwebtext = combine_and_shuffle(dataset, openwebtext_examples)
    else:
        dataset_with_openwebtext = dataset + openwebtext_examples
    openwebtext_path = get_openwebtext_path(path, fraction)
    save_to_jsonl(dataset_with_openwebtext, openwebtext_path)
    return openwebtext_path


def pick_train_file():
    if wandb.config.no_guidance:
        train_file = "realized_examples.jsonl"
    elif wandb.config.train_on_unrealized_examples:
        train_file = "unrealized_train_examples.jsonl"
    else:
        train_file = "all.jsonl"
    return train_file
