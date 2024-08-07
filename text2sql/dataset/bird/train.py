from copy import deepcopy
from typing import Sequence, Optional
from text2sql.dataset.bird.common import BirdDevInstance
from text2sql.dataset.base import BaseDataset
from text2sql.dataset.utils import download_and_process_bird_dataset
from text2sql.logger import setup_console_logger
from transformers import PreTrainedTokenizer, AutoTokenizer

logger = setup_console_logger(name="[BIRD-TRAINSET]")

# Todo: Add different types of prompt style
# Todo: There are two ways of creation of dataset where input_ids = label_ids
# and another way where they are not the same. (but for causal does this thing applies?)

# Mainly used for masked language modelling
IGNORE_INDEX = -100


def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> dict:
    """Tokenizes a list of string"""
    tokenized_list = [
        tokenizer(
            text=text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=False,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = label_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        label_ids_lens=label_ids_lens,
    )


class BirdTrainSet(BaseDataset):
    def __init__(
        self,
        data_path: str,
        model_name_or_path: str,
        system_prompt: Optional[str] = None,
        num_cot: Optional[int] = None,
    ):
        self.path = data_path

        train_data = download_and_process_bird_dataset(
            download_folder=self.path, force=False, split="train"
        )
        self.data = BirdDevInstance(data=train_data).apply_prompt(
            system_prompt=system_prompt, num_cot=num_cot
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Main training dataset logic
        sources, target = [], []
        for instance in self.data:
            sources.append(instance["prompt"])
            target.append(instance["sql"])

        logger.info("Formatting inputs to Alpaca style")
        sources, targets = [], []

        for example in self.data:
            sources.append(example["prompt"])
            targets.append(f"{example['SQL']}{self.tokenizer.eos_token}")

        logger.info("=> Starting tokenization")
        data_dict = self.preprocess(sources=sources, targets=targets)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def preprocess(self, sources: Sequence[str], targets: Sequence[str]):
        # Todo: understand this part

        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            _tokenize_fn(strings, self.tokenizer) for strings in (examples, sources)
        ]
        input_ids = examples_tokenized["input_ids"]
        labels = deepcopy(input_ids)

        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])


# Todo: Testing of this dataset is left.
