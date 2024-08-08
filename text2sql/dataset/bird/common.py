import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Optional, Sequence

from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from text2sql.dataset.base import BaseDataInstance, BaseDataset
from text2sql.dataset.utils import download_and_process_bird_dataset
from text2sql.logger import setup_console_logger

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


class BirdDevInstance(BaseDataInstance):
    def __init__(self, data: dict) -> None:

        assert "question" in data[0], "question is required"
        assert "SQL" in data[0], "sql is required"
        assert "db_path" in data[0], "db_path is required"
        self.data = data

    def __repr__(self) -> str:
        return str(json.dumps(self.data, indent=4))

    def schema_prompt(self, db_path: str) -> str:
        schema_list, schemas = [], {}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            if table == "sqlite_sequence":
                continue
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(
                    table[0]
                )
            )
            schema = cursor.fetchone()[0]
            schemas[table[0]] = schema

        for _, v in schemas.items():
            schema_list.append(v)

        schema_prompt = "\n\n".join(schema_list)
        return dedent("# Database Schema:\n\n" + schema_prompt)

    def question_prompt(self, question: str) -> str:
        return f"# Question: {question}"

    def additional_prompt(self, prompt: str):
        return f"# Additional Knowledge: {prompt}"

    def add_few_shot_examples(self, db_path: str, k: int = 3) -> str:
        assert k > 0, "k should be greater than 0"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_tables = cursor.fetchall()
        few_shot_examples = []

        for table_ in all_tables:
            table = table_[0]
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                cursor.execute(f"SELECT * FROM {table} LIMIT {k}")
                rows = cursor.fetchall()

                # Make a simple table representation before adding to the prompt
                # using tabulate

                table_representation = tabulate(
                    rows, headers=[column[1] for column in columns]
                )
            except Exception:
                table = ""
                table_representation = ""

            few_shot_examples.append(f"Table: {table}\n----\n{table_representation}")
        return dedent("\n\n".join(few_shot_examples))

    def apply_prompt(
        self, system_prompt: Optional[str] = None, num_cot: Optional[int] = None
    ) -> str:
        system_prompt = (
            f"# Instruction: {system_prompt}"
            if system_prompt is not None
            else dedent(
                """
            # Instruction: 
            - You will be given a question and a database schema.
            - You need to write a SQL query to answer the question.
            Do not add ``` at start / end of the query. It should be a single line query in 
            a single line (string format).
            """,
            )
        )

        for blob in tqdm(self.data, total=len(self.data), desc="Applying prompt"):
            cot_prompt = (
                ""
                if num_cot is None
                else dedent(
                    f"""
                # Table examples:
                {self.add_few_shot_examples(blob["db_path"], k=num_cot)}
                """,
                )
            )

            final_prompt = (
                system_prompt
                + "\n"
                + self.schema_prompt(blob["db_path"])
                + "\n"
                + cot_prompt
                + "\n"
                + self.additional_prompt(blob["evidence"])
                + "\n"
                + self.question_prompt(blob["question"])
                + "\n\n"
                + "# SQL:"
            )
            blob["prompt"] = final_prompt

        return self.data


class BirdDatasetBase(BaseDataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        system_prompt: Optional[str] = None,
        num_cot: Optional[int] = None,
        filter_by: Optional[tuple] = None,
        num_rows: Optional[int] = None,
        model_name_or_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        logger_name: str = "[BIRD-DATASET]",
    ):
        self.logger = setup_console_logger(name=logger_name)
        self.path = Path(data_path)

        data = download_and_process_bird_dataset(
            download_folder=self.path, force=False, split=split
        )

        if filter_by is not None:
            filter_key, filter_value = filter_by
            assert filter_key in ["db_id", "difficulty"], ValueError(
                "Filtering is supported for keys: 'db_id' and 'difficulty'"
            )
            if filter_key == "difficulty":
                assert filter_value in [
                    "simple",
                    "moderate",
                    "challenging",
                ], ValueError(
                    "difficulty can either be: 'simple' or 'moderate' or 'challenging'"
                )
            else:
                available_dbs = set([content["db_id"] for content in data])
                assert filter_value in available_dbs, ValueError(
                    f"available_dbs: {', '.join(available_dbs)}"
                )

            data = [content for content in data if content[filter_key] == filter_value]

        if num_rows is not None:
            assert 0 < num_rows <= len(data), ValueError(
                f"num_rows should be more than 0 and less than or equal to {len(data)}"
            )
            data = data[:num_rows]

        bird_instance = BirdDevInstance(data=data)
        self.data = bird_instance.apply_prompt(
            system_prompt=system_prompt, num_cot=num_cot
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
            if model_name_or_path
            else None
        )

        if self.tokenizer:
            self.logger.info("Formatting inputs to Alpaca style")
            sources, targets = [], []

            for example in self.data:
                sources.append(example["prompt"])
                targets.append(f"{example['SQL']}{self.tokenizer.eos_token}")

            self.logger.info("=> Starting tokenization")
            data_dict = self.preprocess(sources=sources, targets=targets)

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
        else:
            self.input_ids, self.labels = None, None

    def preprocess(self, sources: Sequence[str], targets: Sequence[str]):
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
        if not self.tokenizer:
            return self.data[idx]
        return dict(
            input_ids=self.input_ids[idx],
            labels=self.labels[idx],
            raw=dict(**self.data[idx]),
        )
