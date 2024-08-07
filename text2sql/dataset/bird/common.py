import json
import sqlite3
from tqdm import tqdm
from textwrap import dedent
from typing import Optional
from tabulate import tabulate
from text2sql.dataset.base import BaseDataInstance


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
