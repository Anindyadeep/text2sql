import os
import re
import json
import shutil
import sqlite3
import requests
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile
from typing import Optional
from datasets import load_dataset

from text2sql.logger import setup_console_logger
from text2sql.constants import RESERVED_KEYWORDS

logger = setup_console_logger(name="[DATA-UTILS]")

# BIRDBench utility functions
# It has been modified a bit to make it more readable and to remove the unnecessary parts

# Set of Commands for train
# unzip train.zip
# cd train
# unzip train_databases.zip
# rm -rf __MACOSX
# rm -rf train_databases.zip


def _bird_bench_train_dataset_steps(download_path: Path, force: Optional[bool] = False):
    url = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"

    zip_file = download_path / "train.zip"
    inner_zips = download_path / "train"
    inner_zip_file = download_path / "train_databases.zip"
    unzip_dir = download_path / "train_databases"
    inner_train_databases = unzip_dir / "train_databases"
    macosx_dir = download_path / "__MACOSX"
    inner_macosx_dir = download_path / "train_databases" / "__MACOSX"

    if not (download_path).exists() or force:
        if not zip_file.exists():
            logger.info(
                "=> Starting to download the dataset for training [BirdBench trainset]."
            )
            response = requests.get(url)
            with open(zip_file, "wb") as f:
                f.write(response.content)

        # Do the initial extraction
        logger.info("=> Extracting BirdBench training dataset.")
        with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(zip_file)

        # Then remove the macosx dir (which is the initial one)
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir, ignore_errors=True)

        # Now move all contents from download_path / train to download_path
        for file in inner_zips.iterdir():
            shutil.move(file, download_path)

        # Now remove download_path / train
        shutil.rmtree(inner_zips, ignore_errors=True)

        # Now extract download_path / train_databases.zip to download_path / train_databases
        with ZipFile(inner_zip_file, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
        os.remove(inner_zip_file)

        # Now delete __MACOSX inside download_path / train_databases

        if inner_macosx_dir.exists():
            shutil.rmtree(inner_macosx_dir, ignore_errors=True)

        # first rename download_path / train_databases to download_path / tmp
        # then move the folder inside download_path / tmp to download_path
        # and delete download_path / tmp
        tmp_dir = download_path / "tmp"
        inner_train_databases.rename(tmp_dir)

        if unzip_dir.exists():
            shutil.rmtree(unzip_dir, ignore_errors=True)
        tmp_dir.rename(unzip_dir)
        logger.info("=> Finished extraction.")

    else:
        logger.info("=> Dataset for training [BirdBench trainset] already exists.")


def _bird_bench_dev_dataset_steps(download_path: Path, force: Optional[bool] = False):
    url = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
    zip_file = download_path / "dev.zip"
    unzip_dir = download_path / "dev_20240627"
    inner_zip_file = unzip_dir / "dev_databases.zip"
    macosx_dir = download_path / "__MACOSX"

    if not (download_path).exists() or force:
        logger.info(
            "=> Starting to download the dataset for evaluation [BirdBench devset]."
        )
        response = requests.get(url)

        with open(zip_file, "wb") as f:
            f.write(response.content)

        # Now extract
        with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(zip_file)

        with ZipFile(inner_zip_file, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
        os.remove(inner_zip_file)

        # Move things
        for item in os.listdir(unzip_dir):
            shutil.move(unzip_dir / item, download_path)
        os.rmdir(unzip_dir)

        if macosx_dir.exists():
            shutil.rmtree(macosx_dir, ignore_errors=True)

        logger.info(
            "=> Finished downloading the dataset for evaluation [BirdBench devset]."
        )
    else:
        logger.info("=> Dataset for evaluation [BirdBench devset] already exists.")


def download_and_process_bird_dataset(
    split: Optional[str] = "train",
    download_folder: Optional[str] = "./data",
    force: Optional[bool] = False,
):
    assert split in ["train", "validation"], "Invalid split name"

    download_folder = Path(download_folder)
    download_path = download_folder / "bird" / split

    if not download_path.exists():
        download_path.mkdir(parents=True, exist_ok=True)

    if split == "train":
        _bird_bench_train_dataset_steps(download_path, force)
    else:
        _bird_bench_dev_dataset_steps(download_path, force)

    data_split = "train" if split == "train" else "dev"
    dataset = json.load(open(download_path / f"{data_split}.json", "r"))

    for blob in dataset:
        blob["db_path"] = str(
            download_path / "dev_databases" / blob["db_id"] / f"{blob['db_id']}.sqlite"
        )
    return dataset


# WikiSQL utility functions
# It has been modified a bit to make it more readable and to remove the unnecessary parts
# Instead of the returning the dataset make sure to make a split.json file under a separate folder


def sanitize_column_name_for_wikisql(column_name: str) -> str:
    sanitized_name = re.sub(r"\W+", "_", column_name)
    if sanitized_name[0].isdigit():
        sanitized_name = "_" + sanitized_name
    if sanitized_name.upper() in RESERVED_KEYWORDS:
        sanitized_name = sanitized_name + "_"
    return sanitized_name


def ensure_unique_column_names(columns):
    seen = {}
    unique_columns = []
    for col in columns:
        counter = seen.get(col, 0)
        new_col = col if counter == 0 else f"{col}_{counter}"
        seen[col] = counter + 1
        seen[new_col] = 1
        unique_columns.append(new_col)
    return unique_columns


def _process_wikisql_tables(raw_dataset, download_path: Path, split: str):
    dataset = []
    db_path = download_path / f"{split}.db"

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
    except sqlite3.Error as e:
        logger.error("Failed to connect to the database. Error: %s", e)
        return dataset

    for i, example in tqdm(enumerate(raw_dataset), total=len(raw_dataset)):
        table_name = f"table_{example['table']['id'].replace('-', '_')}"
        content = {
            "question": example["question"],
            "table": table_name,
            "db_path": str(db_path),
            "sql": example["sql"]["human_readable"].replace("table", table_name),
        }

        table_columns = example["table"]["header"]
        table_rows = example["table"]["rows"]
        table_column_types = example["table"]["types"]
        sanitized_columns = [
            sanitize_column_name_for_wikisql(col) for col in table_columns
        ]
        unique_columns = ensure_unique_column_names(sanitized_columns)

        # Create the table schema
        try:
            if (
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                    (table_name,),
                ).fetchone()
                is None
            ):
                create_table_sql = f"""
                    CREATE TABLE {content['table']} (
                        {', '.join([f'{col} {type_}' for col, type_ in zip(
                            unique_columns, table_column_types
                        )])}
                    )
                """
                cursor.execute(create_table_sql)
        except sqlite3.Error as e:
            logger.error(f"Failed to create table {content['table']}. Error: {e}")
            continue

        # Insert rows into the table
        try:
            for row in table_rows:
                insert_row_sql = f"""
                    INSERT INTO {content['table']} VALUES (
                        {', '.join(['?' for _ in row])}
                    )
                """
                cursor.execute(insert_row_sql, row)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(
                f"Failed to insert rows into the table {content['table']} number: {i}. Error: {e}"
            )
            continue

        # now append things
        dataset.append(content)

    conn.close()
    return dataset


def download_and_process_wikisql_dataset(
    download_folder: Optional[str] = "./data",
    split: Optional[str] = "train",
    force: Optional[bool] = False,
):
    assert split in ["train", "validation", "test"], "Invalid split name"

    download_path = Path(download_folder) / "wikisql" / split
    if not download_path.exists() or force:
        download_path.mkdir(parents=True, exist_ok=True)

        raw_dataset = load_dataset("Salesforce/wikisql")[split]
        dataset = _process_wikisql_tables(raw_dataset, download_path, split)

        # Write this list of dict jsonl using json.dump
        with open(download_path / f"{split}.json", "w") as f:
            json.dump(dataset, f, indent=4)

        logger.info(f"Finished processing the WikiSQL {split} dataset.")

    else:
        logger.info(f"WikiSQL {split} dataset already exists.")
        dataset = json.load(open(download_path / f"{split}.json", "r"))
    return dataset
