



# =====


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

def map_original_to_sanitized(original_columns, sanitized_columns):
    return {sanitized: original for original, sanitized in zip(original_columns, sanitized_columns)}

def correct_sql_column_names(sql_query, column_mapping):
    for sanitized, original in column_mapping.items():
        sql_query = re.sub(r'\b{}\b'.format(re.escape(original)), sanitized, sql_query)
    return sql_query

def quote_sql_values(sql_query):
    # This regex matches values after '=' and before any spaces or closing parentheses
    return re.sub(r"(\s=\s)([^'\s]+)", r'\1"\2"', sql_query)

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
        table_columns = example["table"]["header"]
        table_rows = example["table"]["rows"]
        table_column_types = example["table"]["types"]

        sanitized_columns = [
            sanitize_column_name_for_wikisql(col) for col in table_columns
        ]
        unique_columns = ensure_unique_column_names(sanitized_columns)
        column_mapping = map_original_to_sanitized(table_columns, unique_columns)

        # Create the table schema
        try:
            if cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", 
                (table_name,)
            ).fetchone() is None:
                create_table_sql = f"""
                    CREATE TABLE {table_name} (
                        {', '.join([f'{col} {type_}' for col, type_ in zip(
                            unique_columns, table_column_types
                        )])}
                    )
                """
                cursor.execute(create_table_sql)
        except sqlite3.Error as e:
            logger.error(f"Failed to create table {table_name}. Error: {e}")
            continue

        # Insert rows into the table
        try:
            for row in table_rows:
                insert_row_sql = f"""
                    INSERT INTO {table_name} VALUES (
                        {', '.join(['?' for _ in row])}
                    )
                """
                cursor.execute(insert_row_sql, row)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(
                f"Failed to insert rows into the table {table_name} number: {i}. Error: {e}"
            )
            continue

        # Correct the SQL query
        original_sql_query = example["sql"]["human_readable"]
        corrected_sql_query = correct_sql_column_names(original_sql_query, column_mapping)
        corrected_sql_query = quote_sql_values(corrected_sql_query)
        corrected_sql_query = corrected_sql_query.replace("table", table_name)

        content = {
            "question": example["question"],
            "table": table_name,
            "db_path": str(db_path),
            "sql": corrected_sql_query,
        }

        # Append to dataset
        dataset.append(content)

    conn.close()
    return dataset

