import pandas as pd
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_into_db(filename, table_name, schema_name, delimiter=","):
    conn = None
    try:
        # Create connection to SQLite database
        conn = sqlite3.connect(f'{schema_name}.sqlite')
        logging.info(f"Connected to the database {schema_name}.db successfully.")

        # Read data from file with specified delimiter
        df = pd.read_csv(filename, sep=delimiter)
        logging.info(f"Data loaded from {filename}: \n{df.head()}")
        if "Unnamed: 0 " in df.columns:
            df = df.drop(["Unnamed: 0"],axis=1)

        # Determine data types for SQL
        data_types = {
            'int64': 'INTEGER', 'int32': 'INTEGER',
            'float64': 'REAL', 'float32': 'REAL',
            'object': 'TEXT', 'bool': 'INTEGER',  # Assuming boolean values will be stored as integers (0 or 1)
        }

        # Create table if it does not exist
        column_defs = ', '.join([f"{col} {data_types[str(dtype)]}" for col, dtype in zip(df.columns, df.dtypes)])
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_defs})"
        print(create_table_sql)
        conn.execute(create_table_sql)
        conn.commit()
        logging.info(f"Table {table_name} SQL: {create_table_sql}")

        # Prepare the insert statement
        columns = ', '.join(df.columns)
        placeholders = ', '.join(['?'] * len(df.columns))
        insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        # Insert data into the table
        conn.executemany(insert_sql, df.values.tolist())
        conn.commit()
        logging.info(f"Data inserted into {table_name} successfully.")

    except pd.errors.EmptyDataError:
        logging.error("No data: The file is empty.")
    except pd.errors.ParserError:
        logging.error("Parsing error: Check the delimiter or file format.")
    except sqlite3.DatabaseError as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

    return f"Data ingested into table: {table_name}"

