import pandas as pd
import ibm_db
import configparser
import db2_connector as dbcon

import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Read configuration from expertConfig.ini file
expertConfig = configparser.ConfigParser()
expertConfig.read('./../expertConfig.ini')

simple_config = configparser.ConfigParser()
simple_config.read('./../simpleConfig.ini')

delimiter = expertConfig['DataIngestion']["delimiter"]

filename = simple_config['DataIngestion']["filename"]#'/home/askyourcorpus/owais_querycraft_experiments/QueryCraft-fork/synthetic_data.csv'
table_name = simple_config['DataIngestion']["table_name"]#'contracts_database_owais'
schema_name = simple_config['DataIngestion']["schema_name"]#'test'


def load_data_into_db(filename, table_name, schema_name, delimiter=delimiter):

    conn = dbcon.db2_connector()

    try:
        drop_table_sql = "DROP TABLE {}.{}".format(schema_name,table_name)
        ibm_db.exec_immediate(conn, drop_table_sql)
    except:
        print("Table does not exist in database, so it was not dropped")
        pass
    try:
        # Read data from file with specified delimiter
        df = pd.read_csv(filename, sep=delimiter)
        df = df.dropna(axis=1, how='any')
        print(df.head())
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    try:
        drop_table_sql = "DROP TABLE {}.{}".format(schema_name,table_name)
        ibm_db.exec_immediate(conn, drop_table_sql)
    except:
        print("Table does not exist in database, so it was not dropped")
        pass
    # Check if the table exists
    def table_exists(conn, table_name, schema_name):
        sql = """
        SELECT 1 FROM SYSCAT.TABLES
        WHERE TABSCHEMA = ? AND TABNAME = ?
        FETCH FIRST ROW ONLY
        """
        try:
            stmt = ibm_db.prepare(conn, sql)
            
            ibm_db.bind_param(stmt, 1, schema_name.upper())
            ibm_db.bind_param(stmt, 2, table_name.upper())
            
            ibm_db.execute(stmt)
            
            result = ibm_db.fetch_assoc(stmt)            
            return result
        except Exception as e:
            print(f"Error checking if table exists: {e}")
            return False
    
    columns_schema = [(column, 'INT' if dtype == 'int64' or dtype == 'int32' else 'FLOAT' if dtype == 'float64' or dtype == 'float32' else 'VARCHAR(1000)') for column, dtype in zip(df.columns, df.dtypes)]
    print(table_exists(conn, table_name, schema_name))
    if not table_exists(conn, table_name, schema_name):
        create_table_sql = "CREATE TABLE {}.{} (".format(schema_name, table_name)
        
        for column, data_type in columns_schema:
            create_table_sql += "{} {}, ".format(column, data_type)
        create_table_sql = create_table_sql[:-2] + ")"
        try:
            stmt = ibm_db.exec_immediate(conn, create_table_sql)
        except Exception as e:
            print(f"Error creating table: {e}")
            return
    
    columns = ','.join(df.columns)
    placeholders = ','.join(['?'] * len(df.columns))
    insertSQL = f'INSERT INTO {schema_name}.{table_name} ({columns}) VALUES({placeholders})'
    
    try:
        stmt = ibm_db.prepare(conn, insertSQL)
    except Exception as e:
        print(f"Error preparing SQL statement: {e}")
        return

    # Insert data into the table
    try:
        for n in range(len(df)):
            for i, col in enumerate(df.columns, start=1):
                value = df.at[n, col]
                if isinstance(value, int):
                    data_type = ibm_db.SQL_INTEGER
                elif isinstance(value, float):
                    data_type = ibm_db.SQL_DECIMAL
                else:
                    data_type = ibm_db.SQL_VARCHAR
                ibm_db.bind_param(stmt, i, str(value), ibm_db.SQL_PARAM_INPUT, data_type)
            ibm_db.execute(stmt)
    except Exception as e:
        print(f"Error inserting data: {e}")
        return

    try:
        ibm_db.close(conn)
    except Exception as e:
        print(f"Error closing the connection: {e}")

    return "Data Ingested into table: {}.{}".format(schema_name, table_name)

# filename = '/home/askyourcorpus/owais_querycraft_experiments/QueryCraft-fork/synthetic_data.csv'
# table_name = 'contracts_database_owais'
# schema_name = 'test'