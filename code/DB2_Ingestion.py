import pandas as pd
import ibm_db
import configparser
import db2_connector as dbcon

# Read configuration from config.ini file
config = configparser.ConfigParser()
config.read('./../config.ini')


delimiter = config['DataIngestion']["delimiter"]
conn = dbcon.db2_connector()

def load_data_into_db(filename,table_name,schema_name,delimiter = delimiter):

    # Read data from file with specified delimiter
    df = pd.read_csv(filename,sep= delimiter)
    # df = df[0:20]

    # Check if the table exists
    def table_exists(conn, table_name,schema_name):
        sql = "SELECT * FROM {}.{}".format(schema_name, table_name)
        try:
            stmt = ibm_db.exec_immediate(conn, sql)

            while ibm_db.fetch_row(stmt) != False:
                return True
        except:
            return False

    columns_schema = [(column, 'INT' if dtype == 'int64' or dtype == 'int32' else 'FLOAT' if dtype == 'float64' or dtype == 'float32' else 'VARCHAR(1000)') for column, dtype in zip(df.columns, df.dtypes)]

    if not table_exists(conn, table_name,schema_name):
        create_table_sql = "CREATE TABLE {}.{} (".format(schema_name, table_name)
        for column, data_type in columns_schema:
            create_table_sql += "{} {}, ".format(column, data_type)
        create_table_sql = create_table_sql[:-2] + ")" 
        stmt = ibm_db.exec_immediate(conn, create_table_sql)

    columns = ','.join(df.columns)

    placeholders = ','.join(['?'] * len(df.columns))

    insertSQL = f'INSERT INTO {schema_name}.{table_name} ({columns}) VALUES({placeholders})'
    stmt = ibm_db.prepare(conn, insertSQL)

    # Insert data into the table
    for n in range(len(df)):  # start from 0 if you want to include all rows
        #Dynamically bind parameters based on the DataFrame's columns
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

    ibm_db.close(conn)
    
    return "Data Ingested into table: "+schema_name+"."+table_name 
    # Dynamically bind parameters based on the DataFrame's columns
    for i, col in enumerate(df.columns, start=1):
        value = df.at[n, col]
        if isinstance(value, int):
            data_type = ibm_db.SQL_INTEGER
        elif isinstance(value, float):
            data_type = ibm_db.SQL_DECIMAL
        else:
            data_type = ibm_db.SQL_VARCHAR
        # Quote column names properly
        col_name = f'"{col}"'
        ibm_db.bind_param(stmt, i, str(value), ibm_db.SQL_PARAM_INPUT, data_type)
    ibm_db.execute(stmt)
# load_data_into_db(filename, table_name,schema_name,delimiter)