import pandas as pd
import ibm_db
import sqlite3
import configparser
import glob

config = configparser.ConfigParser()
config.read('../config.ini')

# User Config
dsn_database = config['DataIngestion']["dsn_database"]
dsn_uid = config['DataIngestion']["dsn_uid"]
dsn_pwd = config['DataIngestion']["dsn_pwd"]
dsn_hostname = config['DataIngestion']["dsn_hostname"]
dsn_port = config['DataIngestion']["dsn_port"]
dsn_protocol = config['DataIngestion']["dsn_protocol"]
dsn_driver = config['DataIngestion']["dsn_driver"]

output_file = config['ContextRetriever']["output_file"]


def funcContextRetriever(db_type, input_file, database_folder='input/spider/database/'):
    
    def contextFinder(db):
        if db_type == 'db2':
            return contextFinderDb2(db)
        elif db_type == 'sqlite':
            return contextFinderSqlite(db,database_folder)
        else:
            raise ValueError("Invalid database type. Supported types: 'db2', 'sqlite'")
    
            
    database_folder = config['Default']['home_dir'] + database_folder
    df_train = pd.read_csv(input_file)
    
    df_train["context"] = df_train["db_id"].apply(contextFinder)
    df_train.to_csv("../input/datasets/"+output_file)
    
    return "Generated Retriver file: "+"../input/datasets/"+output_file
    
    
def contextFinderDb2(table_name):
    dsn = ("DRIVER={{IBM DB2 ODBC DRIVER}};" "DATABASE={0};" "HOSTNAME={1};" "PORT={2};" "PROTOCOL=TCPIP;" "UID={3};" "PWD={4};SECURITY=SSL").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd)
    options = {ibm_db.SQL_ATTR_AUTOCOMMIT: ibm_db.SQL_AUTOCOMMIT_ON}
    conn = ibm_db.connect(dsn, "", "", options)
    
    sql = f"""    
        SELECT 
        c.colname AS column_name,
        c.typename AS data_type,
        c.length,
        CASE 
            WHEN k.colname IS NOT NULL THEN 'YES'
            ELSE 'NO'
        END AS is_primary_key
    FROM 
        syscat.columns c
    LEFT JOIN 
        syscat.keycoluse k ON c.tabschema = k.tabschema 
                            AND c.tabname = k.tabname 
                            AND c.colname = k.colname
    INNER JOIN 
        syscat.tables t ON t.tabschema = c.tabschema 
                        AND t.tabname = c.tabname
    WHERE 
        t.type = 'T' 
        AND c.tabname = '{table_name}'
    ORDER BY 
        c.tabschema, c.tabname;
    """
    
    stmt = ibm_db.exec_immediate(conn, sql)
    columns = []
    primary_keys = []

    while True:
        row = ibm_db.fetch_tuple(stmt)
        if not row:
            break
        column_name, data_type, length, is_primary_key = row
        columns.append((column_name, data_type, length))
        if is_primary_key == 'YES':
            primary_keys.append(column_name)

    create_table_sql = f"CREATE TABLE {table_name} (\n"
    for column in columns:
        column_name, data_type, length = column
        create_table_sql += f"\t{column_name} {data_type}"
        if data_type in ["VARCHAR", "CHARACTER", "CHAR"]:
            create_table_sql += f"({length})"
        create_table_sql += ",\n"
    if primary_keys:
        create_table_sql += f"\tPRIMARY KEY ({', '.join(primary_keys)})\n"
    create_table_sql = create_table_sql.rstrip(",\n")
    create_table_sql += "\n)"
    print(create_table_sql)
    ibm_db.close(conn)

    return create_table_sql

def contextFinderSqlite(db,database_folder):

    file_path = f"{database_folder}{db}/{db}.sqlite"
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    table_list = [row[0] for row in cur.execute(query)]
    tables_create_queries = []
    for table in table_list:
        cursor = con.cursor()
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
        ddl_query = cursor.fetchone()
        cursor.close()
        if ddl_query:
            temp = ddl_query[0].replace("\n"," ")
            temp = " ".join(temp.split())
            tables_create_queries.append(temp)
        else:
            print(f"Table '{table}' not found in the database.")
    con.close()
    return tables_create_queries
