import pandas as pd
import ibm_db
import sqlite3
import configparser
import glob
import db2_connector as dbcon

config = configparser.ConfigParser()
config.read('./../config.ini')

super_config = configparser.ConfigParser()
super_config.read('./../superConfig.ini')
home_dir  = super_config['Default']['home_dir']

def funcContextRetriever(exp_name,db_type, input_file, database_folder='../input/spider/database/'):
    output_file = home_dir+"input/datasets/"+exp_name+"_contextRetriever.csv"
    def contextFinder(db):
        if db_type == 'db2':
            return contextFinderDb2(db)
        elif db_type == 'sqlite':
            return contextFinderSqlite(db,database_folder)
        else:
            raise ValueError("Invalid database type. Supported types: 'db2', 'sqlite'")
            
    df_train = pd.read_csv(input_file)
    df_train["context"] = df_train["db_id"].apply(contextFinder)
    df_train.to_csv(output_file)
    
    return "Generated Retriver file:"+output_file

def contextFinderDb2(schema_name):
    conn = dbcon.db2_connector()
    schema_name = schema_name.upper()
    # Querying list of tables within the specified schema
    tables_query = f"SELECT TABNAME FROM SYSCAT.TABLES WHERE TABSCHEMA = '{schema_name}'"
    tables_stmt = ibm_db.exec_immediate(conn, tables_query)
    
    tables = []
    while True:
        row = ibm_db.fetch_tuple(tables_stmt)
        if not row:
            break
        tables.append(row[0])
    create_tables_sql = []
    for table_name in tables:
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
        create_tables_sql.append(create_table_sql)
        
    ibm_db.close(conn)
    combined_sql = "\n".join(create_tables_sql)
    return combined_sql


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
