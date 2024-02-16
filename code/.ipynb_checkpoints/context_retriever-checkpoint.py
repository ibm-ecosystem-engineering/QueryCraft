from datasets import load_dataset
from math import ceil
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.sql_database import SQLDatabase
import sqlite3
import configparser
import logging
import pandas as pd
######################################################################################################
config = configparser.ConfigParser()
config.read('config.ini')
config.sections()


database_folder = config['Default']['home_dir']+config['ContextRetriever']['input_database_folder']
input_data_file = config['Default']['home_dir']+config['ContextRetriever']['input_data_file']
output_file = config['Default']['home_dir']+"input/datasets/"+config['ContextRetriever']['output_file']+".csv"

data = pd.read_csv(input_data_file)
num_samples = len(data)
print(f"Loaded {num_samples} samples. ")
df_train = data

def contextFinderSqlite(db):
    file_path = f"{database_folder}{db}/{db}.sqlite"
    print(file_path)
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    #Get all the tables in a db
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    table_list = []
    for row in cur.execute(query):
        table_list.append(row[0])
    tables_create_queries = []
    for table in table_list:
        table_name = table
        # Connect to the SQLite database
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        # Get the DDL query for the specified table
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        ddl_query = cursor.fetchone()

        # Close the cursor and the connection
        cursor.close()
        conn.close()

        if ddl_query:
            temp = ddl_query[0].replace("\n"," ") 
            temp = " ".join(temp.split())
            #tables_create_queries.append(ddl_query[0])
            tables_create_queries.append(temp)
        else:
            print(f"Table '{table_name}' not found in the database.")
    con.close()
    return tables_create_queries


def filter_strings_with_keyword(strings, keyword):
    filtered_strings = [s for s in strings if s.startswith(keyword)]
    return filtered_strings

def contextFinder(db):
    print(f"{database_folder}{db}/{db}.sql")
    file = glob.glob(f"{database_folder}{db}/{db}.sql")
    try:
        fd = open(file[0], 'r')
        sqlFile = fd.read()
        fd.close()
        sqlFile = sqlFile.replace("\n","")
        # all SQL commands (split on ';')
        sqlCommands = sqlFile.split(';')
        filtered_strings = filter_strings_with_keyword(sqlCommands,"CREATE")
        filtered_strings = [x+";" for x in filtered_strings]
        return filtered_strings
    except:
        return contextFinderSqlite(db)

df_train["context"] = df_train["db_id"].apply(contextFinder)
df_train.to_csv(output_file)