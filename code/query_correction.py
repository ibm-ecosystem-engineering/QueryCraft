from __future__ import print_function
import os, sys
import json
import sqlite3
import traceback
import argparse
from tqdm import tqdm
from itertools import product
from collections import defaultdict
import random
from datetime import datetime
import os
from math import ceil
import random
from typing import Optional
from pathlib import Path
import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
# from process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql
import pandas as pd
import configparser
import logging
import re
######################################################################################################

######################################################################################################
# EX Match Logic
######################################################################################################

def reformat_query(query: str) -> str:
    t_stars = ["t1.*", "t2.*", "t3.*", "T1.*", "T2.*", "T3.*"]
    for ts in t_stars:
        query = query.replace(ts, "*")
    return query


def replace_cur_year(query: str) -> str:
    return re.sub(
        "YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
    )

def query_processing(row):
    g_str =''
    p_str=''
    if ';' not in row["query"]:
        g_str = row["query"]+" ;"
    else:
        g_str = row["query"]
    
    if ';' not in row["model_op"]:
        p_str = row["model_op"]+" ;"
    else:
        p_str = row["model_op"].split(";")[0]
        
    p_str = p_str.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    
    g_str = g_str.replace('``` ',"").replace('`',"")
    p_str = p_str.replace('``` ',"").replace('`',"")
    p_str = p_str.replace('### Expected Output:   ',"").replace('`',"")
    p_str = p_str.replace('Note:',"")
    p_str = p_str.replace(' Ex',"")
    p_str = p_str.replace('Here is the',"")
    p_str = p_str.split("### Explanation:")[0]
    p_str = p_str.split("Explanation: ")[0]
    p_str = p_str.split(": Explanation:")[0]
    p_str = p_str.split("Explanation:")[0]
    
    p_str = p_str.replace('ILIKE',"LIKE")
    p_str = p_str.replace('ilike',"LIKE")
    
    if "### Response:" in p_str:
        p_str = p_str.split("### Response:")[1]
    p_str = p_str.replace("###","")
    
    
   
    p_str_val = p_str.split(": Answer:")
    if len(p_str_val) ==2:
        p_str = p_str_val[1]
    p_str_val = p_str.split(": Query:")
    if len(p_str_val) ==2:
        p_str = p_str_val[1]
    
    if "This query" in p_str:
         p_str = p_str.split("This query")[0]
    if "The query" in p_str:
         p_str = p_str.split("The query")[0]     
    if "The above query" in p_str:
         p_str = p_str.split("The above query")[0]
    if "planation:" in p_str:
         p_str = p_str.split("planation:")[0]
    if "This queries" in p_str:
         p_str = p_str.split("This queries")[0]
    if "noqa: E501" in p_str:
         p_str = p_str.split("noqa: E501")[0]
   


    p_str = p_str.split(": Result:")[0]
    p_str = p_str.split("INST ")[0]
    p_str = p_str.split(" INST")[0]
    p_str = p_str.split(" find ")[0]
    p_str = p_str.split(" INST)")[0]
    
    
    p_str = p_str.strip()
    g_str = g_str.strip()
    p_str = p_str.replace("#","")
    p_str = reformat_query(p_str)
    p_str = replace_cur_year(p_str)
    
    if "select" in p_str.lower():
        if ':' in p_str:
            p_str=p_str.replace(":","")
        if ';' not in p_str:
            p_str=p_str+' ;'
    return g_str, p_str

  
def funcQueryCorrection(exp_name ='exp_codellama-13b_spider_0412',input_dataset='/data/rlhf/amit/QueryCraft-The-SuperKnowa-SQL-Sculptor/output/inference/exp_codellama-13b_spider_0412.csv'):
    config_filePath="./../config.ini"
    config = configparser.ConfigParser()
    config.read(config_filePath)
    config.sections()
    super_config = configparser.ConfigParser()
    super_config.read('./../superConfig.ini')
    home_dir  = super_config['Default']['home_dir']

    logging_path = home_dir+config['logs']['log_folder']+"/"+ exp_name +"_EX"
    logging.basicConfig(filename=logging_path+".log", level=logging.INFO)
    df = pd.read_csv(input_dataset)
    df["model_op"] = df["model_op"].apply(lambda x : x.replace("{",""))
    df["model_op"] = df["model_op"].apply(lambda x : x.replace("}",""))
    for index, row in df.iterrows():
        g_str, p_str = query_processing(row)
        df.at[index,"query"] = g_str
        df.at[index,"model_op1"] = p_str
    df.to_csv(input_dataset)
    print("File saved succesfully")
