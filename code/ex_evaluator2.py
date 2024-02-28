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

import db2_connector
import ibm_db

######################################################################################################
# EX Match Logic
######################################################################################################
def hello(name):
    print("helloooo",name)

def error_handling(e):
    error ="None"
    if 'no such column' in e:
            error ="No such column"
    elif 'syntax error' in e:
            error = "Syntax error"
    elif 'no such table' in e:
            error = "No such table"
    elif 'ambiguous column name' in e:
            error = "Ambiguous column name"
    else:
        error = e
    return error

def reformat_query(query: str) -> str:
    t_stars = ["t1.*", "t2.*", "t3.*", "T1.*", "T2.*", "T3.*"]
    for ts in t_stars:
        query = query.replace(ts, "*")
    return query


def isValidSQL(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    except:
        return False
    return True

def unorder_row(row):
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))

def quick_rej(result1, result2, order_matters):
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)
    
def get_constraint_permutation(tab1_sets_by_columns, result2):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)

def permute_tuple(element, perm):
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])

def multiset_eq(l1, l2):
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def result_eq(result1, result2, order_matters):
    result ="None"
    if len(result1) == 0 and len(result2) == 0:
        result = "same"
        return True,result

    # if length is not the same, then they are definitely different bag of rows
    status =0
    if len(result1) != len(result2):
        if len(result1)==0:
            result = "P result zero"
        elif len(result2)==0:
            result = "Q result zero"
        elif len(result1) > len(result2):
            for res in result2:
                if res in result1:
                    status =1
            if status ==1:
                result = "Partial Match"
            else:
                result = "P result greater"
                
        elif len(result1) < len(result2):
            for res in result1:
                if res in result2:
                    status =1
            if status ==1:
                result = "Partial Match"
            else:   
                result = "Q result greater"    
        return False,result

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        result = "column length different"
        return False,result

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        count =0
        for res in result2:
                if res in result1:
                    count =1
        if count ==1:
            result = "Partial Match"
        else:
            result = "order or result different"
        return False,result

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                result ="same"
                return True,result
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                result ="same"
                return True,result
    return False,result


def eval_exec_match_sqlite(db, db2, p_str, g_str):
    """
    return 1 if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """
    error ='None'
    result = "error"
    conn = sqlite3.connect(db2)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    cursor = conn.cursor()
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
    except Exception as e:
        # import ipdb; ipdb.set_trace()
        error =error_handling(str(e))
        return False,error,result

    conn = sqlite3.connect(db)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    cursor = conn.cursor()
    try:
        cursor.execute(g_str)
    except Exception as e:
        error =error_handling(str(e))
        return False,error,result
    q_res = cursor.fetchall()

    ##orders_matter = 'order by' in g_str.lower()
    orders_matter = False
    value,result = result_eq(p_res, q_res, order_matters=orders_matter)
    return value,error,result




def eval_exec_match_db2(db2_conn,db2_conn1, p_str, g_str):
    import ibm_db
    """
    return 1 if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """
    error ='None'
    result = "error"
    try:
        stmt = ibm_db.exec_immediate(db2_conn, p_str)
        p_res = ibm_db.fetch_assoc(stmt)
    except Exception as e:
        # import ipdb; ipdb.set_trace()
        error =error_handling(str(e))
        return False,error,result
    try:
        stmt = ibm_db.exec_immediate(db2_conn1, g_str)
        q_res = ibm_db.fetch_assoc(stmt)
    except Exception as e:
        error =error_handling(str(e))
        return False,error,result
    
    ##orders_matter = 'order by' in g_str.lower()
    orders_matter = False
    value,result = result_eq(p_res, q_res, order_matters=orders_matter)
    return value,error,result


def formaterAndCaller(row,database_folder):
    db = database_folder+row["db_id"]+"/"+row["db_id"]+".sqlite"
    g_str = row["query"]+";"
    p_str =row["model_op"]
    
    ## For query correction:
    p_str_p =row["model_op1"]
    print("I am at row:",row["Sno"])
    eval_score,e,r = eval_exec_match(db,db,p_str, g_str)
    eval_score1 ,error,result = eval_exec_match(db,db,p_str_p, g_str)
    
    return eval_score,eval_score1,error,result
  
def formaterAndCaller_db2(row):
    conn = db2_connector.db2_connector()

    g_str = row["query"]+";"
    p_str =row["model_op"]
    
    ## For query correction:
    p_str_p =row["model_op1"]
    print("I am at row:",row["Sno"])
    eval_score,e,r = eval_exec_match_db2(conn,conn,p_str, g_str)
    eval_score1 ,error,result = eval_exec_match_db2(conn,conn,p_str_p, g_str)
    
    return eval_score,eval_score1,error,result
    

    
def ex_evalution(dbType,input_dataset,database_folder):
    print("readingggggg eval")
    config_filePath="./../config.ini"
    config = configparser.ConfigParser()
    config.read(config_filePath)
    config.sections()
    logging_path = config['Default']['home_dir']+config['logs']['log_folder']+"/"+ config['EXEvaluator']['EXP'] +"_EX"
    logging.basicConfig(filename=logging_path+".log", level=logging.INFO)

    ######################################################################################################
    # Read the inference dataset
    ######################################################################################################
    df = pd.DataFrame()
    if input_dataset == '':
        df = pd.read_csv(config['Default']['home_dir']+config['EXEvaluator']['input_dataset'])
    else:
        df = pd.read_csv(input_dataset)
        
    if dbType =='sqlite':
        if database_folder == '':
            database_folder = config['Default']['home_dir']+config['ContextRetriever']['input_database_folder']            
        print("reading the file :",config['EXEvaluator']['input_dataset'])
        for index, row in df.iterrows():
            print("Helloo---")
            evalScore,value,error,result = formaterAndCaller_sqlite(row,database_folder)
            df.at[index,"evalScore"] = evalScore
            df.at[index,"evalScorePostProcessing"] = value
            df.at[index,"error_type"] = error
            df.at[index,"result"] = result
    else :
        for index, row in df.iterrows():
            evalScore,value,error,result = formaterAndCaller_db2(row)
            df.at[index,"evalScore"] = evalScore
            df.at[index,"evalScorePostProcessing"] = value
            df.at[index,"error_type"] = error
            df.at[index,"result"] = result
    EXAccuracy = sum(df["evalScore"])/len(df["evalScore"])
    EXAccuracyPP = sum(df["evalScorePostProcessing"])/len(df["evalScorePostProcessing"])
    logging.info("EX Accuracy :"+str(EXAccuracy))
    logging.info("PP EX Accuracy :"+str(EXAccuracyPP))
    print("PP EX Accuracy :",str(EXAccuracyPP))
    print("EX Accuracy :",str(EXAccuracy))
    df.to_csv(config['Default']['home_dir']+"output/evalResults/"+config['EXEvaluator']['EXP']+"_EX.csv")
    print("File saved succesfully")
