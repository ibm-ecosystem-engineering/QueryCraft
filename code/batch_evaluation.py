import pandas as pd
import os
import ex_evaluator
files = os.listdir("../input/qc_csvs_previous")
db_type = "db2"
for file in files:
    print(f"for {file}")
    exp_name=file.split(".csv")[0]
    input_dataset="../input/qc_csvs_previous/"+file
    database_path = "" # for sqlite
    expected_query_column = "expected query"
    generated_query_column = "generated query"
    schema_name = "SJC11494"
    print(db_type)
    ex_evaluator.ex_evalution(expected_query_column=expected_query_column,
                                generated_query_column=generated_query_column,
                                dbType=db_type,exp_name=exp_name,
                                input_dataset=input_dataset,
                                database_path=database_path,schema_name=schema_name)
