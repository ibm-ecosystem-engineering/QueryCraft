import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR) 
logging.getLogger('vllm._C').setLevel(logging.ERROR)

# find _custom_ops.py and comment out vllm warnnings, # nothing works



import sys
import configparser
# import finetune as ft
import db2_Ingestion as dbin
import sqlite_ingestion as sqlin
import context_retriever as cr
import inference as inf
import query_correction
import ex_evaluator
import traceback as tr
#import streamlit_query_analysis_dashboard as dashboard

from configparser import ConfigParser, ExtendedInterpolation




config_filePath="./../simpleConfig.ini"
simple_config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
simple_config.read(config_filePath)
simple_config.sections()

# Check if arguments are provided
if len(sys.argv) < 1:
    logging.INFO("Usage: python script_name.py <component>")
    sys.exit(1)

# Access arguments
component = sys.argv[1]  # Exclude the script name, which is the first argument

# Process arguments
print("Component to run:", component)

if(component=="all"):
    import streamlit_query_analysis_dashboard as dashboard
    exp_name =simple_config['Default']['exp_name'] 

    input_database_folder = simple_config['Default']['home_dir']+simple_config['ContextRetriever']['input_database_folder']
    input_file = simple_config['Default']['home_dir']+simple_config['ContextRetriever']['input_data_file']
    db_type = simple_config['ContextRetriever']['db_type']
    cr_response = cr.funcContextRetriever(exp_name,db_type ,input_file, input_database_folder)
    print(cr_response)

    data_collator=simple_config['Finetune']['data_collator']
    model_name=simple_config['Finetune']['model_name']
    prompt_file_path=simple_config['Finetune']['prompt_file_path']
    finetune_type=simple_config['Finetune']['finetune_type']
    train_dataset =simple_config['Finetune']['train_dataset']
    ft_response = ft.funcFinetune(exp_name,data_collator,model_name,prompt_file_path,finetune_type,train_dataset)
    print(ft_response)

    model_name=simple_config['Inference']['model_name']
    finetuned_model = simple_config['Inference']['finetuned_model']
    input_dataset = simple_config['Inference']['input_dataset']
    inf_response = inf.executeInference(simple_config)
    print(inf_response)

    input_dataset=simple_config['QueryCorrection']['input_dataset']
    query_correction.funcQueryCorrection(exp_name,input_dataset)

    db_type = simple_config['EXEvaluator']['db_type']
    input_dataset=simple_config['EXEvaluator']['input_dataset']
    input_database_folder = simple_config['EXEvaluator']['input_database_folder']
    ex_evaluator.ex_evalution(db_type,exp_name,input_dataset,input_database_folder)
    
    folder_name = simple_config['Default']['home_dir']+simple_config['QueryAnalysisDashboard']['folder_name']
    dashboard.show_dashboard(folder_name)
    print("To view the query analysis dashboard execute the following command from the terminal: cd code streamlit run streamlit_query_analysis_dashboard.py --server.port 8502 --server.fileWatcherType none")

elif(component=="dataIngestion"):
    try:
        # print(type(simple_config['DataIngestion']))
        # for section in simple_config.sections():
        #     print(f"Section: {section}")
        #     for key, value in simple_config.items(section):
        #         print(f"  {key}: {value}")
        filename = simple_config['DataIngestion']['filename']
        table_name= simple_config['DataIngestion']['table_name']
        schema_name= simple_config['DataIngestion']['schema_name']
        db_type = simple_config['DataIngestion']['db_type']
        if db_type =="db2":        
            dbin_response = dbin.load_data_into_db(filename,table_name,schema_name)
            print(dbin_response)
        elif db_type == "sqlite":
            dbin_response = sqlin.load_data_into_db(filename,table_name,schema_name)
            print(dbin_response)


    except:
        print(tr.format_exc())

    
elif(component=="contextRetriever"):
    exp_name =simple_config['Default']['exp_name'] 
    input_database_folder = simple_config['Default']['home_dir']+simple_config['ContextRetriever']['input_database_folder']
    input_file = simple_config['Default']['home_dir']+simple_config['ContextRetriever']['input_data_file']
    db_type = simple_config['ContextRetriever']['db_type']
    
    cr_response = cr.funcContextRetriever(exp_name,db_type ,input_file, input_database_folder)
    print(cr_response)

elif(component=="finetune"):
    exp_name=simple_config['Default']['exp_name']
    data_collator=simple_config['Finetune']['data_collator']
    model_name=simple_config['Finetune']['model_name']
    prompt_file_path=simple_config['Finetune']['prompt_file_path']
    finetune_type=simple_config['Finetune']['finetune_type']
    train_dataset =simple_config['Finetune']['train_dataset']
    
    ft_response = ft.funcFinetune(exp_name,data_collator,model_name,prompt_file_path,finetune_type,train_dataset)
    print(ft_response)
    
elif(component=="inference"):
    exp_name=simple_config['Default']['exp_name']
    model_name=simple_config['Inference']['model_name']
    finetuned_model = simple_config['Inference']['finetuned_model']
    input_dataset = simple_config['Inference']['input_dataset']
    
    inf_response = inf.executeInference(simple_config)
    print(inf_response)
    
elif(component=="querycorrection"):
    exp_name=simple_config['Default']['exp_name']
    input_dataset=simple_config['QueryCorrection']['input_dataset']
    print("input_dataset",input_dataset)
    query_correction.funcQueryCorrection(exp_name,input_dataset)
    
elif(component=="evaluation"):
    db_type = simple_config['EXEvaluator']['db_type']
    exp_name=simple_config['Default']['exp_name']
    input_dataset=simple_config['EXEvaluator']['input_dataset']
    database_path = simple_config['EXEvaluator']['database_path']
    expected_query_column = simple_config['EXEvaluator']['expected_query_column']
    generated_query_column = simple_config['EXEvaluator']['generated_query_column']
    schema_name = simple_config['EXEvaluator']['schema_name']
    print(db_type)
    ex_evaluator.ex_evalution(expected_query_column=expected_query_column,
                              generated_query_column=generated_query_column,
                              dbType=db_type,exp_name=exp_name,
                              input_dataset=input_dataset,
                              database_path=database_path,schema_name=schema_name)

elif(component=="queryanalysisDashboard"):
    import streamlit_query_analysis_dashboard as dashboard
    folder_name = simple_config['Default']['home_dir']+simple_config['QueryAnalysisDashboard']['folder_name']
    dashboard.show_dashboard(folder_name)
    print("To view the query analysis dashboard execute the following command from the terminal: cd code streamlit run streamlit_query_analysis_dashboard.py --server.port 8502 --server.fileWatcherType none")
    
