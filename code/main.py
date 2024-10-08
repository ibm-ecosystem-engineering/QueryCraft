import sys
import configparser
import finetune as ft
import db2_Ingestion as dbin
import context_retriever as cr
import inference as inf
import query_correction
import ex_evaluator
#import streamlit_query_analysis_dashboard as dashboard

from configparser import ConfigParser, ExtendedInterpolation

config_filePath="./../simpleConfig.ini"
expertConfig = configparser.ConfigParser(interpolation=ExtendedInterpolation())
expertConfig.read(config_filePath)
expertConfig.sections()

# Check if arguments are provided
if len(sys.argv) < 1:
    print("Usage: python script_name.py <component>")
    sys.exit(1)

# Access arguments
component = sys.argv[1]  # Exclude the script name, which is the first argument

# Process arguments
print("Component to run:", component)

if(component=="all"):
    import streamlit_query_analysis_dashboard as dashboard
    exp_name =expertConfig['Default']['exp_name'] 

    input_database_folder = expertConfig['Default']['home_dir']+expertConfig['ContextRetriever']['input_database_folder']
    input_file = expertConfig['Default']['home_dir']+expertConfig['ContextRetriever']['input_data_file']
    db_type = expertConfig['ContextRetriever']['db_type']
    cr_response = cr.funcContextRetriever(exp_name,db_type ,input_file, input_database_folder)
    print(cr_response)

    data_collator=expertConfig['Finetune']['data_collator']
    model_name=expertConfig['Finetune']['model_name']
    prompt_file_path=expertConfig['Finetune']['prompt_file_path']
    finetune_type=expertConfig['Finetune']['finetune_type']
    train_dataset =expertConfig['Finetune']['train_dataset']
    ft_response = ft.funcFinetune(exp_name,data_collator,model_name,prompt_file_path,finetune_type,train_dataset)
    print(ft_response)

    model_name=expertConfig['Inference']['model_name']
    finetuned_model = expertConfig['Inference']['finetuned_model']
    input_dataset = expertConfig['Inference']['input_dataset']
    inf_response = inf.executeInference(expertConfig)
    print(inf_response)

    input_dataset=expertConfig['QueryCorrection']['input_dataset']
    query_correction.funcQueryCorrection(exp_name,input_dataset)

    db_type = expertConfig['EXEvaluator']['db_type']
    input_dataset=expertConfig['EXEvaluator']['input_dataset']
    input_database_folder = expertConfig['EXEvaluator']['input_database_folder']
    expected_query_column = expertConfig['EXEvaluator']['expected_query_column']
    generated_query_column = expertConfig['EXEvaluator']['generated_query_column']
    ex_evaluator.ex_evalution(db_type, exp_name, input_dataset, input_database_folder, expected_query_column, generated_query_column)
    
    folder_name = expertConfig['Default']['home_dir']+expertConfig['QueryAnalysisDashboard']['folder_name']
    dashboard.show_dashboard(folder_name)
    print("To view the query analysis dashboard execute the following command from the terminal: cd code streamlit run streamlit_query_analysis_dashboard.py --server.port 8502 --server.fileWatcherType none")

elif(component=="dataIngestion"):

    filename = expertConfig['Default']['home_dir']+expertConfig['DataIngestion']['filename']
    table_name= expertConfig['DataIngestion']['table_name']
    schema_name= expertConfig['DataIngestion']['schema_name']
    
    dbin_response = dbin.load_data_into_db(filename,table_name,schema_name)
    print(dbin_response)
    
    
elif(component=="contextRetriever"):
    exp_name =expertConfig['Default']['exp_name'] 
    input_database_folder = expertConfig['Default']['home_dir']+expertConfig['ContextRetriever']['input_database_folder']
    input_file = expertConfig['Default']['home_dir']+expertConfig['ContextRetriever']['input_data_file']
    db_type = expertConfig['ContextRetriever']['db_type']
    
    cr_response = cr.funcContextRetriever(exp_name,db_type ,input_file, input_database_folder)
    print(cr_response)

elif(component=="finetune"):
    exp_name=expertConfig['Default']['exp_name']
    data_collator=expertConfig['Finetune']['data_collator']
    model_name=expertConfig['Finetune']['model_name']
    prompt_file_path=expertConfig['Finetune']['prompt_file_path']
    finetune_type=expertConfig['Finetune']['finetune_type']
    train_dataset =expertConfig['Finetune']['train_dataset']
    
    ft_response = ft.funcFinetune(exp_name,data_collator,model_name,prompt_file_path,finetune_type,train_dataset)
    print(ft_response)
    
elif(component=="inference"):
    exp_name=expertConfig['Default']['exp_name']
    model_name=expertConfig['Inference']['model_name']
    finetuned_model = expertConfig['Inference']['finetuned_model']
    input_dataset = expertConfig['Inference']['input_dataset']
    
    inf_response = inf.executeInference(expertConfig)
    print(inf_response)
    
elif(component=="querycorrection"):
    exp_name=expertConfig['Default']['exp_name']
    input_dataset=expertConfig['QueryCorrection']['input_dataset']
    query_correction.funcQueryCorrection(exp_name,input_dataset)
    
elif(component=="evaluation"):
    db_type = expertConfig['EXEvaluator']['db_type']
    exp_name=expertConfig['Default']['exp_name']
    input_dataset=expertConfig['EXEvaluator']['input_dataset']
    input_database_folder = expertConfig['EXEvaluator']['input_database_folder']
    expected_query_column= expertConfig['EXEvaluator']['expected_query_column']
    generated_query_column= expertConfig['EXEvaluator']['generated_query_column']
    ex_evaluator.ex_evalution(db_type,exp_name,input_dataset,input_database_folder,expected_query_column,generated_query_column)

elif(component=="queryanalysisDashboard"):
    import streamlit_query_analysis_dashboard as dashboard
    folder_name = expertConfig['Default']['home_dir']+expertConfig['QueryAnalysisDashboard']['folder_name']
    dashboard.show_dashboard(folder_name)
    print("To view the query analysis dashboard execute the following command from the terminal: cd code streamlit run streamlit_query_analysis_dashboard.py --server.port 8502 --server.fileWatcherType none")
    
