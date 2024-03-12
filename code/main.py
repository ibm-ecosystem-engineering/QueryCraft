import sys
import configparser
import finetune as ft
import DB2_Ingestion as dbin
import context_retriever as cr
import inference as inf
import query_correction
import ex_evaluator
import streamlit_query_analysis_dashboard as dashboard

from configparser import ConfigParser, ExtendedInterpolation

config_filePath="./../superConfig.ini"
config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_filePath)
config.sections()

# Check if arguments are provided
if len(sys.argv) < 1:
    print("Usage: python script_name.py <component>")
    sys.exit(1)

# Access arguments
component = sys.argv[1]  # Exclude the script name, which is the first argument

# Process arguments
print("Compnent to run:", component)

if(component=="all"):
    exp_name =config['Default']['exp_name'] 

    input_database_folder = config['Default']['home_dir']+config['ContextRetriever']['input_database_folder']
    input_file = config['Default']['home_dir']+config['ContextRetriever']['input_data_file']
    db_type = config['ContextRetriever']['db_type']
    cr_response = cr.funcContextRetriever(exp_name,db_type ,input_file, input_database_folder)
    print(cr_response)

    data_collator=config['Finetune']['data_collator']
    model_name=config['Finetune']['model_name']
    prompt_file_path=config['Finetune']['prompt_file_path']
    finetune_type=config['Finetune']['finetune_type']
    train_dataset =config['Finetune']['train_dataset']
    ft_response = ft.funcFinetune(exp_name,data_collator,model_name,prompt_file_path,finetune_type,train_dataset)
    print(ft_response)

    model_name=config['Inference']['model_name']
    finetuned_model = config['Inference']['finetuned_model']
    input_dataset = config['Inference']['input_dataset']
    inf_response = inf.funcInference(exp_name,model_name,finetuned_model,input_dataset)
    print(inf_response)

    input_dataset=config['QueryCorrection']['input_dataset']
    query_correction.funcQueryCorrection(exp_name,input_dataset)

    db_type = config['EXEvaluator']['db_type']
    input_dataset=config['EXEvaluator']['input_dataset']
    input_database_folder = config['EXEvaluator']['input_database_folder']
    ex_evaluator.ex_evalution(db_type,exp_name,input_dataset,input_database_folder)
    
    folder_name = config['Default']['home_dir']+config['QueryAnalysisDashboard']['folder_name']
    dashboard.show_dashboard(folder_name)
    print("To view the query analysis dashboard execute the following command from the terminal: streamlit run streamlit_query_analysis_dashboard.py --server.port 8052 --server.fileWatcherType none")

elif(component=="dataIngestion"):

    filename = config['Default']['home_dir']+config['DataIngestion']['filename']
    table_name= config['DataIngestion']['table_name']
    
    dbin_response = dbin.load_data_into_db(filename,table_name)
    print(dbin_response)
    
    
elif(component=="contextRetriever"):
    exp_name =config['Default']['exp_name'] 
    input_database_folder = config['Default']['home_dir']+config['ContextRetriever']['input_database_folder']
    input_file = config['Default']['home_dir']+config['ContextRetriever']['input_data_file']
    db_type = config['ContextRetriever']['db_type']
    
    cr_response = cr.funcContextRetriever(exp_name,db_type ,input_file, input_database_folder)
    print(cr_response)

elif(component=="finetune"):
    exp_name=config['Default']['exp_name']
    data_collator=config['Finetune']['data_collator']
    model_name=config['Finetune']['model_name']
    prompt_file_path=config['Finetune']['prompt_file_path']
    finetune_type=config['Finetune']['finetune_type']
    train_dataset =config['Finetune']['train_dataset']
    
    ft_response = ft.funcFinetune(exp_name,data_collator,model_name,prompt_file_path,finetune_type,train_dataset)
    print(ft_response)
    
elif(component=="inference"):
    exp_name=config['Default']['exp_name']
    model_name=config['Inference']['model_name']
    finetuned_model = config['Inference']['finetuned_model']
    input_dataset = config['Inference']['input_dataset']
    
    inf_response = inf.funcInference(exp_name,model_name,finetuned_model,input_dataset)
    print(inf_response)
    
elif(component=="querycorrection"):
    exp_name=config['Default']['exp_name']
    input_dataset=config['QueryCorrection']['input_dataset']
    query_correction.funcQueryCorrection(exp_name,input_dataset)
    
elif(component=="evaluation"):
    db_type = config['EXEvaluator']['db_type']
    exp_name=config['Default']['exp_name']
    input_dataset=config['EXEvaluator']['input_dataset']
    input_database_folder = config['EXEvaluator']['input_database_folder']
    ex_evaluator.ex_evalution(db_type,exp_name,input_dataset,input_database_folder)

elif(component=="queryanalysisDashboard"):
    import streamlit_query_analysis_dashboard as dashboard
    folder_name = config['Default']['home_dir']+config['QueryAnalysisDashboard']['folder_name']
    dashboard.show_dashboard(folder_name)
    print("To view the query analysis dashboard execute the following command from the terminal: streamlit run code/streamlit_query_analysis_dashboard.py --server.port 8052 --server.fileWatcherType none")
    
