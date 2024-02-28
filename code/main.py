import sys
import configparser
import finetune_new as ft
import DB2_Ingestion as dbin
import context_retriever as cr
import inference_new as inf
import query_correction
import ex_evaluator
import streamlit_query_analysis_dashboard as dashboard



config_filePath="./../superConfig.ini"
config = configparser.ConfigParser()
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
    pass

elif(component=="dataIngestion"):

    filename = config['DataIngestion']['filename']
    table_name= config['DataIngestion']['table_name']
    
    dbin_response = dbin.load_data_into_db(filename,table_name)
    print(dbin_response)
    
    
elif(component=="contextRetriever"):
    
    input_database_folder = config['ContextRetriever']['input_database_folder']
    input_file = config['ContextRetriever']['input_data_file']
    db_type = config['ContextRetriever']['db_type']
    
    cr_response = cr.funcContextRetriever(db_type ,input_file, input_database_folder)
    print(cr_response)

elif(component=="finetune"):
    exp_name=config['Finetune']['exp_name']
    data_collator=config['Finetune']['data_collator']
    model_name=config['Finetune']['model_name']
    prompt_file_path=config['Finetune']['prompt_file_path']
    finetune_type=config['Finetune']['finetune_type']
    train_dataset =config['Finetune']['train_dataset']
    
    ft_response = ft.funcFinetune(exp_name,data_collator,model_name,prompt_file_path,finetune_type,train_dataset)
    print(ft_response)
    
elif(component=="inference"):
    exp_name=config['Inference']['exp_name']
    model_name=config['Inference']['model_name']
    finetuned_model = config['Inference']['finetuned_model']
    input_dataset = config['Inference']['input_dataset']
    
    inf_response = inf.funcInference(exp_name,model_name,finetuned_model,input_dataset)
    print(inf_response)
    
elif(component=="querycorrection"):
    exp_name=config['QueryCorrection']['exp_name']
    input_dataset=config['Default']['home_dir']+config['QueryCorrection']['input_dataset']
    query_correction.funcQueryCorrection(exp_name,input_dataset)
    
elif(component=="evaluation"):
    db_type = config['EXEvaluator']['db_type']
    exp_name=config['EXEvaluator']['exp_name']
    input_dataset=config['Default']['home_dir']+config['EXEvaluator']['input_dataset']
    input_database_folder = config['Default']['home_dir']+config['EXEvaluator']['input_dataset_folder']
    ex_evaluator.ex_evalution(db_type,exp_name,input_dataset,input_database_folder)

elif(component=="queryanalysisDashboard"):
    folder_name = config['Default']['home_dir']+config['QueryAnalysisDashboard']['folder_name']
    dashboard.show_dashboard(folder_name)
    
