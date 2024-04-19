import argparse
import json
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import tempfile
import glob
import matplotlib
import configparser
import ast


matplotlib.use('Agg')

def create_result():
    ###
    expertConfig = configparser.ConfigParser()
    expertConfig.read('./../expertConfig.ini')
    expertConfig.sections()
    
    
    simpleConfig = configparser.ConfigParser()
    simpleConfig.read('./../simpleConfig.ini')
    simpleConfig.sections()

    ### EXP FILE
    EXP = simpleConfig['Default']['exp_name'] 
    home_dir = simpleConfig['Default']['home_dir']


    # Start MLflow run with a dynamic run name
    text2sql_exp_file = home_dir+ expertConfig['QueryAnalysisDashboard']['text2sql_exp_file']

    # Check if SQL_Leaderboard.csv file exists in the artifacts directory
    if os.path.isfile(text2sql_exp_file):
        # Load the existing leaderboard from the CSV file
        SQL_leaderboard = pd.read_csv(text2sql_exp_file)
    else:
        # Create an empty DataFrame for the leaderboard if it doesn't exist
        SQL_leaderboard = pd.DataFrame(columns=['Base Model','Evaluation set','Ex-accuracy','PP-Ex-accuracy','R', "precision", "Training Set",'LORA Alpha', 'LORA Dropout',
                                                    'Finetune Strategy', 'Target Modules', 'Task Type',
                                                    'Epoch', 'Learning Rate', 'Loss', 'Eval Loss',
                                                    'Eval Runtime', 'Eval Samples/Second',
                                                    'Eval Steps/Second', 'Logging Steps', 'Max Steps'])


    #step-1
    #User expertConfig
    trainDataset = simpleConfig['Finetune']['train_dataset']  
    base_model = simpleConfig['Finetune']['model_name'] 
    finetuningMethod = simpleConfig['Finetune']['finetune_type'] 
    precision = expertConfig['Finetune']['precision'] 
    LoRA_r = int(expertConfig['Finetune']['LoRA_r'])
    LoRA_dropout = float(expertConfig['Finetune']['LoRA_dropout'])
    batch_size = int(expertConfig['Finetune']['batch_size'])
    per_device_train_batch_size = int(expertConfig['Finetune']['per_device_train_batch_size'])
    output_dir =  home_dir+"output/model/"+ simpleConfig['Default']['exp_name']
    target_modules = expertConfig['Finetune']['target_modules']
    logging_path = home_dir+expertConfig['logs']['log_folder']+ simpleConfig['Default']['exp_name']

    #step-2
    # Set the path to the directory containing checkpoint folders
    checkpoint_dir = home_dir + "/output/model/" + simpleConfig['Default']['exp_name']

    # Function to extract the checkpoint number from a folder name
    def get_checkpoint_number(folder_name):
        try:
            return int(folder_name.split('-')[1])
        except ValueError:
            return -1
    
    # Get a list of checkpoint folders
    try:
        checkpoint_folders = [folder for folder in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, folder))]

        # Find the folder with the maximum checkpoint number
        max_checkpoint_folder = max(checkpoint_folders, key=get_checkpoint_number)

        # Form the full path to the max checkpoint folder
        max_checkpoint_folder_path = os.path.join(checkpoint_dir, max_checkpoint_folder)

        #Read adapter_config.json
        adapter_config_file_path = os.path.join(max_checkpoint_folder_path, "adapter_config.json")
        #print(checkpoint_dir)
        #adapter_config_file_path = checkpoint_dir+ "/adapter_config.json"
        with open(adapter_config_file_path, 'r') as adapter_config_file:
            adapter_config_data = json.load(adapter_config_file)

            base_model = adapter_config_data.get('base_model_name_or_path', '')
            lora_alpha = adapter_config_data.get('lora_alpha', '')
            lora_dropout = adapter_config_data.get('lora_dropout', '')
            peft_type = adapter_config_data.get('peft_type', '')
            r = adapter_config_data.get('r', '')
            target_modules = adapter_config_data.get('target_modules', [])
            task_type = adapter_config_data.get('task_type', '')

        # Read trainer_state.json
        trainer_state_file_path = os.path.join(max_checkpoint_folder_path, "trainer_state.json")
        with open(trainer_state_file_path, 'r') as trainer_state_file:
            trainer_state_data = json.load(trainer_state_file)

            max_steps = trainer_state_data.get('max_steps', 0)
            log_history = trainer_state_data.get('log_history', [])

            if log_history:
                last_log = log_history[-1]
                epoch = last_log.get('epoch', '')
                learning_rate = log_history[-2].get('learning_rate', '')
                loss = log_history[-2].get('loss', '')
                eval_loss = last_log.get('eval_loss', '')
                eval_runtime = last_log.get('eval_runtime', '')
                eval_samples_per_second = last_log.get('eval_samples_per_second', '')
                eval_steps_per_second = last_log.get('eval_steps_per_second', '')

            logging_steps = trainer_state_data.get('logging_steps', '')
    except:
        print("Fine tune model folder not found /or it's not fine-tune model")


    #step-3
    try:
        log_file_path = home_dir + expertConfig['logs']['log_folder'] + simpleConfig['Default']['exp_name'] + ".log"

        # Define a dictionary to store the parameters and their values
        parameters = {}

        # Open and read the log file
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                # Remove "INFO:root:" from the beginning of each line
                line = line.replace("INFO:root:", "").strip()
                if line:
                    # Split the line into parameter name and parameter value
                    parts = line.split(":")
                    if len(parts) == 2:
                        parameter_name = parts[0].strip()
                        parameter_value = parts[1].strip()
                        parameters[parameter_name] = parameter_value

        # Now, the parameters and their values are stored in the 'parameters' dictionary
        # for parameter, value in parameters.items():
        #     print(f"{parameter}: {value}")
    except:
        print("log file path is not available")

    #Step-4 Ex_accuracy

    try:
        log_file_path = home_dir + expertConfig['logs']['log_folder']  + simpleConfig['Default']['exp_name'] + "_EX.log"

        Ex_accuracy = None
        PP_Ex_accuracy = None
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                line = line.replace("INFO:root:", "").strip()
                if line.startswith("EX Accuracy :"):
                    parts = line.split(":")
                    if len(parts) == 2:
                        ex_parameter_name = parts[0].strip()
                        ex_parameter_value = parts[1].strip()
                        Ex_accuracy = round(float(ex_parameter_value) * 100, 2)
                if line.startswith("PP EX Accuracy :"):
                    parts = line.split(":")
                    if len(parts) == 2:
                        pp_ex_parameter_name = parts[0].strip()
                        pp_ex_parameter_value = parts[1].strip()
                        PP_Ex_accuracy = round(float(pp_ex_parameter_value) * 100, 2)

        print(f"EX Accuracy: {Ex_accuracy}")

        target_modules = str(target_modules)

        # Define the parameters and their corresponding values
        data = {
            'Base Model': os.path.basename(base_model),
            'Evaluation set':'Spider dev',
            'Ex-accuracy': Ex_accuracy,
            'PP-Ex-accuracy': PP_Ex_accuracy,
            'R': r,
            'precision': precision,
            'Training Set': os.path.basename(trainDataset),
            'LORA Alpha': lora_alpha,
            'LORA Dropout': lora_dropout,
            'Finetune Strategy': peft_type,
            'Target Modules': target_modules,
            'Task Type': task_type,
            'Epoch': epoch,
            'Learning Rate': learning_rate,
            'Loss': loss,
            'Eval Loss': eval_loss,
            'Eval Runtime': eval_runtime,
            'Eval Samples/Second': eval_samples_per_second,
            'Eval Steps/Second': eval_steps_per_second,
            'Logging Steps': logging_steps,
            'Max Steps': max_steps,
        }


        # print(data)
        # Create a DataFrame from the data
        new_leaderboard = pd.DataFrame(data,index=[0])
        new_leaderboard = new_leaderboard.sort_values(by='Base Model')
        print(new_leaderboard)

        # Get the selected columns from the expertConfig file
        selected_columns = expertConfig['QueryAnalysisDashboard']['selected_columns'].split(', ')
        # Replace underscores with spaces in column names
        selected_columns = [column.replace('_', ' ') for column in selected_columns]
        # Filter the new_leaderboard DataFrame based on the selected columns
        new_leaderboard = new_leaderboard[selected_columns]
        print(new_leaderboard)

        # Define the columns for comparison
        # columns_to_compare = ['Base Model', 'Training Set', 'LORA Alpha', 'LORA Dropout', 'Finetune Strategy', 'R']

        # Check if new_leaderboard already exists in SQL_leaderboard based on the specified columns
        # matching_rows = SQL_leaderboard[SQL_leaderboard[columns_to_compare].eq(new_leaderboard[columns_to_compare].iloc[0]).all(axis=1)]

        #if matching_rows.empty:
        SQL_leaderboard = pd.concat([SQL_leaderboard, new_leaderboard], ignore_index=True)
        SQL_leaderboard = SQL_leaderboard.sort_values(by=SQL_leaderboard.columns.tolist())
        # SQL_leaderboard = SQL_leaderboard.drop_duplicates(subset=columns_to_compare, keep="last")
        #else:
        #    print("new_leaderboard already exists in SQL_leaderboard and will not be concatenated.")

        print(SQL_leaderboard)

        # Save SQL leaderboard to the CSV file
        SQL_leaderboard.to_csv(text2sql_exp_file, index=False)
        print("Save ---",text2sql_exp_file)
    except Exception as e:
        SQL_leaderboard.to_csv(text2sql_exp_file, index=False)
        print("No new experiment available",text2sql_exp_file)
