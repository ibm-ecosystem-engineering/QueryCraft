import pandas as pd
import streamlit as st
from datetime import datetime
import os
import sys
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
from math import ceil
import configparser
import logging

import numpy as np
import random
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yaml
from yaml.loader import SafeLoader
import time
import ast
import configparser
import logging
from pipeline_result_csv_gen import create_result


######################################################################################################

################################################################################################
st.set_page_config(layout="wide")


################################################################################################
query_keywords_list = ["TOP","EXISTS","INTERSECT","SELECT","DISTINCT","TOP","RANK","AS","WHERE","AND","OR","BETWEEN","LIKE","COUNT","SUM","AVG","MIN","MAX","GROUP BY","ORDER BY","DESC","OFFSET","FETCH","INNER JOIN","LEFT JOIN","RIGHT JOIN","FULL JOIN","UNION","HAVING","JOIN"]

aggregate_keywords = ["COUNT","SUM","AVG","MIN","MAX","TOP"]
rank_keywords = ["RANK"]
fillter_keywords = ["GROUP BY","ORDER BY","FILTER","HAVING","EXISTS"]
join_keywords = ["JOIN","INNER JOIN","LEFT JOIN","RIGHT JOIN","FULL JOIN","UNION","INTERSECT"]


orderby_keywords = ["ORDER BY"]
groupby_keywords = ["GROUP BY"]
where_keywords =["WHERE"]
date_keywords =["NOW","GETDATE","CURRENT_TIMESTAMP","DATEDIFF","DATEADD","YEAR","DAY","MONTH"]


### Methods 

def calculate_classification_new(df):
    ## Create new 3 columns 
    df["count"] = df.index
    df["difficulty"] = df.index
    df["classification_new"] = df.index
    for index, row in df.iterrows():
        sql = row["query"]
        count =0
        classification =""
        for keyword in query_keywords_list:
            if keyword in sql:
                count=count+1
                if keyword in orderby_keywords:
                    classification = "ORDER BY"
                elif keyword in groupby_keywords:
                    classification = "GROUP BY"
                elif keyword in aggregate_keywords:
                    classification = "AGGREGATE/RATIO"
                elif keyword in join_keywords:
                    classification = "JOIN"
                elif keyword in where_keywords:
                    classification = "WHERE"
                elif keyword in date_keywords:
                    classification = "DATE"
                ## join  for category 
                ## pre-trained codellama model what type of query are not correct.

        if count < 6:
            df.at[index,'difficulty'] ="simple"
        elif count > 5 and count < 9:
            df.at[index,'difficulty'] ="moderate"
        else:
            df.at[index,'difficulty'] ="challenging"
        if classification == '':
            classification = 'SELECT'

        df.at[index,'classification_new'] = classification
        df.at[index,'count'] =count
    return df

def calculate_classification(df):
    ## Create new 3 columns 
    df["count"] = df.index
    df["difficulty"] = df.index
    df["classification"] = df.index
    for index, row in df.iterrows():
        sql = row["query"]
        count =0
        classification =""
        for keyword in query_keywords_list:
            if keyword in sql:
                count=count+1
                if keyword in rank_keywords:
                    classification = "RANK"
                elif keyword in fillter_keywords:
                    classification = "FILTER"
                elif keyword in aggregate_keywords:
                    classification = "AGGREGATE"
                elif keyword in join_keywords:
                    classification = "JOIN"
                ## join  for category 
                ## pre-trained codellama model what type of query are not correct.

        if count < 6:
            df.at[index,'difficulty'] ="simple"
        elif count > 5 and count < 9:
            df.at[index,'difficulty'] ="moderate"
        else:
            df.at[index,'difficulty'] ="challenging"
        if classification == '':
            classification = 'SELECT'

        df.at[index,'classification'] = classification
        df.at[index,'count'] =count
    return df

###############################################################################################


#Query Analysis 
########################################################################################################
      
def create_graph(df):
    classification = df['classification'].unique()
    st.subheader('Query classification Analysis', divider='rainbow')
    df_grouped = df.groupby(['classification', 'evalScore'])["evalScore"].count()
    df_class= df_grouped/ df_grouped.groupby('classification').transform("sum")*100
    #st.write(df_grouped)
    
    
    value_true =[]
    value_false =[]
    for clas in classification:
        value = df_class.get(key = clas)
        print("Eval Score ---",value)
        if len(value) == 2:
            print("valueee vvv",value[0])
            value_false.append(value[0])
            value_true.append(value[1])
        else:
            if "True" in value:
                print("valueee",value)
                value_true.append(value[0])
                value_false.append(0.0)
            else:
                value_false.append(value[0]) 
                value_true.append(0.0)

    print(value_true)
    print(value_false)
    fig, ax = plt.subplots()
    
    # Stacked bar chart
    ax1 =ax.bar(classification, value_true, label = "True")
    ax2 =ax.bar(classification, value_false, bottom = value_true, label = "False")
    
    for r1, r2 in zip(ax1, ax2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%.2F" % h1, ha="center", va="bottom", color="white", fontsize=14, fontweight="bold")
        plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%.2F" % h2, ha="center", va="bottom", color="white", fontsize=14, fontweight="bold")

    

    ax.legend()
    ax.set_ylabel('Eval Score %')
    ax.set_xlabel('Query Classiifcation')
    #fig.layout.height = 400
    #fig.layout.width = 1100
    st.pyplot(fig)
    
    
def create_graph_new(df):
    classification = df['classification_new'].unique()
    st.subheader('New Query classification Analysis', divider='rainbow')
    df_grouped = df.groupby(['classification_new', 'evalScore'])["evalScore"].count()
    df_class= df_grouped/ df_grouped.groupby('classification_new').transform("sum")*100
    #st.write(df_grouped)
    
    value_true =[]
    value_false =[]
    for clas in classification:
        value = df_class.get(key = clas)
        print("------------",value)
        
        if len(value) == 2:
            value_false.append(value[0])
            value_true.append(value[1])
        else:
            if "True" in value:
                value_true.append(value[0])
                value_false.append(0.0)
                
            else:
                value_false.append(value[0])
                value_true.append(0.0)

    fig, ax = plt.subplots()
    
    # Stacked bar chart
    ax1 =ax.bar(classification, value_true, label = "True")
    ax2 =ax.bar(classification, value_false, bottom = value_true, label = "False")
    
    for r1, r2 in zip(ax1, ax2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%.2F" % h1, ha="center", va="bottom", color="white", fontsize=14, fontweight="bold")
        plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%.2F" % h2, ha="center", va="bottom", color="white", fontsize=14, fontweight="bold")

    

    ax.legend()
    ax.set_ylabel('Eval Score %')
    ax.set_xlabel('Query Classiifcation')
    st.pyplot(fig)




########################################################################################################

#Query Analysis Dashboard Heatmap

########################################################################################################

def clean_model_name(file):
    # Define the patterns to remove from the file name
    patterns_to_remove = ['exp_', '.csv', 'df_output_finetune_exp_', 'df_','_EX','_EX1','_EXP1','_EXP2','_EXP3','dfoutput']

    # Apply the cleaning patterns
    cleaned_name = file
    for pattern in patterns_to_remove:
        cleaned_name = cleaned_name.replace(pattern, '')

    return cleaned_name

def get_heatmap(folder_name,files):
    y = []
    z = []
    x = []

    for file in files:
            y.append(clean_model_name(file))
            df_eval = pd.read_csv(folder_name + file)
            df_eval = calculate_classification(df_eval)
            df_eval = calculate_classification_new(df_eval)
            classification = df_eval['classification_new'].unique()
            df_grouped = df_eval.groupby(['classification_new', 'evalScore'])["evalScore"].count()
            df_class = df_grouped / df_grouped.groupby('classification_new').transform("sum") * 100
            value_true = []
            value_false = []
            for clas in classification:
                value = df_class.get(key=clas)
                if len(value) == 2:
                    value_false.append(value[0])
                    value_true.append(str(round(value[1], 2)))
                else:
                    if "True" in value:
                        value_true.append(str(round(value[1], 2)))
                    else:
                        value_false.append(value)
            z.append(value_true)
            x = classification

    # Define a custom color scale
    colorscale = [
        [0.0, 'red'],
        [1.0, 'green']
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        texttemplate="%{z}",
        textfont={"size": 14},
        hoverongaps=False,
        colorscale=colorscale
    ))
    fig.update_xaxes(side="top")
    fig.layout.height = 800
    fig.layout.width = 1100
    st.plotly_chart(fig)

def get_heatmap_new(folder_name,files):
    y =[]
    z =[]
    x =[]
    for file in files:
        y.append(clean_model_name(file))
        df_eval = pd.read_csv(folder_name+file)
        df_eval = calculate_classification(df_eval)
        df_eval = calculate_classification_new(df_eval)
        classification = df_eval['classification_new'].unique()
        df_grouped = df_eval.groupby(['classification_new', 'evalScorePostProcessing'])['evalScorePostProcessing'].count()
        df_class= df_grouped/ df_grouped.groupby('classification_new').transform("sum")*100
        value_true =[]
        value_false =[]
        for clas in classification:
            value = df_class.get(key = clas)
            if len(value) == 2:
                value_false.append(value[0])
                value_true.append(str(round(value[1], 2)))
            else:
                if "True" in value:
                    value_true.append(str(round(value[1], 2)))
                else:
                    value_false.append(value)  
            z.append(value_true)
            x = classification
        
    # Define a custom color scale
    colorscale = [
        [0.0, 'red'],
        [1.0, 'green']
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        texttemplate="%{z}",
        textfont={"size": 14},
        hoverongaps=False,
        colorscale=colorscale
    ))
    fig.update_xaxes(side="top")
    fig.layout.height = 800
    fig.layout.width = 1100
    st.plotly_chart(fig)
    
    
def getErrorAndResultAnalysis(folder_name,files):
    select_files =[]
    for file in files:
        select_files.append(file)
                     

    model_names =[]
    for files in select_files:
        model_name = files.split("_")
        if(len(model_name)>2):
            model_names.append(model_name[1])
    
    
    model_names_options = tuple(set(model_names))
    model_names_option1 = st.selectbox('Select a model: ',model_names_options,index=0)
    
    select_files_ep =[]
    for file in select_files:
        if model_names_option1 in file:
            select_files_ep.append(file)
    
    
    eval_data_options = tuple(select_files_ep)
    eval_data_option = st.selectbox('Select a input data source: ',eval_data_options,index=0)
    
    
    df_eval = pd.read_csv(folder_name+eval_data_option)
    
    df_eval = calculate_classification(df_eval)
    df_eval = calculate_classification_new(df_eval)
    
    
    
    st.write("EX Accuracy Post Processing = ", sum(df_eval["evalScorePostProcessing"])/len(df_eval["evalScorePostProcessing"]))
    st.subheader('Query Classification Analysis', divider='rainbow')
    df_grouped = df_eval.groupby(['classification', 'evalScorePostProcessing'])["evalScorePostProcessing"].count()
    df_class= df_grouped/ df_grouped.groupby('classification').transform("sum")*100
    
    col5, col6 = st.columns(2)
    col5.markdown("### Query Classification Analysis Count")
    col5.dataframe(df_grouped)

    col6.markdown("### Query Classification Analysis %")
    col6.dataframe(df_class)
    st.divider()
    
    st.subheader('New Query classification Analysis', divider='rainbow')
    
    df_grouped = df_eval.groupby(['classification_new', 'evalScorePostProcessing'])["evalScorePostProcessing"].count()
    df_class= df_grouped/ df_grouped.groupby('classification_new').transform("sum")*100
    
    col7, col8 = st.columns(2)
    col7.markdown("### New Query Classification Analysis Count")
    col7.dataframe(df_grouped)

    col8.markdown("### New Query Classification Analysis %")
    col8.dataframe(df_class)
    st.divider()
    
    st.subheader("Error Type Analysis", divider='rainbow')   
    col3, col4 = st.columns(2)
    col3.markdown("### Error Type Analysis Count")
    col3.dataframe(df_eval["error_type"].value_counts())

    col4.markdown("### Error Type Analysis %")
    col4.dataframe(df_eval["error_type"].value_counts()/len(df_eval["error_type"])*100)
    
    st.subheader("Result Analysis", divider='rainbow')
    col1, col2 = st.columns(2)
    col1.markdown("### Result Analysis Count")
    col1.dataframe(df_eval["result"].value_counts())

    col2.markdown("### Result Analysis %")
    col2.dataframe(df_eval["result"].value_counts()/len(df_eval["result"])*100)
    
    
    
def getEvaluationAnalysis(folder_name,select_files):
    model_names =[]
    for files in select_files:
        model_name = files.split("_")
        if(len(model_name)>2):
            model_names.append(model_name[1])
    
    
    model_names_options = tuple(set(model_names))
    key=[]
    for i in range(len(model_names_options)):
        key.append("count"+str(i))
    model_names_option = st.selectbox('Select a model: ',model_names_options,index=0,key=key)
    
    select_files_ep =[]
    for file in select_files:
        if model_names_option in file:
            select_files_ep.append(file)
    
    
    eval_data_options = tuple(select_files_ep)
    
    keyOp=[]
    for i in range(len(eval_data_options)):
        keyOp.append("data"+str(i))
    eval_data_option = st.selectbox('Select a input data source: ',eval_data_options,index=0,key=keyOp)
    
    df_eval = pd.read_csv(folder_name+eval_data_option)
    df_eval = df_eval[['query','question','context','model_op','evalScore']]
    st.write("EX Accuracy = ", sum(df_eval["evalScore"])/len(df_eval["evalScore"])*100)
    df_eval = calculate_classification(df_eval)
    df_eval = calculate_classification_new(df_eval)
    create_graph(df_eval)
    st.divider()
    create_graph_new(df_eval)

def get_evaluationscore(folder_name,files):
 
    eval_score_list = []
    for file in files:
        if '.DS_Store' not in file:
            print('file-name',file)
            df_eval = pd.read_csv(folder_name+file,encoding='utf-8')
            ## upadate the dict
            model_name = file.split("_")
            eval_score = sum(df_eval["evalScore"])/len(df_eval["evalScore"])*100
            evalScorePostProcessing = sum(df_eval["evalScorePostProcessing"])/len(df_eval["evalScorePostProcessing"])*100
            result = df_eval["result"].value_counts()
            if 'Partial Match' in result:
                value = (result['Partial Match'])/len(df_eval["evalScore"])*100
            else:
                value =0
            eval_score_list.append({'Model name': str(model_name[1].replace("finetune-","")), 'Accuracy': str(round(eval_score, 2)), 'Post processing accuracy': str(round(evalScorePostProcessing, 2)), 'Partial Match': str(round(value, 2)),'File_name': str(file)})


        eval_score_df = pd.DataFrame(eval_score_list)
        st.dataframe(eval_score_df)

def get_evaluationscoreCheckBox(folder_name,files,finetune_data_file):
 
    eval_score_list = []
    for file in files:
        df_eval = pd.read_csv(folder_name+file)
        ## upadate the dict
        model_name = file.split("_")
        eval_score = sum(df_eval["evalScore"])/len(df_eval["evalScore"])*100
        evalScorePostProcessing = sum(df_eval["evalScorePostProcessing"])/len(df_eval["evalScorePostProcessing"])*100
        result = df_eval["result"].value_counts()
        if 'Partial Match' in result:
            value = (result['Partial Match'])/len(df_eval["evalScore"])*100
        else:
            value =0
        eval_score_list.append({'Select':False,'Model name': str(model_name[1].replace("finetune-","")), 'Accuracy': str(round(eval_score, 2)), 'Post processing accuracy': str(round(evalScorePostProcessing, 2)), 'Partial Match': str(round(value, 2)),'File_name': str(file)})
     
    
    eval_score_df = pd.DataFrame(eval_score_list)
    #st.dataframe(eval_score_df)
    
    edited_df = st.data_editor(
        eval_score_df,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)}
    )
    
    selected_rows = edited_df[edited_df.Select]
    
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=selected_rows['File_name'], y=selected_rows['Accuracy']),
        go.Bar(name='Post processing accuracy', x=selected_rows['File_name'], y=selected_rows['Post processing accuracy']),
        go.Bar(name='Partial Match', x=selected_rows['File_name'], y=selected_rows['Partial Match'])
    ])
    
    # Change the bar mode
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    finetune_data = pd.read_csv(finetune_data_file)
    
    finetune_df =pd.DataFrame()
    values = selected_rows['File_name']
    for file_name in values:
        finetune_df = pd.concat([finetune_df, finetune_data.loc[finetune_data['File-Name'] == file_name]], ignore_index=True)
            
        
    st.write(finetune_df)
    
        
        

def getQueryAnalysisdashboard(folder_name,select_files):
    model_names_option = st.selectbox('Select a Method: ',["Pre-Trained","Finetune"],index=0)
    
    files = []
    if model_names_option == "Pre-Trained":
        for file in select_files:
            if "finetune" not in file: 
                files.append(file)
    else:
        for file in select_files:
            if "finetune" in file: 
                files.append(file)
                
    st.subheader("Evalution score analysis", divider='rainbow') 
    get_evaluationscore(folder_name,files)
    st.subheader("Query classification level analysis", divider='rainbow')
    get_heatmap(folder_name,files)
    st.subheader('Query classification level analysis after post processing',divider='rainbow')
    get_heatmap_new(folder_name,files)

def getComparistionAnalysisdashboard(folder_name,select_files,finetune_data_file):
    get_evaluationscoreCheckBox(folder_name,select_files,finetune_data_file)
    

def show_dashboard(folder_name='output/evalResults/'):
        config = configparser.ConfigParser()
        config.read('./../config.ini')
        config.sections()

        superconfig = configparser.ConfigParser()
        superconfig.read('./../superConfig.ini')
        superconfig.sections()
        home_dir = superconfig['Default']['home_dir']
        ## read superconfig add path 
        input_dataset_file = home_dir+config['QueryAnalysisDashboard']['input_dataset_file']
        df_all = pd.read_csv(input_dataset_file)


        #Evaluation Analysis 
        create_result()
        
        
        token_data_file= home_dir+config['QueryAnalysisDashboard']['token_data_file']
        benchmark_image= home_dir+config['QueryAnalysisDashboard']['benchmark_image']
        finetune_data_file= home_dir+config['QueryAnalysisDashboard']['text2sql_exp_file']
        files = tuple(os.listdir(folder_name))
        selected_files=[]

        for file in files:
            if 'ipynb_checkpoints' not in file:
                if '.DS_Store' not in file:
                    selected_files.append(file)
            
        files = tuple(selected_files)
        #######################################################
        
        st.header('TextToSql Analysis Dashboard',divider='rainbow')
        tab1, tab2, tab3,tab4,tab5,tab6 = st.tabs(["Training Data Analysis", "Evaluation Analysis","Evaluation Analysis Post Processing", "Query Classification Dashboard","Comparistion Analysis","Benchmark Analysis"])
        with tab1:
            st.subheader("Context length distribution of queries", divider='rainbow')
            context_data_option = st.selectbox(
            'Select a input data source: ',
                        ('spider_train','spider_dev','bird_train','bird_dev','cosql_dev','sparc_train','sparc_dev','kaggleDBQA'),index=0)

            df_selected = df_all[df_all["source"].isin(list(context_data_option))]

            df_contextLength_selected = df_all[df_all["source"].isin([context_data_option])]

            df_selected = calculate_classification(df_contextLength_selected)
            df_selected = calculate_classification_new(df_selected)

            data_contextLength_selected = Dataset.from_pandas(df_contextLength_selected)

            tonen_len_df = pd.read_csv(token_data_file)
            len_list = ast.literal_eval(tonen_len_df.loc[tonen_len_df["source"] == context_data_option, "token_len"].iloc[0])
            
            fig, ax = plt.subplots()
            ax.hist(len_list, bins=20)
            st.pyplot(fig)
            st.divider()
            st.subheader("Training Data Analysis", divider='rainbow')
            st.write("Length  : ",len(df_selected))
            col1, col2,col3 = st.columns(3)
            col1.markdown("### Difficulty Analysis Count")
            col1.dataframe(df_selected["difficulty"].value_counts())

            col2.markdown("### Query Classification")
            col2.dataframe(df_selected["classification"].value_counts())

            col3.markdown("### New Query Classification")
            col3.dataframe(df_selected["classification_new"].value_counts())    
    

        with tab2:
            getEvaluationAnalysis(folder_name,files)
    
        with tab3:
            getErrorAndResultAnalysis(folder_name,files)
    

        with tab4:
            getQueryAnalysisdashboard(folder_name,files)

        with tab5:
            st.subheader("Comparistion Analysis in b/w models", divider='rainbow')
            getComparistionAnalysisdashboard(folder_name,files,finetune_data_file)
     
    
        with tab6:
            st.subheader("Leaderboard - Execution with Values (Spider Benchmark)", divider='rainbow')
            st.write("Image Credit","https://yale-lily.github.io//spider")
            st.image(benchmark_image, caption='Spider dataset banchmark')

superconfig = configparser.ConfigParser()
superconfig.read('./../superConfig.ini')
superconfig.sections()
home_dir  = superconfig['Default']['home_dir']
folder_name = home_dir+superconfig['QueryAnalysisDashboard']['folder_name']
show_dashboard(folder_name=folder_name)
########################################################################################################

    
    
