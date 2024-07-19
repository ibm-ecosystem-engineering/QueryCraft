import os
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from profiler import ProfileAnalyzer


######################################################################################################
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
        #print("Eval Score ---",value)
        if len(value) == 2:
            #print("valueee vvv",value[0])
            value_false.append(value[0])
            value_true.append(value[1])
        else:
            if "True" in value:
                #print("valueee",value)
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
    #fig.layout.height = 400
    #fig.layout.width = 1100
    st.pyplot(fig)
     

def ensure_series(data):
    if isinstance(data, tuple):
        return pd.Series(data[0], index=data[1])
    elif isinstance(data, pd.Series):
        return data
    else:
        raise ValueError("Input data must be either a tuple or a Pandas Series")


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
        #print("------------",value)
        
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

# Evaluation Data Profiling

########################################################################################################
def display_analysis(analyzer, df):
    df, all_tables, all_columns, num_columns_distribution, nesting_level_distribution = analyzer.analyze_dataframe(df)

    st.title("SQL Query Analysis")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Query Type Counts", 
        "Unique Table Names", 
        "Unique Columns", 
        "Columns Referenced Distribution", 
        "Nesting Levels", 
        "Schema Awareness Check"
    ])

    with tab1:
        st.header("Query Type Counts")
        st.write(df[["has_join", "has_where", "has_groupby", "has_aggregate"]].mean())

    with tab2:
        st.header("Unique Table Names")
        st.write(all_tables)

    with tab3:
        st.header("Unique Columns Across All Queries")
        st.write(all_columns)

    with tab4:
        st.header("Distribution of Number of Columns Referenced")
        st.bar_chart(num_columns_distribution)

    with tab5:
        st.header("Distribution of Nesting Levels")
        st.bar_chart(nesting_level_distribution)

    with tab6:
        st.header("Schema Awareness Check")
        st.write(df["schema_aware"].value_counts())


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
    model_names_option1 = st.selectbox('Select a model: ',model_names_options,index=0,key=7)
    
    select_files_ep =[]
    for file in select_files:
        if model_names_option1 in file:
            select_files_ep.append(file)
    
    
    eval_data_options = tuple(select_files_ep)
    eval_data_option = st.selectbox('Select a input data source: ',eval_data_options,index=0,key=6)
    
    if eval_data_option is not None:
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
    model_names_option = st.selectbox('Select a model: ',model_names_options,index=0,key=5)
    
    select_files_ep =[]
    for file in select_files:
        if model_names_option in file:
            select_files_ep.append(file)
    
    
    eval_data_options = tuple(select_files_ep)
    
    keyOp=[]
    for i in range(len(eval_data_options)):
        keyOp.append("data"+str(i))
    eval_data_option = st.selectbox('Select a input data source: ',eval_data_options,index=0,key=4)
    if eval_data_option is not None:
        df_eval = pd.read_csv(folder_name+eval_data_option)
        df_eval = df_eval[['query','question','context','model_op','evalScore']]
        st.write("EX Accuracy = ", sum(df_eval["evalScore"])/len(df_eval["evalScore"])*100)
        df_eval = calculate_classification(df_eval)
        create_graph(df_eval)
        df_eval = calculate_classification_new(df_eval)
        create_graph(df_eval)
        st.divider()
        create_graph_new(df_eval)

def get_evaluationscore(folder_name,files):
 
    eval_score_list = []
    for file in files:
        if '.DS_Store' not in file:
            #print('file-name',file)
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
        if 'file-name' in finetune_data.columns:
            finetune_df = pd.concat([finetune_df, finetune_data.loc[finetune_data['file-name'] == file_name]], ignore_index=True)
            
        
    st.write(finetune_df)
    
        
        

def getQueryAnalysisdashboard(folder_name,select_files):
    model_names_option = st.selectbox('Select a Method: ',["Pre-Trained","Finetune"],index=0,key=3)
    
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


def get_overall_accuracy_dict_score(base_path = "../output/inference_sirion/accumulated_accuracy/"):
    dict_accuracy_overall = {}

    model_paths = os.listdir(base_path)

    for model_path in model_paths:
        
        df = pd.read_csv(base_path+model_path)
        model_id = model_path.split(".csv")[0]

        dict_accuracy_overall[model_id] =  df[df["category"]=="TOTAL"]["Execution accuracy %"].values[0]


    return dict_accuracy_overall

def get_overall_latency_dict_score(base_path = "../output/inference_sirion/"):
    dict_latency_overall = {}

    model_paths = os.listdir(base_path)
    try:
        model_paths.remove(".DS_Store")
        print(model_paths)
    except:
        pass
    model_paths.remove('accumulated_accuracy')
    for model_path in model_paths:
        print(model_path)
        
        df = pd.read_csv(base_path+model_path)

        model_id = model_path.split(".csv")[0]

        dict_latency_overall[model_id] =  round(df['latency'].mean(),3)


    return dict_latency_overall


def get_overall_accuracy(dict_accuracy_overall: dict):
        sorted_models = dict(sorted(dict_accuracy_overall.items(), key=lambda item: item[1]))

        sorted_model_names = list(sorted_models.keys())
        sorted_scores = list(sorted_models.values())

        sorted_model_names =  [x.capitalize() for x in sorted_model_names]
        data = {
            'Category': sorted_model_names,
            'Values': sorted_scores
        }
        df = pd.DataFrame(data)
        fig = px.bar(df, x='Values', y='Category', orientation='h')
        fig.update_traces(texttemplate='%{x:.4s}%', textposition='outside', marker_color='orange')
        fig.update_layout(xaxis_title='Inference Accuracy Scores', 
                          yaxis_title='Experimental Models',title="Comparison of Model Scores", xaxis_range=[0, 100])
        fig.update_layout(title={'text': 'Comparison of Accuracy Across Different Models', 'x': 0.5, 'xanchor': 'center'})

        st.plotly_chart(fig)
        

def get_overall_latency(dict_latency_overall: dict):
        

        st.header('Comparison of Latency Across Different Models,',divider='rainbow')


        sorted_models = dict(sorted(dict_latency_overall.items(), key=lambda item: item[1]))

        sorted_model_names = list(sorted_models.keys())
        sorted_scores = list(sorted_models.values())

        sorted_model_names =  [x.capitalize() for x in sorted_model_names]
        data = {
            'Category': sorted_model_names,
            'Values': sorted_scores
        }
        df = pd.DataFrame(data)
        st.dataframe(df.rename(columns={"Category": "Models", "Values": "Average Latency (sec)"}))
        # fig = px.bar(df, x='Values', y='Category', orientation='h')
        # fig.update_traces(texttemplate='%{x:.4s}s', textposition='outside')
        # fig.update_layout(xaxis_title='Inference Latency Scores', 
        #                   yaxis_title='Experimental Models',title="Comparisons of Models Latency", xaxis_range=[0, 100])
        # fig.update_layout(title={'text': 'Comparison of Latency Across Different Models', 'x': 0.5, 'xanchor': 'center'})

        # st.plotly_chart(fig)
        

def get_query_analysis(input_path = "../output/inference_sirion/"):
    AGGREGATE_RATIO = []
    GROUP_BY = []
    ORDER_BY = []
    SELECT =[]
    WHERE = []
    TOTAL_AVERAGE = []

    AGGREGATE_RATIO_COUNT = []
    GROUP_BY_COUNT = []
    ORDER_BY_COUNT = []
    SELECT_COUNT =[]
    WHERE_COUNT = []
    TOTAL_CORRECT_COUNT = []


    MODEL_ID_LIST = []

    model_results_dir = os.listdir(input_path)
    model_results_dir.remove("accumulated_accuracy")
    for model_path in model_results_dir:


        df_selected = pd.read_csv(input_path+model_path)
        grouped = df_selected.groupby('Expected classification_new')

        execution_accuracy = {}

        cat_df = pd.DataFrame(columns=["category","Total evaluation records for category","Total correct responses","Execution accuracy %"])
        for name, group in grouped:
            cat_list=[]
            total_true = group['evalScorePostProcessing'].sum()
            total_records = len(group)
            #print(total_true,total_records,"##")
            accuracy = total_true / total_records
            execution_accuracy[name] = accuracy
            cat_list.append(name)
            cat_list.append(total_records)
            cat_list.append(total_true)
            cat_list.append(accuracy)
            #print(cat_list)
            new_row_df = pd.DataFrame([cat_list], columns=cat_df.columns)
            cat_df = pd.concat([cat_df, new_row_df], ignore_index=True)

        total_aggregate = cat_df[cat_df["category"] ==  "TOTAL"]

        total_evaluation_records = cat_df['Total evaluation records for category'].sum()
        cat_df['Weights'] = cat_df['Total evaluation records for category'] / total_evaluation_records

        tot_list = []
        category = "TOTAL"
        total_eval_records = df_selected.shape[0]
        evalScore_counts = df_selected[df_selected['evalScorePostProcessing'] == True]['evalScorePostProcessing'].value_counts()[0]
        exec_accuracy =  evalScore_counts/total_eval_records*100
        tot_list.append(category)
        tot_list.append(total_eval_records)
        tot_list.append(evalScore_counts)
        tot_list.append(exec_accuracy)
        cat_df['Execution accuracy %'] = round(cat_df['Execution accuracy %']*100,2)
        cat_df['Weighted execution accuracy'] = cat_df['Execution accuracy %'] * cat_df['Weights']

        tot_list.append(np.nan)
        tot_list.append(np.nan)
        new_row_df = pd.DataFrame([tot_list], columns=cat_df.columns)
        cat_df = pd.concat([cat_df, new_row_df], ignore_index=True)
        cat_df['Average Latency'] = np.nan
        cat_df['Average Latency'].iloc[-1] = round(df_selected['latency'].mean(),3)

        model_results_dict = dict(zip(cat_df["category"],cat_df['Execution accuracy %']))

        model_count_dict = dict(zip(cat_df["category"],cat_df['Total correct responses']))


        AGGREGATE_RATIO.append(model_results_dict['AGGREGATE/RATIO'])
        GROUP_BY.append(model_results_dict['GROUP BY'])
        ORDER_BY.append(model_results_dict['ORDER BY'])
        SELECT.append(model_results_dict['SELECT'])
        WHERE.append(model_results_dict['WHERE'])
        TOTAL_AVERAGE.append(model_results_dict['TOTAL'])



        AGGREGATE_RATIO_COUNT.append(model_count_dict['AGGREGATE/RATIO'])
        GROUP_BY_COUNT.append(model_count_dict['GROUP BY'])
        ORDER_BY_COUNT.append(model_count_dict['ORDER BY'])
        SELECT_COUNT.append(model_count_dict['SELECT'])
        WHERE_COUNT.append(model_count_dict['WHERE'])




        MODEL_ID_LIST.append(model_path.split(".csv")[0].replace("_exEvaluator",""))


    df_accuracy_query_type = pd.DataFrame({"Model Id":MODEL_ID_LIST,'SELECT': SELECT,'ORDER BY': ORDER_BY,'WHERE': WHERE, 'GROUP BY': GROUP_BY,"AGGREGATE/RATIO":AGGREGATE_RATIO,'TOTAL AVERAGE ACCURACY': TOTAL_AVERAGE,})

    df_count_query_type = pd.DataFrame({"Model Id":MODEL_ID_LIST,'SELECT': SELECT_COUNT,'ORDER BY': ORDER_BY_COUNT,'WHERE': WHERE_COUNT, 'GROUP BY': GROUP_BY_COUNT,"AGGREGATE/RATIO":AGGREGATE_RATIO_COUNT,})


    return df_accuracy_query_type, df_count_query_type

def style_dataframe(df,background_color ="#9EB1CF"):
    return df.style.set_table_styles(
        [{
            'selector': 'th',
            'props': [
                ('background-color', background_color),
                ('color', 'white'),
                ('font-family', 'Arial, sans-serif'),
                ('font-size', '16px')
            ]
        }, 
        {
            'selector': 'td, th',
            'props': [
                ('border', '2px solid #4CAF50')
            ]
        }]
    )




def show_dashboard(base_path = "../output/inference_sirion/"):
        #######################################################
        
        st.header('Sirion Text To SQL Multiple Code Models Analysis Dashboard',divider='rainbow')
        # tab1, tab2, tab3,tab4,tab5,tab6 = st.tabs(["Training Data Analysis", "Evaluation Analysis","Evaluation Analysis Post Processing", "Query Classification Dashboard","Comparistion Analysis","Benchmark Analysis"])
        tab1, tab2,tab3, tab4,tab5 = st.tabs([ "Model Comparison Analysis", "Evaluation Analysis","Evaluation Analysis Post Processing","Analysis by Query Type","SQL Query Analysis"])

        with tab1:
            dict_accuracy_overall = get_overall_accuracy_dict_score()
            dict_latency_overall = get_overall_latency_dict_score()

            get_overall_accuracy(dict_accuracy_overall)
            get_overall_latency(dict_latency_overall)

        with tab2:
            st.subheader("Actuals Vs Predictions Analysis", divider='rainbow')
            context_data_option = st.selectbox(
            'Select a input data source: ',
                        ('mixtral-8x7b-instruct-v01', 'meta-llama-3-70b', 'kaist-ai_prometheus-8x7b-v2', \
                        'ibm_granite-34b-code', 'ibm_granite-20b-code', 'ibm_granite-13b-instruct-v2',\
                        'codellama_codellama-34b', 'deepseek-ai_deepseek-coder-33b', 'ibm_granite-8b-code'
                          ),index=0,key=2)


            input_dataset_file =  base_path+ context_data_option + "_exEvaluator.csv"       
            
            df_selected = pd.read_csv(input_dataset_file)

            # filtered_df = df_selected[df_selected['expected_sql_valid']==True]
            grouped = df_selected.groupby('Expected classification_new')

            execution_accuracy = {}

            cat_df = pd.DataFrame(columns=["category","Total evaluation records for category","Total correct responses","Execution accuracy %"])
            for name, group in grouped:
                cat_list=[]
                total_true = group['evalScore'].sum()
                total_records = len(group)
                #print(total_true,total_records,"##")
                accuracy = total_true / total_records
                execution_accuracy[name] = accuracy
                cat_list.append(name)
                cat_list.append(total_records)
                cat_list.append(total_true)
                cat_list.append(accuracy)
                #print(cat_list)
                new_row_df = pd.DataFrame([cat_list], columns=cat_df.columns)
                cat_df = pd.concat([cat_df, new_row_df], ignore_index=True)

            total_aggregate = cat_df[cat_df["category"] ==  "TOTAL"]

            total_evaluation_records = cat_df['Total evaluation records for category'].sum()
            cat_df['Weights'] = cat_df['Total evaluation records for category'] / total_evaluation_records

            tot_list = []
            category = "TOTAL"
            total_eval_records = df_selected.shape[0]
            evalScore_counts = df_selected[df_selected['evalScore'] == True]['evalScore'].value_counts()[0]
            exec_accuracy =  evalScore_counts/total_eval_records*100
            tot_list.append(category)
            tot_list.append(total_eval_records)
            tot_list.append(evalScore_counts)
            tot_list.append(exec_accuracy)
            cat_df['Execution accuracy %'] = round(cat_df['Execution accuracy %']*100,2)
            cat_df['Weighted execution accuracy'] = cat_df['Execution accuracy %'] * cat_df['Weights']

            tot_list.append(np.nan)
            tot_list.append(np.nan)
            new_row_df = pd.DataFrame([tot_list], columns=cat_df.columns)
            cat_df = pd.concat([cat_df, new_row_df], ignore_index=True)
            cat_df['Average Latency'] = np.nan
            cat_df['Average Latency'].iloc[-1] = round(df_selected['latency'].mean(),3)


            cat_df = cat_df.style.highlight_null(props="color: transparent;")  


            st.subheader("Predictions Analysis", divider='rainbow')
            st.write("Length  : ",len(df_selected))
            col1, col2,col3 = st.columns(3)



            col1.markdown("##### Prediction Difficulty Analysis Count")
            predicted_difficulty_value_counts = ensure_series(df_selected["Predicted difficulty"].value_counts())
            col1.dataframe(predicted_difficulty_value_counts,width=300)

            predicted_classification_value_counts = ensure_series(df_selected["Predicted classification"].value_counts())
            col2.markdown("##### Prediction Query Classification")
            col2.dataframe(predicted_classification_value_counts, width=300)


            predicted_classification_new_value_counts = ensure_series(df_selected["Predicted classification_new"].value_counts())
            col3.markdown("##### New Query Prediction Classification")
            col3.dataframe(predicted_classification_new_value_counts,width=300) 


            col1, col2,col3 = st.columns(3)
            fig = px.pie(predicted_difficulty_value_counts, values=predicted_difficulty_value_counts.values, names=predicted_difficulty_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col1.plotly_chart(fig)

            fig = px.pie(predicted_classification_value_counts, values=predicted_classification_value_counts.values, names=predicted_classification_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col2.plotly_chart(fig)

            fig = px.pie(predicted_classification_new_value_counts, values=predicted_classification_new_value_counts.values, names=predicted_classification_new_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col3.plotly_chart(fig)


            st.subheader("Ground Truths", divider='rainbow')
            st.write("Length  : ",len(df_selected))


            col1, col2,col3 = st.columns(3)
            col1.markdown("##### Actuals Difficulty Analysis Count")

            expected_difficulty_value_counts = ensure_series(df_selected["Expected difficulty"].value_counts())

            col1.dataframe(expected_difficulty_value_counts,width=300)

            expected_classification_value_counts = ensure_series(df_selected["Expected classification"].value_counts())
            col2.markdown("##### Query Actuals Classification")
            col2.dataframe(expected_classification_value_counts,width=300)

            expected_classification_new_value_counts = ensure_series(df_selected["Expected classification_new"].value_counts())
            col3.markdown("##### New Query Actuals Classification")
            col3.dataframe(expected_classification_new_value_counts,width=300)    


            col1, col2,col3 = st.columns(3)
            fig = px.pie(expected_difficulty_value_counts, values=expected_difficulty_value_counts.values, names=expected_difficulty_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col1.plotly_chart(fig,use_container_width=False)

            fig = px.pie(expected_classification_value_counts, values=expected_classification_value_counts.values, names=expected_classification_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col2.plotly_chart(fig,use_container_width=False)

            fig = px.pie(expected_classification_new_value_counts, values=expected_classification_new_value_counts.values, names=expected_classification_new_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col3.plotly_chart(fig,use_container_width=False)


            st.subheader(" ")


            # col4 = st.columns(1)
            st.subheader("Execution Accuracy by Query Type",divider='rainbow')

            st.dataframe(cat_df,width=2000) 

        with tab3:
            st.subheader("Actuals Vs Predictions Analysis with Post Processing", divider='rainbow')
            context_data_option = st.selectbox(
            'Select a input data source: ',
                        ('mixtral-8x7b-instruct-v01', 'meta-llama-3-70b', 'kaist-ai_prometheus-8x7b-v2', \
                        'ibm_granite-34b-code', 'ibm_granite-20b-code', 'ibm_granite-13b-instruct-v2',\
                        'codellama_codellama-34b', 'deepseek-ai_deepseek-coder-33b', 'ibm_granite-8b-code'
                          ),index=0,key=1)


            input_dataset_file =  base_path+ context_data_option + "_exEvaluator.csv"       
            
            df_selected = pd.read_csv(input_dataset_file)

            # filtered_df = df_selected[df_selected['expected_sql_valid']==True]
            grouped = df_selected.groupby('Expected classification_new')

            execution_accuracy = {}

            cat_df = pd.DataFrame(columns=["category","Total evaluation records for category","Total correct responses","Execution accuracy %"])
            for name, group in grouped:
                cat_list=[]
                total_true = group['evalScorePostProcessing'].sum()
                total_records = len(group)
                #print(total_true,total_records,"##")
                accuracy = total_true / total_records
                execution_accuracy[name] = accuracy
                cat_list.append(name)
                cat_list.append(total_records)
                cat_list.append(total_true)
                cat_list.append(accuracy)
                #print(cat_list)
                new_row_df = pd.DataFrame([cat_list], columns=cat_df.columns)
                cat_df = pd.concat([cat_df, new_row_df], ignore_index=True)

            total_aggregate = cat_df[cat_df["category"] ==  "TOTAL"]

            total_evaluation_records = cat_df['Total evaluation records for category'].sum()
            cat_df['Weights'] = cat_df['Total evaluation records for category'] / total_evaluation_records

            tot_list = []
            category = "TOTAL"
            total_eval_records = df_selected.shape[0]
            evalScore_counts = df_selected[df_selected['evalScorePostProcessing'] == True]['evalScorePostProcessing'].value_counts()[0]
            exec_accuracy =  evalScore_counts/total_eval_records*100
            tot_list.append(category)
            tot_list.append(total_eval_records)
            tot_list.append(evalScore_counts)
            tot_list.append(exec_accuracy)
            cat_df['Execution accuracy %'] = round(cat_df['Execution accuracy %']*100,2)
            cat_df['Weighted execution accuracy'] = cat_df['Execution accuracy %'] * cat_df['Weights']

            tot_list.append(np.nan)
            tot_list.append(np.nan)
            new_row_df = pd.DataFrame([tot_list], columns=cat_df.columns)
            cat_df = pd.concat([cat_df, new_row_df], ignore_index=True)
            cat_df['Average Latency'] = np.nan
            cat_df['Average Latency'].iloc[-1] = round(df_selected['latency'].mean(),3)



            cat_df = cat_df.style.highlight_null(props="color: transparent;")  


            st.subheader("Predictions Analysis", divider='rainbow')
            st.write("Length  : ",len(df_selected))
            col1, col2,col3 = st.columns(3)



            col1.markdown("##### Prediction Difficulty Analysis Count")
            predicted_difficulty_value_counts = ensure_series(df_selected["Predicted difficulty"].value_counts())
            col1.dataframe(predicted_difficulty_value_counts,width=300)

            predicted_classification_value_counts = ensure_series(df_selected["Predicted classification"].value_counts())
            col2.markdown("##### Prediction Query Classification")
            col2.dataframe(predicted_classification_value_counts, width=300)


            predicted_classification_new_value_counts = ensure_series(df_selected["Predicted classification_new"].value_counts())
            col3.markdown("##### New Query Prediction Classification")
            col3.dataframe(predicted_classification_new_value_counts,width=300) 


            col1, col2,col3 = st.columns(3)
            fig = px.pie(predicted_difficulty_value_counts, values=predicted_difficulty_value_counts.values, names=predicted_difficulty_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col1.plotly_chart(fig)

            fig = px.pie(predicted_classification_value_counts, values=predicted_classification_value_counts.values, names=predicted_classification_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col2.plotly_chart(fig)

            fig = px.pie(predicted_classification_new_value_counts, values=predicted_classification_new_value_counts.values, names=predicted_classification_new_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col3.plotly_chart(fig)


            st.subheader("Ground Truths", divider='rainbow')
            st.write("Length  : ",len(df_selected))


            col1, col2,col3 = st.columns(3)
            col1.markdown("##### Actuals Difficulty Analysis Count")

            expected_difficulty_value_counts = ensure_series(df_selected["Expected difficulty"].value_counts())

            col1.dataframe(expected_difficulty_value_counts,width=300)

            expected_classification_value_counts = ensure_series(df_selected["Expected classification"].value_counts())
            col2.markdown("##### Query Actuals Classification")
            col2.dataframe(expected_classification_value_counts,width=300)

            expected_classification_new_value_counts = ensure_series(df_selected["Expected classification_new"].value_counts())
            col3.markdown("##### New Query Actuals Classification")
            col3.dataframe(expected_classification_new_value_counts,width=300)    


            col1, col2,col3 = st.columns(3)
            fig = px.pie(expected_difficulty_value_counts, values=expected_difficulty_value_counts.values, names=expected_difficulty_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col1.plotly_chart(fig,use_container_width=False)

            fig = px.pie(expected_classification_value_counts, values=expected_classification_value_counts.values, names=expected_classification_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col2.plotly_chart(fig,use_container_width=False)

            fig = px.pie(expected_classification_new_value_counts, values=expected_classification_new_value_counts.values, names=expected_classification_new_value_counts.index)
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50)) 
            col3.plotly_chart(fig,use_container_width=False)


            st.subheader(" ")


            # col4 = st.columns(1)
            st.subheader("Execution Accuracy by Query Type",divider='rainbow')

            st.dataframe(cat_df,width=2000) 

        with tab4:
            df_accuracy_query_type, df_count_query_type = get_query_analysis()

            st.subheader("Query Type vs Model Accuracy", divider='rainbow')
            df_accuracy_query_type = style_dataframe(df_accuracy_query_type,"#088DA5")
            st.write(df_accuracy_query_type.to_html(), unsafe_allow_html=True)

            # st.dataframe(df_accuracy_query_type)
            

            st.subheader("Query Type vs Model correct responses", divider='rainbow')
            df_count_query_type = style_dataframe(df_count_query_type)
            st.write(df_count_query_type.to_html(), unsafe_allow_html=True)

            # st.dataframe(df_count_query_type)

        with tab5:
            schema = {
                        "Contract": [
                            "Expiration Date", "Supplier", "ID", "TCV", "Term Type",
                            "Reporting Currency", "Status", "Title", "Document Type",
                            "Effective Date", "Functions", "Services", "Regions", "Countries",
                            "Time Zone", "Currencies", "Agreement Type", "Name", "Source Name/Title"
                        ],
                        "Contract Draft Request": [
                            "ID", "Title", "Suppliers", "ESignature Status", "Source Name/Title",
                            "Total Deviations", "Effective Date", "TCV", "Paper Type", "Status",
                            "Regions", "Countries", "Functions", "Services", "Templates",
                            "Counterparty Type", "Agreement Type", "Expiration Date", "Multilingual",
                            "No Touch Contract"
                        ],
                        "CO/CDR": [
                            "Created On", "Created By", "Counterparty", "Reporting Date"
                        ]
                    }
            analyzer = ProfileAnalyzer(schema)
            df_evaluation_dataset = pd.read_csv(r"../input/eval_query.csv")
            df_analyzer, all_tables_analyzer, all_columns_analyzer, num_columns_distribution_analyzer, nesting_level_distribution_analyzer = analyzer.analyze_dataframe(df_evaluation_dataset)

            st.title("SQL Query Analysis Overview")

            st.subheader("Query Type Counts")
            st.write(df_analyzer[["has_join", "has_where", "has_groupby", "has_aggregate"]].mean())

            st.subheader("Unique Table Names")
            st.write(all_tables_analyzer)

            st.subheader("Unique Columns Across All Queries")
            st.write(all_columns_analyzer)

            st.subheader("Schema Awareness Check")
            st.write(df_analyzer["schema_aware"].value_counts())
            
            # Correctly setting x-axis labels for distributions
            st.subheader("Distribution of Number of Columns Referenced")
            num_columns_distribution_df = pd.DataFrame({
                "Count": num_columns_distribution_analyzer.values
            }, index=num_columns_distribution_analyzer.index)
            st.bar_chart(num_columns_distribution_df,color="#ffffbc")

            st.subheader("Distribution of Nesting Levels")
            nesting_level_distribution_df = pd.DataFrame({
                "Count": nesting_level_distribution_analyzer.values
            }, index=nesting_level_distribution_analyzer.index)
            st.bar_chart(nesting_level_distribution_df,color="#00ffbc")
            
            