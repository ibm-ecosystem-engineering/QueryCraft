# **Table of Contents**
[**Create a test environment**](#_toc1711770488)

[***Install the SuperKnowa BYOM JupyterLab extension***](#_toc2114928824)

[***Provision a GPU server***](#_toc410998690)

[***Access the test environment***](#_toc2049952232)

[***Setting up the environment for testing***](#_toc76165070)

[**Configuration Settings**](#_toc1483318699)

[**Step 0. Golden Query Annotation**](#_toc1843293825)

[***Golden Query Annotation:***](#_toc548496318)

[**Step 1. Data Ingestion**](#_toc1471206741)

[***Option 1.1. Delimited file Ingestion***](#_toc212047022)

[***Option 1.2. SQLite Ingestion***](#_toc1138956454)

[**Step 2. Context Retriever**](#_toc1725873645)

[***Option 2.1 Context Retrieval from db2***](#_toc2010877930)

[***Option 2.2 Context Retrieval from SQLite***](#_toc481704433)

[**Step 3. Finetuning**](#_toc347080945)

[**Step 4. Inference**](#_toc1894977564)

[**Step 5. Query Correction**](#_toc890332375)

[***Step 6. Evaluation**](#_toc577012548)

[***Option 6.1 Evaluation on the db2 database***](#_toc1369103650)

[***Option 6.2 Evaluation on sqlite***](#_toc858104986)

[**Step 7. Query Analysis Dashboard**](#_toc766523960)

[**Step 8.  Run pipeline (All)**](#_toc523711792)


---------
# <a name="_toc1711770488"></a>Create a test environment

A GPU server can be easily deployed on IBM Cloud using SuperKnowa BYOM, which provides a JupyterLab extension to provision a GPU-enabled [virtual server instance (VSI) for VPC](https://cloud.ibm.com/docs/vpc?topic=vpc-about-advanced-virtual-servers) in minutes by filling out a simple form in any of your JupyterLab environments.

![Screenshot of Text2Infra](001.png)

## <a name="_toc2114928824"></a>Install the SuperKnowa BYOM JupyterLab extension

1. Prerequisites

   Before you start, make sure you have:

- An [IBM Cloud API Key](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui#create_user_key)
- An [SSH Key for IBM Cloud VPC](https://cloud.ibm.com/docs/vpc?topic=vpc-managing-ssh-keys&interface=ui)
- A GitHub Enterprise (github.ibm.com) [Personal Access Token](https://docs.github.com/en/enterprise-server@3.8/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

### Install the JupyterLab extension

   It’s a good practice to use a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or a [virtual environment](https://docs.python.org/3/library/venv.html) in Python to avoid potential conflict with existing applications and packages installed on your machine.

1. Create and activate a new conda environment

   ```conda create --name superknowa-ext python=3.10

   conda activate superknowa-ext
   ```

2. Create and activate a virtual environment

   ```python3 -m venv .venv

   source .venv/bin/activate
   ```

3. Install JupyterLab if you haven’t done so already:

   `pip3 install jupyterlab`

4. Install the IBM Cloud Schematics Python SDK, which is the Python client library to interact with the [IBM Cloud Schematics API](https://cloud.ibm.com/apidocs/schematics/schematics), with the following command:

   `pip3 install ibm-schematics`

5. Download the packaged JupyterLab extension from the [GitHub repository](https://github.ibm.com/hcbt/SuperKnowa-BYOM/blob/package/superknowa_ext_byom-0.1.0.tar.gz), and install it with the following command, assuming the file is saved to the current directory:

   `pip3 install superknowa_ext_byom-0.1.0.tar.gz`

6. Verify the installation

   Check to make sure the JupyterLab extension is properly installed and enabled:

   #### Frontend component

   `jupyter labextension list`

   #### Backend component

   `jupyter server extension list`

   If they are not automatically enabled, you could try to enable them manually:

   #### Frontend component

   `jupyter labextension enable superknowa_ext_byom`

   ### Backend component

   `jupyter server extension enable superknowa_ext_byom`

7. Now you can start JupyterLab on a port available on your machine (default: 8888):

   `jupyter lab --port=9191`



Once it’s started, a new button should appear in the JupyterLab launcher.

![JupyterLab Launcher with SuperKnowa BYOM icon](002.png)

8. Click the SuperKnowa BYOM button to launch the newly installed JupyterLab extension. If successful, a new tab should open in JupyterLab as shown in the following screenshot.

![Env details in the Jupyter Lab extension](003.png)

## <a name="_toc410998690"></a>Provision a GPU server

1. Simple deployment

   In the SuperKnowa BYOM JupyterLab extension frontend UI, you only need to provide 4 mandatory pieces of information and click the ***Provision*** button to start the provisioning process:

    - A unique name for the workspace
    - Your [IBM Cloud API Key](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui#create_user_key)
    - The SSH Key name for VPC in the IBM Cloud region (Default: Dallas/us-south)
    - A GitHub Enterprise (github.ibm.com) [Personal Access Token](https://docs.github.com/en/enterprise-server@3.8/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

The tool leverages IBM Cloud Schematics API to create a workspace, which pulls a Terraform template from a GitHub repository and set some variables, before provisioning a GPU-enabled VSI for VPC and all the other necessary resources associated with it.

2. Customize the deployment (Optional)

   If you would like to customize your deployment, you can tick the checkbox of ***Customize default settings*** in the JupyterLab extension UI, so that you can override some of the default settings:

    - IBM Cloud region and zone
    - OS image
    - GPU-enabled VPC instance profile
    - Resource group
    - A [Jupyter Docker Stacks image](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)

![Customized deployment](004.png)

Note:

- Not all GPU instance profiles are available in every region/zone.
- We’ve only tested a few Jupyter Docker Stacks images. Some work better than others. Feel free try any of them if you wish. In fact, you could specify any arbitrary container image in the UI.
- You could further customize your deployment in IBM Cloud console where you are able to override the defaults of all the variables available in the Terraform template, including those not exposed in the UI.

3. Monitor the provisioning

   Typically, the provisioning process takes about 15 minutes to deploy the GPU server and run through some initial configuration steps with [Cloud-init](https://cloudinit.readthedocs.io/en/latest/), before it reboots itself in the end. You can monitor the progress by checking the workspace information, server details and activity logs.

   ![Monitor Provisioning](005.png)

   Once the GPU server is successfully deployed, you should see something similar to the following screenshot, with workspace status shown as ***Active***.

    ![Workspace active](006.png)

## <a name="_toc2049952232"></a>Access the test environment

1. SSH into the GPU server

   A couple of minutes after the GPU server is rebooted, should be able to SSH into the GPU server as user ***ubuntu*** or ***root*** with your SSH key.

   `ssh -i </path/to/your/ssh/private/key> ubuntu@<server_public_ip>`

   You would see a container named jupyter-lab with the ***docker ps*** command:

   ```ubuntu@superknowa-byom-kxhuang-vsi:~$ docker ps

   CONTAINER ID   IMAGE                                            COMMAND                  CREATED        STATUS                 PORTS                                                                                  NAMES

   63d825289967   quay.io/jupyter/pytorch-notebook:cuda12-latest   "tini -g -- start.sh…"   11 hours ago   Up 1 hours (healthy)   0.0.0.0:8502->8502/tcp, :::8502->8502/tcp, 0.0.0.0:8888->8888/tcp, :::8888->8888/tcp   jupyter-lab
   ```

1. Access the Jupyter Server

   The URL of the Jupyter server started on the GPU server can be found on the SuperKnowa BYOM JupyterLab extension frontend UI, when you click on ***Server information*** button under ***Workspace details***.

    ![Server information](006.png)

   When you open the URL in your browser, you'd see a warning about the self-signed TLS/SSL certificate used by the Jupyter Server. After accepting the risk and proceed to the login page, you'll be prompted to enter a token or password. By default, a random token is generated when the Jupyter Server first starts. You can run the ***jupyter server list*** command in a terminal session inside the container to see the token, as shown in the following example:

   `ubuntu@ superknowa-byom-kxhuang-vsi:~$ docker exec -it jupyter-lab jupyter server list`

   Currently running servers:

   ```https://jupyter-lab:8888/?token=bed68dacfc4167325817d67c277a600b2d4d7aa84ff0bd8b :: /home/jovyan/work


   After a successful login, you should now see the JupyterLab interface.

## <a name="_toc76165070"></a>Setting up the environment for testing

1. Open a terminal by clicking on the Terminal icon from the launcher:

   ![Terminal icon in BYOM](007.png)

1. Change to your working directory using the following command:

   `cd ~/work`


1. Clone the repo by running the following command in the terminal

`git clone https://github.com/ibm-ecosystem-engineering/SuperKnowa-QueryCraft.git `

Use the following credentials to access the repo:

**Username:** `kevin@ca.ibm.com`

**Personal Access Token:**

##TODO: How to provide this PAT on public GH

4. Change the directory to the GH repo directory if you are not already inside it. 

   `cd SuperKnowa-QueryCraft`

1. Create a new conda environment using the following commands. Replace the “YOURCONDAENVNAME” that you want to provide.

   ```
   conda deactivate
   conda create -n “YOURCONDAENVNAME” python=3.10 
   ```

   Activate the newly created conda environment and install required packages.

   ```
   conda activate “YOURCONDAENVNAME”     
   pip install -r requirements.txt
   ```

   If asked to proceed, enter “y”

   ![Packages installation](008.png)


1. Open the  `SuperKnowa-QueryCraft/superConfig.ini`. The superConfig.ini file has six sections besides the Default section: DataIngestion, ContextRetriever,Finetune, Inference, Logs, QueryCorrection, EXEvaluator, and QueryAnalysisDashboard. The file has comments that can help you understand the fields inside. We will go through each section one by one.

   Use pwd to set the home_dir in the superConfig.

   `pwd`

    Add a `/` to the end of the home_dir.

Also, set a unique experiment name by editing the exp_name variable in the [Default] section of the superConfig.ini file.

![Default section of superConfig](010.png)

## <a name="_toc1483318699"></a>Configuration Settings

There are two configuration files which can be used for tuning the QueryCraft pipeline:

- superConfig.ini allows you to tune the knobs of the QueryCraft pipeline for each of the components. 
- Config.ini provides you with fine-grained control over the parameters of the experiments.

For testing purposes, you can use the default values of config.ini and experiment with different combinations of superConfig.ini fields. The table below lists some of the important fields and values it can take. 



|**Filename**|**Section**|**Field**|**Supported values**|
| :- | :- | :- | :- |
|config.ini|DataIngestion|delimiter|,|
|config.ini|Finetune|precision|32 or 16 or 8|
|config.ini|Finetune|target_modules|attention_linear_layers or all_linear_layers|
|config.ini|QueryCorrection|query_correction|0 or 1|
|superConfig.ini|ContextRetriever|db_type|sqlite or db2|
|superConfig.ini|Finetune|data_collator|DataCollatorForLanguageModeling, DataCollatorForSeq2Seq or DefaultDataCollator|
|superConfig.ini|Finetune|model_name|Any causalLM model on hugging face |
|superConfig.ini|Finetune|finetune_type|LoRA or QLoRA|
|superConfig.ini|Inference|model_name|Any causalLM model on hugging face or IBM’s granite models|
|superConfig.ini|Inference|finetuned_model|NA – if you want to use pretrained model weights<br>` `or <br>Path to finetuned adapter weights folder|
|superConfig.ini|EXEvaluator|db_type|sqlite or db2|
|config.ini|QueryAnalysisDashboard|selected_columns|Base_Model, Evaluation_set, Ex-accuracy, PP-Ex-accuracy, R, precision, Training_Set, LORA_Alpha, LORA_Dropout, Finetune_Strategy, Target_Modules, Task_Type, Epoch, Learning_Rate, Loss, Eval_Loss, Eval_Runtime, Eval Samples/Second, Eval Steps/Second, Logging_Steps, Max_Steps|



For details on other configurable fields, one can refer to the comments across each field in the config.ini and superConfig.ini files.

# <a name="_toc1843293825"></a>Step 0. Golden Query Annotation

There are three options for using your dataset to finetune/evaluate the Text to SQL (QueryCraft) pipeline:

1. Bring your dataset with golden queries in the following format: question, query, and db_id. Instruction for ingesting the dataset is provided in the next Step 1.
1. Curate the golden query dataset using our annotation tool: <https://annotator.superknowa.tsglwatson.buildlab.cloud/>
1. Use the example datasets provided below for testing: Spider and KaggleDBQA

   Unzip the example datasets using the command:

   ```cd ./input

   unzip spider.zip

   unzip KaggleDBQA.zip

   cd ..
   ```

## <a name="_toc548496318"></a>Golden Query Annotation:
1. Go to our annotation tool. <https://annotator.superknowa.tsglwatson.buildlab.cloud/>

![Data annotator view](011.png)

2. Click on the Instruction Manual and follow the instructions for curating the golden queries dataset. <https://annotator.superknowa.tsglwatson.buildlab.cloud/documentation>

![Data annotation instruction manual](012.png)

# <a name="_toc1471206741"></a>Step 1. Data Ingestion

Users have the flexibility to ingest data to both DB2 and SQLite databases, enhancing adaptability and ease of integration. Steps for both the options are explained below.

## <a name="_toc212047022"></a>Option 1.1. Delimited file Ingestion

You can ingest your delimited files to DB2 on IBM cloud. Once the data is properly ingested, you can run the QueryCraft pipeline on your data. 

#### *Prerequisites (First time activity):* 
- Access to IBM Cloud. You can create a free account. <https://cloud.ibm.com/registration>
- Access to a DB2 database. You can provision a free instance: <https://cloud.ibm.com/catalog/services/db2>
- Service credentials for the DB2 database. Get the db2 credentials from the IBM cloud by following the steps here: <https://cloud.ibm.com/docs/Db2onCloud?topic=Db2onCloud-getting-started>


Note: Testers in the Build Lab team can use the [DB Warehouse-SuperKnowa](https://cloud.ibm.com/services/dashdb/crn%3Av1%3Abluemix%3Apublic%3Adashdb%3Aus-south%3Aa%2Fe65910fa61ce9072d64902d03f3d4774%3A9a36e55c-43f7-4867-8cba-b09ef55c44f9%3A%3A) instance to load their data and run the pipeline. 

The db2_Ingestion module offers a streamlined method for inserting data from CSV or any delimiter file into db2 to fine-tune text to SQL pipelines. 

1. First, set the following credentials in the config.ini file under the [**DB2_Credentials**] section:

- **dsn_database**: Name of the database.
- **dsn_uid**: User ID for the database.
- **dsn_pwd**: Password for the database.
- **dsn_hostname**: Hostname or IP address of the database server.
- **dsn_port**: Port number of the database server.
- **dsn_protocol**: Protocol used for communication.
- **dsn_driver**: Driver used for database connection.
**


1. If you don’t have delimited files for your database which also contains golden query dataset, you can use a file from the `/input/dataset` folder from the test env.

    ![Sample dataset](013.png)

1. Now specify the file path, including the file name, in the superConfig.ini file under the `DataIngestion` section. Additionally, indicate the table name that needs to be created in the db2 database. If you are using the salary.csv, TheHistoryofBaseball is the right schema. Ensuring the right schema is important as the Golden query dataset contains this information in the column db_id. This is required to run the context retriever and the execution evaluation service.

Note: The table may already exist. please use a different table_name.

    
    #Relative path (from home_dir) of csv file to be ingested in db2 table

    #CSV file for Loading
    #filename = ../input/datasets/people.csv
    filename = input/datasets/salary.csv

    #Schema name - Database
    schema_name = TheHistoryofBaseball

    # Table name for CSV data
    table_name= querycraft_db2_test
    

If the user needs to import a file specifying the delimiter for files other than CSV, the user can adjust the delimiter from the config.ini file:

`delimiter = ,`

**Usage:**

Run the Data Ingestion module of the QueryCraft pipeline using the runQueryCraft.sh, file with the dataIngestion option after setting the superConfig.ini file to insert salary.csv into the querycraft_db2_test table in db2.  

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`dataIngestion`

You can validate the successful execution of the dataIngestion module from the DB2 UI as well.

![DB2 UI](014.png)


## <a name="_toc1138956454"></a>Option 1.2. SQLite Ingestion
If you using a SQLite database, you can upload the folder containing database dump in .sqlite format to the `/input/`.

# <a name="_toc1725873645"></a>Step 2. Context Retriever

The Context Retriever module offers a convenient solution for accessing context information like DDL schema from both SQLite and db2 databases. 

Configure the ContextRetriever section of superConfig.ini. 

1. input_database_folder: Relative path to the folder containing database dump in .sqlite format. This is required only when using SQLite database
1. input_data_file: Relative path to the golden dataset (CSV file) with columns: question, query, and db_id
1. db_type:  Determines the data source SQLite or db2.


## <a name="_toc2010877930"></a>Option 2.1 Context Retrieval from db2:

For the db2 context retriever, there's no requirement for an input database file like SQLite, as it directly extracts the DDL schema from Db2 tables. Instead, you need to upload the Golden query dataset (input_data_file) for Db2. This Golden query dataset should be uploaded to the input/datasets/ folder. Here is a sample input_data_file named  kaggleDBQASample.csv, and ensure that db_type is mentioned as **db2**.

```
input_database_folder = 
input_data_file = input/datasets/ kaggleDBQASample.csv
db_type = db2
```

## <a name="_toc481704433"></a>Option 2.2 Context Retrieval from SQLite:

For SQLite, the context retriever service expects two inputs:

1. Input Data with golden queries in the form of a CSV file with the following columns: question, query, and db_id
1. Database folder containing .sqlite files.

The output of the context retriever service is a CSV file with the following columns: question, query, db_id, and context. Here, the context includes the DDL schema for the tables in db_id. 

Configure the input data path, database input file, and database type (referred to as db_type) from the superConfig.ini file. The db_type parameter determines the data source for context retrieval, whether it's SQLite or db2.

If you do not have a dataset of your own, we provide either of the two datasets provided as part of this repository—the Spider dataset or the KaggleDBQA dataset.

To use Spider dataset, configure the ContextRetriever section of SuperConfig.ini as:

```
input_database_folder =input/spider/database/
input_data_file = input/datasets/spider.csv
db_type = sqlite
```

To use the KaggleDBQA dataset configure it into ContextRetriever section of SuperConfig.ini as:

```
input_database_folder =input/kaggleDBQA/database/
input_data_file = input/datasets/kaggleDBQA.csv
db_type = sqlite
```

From the example above, you can specify either Spider or KaggleDBQA in the superConfig.ini file.

After updating the **superConfig.ini** as mentioned above, execute the context retriever using the following command.

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`contextRetriever`

The retrieved context file will be generated in the directory input/datasets/, with the filename exp_name_contextRetriever.csv.

![Instruct dataset with golden queries](015.png)

# <a name="_toc347080945"></a>Step 3. Finetuning

The finetuning service can fine-tune your base model for the specific task of SQL query generation given a natural language query. We leverage LoRA/QLoRA, a PEFT (Parameter Efficient Fine Tuning) technique, to fine-tune the LLM. You can use the existing dataset—Spider or KaggleDBQA—for fine-tuning the LLM, or you can configure your own dataset, as explained in the previous section.

The parameters, dataset, and model can be configured in the superConfig.ini file, as explained below.

1. Provide the base model that you want to finetune. You can use any HuggingFace LLM for fine-tuning although a code based LLM is recommended for Text To SQL task.

model_name = codellama/CodeLlama-7b-Instruct-hf

1. Absolute path to the training dataset, we expect a CSV file with the following columns: db_id, question, query, context. 

    `train_dataset = ${Default:home_dir}input/datasets/spiderWithContext.csv`

    or

    `train_dataset = ${Default:home_dir}input/datasets/kaggleDBQAWithContext.csv`

You can keep one of the two lines uncommented to use one of the two datasets for fine-tuning and/or provide your own dataset for fine-tuning.

1. Finetuning method: Currently we only support PEFT fine-tuning using either LoRA or QLoRA.

    `finetune_type = LoRA`

    Or

    `finetune_type = QLoRA`

1. Select the datacollator you want to preprocess your data. The recommended Data collator for Text To SQL fine-tuning is DataCollatorForSeq2Seq.

   `data_collator=DataCollatorForSeq2Seq`

   or

   `data_collator= DefaultDataCollator`

   or

   `data_collator= DataCollatorForLanguageModeling`


1. Provide the Prompt file path. Each LLM behaves differently according to the prompt so you can update your instruction prompt in this file.

    `prompt_file_path =input/prompts/codellama_model.txt`

If you want to change any fine-grained configurations, you can do so by making changes in the config.ini fields. 

To start fine-tuning your LLM for the Text to SQL task, run the below command.

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`finetune`

**Note**: This testing environment is created for one tester at a time. So, if multiple users are accessing the environment, they might get CUDA out of memory error, storage issue, or high CPU consumption.

1. You can view the GPU consumption using the following command in the terminal:

    `watch nvidia-smi`

1. You can kill the running process using the following command to free up GPU consumption by pasting the right PID from the above command.

    `Sudo kill –9 PID`


Note: This GPU env supports PEFT fine-tuning of Models up to 13B parameter with a lower precision of 8 bits.

Check GPU usage and kill the stale processes so that you can use GPU for fine-tuning

This is subjective to settings like: token max length, per_device_train_batch_size, and target modules. With default setting of these fields, you can load a 13B model in 8-bit precision and 7B model in max 16-bit precision on our GPU environment. 

|**Model Size**|**Precision**|
| :- | :- |
|13B|8 bits|
|7B|8 /16 bits|

Note: Check storage (each model download takes up storage)

The output of this service is finetuned adapter weights stored at:

*/output/model/< exp_name >* folder
# <a name="_toc1894977564"></a>Step 4. Inference

If you executed the Context retriever service in step 2, a test dataset was created for you under the folder `/input/datasets/` with name `<exp_name>_validSet.csv`

Ex: `input/datasets/exp_codellama7b_QATesting12Mar_validSet.csv`

The inference service takes the test dataset as input and uses your finetuned/pre-trained model to generate SQL queries. You can configure the Inference service through the superConfig.ini file as shown below:

1. Provide the File path for the data to be used for inference. We expect a CSV file with the following columns: db_id, question, context

   Example

   `input_dataset = input/datasets/exp_codellama7b_QATesting12Mar_validSet.csv`

1. Provide the inference type, expected values are hf_batch_serial and vllm_batch

    `hf_batch_serial uses huggingface library batch inference`

    vllm_batch uses page attention mechanism. 

    Example: `inference_type = hf_batch_serial`

1. Base model with which you want to draw inference.

   `model_name = codellama/CodeLlama-7b-Instruct-hf`

1. Whether you want to merge fine-tuned weights with the base model, 

    `finetuned_model = NA`

    or

    `finetuned _model = ${Default:home_dir}output/model/${Default:exp_name}`

To start the inference service, run the command below.

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`inference`

The output of this service is a CSV file that contains generated SQL queries. This CSV is stored at `output/inference/<exp_name>.csv`.

# <a name="_toc890332375"></a>Step 5. Query Correction
The Query correction service is used to correct the generated queries before you run it against your database. 

The output of the inference service is dynamically read as the input of the Query correction service and the same CSV is updated.

The input for the query correction can be configured (if required) in the QueryCorrection section of the superConfig.ini file:

`input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv`

To start the query correction service, run the command below.

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`querycorrection`

# <a name="_toc577012548"></a>Step 6. Evaluation

The evaluation service will run the corrected queries against your database and compare the results of the generated query with the golden query to calculate execution accuracy. You can configure the evaluation service using the superConfig.ini file.



The output of the QueryCorrection service is dynamically read as the input of the Query correction service and the same CSV is updated.

## <a name="_toc1369103650"></a>Option 6.1 Evaluation on the db2 database:

Configure the Evaluation service by updating the EXEvaluator section of the superConfig file for the following parameters:

1. Set the database type for db2.

   `db_type = ‘db2’`

1. Validate the absolute path of the file on which you want to get an execution accuracy score.

    `input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv`

Note: You must have completed the Step 1. Data Ingestion section for DB2.

## <a name="_toc858104986"></a>Option 6.2 Evaluation on sqlite:

Configure the Evaluation service by updating the EXEvaluator section of the superConfig file for the following parameters:

1. Set the database type for db2.

    `db_type = 'sqlite'`

1. Database path: Provide the relative path (from home_dir) to the folder containing the database dump in .sqlite format. Example for Spider SQL database:

    `input_database_folder=${Default:home_dir}input/spider/database/`

1. Inference dataset path: Validate the absolute path of the file on which you want to get an execution accuracy score.

    `input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv`



Evaluate the performance of your model against the SQLite database or DB2 by running the below command:

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`evaluation`

# <a name="_toc766523960"></a>Step 7. Query Analysis Dashboard

Query analysis dashboard service can be used to analyze the generated queries, categories of the incorrect queries, evaluation results and comparison between the results of multiple LLMs. 

The output of the Evaluation service is dynamically read as the input of the QueryAnalysisDashboard.

The input for the query correction can be configured (if required) in the QueryAnalysisDashboard section of the superConfig.ini file:

`folder_name =output/evalResults/`

Launch the dashboard for visual analysis of your fine-tuning experiments and generated SQL queries:

`sh runQueryCraft.sh`

Enter the name of the component you want to run:

`queryanalysisDashboard`

Use one of the following ports for exposing your dashboard: 8506-8510.

Please ensure that the URL is not already under use for the port you are selecting: [http://169.46.68.130:port_number]()

Ex: <http://169.46.68.130:8501>

Then you can run the following command with the **right port number** to run the dashboard and access it in your browser.

```
cd code
streamlit run streamlit_query_analysis_dashboard.py --server.port 8502 --server.fileWatcherType none
```

# <a name="_toc523711792"></a>Step 8.  Run pipeline (All)

To run all components together, you can change the required parameters in `superConfig.ini`. You must set the default path as shown in the designated section below. 

[Finetune] 

`train_dataset=${Default:home_dir}input/datasets/${Default:exp_name}_contextRetriever.csv` 



[Inference] 

`input_dataset = input/datasets/${Default:exp_name}_validSet.csv`



[QueryCorrection] 

`input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv` 



[EXEvaluator] 

`input_dataset = ${Default:home_dir}output/inference/${Default:exp_name}_inference.csv`


Run the below command:

`sh runQueryCraft.sh`

Enter the command 

`all`
