# Setting up environment

## Step 1. Create your GPU-enabled environment

You can run SuperKnowa-QueryCraft in any GPU-enabled environment, whether it's cloud-based, on-premises, or both. You can leverage [watsonx.ai](https://www.ibm.com/products/watsonx-ai) to train, validate, tune and deploy foundation and machine learning models, with both installed software and SaaS offerings. Or, you can simply deploy a GPU server on IBM Cloud using SuperKnowa BYOM, which provides a JupyterLab extension to provision a GPU-enabled [virtual server instance (VSI) for VPC](https://cloud.ibm.com/docs/vpc?topic=vpc-about-advanced-virtual-servers) in minutes by filling out a simple form in any of your JupyterLab environments.

## Step 2. Optional: Install the JupyterLab extension

You can leverage our JupyterLab extension to provision a GPU server. Please refer to the SuperKnowa Text2Infra repo to learn more about it.   

## Step 3. Configuration Settings

There are two configuration files which can be used for tuning the QueryCraft pipeline:

- simpleConfig.ini allows you to tune the knobs of the QueryCraft pipeline for each of the components. 
- expertConfig.ini provides you with fine-grained control over the parameters of the experiments.

For testing purposes, you can use the default values of expertConfig.ini and experiment with different combinations of simpleConfig.ini fields. The table below lists some of the important fields and values it can take. 

1. Open the  `SuperKnowa-QueryCraft/simpleConfig.ini`. The simpleConfig.ini file has six sections besides the Default section: DataIngestion, ContextRetriever,Finetune, Inference, Logs, QueryCorrection, EXEvaluator, and QueryAnalysisDashboard. The file has comments that can help you understand the fields inside. We will go through each section one by one.

   Use pwd to set the home_dir in the simpleConfig.

   `pwd`

    Add a `/` to the end of the home_dir.

Also, set a unique experiment name by editing the exp_name variable in the [Default] section of the simpleConfig.ini file.

![Default section of simpleConfig](../image/010.png)



|**Filename**|**Section**|**Field**|**Supported values**|
| :- | :- | :- | :- |
|expertConfig.ini|DataIngestion|delimiter|,|
|expertConfig.ini|Finetune|precision|32 or 16 or 8|
|expertConfig.ini|Finetune|target_modules|attention_linear_layers or all_linear_layers|
|expertConfig.ini|QueryCorrection|query_correction|0 or 1|
|simpleConfig.ini|ContextRetriever|db_type|sqlite or db2|
|simpleConfig.ini|Finetune|data_collator|DataCollatorForLanguageModeling, DataCollatorForSeq2Seq or DefaultDataCollator|
|simpleConfig.ini|Finetune|model_name|Any causalLM model on hugging face |
|simpleConfig.ini|Finetune|finetune_type|LoRA or QLoRA|
|simpleConfig.ini|Inference|model_name|Any causalLM model on hugging face or IBM’s granite models|
|simpleConfig.ini|Inference|finetuned_model|NA – if you want to use pretrained model weights<br>` `or <br>Path to finetuned adapter weights folder|
|simpleConfig.ini|EXEvaluator|db_type|sqlite or db2|
|expertConfig.ini|QueryAnalysisDashboard|selected_columns|Base_Model, Evaluation_set, Ex-accuracy, PP-Ex-accuracy, R, precision, Training_Set, LORA_Alpha, LORA_Dropout, Finetune_Strategy, Target_Modules, Task_Type, Epoch, Learning_Rate, Loss, Eval_Loss, Eval_Runtime, Eval Samples/Second, Eval Steps/Second, Logging_Steps, Max_Steps|



For details on other configurable fields, one can refer to the comments across each field in the expertConfig.ini and simpleConfig.ini files.

