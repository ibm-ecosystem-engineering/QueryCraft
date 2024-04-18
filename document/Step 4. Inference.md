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