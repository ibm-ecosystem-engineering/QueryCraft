# <a name="_toc347080945"></a>Step 3. Finetuning

The finetuning service can fine-tune your base model for the specific task of SQL query generation given a natural language query. We leverage LoRA/QLoRA, a PEFT (Parameter Efficient Fine Tuning) technique, to fine-tune the LLM. You can use the existing dataset—Spider or KaggleDBQA—for fine-tuning the LLM, or you can configure your own dataset, as explained in the previous section.

The parameters, dataset, and model can be configured in the simpleConfig.ini file, as explained below.

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

If you want to change any fine-grained configurations, you can do so by making changes in the expertConfig.ini fields. 

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