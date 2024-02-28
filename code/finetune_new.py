from datetime import datetime
import os
import sys
import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,BitsAndBytesConfig
from datasets import Dataset, DatasetDict
from math import ceil
import configparser
import logging,transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

def funcFinetune(exp_name="expDummy",data_collator="DataCollatorForSeq2Seq",model_name="codellama/CodeLlama-7b-Instruct-hf",prompt_file_path="input/prompts/codellama_model.txt",finetune_type="LoRA",train_dataset ="input/datasets/spiderTrainSetNewContext.csv"):
    if("granite" in model_name):
        return("The fine-tuning service does not offer support for models from the 'Granite' family. Please consider using models provided by Hugging Face instead.")
        
    else:
        start_time = time.time()
        config_filePath="./../config.ini"
        config = configparser.ConfigParser()
        config.read(config_filePath)
        config.sections()

        super_config = configparser.ConfigParser()
        super_config.read('./../superConfig.ini')
        home_dir  = super_config['Default']['home_dir']

        #User Config
        logging_path = home_dir+config['logs']['log_folder']+ exp_name

        base_model = model_name
        finetuningMethod = finetune_type
        precision = int(config['Finetune']['precision'])
        tokenizeMaxLength = int(config['Finetune']['tokenizeMaxLength'])
        


        batch_size = int(config['Finetune']['batch_size'])
        num_train_epochs = int(config['Finetune']['num_train_epochs'])
        per_device_train_batch_size = int(config['Finetune']['per_device_train_batch_size'])
        output_dir =  home_dir+"output/model/"+ exp_name
        prompt_file_path = home_dir+prompt_file_path

        LoRA_r = int(config['Finetune']['LoRA_r'])
        LoRA_dropout = float(config['Finetune']['LoRA_dropout'])
        LoRA_alpha = float(config['Finetune']['LoRA_alpha'])
        target_modules = config['Finetune']['target_modules']
        #LoRA_taskType = config['Finetune']['LoRA_taskType']

        #train_dataset = home_dir+train_dataset

        logging.basicConfig(filename=logging_path+".log", level=logging.INFO)

        logging.info("EXPERIMENT :"+ exp_name)
        logging.info(" Training Set : "+ train_dataset)
        logging.info(" Base Model : "+ base_model)
        logging.info(" Finetuning Method : "+finetuningMethod)
        logging.info(" Precision : "+ str(precision))
        logging.info(" Max length in tokenizer : "+ str(tokenizeMaxLength))
        logging.info(" LoRA_r  : "+ str(LoRA_r))
        logging.info(" LoRA_dropout  : "+ str(LoRA_dropout))
        #logging.info(" task_type  : "+ str(LoRA_taskType))
        logging.info(" LoRA_alpha  : "+ str(LoRA_alpha))
        logging.info(" Batch Size  : "+ str(batch_size))
        logging.info(" Number of train epochs  : "+ str(num_train_epochs))
        logging.info(" per_device_train_batch_size  : "+ str(per_device_train_batch_size))
        logging.info(" Output Directory : "+ output_dir)
        logging.info(" Target Modules: "+ str(target_modules))

        df = pd.read_csv(train_dataset)
        data = Dataset.from_pandas(df)
        num_samples = len(data)
        val_set_size = ceil(0.1 * num_samples)
        logging.info(" Number of samples for training: "+ str(num_samples))
        logging.info(" Number of samples for validation: "+ str(val_set_size))

        if (target_modules == "all_linear_layers"):
            target_modules = ['gate_proj',
            'down_proj',
            'v_proj',
            'q_proj',
            'k_proj',
            'o_proj',
            'lm_head',
            'up_proj']
        if (target_modules == "attention_linear_layers"):
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.add_eos_token = True
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"


        def tokenize(prompt):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=tokenizeMaxLength,
                padding="max_length",
                return_tensors=None,
            )

            # "self-supervised learning" means the labels are also the inputs:
            result["labels"] = result["input_ids"].copy()
            return result

        def generate_and_tokenize_prompt(data_point):
            prompt_file = open(prompt_file_path, "r")
            full_prompt = prompt_file.read()
            full_prompt = full_prompt.replace("{context}",data_point["context"])
            full_prompt = full_prompt.replace("{question}",data_point["question"])
            full_prompt = full_prompt.replace("{query}",data_point["query"])
            #print("**********************************",full_prompt)
            return tokenize(full_prompt)

        dataTrainTest = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        tokenized_train_dataset = dataTrainTest["train"].shuffle().map(generate_and_tokenize_prompt)
        tokenized_val_dataset = dataTrainTest["test"].shuffle().map(generate_and_tokenize_prompt)
        dataTrainTest["test"].to_csv(home_dir+"input/datasets/"+exp_name+"_validSet.csv")



        if(finetune_type=="QLoRA"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ) # setup bits and bytes config

            model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map={"":0})
            
        else:
            if (precision==8):
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    load_in_8bit=True,
                    device_map="auto",
                )
            if (precision==32):
                model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                )
            if (precision==16):
                model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                )
        
        
        lora_config = LoraConfig(
            r=LoRA_r, 
            lora_alpha=LoRA_alpha, 
            target_modules=target_modules, 
            lora_dropout=LoRA_dropout, 
            bias="none", 
            task_type="CAUSAL_LM",
        )
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model) # prepares the whole model for kbit training
        model = get_peft_model(model, lora_config) # Now you get a model ready for QLoRA training



        if torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

        #Training Arguments
        gradient_accumulation_steps = batch_size // per_device_train_batch_size
        training_args = TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=5,
                max_steps=50,
                learning_rate=3e-4,
                logging_steps=50,
                optim="adamw_torch",
                evaluation_strategy="steps", # if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=10,
                save_steps=20,
                #num_train_epochs = num_train_epochs,
                output_dir=output_dir,
                load_best_model_at_end=True, #
                group_by_length=True, # group sequences of roughly the same length together to speed up training
                report_to="none", # if use_wandb else "none",
            )

        if(data_collator == "DataCollatorForSeq2Seq"):
            trainer = Trainer(
                model=model,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_val_dataset,
                args=training_args,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )
        elif(data_collator == "DataCollatorForLanguageModeling"):
            trainer = Trainer(
                model=model,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_val_dataset,
                args=training_args,
                data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
        elif(data_collator == "DefaultDataCollator"):
            trainer = Trainer(
                model=model,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_val_dataset,
                args=training_args,
                data_collator=transformers.DefaultDataCollator(),
            )

        #pytorch-related optimisation (which just make training faster but don't affect accuracy):
        model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            print("compiling the model")
            model = torch.compile(model)
        with torch.autocast("cuda"):
            trainer.train()
            model.save_pretrained(output_dir)

        end_time = time.time()
        total_time = end_time - start_time
        logging.info("Time taken to run in seconds: :"+ str(total_time))
        return ("Finetuning completed successfully, finetuned model is saved at:"+output_dir)
