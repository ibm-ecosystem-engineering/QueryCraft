from datetime import datetime
import os
import sys
import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
from math import ceil
import configparser
import logging,transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

start_time = time.time()

config = configparser.ConfigParser()
config.read('config.ini')
config.sections()

#User Config
trainDataset = config['Default']['home_dir']+config['FinetuneLoRA']['trainDataset']  
base_model = config['FinetuneLoRA']['base_model'] 
finetuningMethod = config['FinetuneLoRA']['finetuningMethod'] 
precision = int(config['FinetuneLoRA']['precision'])
tokenizeMaxLength = int(config['FinetuneLoRA']['tokenizeMaxLength'])
LoRA_r = int(config['FinetuneLoRA']['LoRA_r'])
LoRA_dropout = float(config['FinetuneLoRA']['LoRA_dropout'])
batch_size = int(config['FinetuneLoRA']['batch_size'])
num_train_epochs = int(config['FinetuneLoRA']['num_train_epochs'])
per_device_train_batch_size = int(config['FinetuneLoRA']['per_device_train_batch_size'])
output_dir =  config['Default']['home_dir']+"output/model/"+ config['FinetuneLoRA']['EXP']
target_modules = config['FinetuneLoRA']['target_modules']
logging_path = config['Default']['home_dir']+config['logs']['log_folder']+ config['FinetuneLoRA']['EXP']
prompt_file_path  = config['Default']['home_dir']+config['FinetuneLoRA']['prompt_path']
logging.basicConfig(filename=logging_path+".log", level=logging.INFO)

logging.info("EXPERIMENT :"+ config['FinetuneLoRA']['EXP'])
logging.info(" Training Set : "+ trainDataset)
logging.info(" Base Model : "+ base_model)
logging.info(" Finetuning Method : "+finetuningMethod)
logging.info(" Precision : "+ str(precision))
logging.info(" Max length in tokenizer : "+ str(tokenizeMaxLength))
logging.info(" LoRA_r  : "+ str(LoRA_r))
logging.info(" LoRA_dropout  : "+ str(LoRA_dropout))
logging.info(" Batch Size  : "+ str(batch_size))
logging.info(" Number of train epochs  : "+ str(num_train_epochs))
logging.info(" per_device_train_batch_size  : "+ str(per_device_train_batch_size))
logging.info(" Output Directory : "+ output_dir)
logging.info(" Target Modules: "+ str(target_modules))

df = pd.read_csv(trainDataset)
data = Dataset.from_pandas(df)
num_samples = len(data)
val_set_size = ceil(0.1 * num_samples)
logging.info(" Number of samples for training: "+ str(num_samples))
logging.info(" Number of samples for validation: "+ str(val_set_size))


#Load Model
if (precision==8):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
    )
if (precision==32):
    model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizeMaxLength,
        padding=False,
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
dataTrainTest["test"].to_csv(config['Default']['home_dir']+"input/datasets/"+config['FinetuneLoRA']['EXP']+"_validSet.csv")

#Setup LoRA
model.train() # put model back into training mode
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

lora_config = LoraConfig(
    r=LoRA_r,
    target_modules=target_modules,
    lora_dropout=LoRA_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True


# In[9]:


#Training Arguments

gradient_accumulation_steps = batch_size // per_device_train_batch_size
training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        #max_steps=20,
        learning_rate=3e-4,
        logging_steps=50,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=100,
        save_steps=50,
        num_train_epochs = num_train_epochs,
        output_dir=output_dir,
        load_best_model_at_end=True, #
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="none", # if use_wandb else "none",
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
     ),
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
