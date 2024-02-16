from datasets import load_dataset
from math import ceil
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import configparser
import logging

config = configparser.ConfigParser()
config.read('config.ini')
config.sections()

EXP = config['Inference']['EXP'] 
input_dataset = config['Default']['home_dir']+config['Inference']['ip_dataset'] 
base_model = config['Inference']['base_model'] 
load_finetuned_model = config['Inference']['load_finetuned_model'] 
finetuned_model = config['Default']['home_dir']+config['Inference']['finetuned_model'] 

logging_path = config['Default']['home_dir']+config['logs']['log_folder']+ config['Inference']['EXP']+"_infer"
logging.basicConfig(filename=logging_path+".log", level=logging.INFO)

logging.info("EXPERIMENT :"+EXP)
logging.info(" Input Set : "+ input_dataset)
logging.info(" base_model  : "+ base_model)
logging.info(" finetuned_model : "+ finetuned_model)
logging.info(" load_finetuned_model : "+ load_finetuned_model)

df_validation = pd.read_csv(input_dataset)
device_map = "auto"
#df_validation = df_validation[900:901]

logging.info("Number of samples to be inferred : "+ str(len(df_validation)))
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    #torch_dtype=torch.float16,
    device_map=device_map,
)

if(load_finetuned_model=="Y"):
    model = PeftModel.from_pretrained(model, finetuned_model)
    
tokenizer = AutoTokenizer.from_pretrained(base_model)

def resultGenerator(row):
    question=row["question"]
    context=row["context"]  
    text =  f"""
    [INST] Write SQLite query to answer the following question given the database schema. Please wrap your code answer using ```: Schema: {context} [/INST] Here is the SQLite query to answer to the question:{question} ```
    """
    eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
    **inputs,
    num_return_sequences=1,
    eos_token_id=eos_token_id,
    pad_token_id=eos_token_id,
    max_new_tokens=400,
    do_sample=False,
    num_beams=5
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    result = outputs[0][len(text):].split("```")[0]
    print("SNo :"+str(row["Sno"]))
    logging.info("SNo :"+str(row["Sno"]))

    logging.info("Question: "+row["question"])
    logging.info("context: "+row["context"])
    logging.info("result: "+result)
    logging.info("output: "+outputs[0])
    print("result: "+result)
    logging.info("*******************")
    return result

eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
df_validation["Sno"] = df_validation.index

df_validation["model_op"] = df_validation.apply(resultGenerator,axis=1)
df_validation.to_csv(config['Default']['home_dir']+"output/inference/"+EXP+".csv")