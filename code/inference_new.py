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
import logging,requests,json
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams



def funcInference(exp_name="expDummy",
                  model_name="codellama/CodeLlama-7b-Instruct-hf",
                  finetuned_model = "NA",
                  input_dataset = "input/datasets/spiderTrainSetNewContext.csv"):
    config = configparser.ConfigParser()
    config.read('./../config.ini')
    config.sections()
    
    input_dataset = config['Default']['home_dir']+input_dataset
    base_model = model_name
    logging_path = config['Default']['home_dir']+config['logs']['log_folder']+ exp_name+"_infer"
    logging.basicConfig(filename=logging_path+".log", level=logging.INFO)
    logging.info("EXPERIMENT :"+exp_name)
    logging.info(" Input Set : "+ input_dataset)
    logging.info(" base_model  : "+ base_model)
    logging.info(" finetuned_model : "+ finetuned_model)

#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': "",
#     }

    df_validation = pd.read_csv(input_dataset)
    device_map = "auto"
    df_validation = df_validation[0:3]
    logging.info("Number of samples to be inferred : "+ str(len(df_validation)))


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
    


    def get_prompt_granite(context, question):
        input = f"""
        SQL query to answer the following question:
        {question}

        Database schema:
        {context}
        """
        return input

    def resultGeneratorGranite(row):
        question=row["question"]
        context=row["context"]
        prompt_txt = [get_prompt_granite(context.replace('`',''),question)]
        generated_response = model.generate( prompt_txt)
        result = generated_response[0]["results"][0]["generated_text"]
        print(result)
        print("SNo: ",row["Sno"])
        print("Question: ",question)
        print("context: ",context)
        print("result: ", result)
        print("*******************")
        return result
    
    
    df_validation["Sno"] = df_validation.index

    if("granite" in base_model):
        from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
        from ibm_watsonx_ai.foundation_models import Model
        credentials = {
        "url": config["Inference"]["watsonx_url"],
        "apikey": config["Inference"]["watsonx_apikey"]
        }
        print(credentials)
        project_id = config["Inference"]["watsonx_projectID"]
        
        #model_id    = ModelTypes.FLAN_T5_XXL        
        parameters = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 200,
            GenParams.STOP_SEQUENCES: ["<end·of·code>"]
        }
        model = Model(
        model_id=base_model, 
        params=parameters, 
        credentials=credentials,
        project_id=project_id
        )
        df_validation["model_op"] = df_validation.apply(resultGeneratorGranite,axis=1)
    else:
        print("I am outside Granite")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map=device_map,
            )
        if(finetuned_model!="NA"):
            finetuned_model = config['Default']['home_dir']+finetuned_model
            model = PeftModel.from_pretrained(model, finetuned_model)
        eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
        df_validation["model_op"] = df_validation.apply(resultGenerator,axis=1)
        
    df_validation.to_csv(config['Default']['home_dir']+"output/inference/"+exp_name+".csv")
    return ("Inference completed successfully, output file is saved at :",config['Default']['home_dir']+"output/inference/"+exp_name+".csv")