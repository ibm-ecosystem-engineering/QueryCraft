from InferenceStrategy import InferenceStrategy
import logging
from datasets import load_dataset
from math import ceil
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import logging,json
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

class HuggingfaceBatchSerial(InferenceStrategy):

    def __init__(self):
        pass

    def infer(self, config):
        ## Extract all the values from the configuration file
        ## set them in variables
        ## config = self.infer_configuration()
        base_model = config["base_model"]
        finetuned_model = config["finetuned_model"]
        input_dataset = config["input_dataset"]
        logging_path = config["logging_path"]
        output_location = config["output_location"]

        ## set the logging path
        logging.basicConfig(filename=logging_path+".log", level=logging.INFO)

        ## Model inference by creating tokenizer and model object
        device_map = "auto"
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map=device_map,
            )
        
        ## check if finetuned model is set
        if(finetuned_model is not None):
            model = PeftModel.from_pretrained(model, finetuned_model)

        df_validation = pd.read_csv(input_dataset)
        df_validation["model_op"] = df_validation.apply(self.resultGenerator, args=( tokenizer, model), axis=1)
        df_validation.to_csv(output_location)
        return ("Huggingface batch inference completed successfully, output file is saved at :", output_location)
        

    def resultGenerator(row, tokenizer, model):
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
    