from inference.InferenceStrategy import InferenceStrategy
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import logging
import torch

class HuggingfaceBatchSerial(InferenceStrategy):

    def infer(self, expertConfig):
        ## Extract all the values from the configuration file
        ## set them in variables
        ## expertConfig = self.infer_configuration()
        base_model = expertConfig["base_model"]
        finetuned_model = expertConfig["finetuned_model"]
        input_dataset = expertConfig["input_dataset"]
        logging_path = expertConfig["logging_path"]
        output_location = expertConfig["output_location"]

        ## set the logging path
        logging.basicConfig(filename=logging_path+".log", level=logging.INFO)

        ## Model inference by creating tokenizer and model object
        device_map = "auto"
        tokenizer = AutoTokenizer.from_pretrained(expertConfig["base_model"])
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map=device_map,
            )
        eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
        ## check if finetuned model is set
        if(finetuned_model is not None):
            logging.info(f"Finetuned model found, started loading ... {finetuned_model}")
            print("Finetuned model found and loadding...")
            model = PeftModel.from_pretrained(model, finetuned_model)

        df_validation = pd.read_csv(input_dataset)
        #df_validation = df_validation[1:3]
        df_validation["model_op"] = df_validation.apply(lambda row : self.resultGenerator(row, tokenizer=tokenizer, model=model, eos_token_id=eos_token_id), axis=1)
        df_validation.to_csv(output_location)
        return ("Huggingface batch inference completed successfully, output file is saved at :", output_location)
        

    def resultGenerator(self, row, tokenizer, model, eos_token_id):
        
        question=row["question"]
        context=row["context"]  
        print(f"executing question ... {question}")

        text =  f"""
        [INST] Write SQLite query to answer the following question given the database schema. Please wrap your code answer using ```: Schema: {context} [/INST] Here is the SQLite query to answer to the question:{question} ```
        """
        input_tokens = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            sequences = model.generate(
                **input_tokens,
                num_return_sequences=1,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
                max_new_tokens=400,
                do_sample=False
                ## num_beams=5
            )
        outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        torch.cuda.empty_cache()

        result = outputs[0][len(text):].split("```")[0]
        print("result: "+result)
        logging.info("*******************")
        return result
    