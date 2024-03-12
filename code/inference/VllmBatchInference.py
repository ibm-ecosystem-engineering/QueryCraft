from InferenceStrategy import InferenceStrategy
from pathlib import Path
import torch
import logging
from vllm import LLM, SamplingParams
import pandas as pd

class VllmBatchInference(InferenceStrategy):

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

        sampling_params = SamplingParams(
            temperature=0.7, 
            max_tokens=400,
            #use_beam_search=True,
            #best_of=2,
        )

        llm = LLM(
            model=base_model, 
            dtype=torch.float16,
        )

        df_validation = pd.read_csv(input_dataset)
        prompts = [self.create_prompt(row) for _index, row in df_validation.iterrows()]
        ## Batch inference just simply pass all an array of prompts
        vllm_batch_outputs = llm.generate(prompts, sampling_params)
        df_validation = self.save_batch_inference(vllm_batch_outputs, df_validation)
        df_validation.to_csv(output_location)
        return ("vLLM batch inference completed successfully, output file is saved at :", output_location) 

    
    def save_batch_inference(results, df_validation):
        outputs = []
        for result in results:
            generated_text = result.outputs[0].text
            outputs.append(generated_text)
        
        df_validation["model_op"] = outputs
        return df_validation
        
    def prepare_context(self, question: str, context: str):
        text = f"""
        [INST] Write SQLite query to answer the following question given the database schema. Please wrap your code answer using ```: Schema: {context} [/INST] Here is the SQLite query to answer to the question:{question} ```
        """
        return text

    def create_prompt(self, row) -> str:
        return self.prepare_context(row["question"], row["context"])