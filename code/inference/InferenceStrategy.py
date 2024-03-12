# Define the interface
from abc import ABC, abstractmethod
import configparser
import pandas as pd
import os

class InferenceStrategy(ABC):

    @abstractmethod
    def infer(self, config):
        pass

class InferenceConfiguration():
     
     def __init__(self, config) -> None:
         self.config = config
     
     def build(self) -> dict:

        ## Get configuration from superConfig.ini
        super_config = configparser.ConfigParser()
        super_config.read('./../../superConfig.ini')

        ## Home directory
        home_dir  = super_config['Default']['home_dir']
        exp_name  = super_config['Default']['exp_name']

        inference_type = super_config["Inference"]["inference_type"]

        # input_dataset = home_dir+input_dataset
        input_dataset = home_dir+super_config['Inference']['input_dataset']

        # Base model identifed
        base_model = super_config['Inference']['model_name']

        # Looking for finetuned model
        finetuned_model = super_config['Inference']['finetuned_model']
        if(os.path.exists(finetuned_model) is False):
            finetuned_model = None

        logging_path = home_dir+ self.config['logs']['log_folder']+ exp_name+"_infer"
        output_location = home_dir+"output/inference/"+exp_name+"_inference.csv"

        ## Watsonx details
        watsonx_url = self.config["Inference"]["watsonx_url"],
        watsonx_apikey =  self.config["Inference"]["watsonx_apikey"]
        watsonx_projectId = self.config["Inference"]["watsonx_projectID"]

        return {
            "base_model": base_model,
            "input_dataset": input_dataset,
            "logging_path": logging_path,
            "finetuned_model": finetuned_model,
            "output_location": output_location,
            "watsonx_url": watsonx_url,
            "watsonx_apikey": watsonx_apikey,
            "watsonx_projectId": watsonx_projectId,
            "inference_type": inference_type,
            "exp_name": exp_name
        }

