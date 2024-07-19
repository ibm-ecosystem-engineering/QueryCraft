#  This code works for the updated V2 api of the ibm-generative-ai library as of 2024.
import os
import logging
from genai import Client
from genai.credentials import Credentials
from dotenv import load_dotenv

DEFAULT_LLM_PARAMS = {
    'decoding_method':"greedy",
    'min_new_tokens':1, 
    'max_new_tokens':400, 
    'repetition_penalty':1.0,
    'random_seed':42, 
    'stop_sequences':["</SQL>","</ENTITIES>"]
}

DEFAULT_PROMPT="""
<s>[INST]<<SYS>>
Read the context and answer the question. If it cannot be answered, only say: 'Unanswerable'. Answer should be concise and professional. Make sure response is not cut off, and do not give an empty response.
Guidelines for Answering:
1. Understand the Context
2. Base answers solely on the information within the given context; do not rely on external knowledge.
3. Craft responses in full sentences to enhance clarity.
4. Be Concise and Relevant. Avoid unnecessary elaboration.
5. Provide answers without personal opinions or interpretations.
6. Keep your response format consistent, adapting it to fit the nature of the question
7. Rely solely on the provided context. Do not introduce external information.
<</SYS>>
#### START CONTEXT
Context: 
{context}
#### END CONTEXT
Question: 
{question}
Answer [/INST]
"""

class LLMBackendGenAI:
    """
    Implementation of the LLM call from IBM watsonx.ai on IBM Research BAM.
    """

    def __init__(self, model_id='meta-llama/llama-3-70b-instruct',llm_params=DEFAULT_LLM_PARAMS):
        
        """
        Initializes the Client instance, model_id and generation parameters

        Returns: None
        """
        load_dotenv()             
        genai_key = os.getenv("GENAI_KEY", None) 
        genai_api = os.getenv("GENAI_API", None)     
        creds = Credentials(api_key=genai_key, api_endpoint=genai_api)
        self.client = Client(credentials=creds)                  
        self.model_params= llm_params      
        self.model = model_id    
        
    def generate_response(self, prompt, text_only = True):         
        """
        Generate a response using llm based on the prompt.    
        Args:         
            prompt_text (str, optional): The prompt for generating response from llm. Defaults to DEFAULT_PROMPT.
            text_only (bool, optional): Flag to indicate if only text response is desired. Defaults to True.
            **kwargs: Additional keyword arguments for customization.    
        Returns:
            str: The generated response.
        """
        
        if isinstance(prompt, str):            
            try:
                response=list(self.client.text.generation.create(model_id = self.model, inputs = prompt, parameters = self.model_params))
                if text_only:                 
                    return response[0].results[0].generated_text
                else:
                    return response[0]
            except Exception:               
                logging.exception("An error occurred while fetching results from llm")
        
        if isinstance(prompt, list):
            try:                
                outputs = []
                response=list(self.client.text.generation.create(model_id = self.model, inputs = prompt, parameters = self.model_params))                
                if text_only:
                    outputs.append(response[0].results[0].generated_text)
                else:
                    outputs.append(response[0])
                return outputs
            except Exception:                
                logging.exception("An error occurred while fetching results from llm")
                
if __name__ == '__main__':
    genai_obj = LLMBackendGenAI()
    answer = genai_obj.generate_response(prompt="What is 1 + 4",text_only=True)