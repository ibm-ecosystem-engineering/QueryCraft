from inference.InferenceStrategy import InferenceStrategy
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import pandas as pd

class GraniteInference(InferenceStrategy):

    def infer(self, expertConfig):
        ## Extract all the values from the configuration file
        ## set them in variables
        ## expertConfig = self.infer_configuration()
        credentials = {
            "url": expertConfig["watsonx_url"],
            "apikey": expertConfig["watsonx_apikey"]
        }
        project_id = expertConfig["watsonx_projectId"]
        
        parameters = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 200,
            GenParams.STOP_SEQUENCES: ["<end·of·code>"]
        }

        base_model = expertConfig["base_model"]
        input_dataset = expertConfig["input_dataset"]
        output_location = expertConfig["output_location"]

        model = Model(
            model_id=base_model, 
            params=parameters, 
            credentials=credentials,
            project_id=project_id
        )
        df_validation = pd.read_csv(input_dataset)
        df_validation["model_op"] = df_validation.apply(self.resultGeneratorGranite, args=(model), axis=1)
        df_validation.to_csv(output_location)
        return ("Granatine batch inference completed successfully, output file is saved at :", output_location)


    def resultGeneratorGranite(self, row, model):
        question=row["question"]
        context=row["context"]
        prompt_txt = [self.get_prompt_granite(context.replace('`',''),question)]
        generated_response = model.generate( prompt_txt)
        result = generated_response[0]["results"][0]["generated_text"]
        print(result)
        print("SNo: ",row["Sno"])
        print("Question: ",question)
        print("context: ",context)
        print("result: ", result)
        print("*******************")
        return result
    
    def get_prompt_granite(self, context, question):
        input = f"""
        SQL query to answer the following question:
        {question}

        Database schema:
        {context}
        """
        return input