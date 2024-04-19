from enum import Enum
from inference.HuggingfaceBatchSerial import HuggingfaceBatchSerial
from inference.GraniteInference import GraniteInference
from inference.VllmBatchInference import VllmBatchInference
from inference.InferenceStrategy import InferenceStrategy
from inference.InferenceConfiguration import InferenceConfiguration
import json

class InferenceType(Enum):
    HF_BATCH_SERIAL = "hf_batch_serial"
    VLLM_BATCH = "vllm_batch"
    HF_BATCH_MULTITHREAD = "hf_batch_multhread"
    HF_BATCH_ACCELARATOR = "hf_batch_accelarator"

class ModelType(Enum):
    GRANITE = "granite"

def infer_type_factory(inference_type, base_model):
    if(ModelType.GRANITE.value in base_model):
        return GraniteInference()

    if(inference_type == InferenceType.HF_BATCH_SERIAL.value):
        return HuggingfaceBatchSerial()
    elif(inference_type == InferenceType.VLLM_BATCH.value):
        return VllmBatchInference()
    else:
        return None

class InferenceContext():

    def __init__(self, strategy: InferenceStrategy) -> None:
        self._strategy = strategy

    def strategy(self, strategy: InferenceStrategy) -> None:
        self._strategy = strategy
    
    def execute(self, expertConfig: dict) -> any:
        return self._strategy.infer(expertConfig)

def executeInference(expertConfig):
    expertConfig = InferenceConfiguration(expertConfig).build()
    model_name = expertConfig["base_model"]
    inference_type = expertConfig["inference_type"]

    inference = infer_type_factory(inference_type, model_name)
    if(inference is not None):
        rs = InferenceContext(inference).execute(expertConfig)
        return rs
    else:
        return "No inference type found"
    
    

