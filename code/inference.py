from enum import Enum
from GraniteInference import GraniteInference
from HuggingfaceBatchSerial import HuggingfaceBatchSerial
from InferenceStrategy import InferenceConfiguration, InferenceStrategy
from VllmBatchInference import VllmBatchInference

class InferenceType(Enum):
    HF_BATCH_SERIAL = "hf_batch_serial"
    VLLM_BATCH = "vllm_batch"
    HF_BATCH_MULTITHREAD = "hf_batch_multhread"
    HF_BATCH_SERIAL = "hf_batch_accelarator"

class ModelType(Enum):
    GRANITE = "granite"

def infer_type_factory(inference_type, base_model):
    if(ModelType.GRANITE in base_model):
        return GraniteInference()

    if(inference_type == InferenceType.HF_BATCH_SERIAL):
        return HuggingfaceBatchSerial()
    elif(inference_type == InferenceType.VLLM_BATCH):
        return VllmBatchInference()
    else:
        return None

class InferenceContext():

    def __init__(self, strategy: InferenceStrategy) -> None:
        self._strategy = strategy

    def strategy(self, strategy: InferenceStrategy) -> None:
        self._strategy = strategy
    
    def execute(self, config: dict) -> any:
        self._strategy.infer(config)

def funcInference(inference_type="expDummy",
                  model_name="codellama/CodeLlama-7b-Instruct-hf"):
    
    config = InferenceConfiguration().build()

    inference = infer_type_factory(inference_type, model_name)
    rs = InferenceContext(inference).execute(config)
    return rs


