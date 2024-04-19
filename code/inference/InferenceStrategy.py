# Define the interface
from abc import ABC, abstractmethod

class InferenceStrategy(ABC):

    @abstractmethod
    def infer(self, expertConfig):
        pass
