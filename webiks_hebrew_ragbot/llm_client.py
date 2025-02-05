from abc import ABC, abstractmethod
from .document import document_definition_factory

definitions = document_definition_factory()

class LLMClient(ABC):
    def __init__(self):
        self.field_for_answer = definitions.field_for_llm


    @abstractmethod
    def answer(self, _question, _top_k_docs) -> tuple[str, float, int]:
        pass
