# tokenizer/base.py
from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to token list (ids)"""
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert token list (ids) to text"""
        pass
