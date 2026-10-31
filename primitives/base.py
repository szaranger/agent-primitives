from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch

class LatentMessage:
    """Simulates direct KV Cache / hidden-state tensor communication between agents."""
    def __init__(self, tensor_state: torch.Tensor, metadata: Optional[Dict[str, Any]] = None):
        self.tensor_state = tensor_state
        self.metadata = metadata or {}

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, input_data: Any) -> LatentMessage:
        pass
