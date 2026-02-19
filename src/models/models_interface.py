from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

class ModelTrainerInterface(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Train the model using the provided DataFrame.
        
        Args:
            df (pd.DataFrame): The input data for training.
        
        Returns:
            Tuple[float, float]: A tuple containing training and validation accuracy.
        """
        pass