import numpy as np

class transform_functions:
    def __init__(self) -> None:
        pass
        
    @staticmethod
    def hardlims(input :np.ndarray) -> np.ndarray:
        return np.where(input > 0, 1, -1)
    
    @staticmethod
    def Sigmoid(x :np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def Softmax(x :np.ndarray) -> np.ndarray:
        x  = np.subtract(x, np.max(x))        # prevent overflow
        ex = np.exp(x)

        return ex / np.sum(ex)