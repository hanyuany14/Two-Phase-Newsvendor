import numpy as np
from datetime import datetime


class Base:
    def __init__(self, current_timestamp: datetime):
        self.data_size = 200
        self.train_size = 0.5
        self.testing_size = 0.5
        self.k_fold = 2
        self.current_timestamp = current_timestamp

        self.assigned_Ts = list(range(2, self.T))
        self.assigned_Fs = np.arange(0.1, 1.0, 0.1)

        self.T = 10
        self.M = 500000000
