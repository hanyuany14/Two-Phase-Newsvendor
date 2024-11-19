import gurobipy as gp
from gurobipy import GRB
import gurobipy_pandas as gppd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import min_, max_
from scipy.stats import multivariate_normal, norm
import pickle
import os
import glob
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
import math
import seaborn as sns

from src.modules.base import Base


class ModelUtils(Base):
    def __init__(self, current_timestamp: datetime):
        super().__init__(current_timestamp)
        self.current_timestamp = current_timestamp

    def save_model_parameters(
        self,
        name: str,
        alpha_values=None,
        beta_values=None,
        f_values=None,
        tau_values=None,
    ):
        os.makedirs("models", exist_ok=True)

        params = {}
        if alpha_values is not None:
            params["alpha"] = alpha_values
        if beta_values is not None:
            params["beta"] = beta_values
        if f_values is not None:
            params["f_values"] = f_values
        if tau_values is not None:
            params["tau_values"] = tau_values

        if params:
            with open(
                f"models/{name}_{self.data_size}_{self.current_timestamp}.pkl", "wb"
            ) as f:
                pickle.dump(params, f)
            print(
                f"Model parameters saved as models/{name}_{self.data_size}_{self.current_timestamp}.pkl"
            )
        else:
            print("No parameters provided to save.")

    def delete_model_parameters(self, name: str):

        file_path = f"models/{name}_{self.data_size}_{self.current_timestamp}.pkl"

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Model parameters file '{file_path}' has been deleted.")
        else:
            print(f"File '{file_path}' does not exist.")

    def show_models(self, model_prefix: str):
        file_paths = sorted(glob.glob(f"models/{model_prefix}_*.pkl"))

        for file_path in file_paths:
            with open(file_path, "rb") as f:
                params = pickle.load(f)
                print(f"Contents of {file_path}:")
                print(params)
                print()
