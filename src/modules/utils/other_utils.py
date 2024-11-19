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

    def replace_negative_with_zero(self, df):
        return df.applymap(lambda x: max(x, 0))

    def check_values(
        self,
        Q1_vars,
        Q_hat_adjusteds,
        Q0_vars,
        Sold_0s,
        total_demand_up_to_k_minus_1_vars,
        Sold_1s,
        total_demand_from_k_to_T_vars,
        Q1_plus_lefts,
        Left_0s,
        Lost_0s,
        Left_1s,
        Lost_1s,
    ):

        # 用於存儲每個條件的統計結果
        results = {
            "Condition": [],
            "Average_Error_Percentage": [],
            "Max_Error_Percentage": [],
            "Min_Error_Percentage": [],
            "Max_Error": [],
            "Min_Error": [],
        }

        # 定義存儲每個條件下的誤差和誤差百分比
        conditions_errors = {
            "Q1_vars": [],
            "Sold_0s": [],
            "Sold_1s": [],
            "Left_0s": [],
            "Left_1s": [],
            "Lost_0s": [],
            "Lost_1s": [],
        }

        # 存儲每個條件下的誤差百分比
        conditions_error_percentage = {
            "Q1_vars": [],
            "Sold_0s": [],
            "Sold_1s": [],
            "Left_0s": [],
            "Left_1s": [],
            "Lost_0s": [],
            "Lost_1s": [],
        }

        # 遍歷每一個變量集合
        for i in range(len(Q1_vars)):
            # 提取變量的值
            Q1 = Q1_vars[i].X
            Q_hat_adjusted = Q_hat_adjusteds[i].X
            Q0 = Q0_vars[i].X
            Sold_0 = Sold_0s[i].X
            total_demand_up_to_k_minus_1 = total_demand_up_to_k_minus_1_vars[i].X
            Sold_1 = Sold_1s[i].X
            total_demand_from_k_to_T = total_demand_from_k_to_T_vars[i].X
            Q1_plus_left = Q1_plus_lefts[i].X
            Left_0 = Left_0s[i].X
            Lost_0 = Lost_0s[i].X
            Left_1 = Left_1s[i].X
            Lost_1 = Lost_1s[i].X

            # 計算理論值
            theoretical_sold_0 = min(total_demand_up_to_k_minus_1, Q0)
            theoretical_left_0 = max(Q0 - theoretical_sold_0, 0)
            theoretical_Q1_plus_left = Q1 + theoretical_left_0  # Q1_plus_left 的理論值
            theoretical_sold_1 = min(total_demand_from_k_to_T, theoretical_Q1_plus_left)
            theoretical_left_1 = max(theoretical_Q1_plus_left - theoretical_sold_1, 0)
            theoretical_lost_0 = max(total_demand_up_to_k_minus_1 - Q0, 0)
            theoretical_lost_1 = max(total_demand_from_k_to_T - theoretical_Q1_plus_left, 0)

            # 檢查條件 2：Sold_0 一定等於理論值
            if not (Sold_0 == theoretical_sold_0):
                error = abs(Sold_0 - theoretical_sold_0)
                conditions_errors["Sold_0s"].append(error)
                # 計算誤差百分比
                conditions_error_percentage["Sold_0s"].append(
                    (error / theoretical_sold_0) * 100 if theoretical_sold_0 != 0 else 0
                )

            # 檢查條件 3：Sold_1 一定等於理論值
            if not (Sold_1 == theoretical_sold_1):
                error = abs(Sold_1 - theoretical_sold_1)
                conditions_errors["Sold_1s"].append(error)
                # 計算誤差百分比
                conditions_error_percentage["Sold_1s"].append(
                    (error / theoretical_sold_1) * 100 if theoretical_sold_1 != 0 else 0
                )

            # 檢查條件 4：Left_0 一定等於理論值
            if not (Left_0 == theoretical_left_0):
                error = abs(Left_0 - theoretical_left_0)
                conditions_errors["Left_0s"].append(error)
                # 計算誤差百分比
                conditions_error_percentage["Left_0s"].append(
                    (error / theoretical_left_0) * 100 if theoretical_left_0 != 0 else 0
                )

            # 檢查條件 5：Left_1 一定等於理論值
            if not (Left_1 == theoretical_left_1):
                error = abs(Left_1 - theoretical_left_1)
                conditions_errors["Left_1s"].append(error)
                # 計算誤差百分比
                conditions_error_percentage["Left_1s"].append(
                    (error / theoretical_left_1) * 100 if theoretical_left_1 != 0 else 0
                )

            # 檢查條件 6：Lost_0 一定等於理論值
            if not (Lost_0 == theoretical_lost_0):
                error = abs(Lost_0 - theoretical_lost_0)
                conditions_errors["Lost_0s"].append(error)
                # 計算誤差百分比
                conditions_error_percentage["Lost_0s"].append(
                    (error / theoretical_lost_0) * 100 if theoretical_lost_0 != 0 else 0
                )

            # 檢查條件 7：Lost_1 一定等於理論值
            if not (Lost_1 == theoretical_lost_1):
                error = abs(Lost_1 - theoretical_lost_1)
                conditions_errors["Lost_1s"].append(error)
                # 計算誤差百分比
                conditions_error_percentage["Lost_1s"].append(
                    (error / theoretical_lost_1) * 100 if theoretical_lost_1 != 0 else 0
                )

        # 計算每個條件的統計結果
        for condition, errors in conditions_errors.items():
            error_percentages = conditions_error_percentage[condition]
            if errors:
                # 統計數據，並將所有數值四捨五入至小數點后三位
                avg_error_percentage = (
                    round(sum(error_percentages) / len(error_percentages), 3)
                    if error_percentages
                    else 0.0
                )
                max_error_percentage = (
                    round(max(error_percentages), 3) if error_percentages else 0.0
                )
                min_error_percentage = (
                    round(min(error_percentages), 3) if error_percentages else 0.0
                )
                max_error = round(max(errors), 3) if errors else 0.0
                min_error = round(min(errors), 3) if errors else 0.0

                # 存儲結果
                results["Condition"].append(condition)
                results["Average_Error_Percentage"].append(avg_error_percentage)
                results["Max_Error_Percentage"].append(max_error_percentage)
                results["Min_Error_Percentage"].append(min_error_percentage)
                results["Max_Error"].append(max_error)
                results["Min_Error"].append(min_error)

        # 轉換為 DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    # Calculate service level
    def calculate_service_level(self, *, salvage_value, cost, price):

        cu = price - cost
        co = cost - salvage_value
        service_lv = cu / (co + cu)

        return service_lv

    def make_s3_related_strtegies_result(
        self,
        *,
        all_Rs,
        losses,
        lefts,
        profits,
        operation_profits,
        alpha_values,
        beta_values,
        F_vars,
        Q0_vars,
        Q1_vars,
        f_values,
        tau_values,
    ):

        results_dict = {
            "average_profits": [sum(profits) / len(profits) if profits else 0],
            "average_losses": [sum(losses) / len(losses) if losses else 0],
            "average_lefts": [sum(lefts) / len(lefts) if lefts else 0],
            "average_operation_profits": [
                sum(operation_profits) / len(operation_profits) if operation_profits else 0
            ],
            "alpha_values": [alpha_values],
            "beta_values": [beta_values],
            "tau_values": [tau_values],
        }
        stimulations_result = {
            "R(T)": all_Rs,
            "R": [x - 2 for x in all_Rs],
            "F": F_vars,
            "f_values": f_values,
            "profits": profits,
            "losses": losses,
            "lefts": lefts,
            "operation_profits": operation_profits,
            "Q0": Q0_vars,
            "Q1": Q1_vars,
        }

        return pd.DataFrame(results_dict).sort_values(
            by="average_profits", ascending=False
        ), pd.DataFrame(stimulations_result)