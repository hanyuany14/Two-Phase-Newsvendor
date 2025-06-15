import gurobipy as gp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

T = 10
ASSIGNED_FS = np.arange(0.1, 1.0, 0.1)
ASSIGNED_TS = list(range(2, T))  # 2 到 T-1

np.random.seed(0)

class S1_Model():
    def __init__(self):
        pass


    def cal_test_fixed_F_fixed_R(
        self,
        assigned_T,
        assigned_F,
        cost,
        price,
        salvage_value,
        Qk_hat_df_test,
        demand_df_test,
        Q_star,
    ):
        assigned_R = assigned_T - 2
        result, stimulation_result = self.__cal_fixed_F_fixed_R(
            Q_star,
            assigned_F,
            assigned_R,
            demand_df_test,
            cost,
            price,
            salvage_value,
            Qk_hat_df_test,
        )

        results_df_1 = pd.DataFrame([result]).sort_values(
            by="average_profits", ascending=False
        )

        return results_df_1, pd.DataFrame(stimulation_result)
    
    def grid_fixed_F_fixed_R(
        self,
        assigned_Ts,
        assigned_Fs,
        cost,
        price,
        salvage_value,
        Qk_hat_df,
        demand_df_train,
        Q_star,
    ):

        results_list = []
        max_profit = None
        max_profit_stimulation_result = {}

        for assigned_T in assigned_Ts:
            for assigned_F in assigned_Fs:
                assigned_R = assigned_T - 2
                mean_result, stimulation_result = self.__cal_fixed_F_fixed_R(
                    Q_star,
                    assigned_F,
                    assigned_R,
                    demand_df_train,
                    cost,
                    price,
                    salvage_value,
                    Qk_hat_df,
                )
                results_list.append(mean_result)

                if max_profit is None or max_profit < mean_result["average_profits"]:
                    max_profit = mean_result["average_profits"]
                    max_profit_stimulation_result = stimulation_result

        results_df_1 = pd.DataFrame(results_list).sort_values(
            by="average_profits", ascending=False
        )

        return results_df_1, pd.DataFrame(max_profit_stimulation_result)

    def __cal_fixed_F_fixed_R(
        self, Q_star, assigned_F, assigned_R, demand_df, cost, price, salvage_value, Qk_hat_df
    ):
        all_losses = []
        all_lefts = []
        all_left0s = []
        all_left1s = []
        all_operation_profits = []
        all_profits = []
        all_q0s = []
        all_q1s = []

        Q0 = assigned_F * Q_star  # 期初庫存

        for i, row in demand_df.iterrows():

            # 第一階段計算
            total_sold_0 = min(Q0, row[: assigned_R + 1].sum())  # 第一階段售出量
            left_0 = max(Q0 - total_sold_0, 0)  # 第一階段剩餘
            lost_0 = max((row[: assigned_R + 1].sum() - Q0), 0)

            # 第二階段開始補貨，根據指定的 R
            Qk_hat = Qk_hat_df.iloc[i, assigned_R]
            Q1 = max((Qk_hat - Q0), 0)  # 二次訂貨量
            total_sold_1 = min(Q1 + left_0, row[assigned_R + 1 :].sum())  # 第二階段售出量
            left_1 = max((Q1 + left_0) - total_sold_1, 0)  # 第二階段剩餘
            lost_1 = max(row[assigned_R + 1 :].sum() - (Q1 + left_0), 0)

            # 統計
            total_sold = total_sold_0 + total_sold_1
            total_lost = lost_0 + lost_1
            total_left = left_0 + left_1

            # 計算運營利潤和總利潤
            operation_profit = (price - cost) * total_sold

            left_penalty_cost = (cost - salvage_value) * left_1
            # left_penalty_cost = (cost - salvage_value) * total_left
            lost_penalty_cost = (price - cost) * total_lost

            profit = operation_profit - left_penalty_cost - lost_penalty_cost

            all_losses.append(total_lost)
            all_lefts.append(total_left)
            all_operation_profits.append(operation_profit)
            all_profits.append(profit)
            all_q0s.append(Q0)
            all_q1s.append(Q1)
            all_left0s.append(left_0)
            all_left1s.append(left_1)

        result_df = {
            "R(T)": assigned_R + 2,
            "F": assigned_F,
            "Q0": all_q0s,
            "Q1": all_q1s,
            "average_profits": np.mean(all_profits),
            "average_losses": np.mean(all_losses),
            "average_lefts": np.mean(all_lefts),
            "average_operation_profits": np.mean(all_operation_profits),
        }

        stimulation_result = {
            "R(T)": assigned_R + 2,
            "F": assigned_F,
            "profits": all_profits,
            "losses": all_losses,
            "lefts": all_lefts,
            "Left0s": all_left0s,
            "Left1s": all_left1s,
            "operation_profits": all_operation_profits,
            "Q0": all_q0s,
            "Q1": all_q1s,
        }

        return result_df, stimulation_result
