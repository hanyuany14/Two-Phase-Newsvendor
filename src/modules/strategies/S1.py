

import numpy as np
import pandas as pd

from src.modules.strategies.train_base import Train

class Train_S1(Train):
    def __init__(self):
        super().__init__()
    
    def do_train(self,assigned_Ts,assigned_Fs):
        return self.grid_fixed_F_fixed_R(assigned_Ts, assigned_Fs)

    def grid_fixed_F_fixed_R(
        self,
        assigned_Ts,
        assigned_Fs,
    ):
        results_list = []
        max_profit = 0
        max_profit_stimulation_result = {}

        for assigned_T in assigned_Ts:
            for assigned_F in assigned_Fs:
                assigned_R = assigned_T - 2
                mean_result, stimulation_result = self.cal_fixed_F_fixed_R(
                    assigned_F,
                    assigned_R,
                )
                results_list.append(mean_result)

                if max_profit < mean_result["average_profits"]:
                    print(
                        f"max_profit is changed from {max_profit} to {mean_result['average_profits']}"
                    )
                    max_profit = mean_result["average_profits"]
                    max_profit_stimulation_result = stimulation_result

        results_df_1 = pd.DataFrame(results_list).sort_values(
            by="average_profits", ascending=False
        )

        return results_df_1, pd.DataFrame(max_profit_stimulation_result)
    
    def cal_fixed_F_fixed_R(
        self, assigned_F, assigned_R
    ):
        all_losses = []
        all_lefts = []
        all_operation_profits = []
        all_profits = []
        all_q0s = []
        all_q1s = []

        Q0 = assigned_F * self.Q_star  # 期初庫存

        # print(f"\n")
        # print(f"====" * 10)
        # print(f"\n")

        for i, row in self.demand_df.iterrows():

            # 第一階段計算
            total_sold_0 = min(Q0, row[: assigned_R + 1].sum())  # 第一階段售出量
            left_0 = max(Q0 - total_sold_0, 0)  # 第一階段剩餘
            lost_0 = max(row[: assigned_R + 1].sum() - Q0, 0)

            # 第二階段開始補貨，根據指定的 R
            Qk_hat = self.Qk_hat_df.iloc[i, assigned_R]
            Q1 = max((Qk_hat - Q0), 0)  # 二次訂貨量
            total_sold_1 = min(Q1 + left_0, row[assigned_R + 1 :].sum())  # 第二階段售出量
            left_1 = max((Q1 + left_0) - total_sold_1, 0)  # 第二階段剩餘
            lost_1 = max(row[assigned_R + 1 :].sum() - (Q1 + left_0), 0)

            # 統計
            total_sold = total_sold_0 + total_sold_1
            total_lost = lost_0 + lost_1

            # 計算運營利潤和總利潤
            operation_profit = (self.price - self.cost) * total_sold
            left_penalty_cost = (self.cost - self.salvage_value) * left_1
            lost_penalty_cost = (self.price - self.cost) * total_lost
            profit = operation_profit - left_penalty_cost - lost_penalty_cost

            all_losses.append(total_lost)
            all_lefts.append(left_1)
            all_operation_profits.append(operation_profit)
            all_profits.append(profit)
            all_q0s.append(Q0)
            all_q1s.append(Q1)

            # print(f"這是第 {i+1} 筆模擬資料\n")
            # print(f"F: {assigned_F}, R: {assigned_R+2}")
            # print(f"self.Q_star 為 {self.Q_star}")
            # print(f"期初庫存 Q0: {Q0}")
            # print(f"重新估計量 Qk_hat: {Qk_hat}")
            # print(f"訂貨量 Q1 為 {Q1}\n")

            # print(
            #     f"第一階段：期初庫存 Q0: {Q0}，需求量為 {row[:assigned_R + 1].sum()}，Sold_0 為 {total_sold_0}，Left_0 為 {left_0}，Lost_0 為 {lost_0}"
            # )
            # print(
            #     f"第二階段：期初庫存 Q1+left_0 為 {Q1+left_0}，需求量為 {row[assigned_R + 1:].sum()}，Sold_1 為 {total_sold_1}，Left_1 為 {left_1}，Lost_1 為 {lost_1}\n"
            # )
            # print(
            #     f"統計結果：Sold 為 {total_sold}, Lost 為 {total_lost} Left_Penalty_Cost 為 {left_penalty_cost}，Lost_Penalty_Cost 為 {lost_penalty_cost}，Profit 為 {profit}"
            # )
            # print("----" * 10)

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
            "operation_profits": all_operation_profits,
            "Q0": all_q0s,
            "Q1": all_q1s,
        }

        return result_df, stimulation_result
