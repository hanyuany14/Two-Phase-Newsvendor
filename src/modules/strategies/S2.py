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

from src.modules.strategies.train_base import Train


class Train_S2(Train):
    def __init__(self):
        super().__init__()

    def do_train(self):
        return self.grid_flexible_F_fixed_R(self.assigned_Ts)

    def grid_flexible_F_fixed_R(self, assigned_Ts):
        results_dict = {
            "R(T)": [],
            "R": [],
            "average_profits": [],
            "average_losses": [],
            "average_lefts": [],
            "average_operation_profits": [],
            "alpha_values": [],
            "F_vars": [],
            "Q0_vars": [],
            "Q1_vars": [],
        }

        max_profit = 0
        max_profit_stimulation_result = {}

        for assigned_T in assigned_Ts:
            print(
                f"+++++++++++++++++++++++++++++++++++++++ THis is T={assigned_T} +++++++++++++++++++++++++++++++++++++++++++++++++"
            )

            assigned_R = assigned_T - 2
            result = self.cal_flexible_F_fixed_R(assigned_R=assigned_R)

            if result is None:
                print(f"模型沒有最佳解")

            else:
                (
                    all_Rs,
                    losses,
                    lefts,
                    profits,
                    operation_profits,
                    alpha_values,
                    F_vars,
                    Q0_vars,
                    Q1_vars,
                ) = result

                # 计算平均值
                average_losses = sum(losses) / len(losses) if losses else 0
                average_lefts = sum(lefts) / len(lefts) if lefts else 0
                average_profits = sum(profits) / len(profits) if profits else 0
                average_operation_profits = (
                    sum(operation_profits) / len(operation_profits)
                    if operation_profits
                    else 0
                )

                results_dict["R(T)"].append(assigned_T)
                results_dict["R"].append([x - 2 for x in all_Rs])
                results_dict["average_losses"].append(average_losses)
                results_dict["average_lefts"].append(average_lefts)
                results_dict["average_profits"].append(average_profits)
                results_dict["average_operation_profits"].append(
                    average_operation_profits
                )
                results_dict["alpha_values"].append(alpha_values)
                results_dict["F_vars"].append(F_vars)
                results_dict["Q0_vars"].append(Q0_vars)  # 紀錄該策略組合每一次模擬的 Q0
                results_dict["Q1_vars"].append(Q1_vars)  # 紀錄該策略組合每一次模擬的 Q1

                if max_profit < average_profits:
                    print(
                        f"max_profit is changed from {max_profit} to {average_profits}"
                    )
                    max_profit = average_profits
                    max_profit_stimulation_result = {
                        "R": [x - 2 for x in all_Rs],
                        "F": F_vars,
                        "profits": profits,
                        "losses": losses,
                        "lefts": lefts,
                        "operation_profits": operation_profits,
                        "Q0": Q0_vars,
                        "Q1": Q1_vars,
                    }

        return pd.DataFrame(results_dict).sort_values(
            by="average_profits", ascending=False
        ), pd.DataFrame(max_profit_stimulation_result)

    def cal_flexible_F_fixed_R(self, assigned_R):
        model = self.model

        # add variables
        self.alphas = model.addVars(self.features_num + 1, name="alphas")

        # ======================= Start Stimulation! =======================

        for i, row in self.demand_df.iterrows():

            ### Data for this stimulation
            demand_row = self.demand_df.iloc[i]
            Qk_hat_df_row = self.Qk_hat_df.iloc[i]
            X_data = self.training_df.iloc[i].tolist()
            X_data.append(1)

            # =================== Model 1: Optimal Fraction Model ===================

            ### 用線性回歸計算F_var
            model.addConstr(
                self.f_vars[i]
                == gp.quicksum(
                    X_data[j] * self.alphas[j] for j in range(self.features_num + 1)
                )
            )
            model.addGenConstrLogistic(
                xvar=self.f_vars[i],
                yvar=self.F_vars[i],
                name=f"logistic_constraint_{i}",
            )

            ### Q0_var = F_vars * self.Q_star
            model.addConstr(
                self.Q0_vars[i] == self.F_vars[i] * self.Q_star, f"Q0_upper_bound_{i}"
            )

            # =================== Model 2: Optimal Order Time Model ===================

            ## 只會有一個 R 為 1
            model.addConstr(self.R_vars[i, assigned_R] == 1, name=f"Set_R_{i}_0_To_1")
            model.addConstr(
                gp.quicksum(self.R_vars[i, k] for k in range(self.K)) == 1,
                name=f"Ensure_only_one_R_true_{i}",
            )  # 0~7

            # ============ Model 3: re-estimate order-up-to-level =================

            ### 計算 Q_hat -> k: 2~9 -> k-2: 0~7
            model.addConstr(
                self.Q_hats[i]
                == gp.quicksum(
                    self.R_vars[i, k - 2] * Qk_hat_df_row[k - 2]
                    for k in range(2, self.T)
                ),
                name=f"Define_Q_hat_{i}",
            )

            ### 將 Q_hats(重新估計的值) 與原先 Q0 值進行比較。如果發現原先估計的比較少，則補足 Q_hat_adjusted，如果不用補充，則為 0
            model.addConstr(
                self.Q_hat_adjusteds[i] == self.Q_hats[i] - self.Q0_vars[i],
                name=f"Adjust_Q_hat_{i}",
            )
            model.addConstr(
                self.Q1_vars[i] >= self.Q_hat_adjusteds[i],
                name=f"Constr_Q1_ge_Q_hat_adjusted_{i}",
            )
            model.addConstr(self.Q1_vars[i] >= 0, name=f"Constr_Q1_ge_0_{i}")

            # =================== Model 4: Maximum Profit Model ===================

            # ### 0~k-1 的需求量
            self.total_demand_up_to_k_minus_1_var = model.addVar(
                name=f"Total_Demand_Up_to_K_Minus_1_{i}"
            )
            model.addConstr(
                self.total_demand_up_to_k_minus_1_var
                == gp.quicksum(
                    self.R_vars[i, k - 2] * demand_row[: k - 1].sum()
                    for k in range(2, self.T)
                ),
                name=f"Constr_Total_Demand_Up_to_K_Minus_1_{i}",
            )
            model.addConstr(
                self.total_demand_up_to_k_minus_1_vars[i]
                == self.total_demand_up_to_k_minus_1_var,
                name=f"Calculate_Total_Demand_Up_to_K_minus_1_{i}",
            )

            # ### k~T 的需求量
            self.total_demand_from_k_to_T_var = model.addVar(
                name=f"Total_Demand_from_K_to_T_{i}"
            )
            model.addConstr(
                self.total_demand_from_k_to_T_var
                == gp.quicksum(
                    self.R_vars[i, k - 2] * demand_row[k - 1 :].sum()
                    for k in range(2, self.T)
                ),
                name=f"Constr_Total_Demand_from_K_to_T_{i}",
            )
            model.addConstr(
                self.total_demand_from_k_to_T_vars[i]
                == self.total_demand_from_k_to_T_var,
                name=f"Calculate_Total_Demand_from_k_to_T_{i}",
            )

            # 計算 Sold_0
            model.addConstr(
                self.Sold_0s[i] <= self.total_demand_up_to_k_minus_1_vars[i],
                name=f"Constr_Sold_0_1_{i}",
            )
            model.addConstr(
                self.Sold_0s[i] <= self.Q0_vars[i], name=f"Constr_Sold_0_2_{i}"
            )

            # 計算 Left_0
            model.addConstr(
                self.Left_0s[i] >= self.Q0_vars[i] - self.Sold_0s[i],
                name=f"Constr_Left_0_1_{i}",
            )
            model.addConstr(self.Left_0s[i] >= 0, name=f"Constr_Left_0_2_{i}")

            # 計算 Lost_0
            model.addConstr(
                self.Lost_0s[i]
                >= self.total_demand_up_to_k_minus_1_vars[i] - self.Q0_vars[i],
                name=f"Constr_Lost_0_1_{i}",
            )
            model.addConstr(self.Lost_0s[i] >= 0, name=f"Constr_Lost_0_2_{i}")

            # 計算 Q1 + left_0
            model.addConstr(
                self.Q1_plus_lefts[i] == self.Q1_vars[i] + self.Left_0s[i],
                name=f"Q1_plus_left_{i}",
            )

            # 計算 Sold_1
            model.addConstr(
                self.Sold_1s[i] <= self.total_demand_from_k_to_T_vars[i],
                name=f"Constr_Sold_1_1_{i}",
            )
            model.addConstr(
                self.Sold_1s[i] <= self.Q1_plus_lefts[i], name=f"Constr_Sold_1_2_{i}"
            )

            # 計算 Left_1
            model.addConstr(
                self.Left_1s[i] >= self.Q1_plus_lefts[i] - self.Sold_1s[i],
                name=f"Constr_Left_1_1_{i}",
            )
            model.addConstr(self.Left_1s[i] >= 0, name=f"Constr_Left_1_2_{i}")

            # 計算 Lost_1
            model.addConstr(
                self.Lost_1s[i]
                >= self.total_demand_from_k_to_T_vars[i] - self.Q1_plus_lefts[i],
                name=f"Constr_Lost_1_1_{i}",
            )
            model.addConstr(self.Lost_1s[i] >= 0, name=f"Constr_Lost_1_2_{i}")

            # 統計本次 Profit for this stimulation
            model.addConstr(
                self.profits_vars[i]
                == (
                    (self.price - self.cost)
                    * (self.Sold_0s[i] + self.Sold_1s[i])  # sold
                    - (self.price - self.cost)
                    * (self.Lost_0s[i] + self.Lost_1s[i])  # lost sales
                    - (self.cost - self.salvage_value)
                    * self.Left_1s[i]  # left, only considering Left_1
                ),
                name=f"Profit_Constraint_{i}",
            )

        #  ======================================= Model optimize =======================================

        model.setObjective(
            gp.quicksum(self.profits_vars[i] for i in range(len(self.demand_df))),
            GRB.MAXIMIZE,
        )

        try:
            model.optimize()

            if model.status == GRB.OPTIMAL:
                print(f"\nmodel.status is optimal: {model.status == GRB.OPTIMAL}")
                print(f"model.status is TIME_LIMIT: {model.status == GRB.TIME_LIMIT}\n")

                print("===================== 找到最佳解 ==================")
                print(f"Q0_optimal（最佳總庫存量）: {self.Q_star}")

                print("Alphas values:")
                for key, alpha in self.alphas.items():
                    print(f"alpha[{key}]: {alpha.X}")

                alpha_values = np.array([alpha.X for key, alpha in self.alphas.items()])

                all_losses = []
                all_lefts = []
                all_operation_profits = []
                all_profits = []
                all_Rs = []
                all_Q0s = []
                all_Q1s = []
                all_Fs = []

                for i in range(len(self.demand_df)):

                    print("----------------------------------------------")
                    print(f"第 {i+1} 筆觀察資料:")

                    sold0 = self.Sold_0s[i].X
                    sold1 = self.Sold_1s[i].X
                    left0 = self.Left_0s[i].X
                    left1 = self.Left_1s[i].X
                    lost0 = self.Lost_0s[i].X
                    lost1 = self.Lost_1s[i].X

                    operation_profit = (self.price - self.cost) * (sold0 + sold1)
                    daily_profit = self.profits_vars[i].X

                    all_losses.append(lost0 + lost1)
                    all_lefts.append(left0 + left1)
                    all_operation_profits.append(operation_profit)
                    all_profits.append(daily_profit)
                    all_Q0s.append(self.Q0_vars[i].X)
                    all_Q1s.append(self.Q1_vars[i].X)
                    all_Fs.append(self.F_vars[i].X)

                    reorder_day = None
                    for k in range(self.K):
                        R_value = self.R_vars[i, k].X
                        print(f"第 {k+2} 天補貨策略: R_vars = {R_value}")

                        if int(R_value) == 1:
                            reorder_day = k + 2
                    print(f"*** 於第[{reorder_day}]天進貨 ***\n")
                    all_Rs.append(reorder_day)

                    demand_row = self.demand_df.iloc[i]
                    # print(f"第一階段需求量: {demand_row[: (assigned_R+1)].sum()}")
                    # print(f"第二階段需求量: {demand_row[(assigned_R+1): ].sum()}\n")

                    # print(f"Q0_optimal（最佳總庫存量）: {self.Q_star}")
                    # print(f"F_var（重新訂貨量佔總訂貨量比例）: {self.F_vars[i].X}")
                    # print(f"Q0_var（期初庫存量）: {self.Q0_vars[i].X}\n")

                    # print(f"Q1_var（二次訂貨量）: {self.Q1_vars[i].X}")

                    total_demand_up = self.total_demand_up_to_k_minus_1_vars[i].X
                    total_demand_down = self.total_demand_from_k_to_T_vars[i].X

                    # check_results_df = check_values(
                    #     Q1_vars=self.Q1_vars,
                    #     Q_hat_adjusteds=self.Q_hat_adjusteds,
                    #     Q0_vars=self.Q0_vars,
                    #     Sold_0s=self.Sold_0s,
                    #     total_demand_up_to_k_minus_1_vars=self.total_demand_up_to_k_minus_1_vars,
                    #     Sold_1s=self.Sold_1s,
                    #     total_demand_from_k_to_T_vars=self.total_demand_from_k_to_T_vars,
                    #     Q1_plus_lefts=self.Q1_plus_lefts,
                    #     Left_0s=self.Left_0s,
                    #     Lost_0s=self.Lost_0s,
                    #     Left_1s=self.Left_1s,
                    #     Lost_1s=self.Lost_1s,
                    # )
                    # print(check_results_df)

                    # for t in range(2):
                    #     if t == 0:
                    #         print(
                    #             f"  第 {t+1} 階段: 本階段期初庫存 = {self.Q0_vars[i].X}, 第一階段總需求 = {total_demand_up}, 銷售量 = {self.Sold_0s[i].X}, 本階段期末剩餘庫存 = {self.Left_0s[i].X}, 本期損失 = {self.Lost_0s[i].X}"
                    #         )
                    #     else:
                    #         print(
                    #             f"  第 {t+1} 階段: 本階段期初庫存 = {self.Q1_plus_lefts[i].X}, 重新預估需求 = {self.Q_hats[i].X}, 第二階段總需求 = {total_demand_down}, 銷售量 = {self.Sold_1s[i].X}, 本階段期末剩餘庫存 = {self.Left_1s[i].X}, 本期損失 = {self.Lost_1s[i].X}"
                    #         )

                    # print(f"  本觀察資料總利潤 = {daily_profit}\n")

                print("==========================================")
                print(f"最佳化模型平均利潤 = {np.mean(all_profits)}")

                return (
                    all_Rs,
                    all_losses,
                    all_lefts,
                    all_profits,
                    all_operation_profits,
                    alpha_values,
                    all_Fs,
                    all_Q0s,
                    all_Q1s,
                )

            else:
                print("===================== 找不到最佳解 ==================")
                print(f"Model is feasible. Status: {model.status}")
                model.computeIIS()
                model.write("model.ilp")

                for constr in model.getConstrs():
                    if constr.IISConstr:
                        print(f"導致不可行的約束： {constr.constrName}")

                for var in model.getVars():
                    if var.IISLB > 0 or var.IISUB > 0:
                        print(
                            f"導致不可行的變量： {var.VarName}, IIS下界： {var.IISLB}, IIS上界： {var.IISUB}"
                        )

                return None

        except gp.GurobiError as e:
            print(f"Error code {str(e.errno)}: {str(e)}")
            return None
