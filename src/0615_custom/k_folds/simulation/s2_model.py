# import gurobipy as gp
# from gurobipy import GRB
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import math

# T = 10
# M = 5000000
# LASSO_BETA = 100
# LASSO_ALPHA = 0.1
# LASSO_BETA_SECOND_TRAIN = 0.9

# ASSIGNED_FS = np.arange(0.1, 1.0, 0.1)
# ASSIGNED_TS = list(range(2, T))  # 2 到 T-1

# np.random.seed(0)

# # Gurobi Model Constants
# THREADS = 12
# TIME_LIMIT = 20000
# MIPGAP = 0.01
# CURRENT_TIMESTAMP = int(datetime.now().strftime("%Y%m%d%H%M"))


# from configs import params
# env = gp.Env(params=params)
# model = gp.Model(env=env)


# class S2_Model():
#     def __init__(self):
#         pass


#     def grid_flexible_F_fixed_R(
#         self,
#         assigned_Ts,
#         salvage_value,
#         cost,
#         price,
#         Q_star,
#         demand_df_train,
#         Qk_hat_df_train,
#         training_df,
#     ):
#         results_dict = {
#             "R(T)": [],
#             "R": [],
#             "average_profits": [],
#             "average_losses": [],
#             "average_lefts": [],
#             "average_operation_profits": [],
#             "alpha_values": [],
#             "F_vars": [],
#             "f_vars": [],
#             "Q0_vars": [],
#             "Q1_vars": [],
#         }

#         max_profit = None
#         max_profit_stimulation_result = {}

#         for assigned_T in assigned_Ts:
#             assigned_R = assigned_T - 2
#             result = self.__cal_flexible_F_fixed_R(
#                 assigned_R=assigned_R,
#                 salvage_value=salvage_value,
#                 cost=cost,
#                 price=price,
#                 Q_star=Q_star,
#                 demand_df_train=demand_df_train,
#                 Qk_hat_df=Qk_hat_df_train,
#                 training_df=training_df,
#                 lambda_alpha=LASSO_ALPHA,
#             )

#             if result is None:
#                 print(f"模型沒有最佳解")
#                 continue

#             (
#                 all_Rs,
#                 losses,
#                 lefts,
#                 profits,
#                 operation_profits,
#                 alpha_values,
#                 F_vars,
#                 f_vars,
#                 Q0_vars,
#                 Q1_vars,
#             ) = result

#             # 計算平均值
#             average_losses = sum(losses) / len(losses) if losses else 0
#             average_lefts = sum(lefts) / len(lefts) if lefts else 0
#             average_profits = sum(profits) / len(profits) if profits else 0
#             average_operation_profits = (
#                 sum(operation_profits) / len(operation_profits) if operation_profits else 0
#             )

#             # 將結果存儲到字典中
#             results_dict["R(T)"].append(assigned_T)
#             results_dict["R"].append(all_Rs)
#             results_dict["average_losses"].append(average_losses)
#             results_dict["average_lefts"].append(average_lefts)
#             results_dict["average_profits"].append(average_profits)
#             results_dict["average_operation_profits"].append(average_operation_profits)
#             results_dict["alpha_values"].append(alpha_values)
#             results_dict["F_vars"].append(F_vars)
#             results_dict["f_vars"].append(f_vars)
#             results_dict["Q0_vars"].append(Q0_vars)
#             results_dict["Q1_vars"].append(Q1_vars)

#             # print(f"The average profits is {average_profits}")

#             if max_profit is None or max_profit < average_profits:
#                 # print(f"max_profit is changed from {max_profit} to {average_profits}")
#                 max_profit = average_profits
#                 max_profit_stimulation_result = {
#                     "R": all_Rs,
#                     "F": F_vars,
#                     "f": f_vars,
#                     "profits": profits,
#                     "losses": losses,
#                     "lefts": lefts,
#                     "operation_profits": operation_profits,
#                     "Q0": Q0_vars,
#                     "Q1": Q1_vars,
#                 }

#         return pd.DataFrame(results_dict).sort_values(
#             by="average_profits", ascending=False
#         ), pd.DataFrame(max_profit_stimulation_result)


#     def cal_test_flexible_F_fixed_R(
#         self,
#         assigned_R,
#         alphas,
#         salvage_value,
#         cost,
#         price,
#         Q_star,
#         demand_df_test,
#         Qk_hat_df_test,
#         testing_df,
#     ):

#         # ======================= Global Variables =======================

#         # Category 1 - Some variables that is important to future work
#         K = T - 2  # this is for k=2~T-1. => if T = 10(1~10), K will be 8. (0~7)
#         n = len(demand_df_test)

#         # Initialize lists or numpy arrays to replace Gurobi variables
#         Sold_0s = np.zeros(n)
#         Left_0s = np.zeros(n)
#         Lost_0s = np.zeros(n)
#         Sold_1s = np.zeros(n)
#         Left_1s = np.zeros(n)
#         Lost_1s = np.zeros(n)
#         all_holding_costs_0 = np.zeros(n)
#         all_holding_costs_1 = np.zeros(n)
#         profits_vars = np.zeros(n)

#         # 1-2. Arrays for demand calculation up to certain periods
#         total_demand_up_to_k_minus_1_vars = np.zeros(n)
#         total_demand_from_k_to_T_vars = np.zeros(n)
#         Q1_plus_lefts = np.zeros(n)

#         # 2. Variables for Model 2: Optimal Fraction Model
#         f_vars = np.zeros(n)
#         F_vars = np.zeros(n)  # Assuming values will be between 0 and 1
#         Q0_vars = np.zeros(n)  # Replace Q_star with a specific value as needed

#         # 3. Variables for Model 3: Optimal Order Time Model (2D array for binary values)
#         R_vars = np.zeros((n, K), dtype=int)  # Use dtype=int to represent binary 0/1 values

#         # 4. Variables for Model 4: Re-estimate order-up-to-level
#         Q1_vars = np.zeros(n)
#         Q_hats = np.zeros(n)
#         Q_hat_adjusteds = np.zeros(n)

#         # ======================= Start Stimulation! =======================

#         for i, row in demand_df_test.iterrows():

#             ### Data for this stimulation
#             demand_row = demand_df_test.iloc[i]
#             Qk_hat_df_test_row = Qk_hat_df_test.iloc[i]
#             X_data = testing_df.iloc[i].tolist()
#             X_data.append(1)

#             # =================== Model 1: Optimal Fraction Model ===================

#             ### 用線性回歸計算F_var
#             f_vars[i] = sum(X_data[j] * alphas[j] for j in range(self.features_num + 1))
#             F_vars[i] = 1 / (1 + np.exp(-(f_vars[i])))
#             print(f"f_vars[i]: {f_vars[i]}, F_vars[i]: {F_vars[i]}")
#             Q0_vars[i] = F_vars[i] * Q_star

#             # =================== Model 2: Optimal Order Time Model ===================

#             # Ensure only one `R` is set to 1 in each row by setting `assigned_R` to 1 and all others to 0
#             R_vars[i, assigned_R] = 1

#             # ============ Model 3: re-estimate order-up-to-level =================

#             Q_hats[i] = sum(
#                 R_vars[i, k - 2] * Qk_hat_df_test_row[k - 2] for k in range(2, T)
#             )
#             Q_hat_adjusteds[i] = Q_hats[i] - Q0_vars[i]
#             Q1_vars[i] = max(Q_hat_adjusteds[i], 0)

#             # =================== Model 4: Maximum Profit Model ===================

#             # Calculate the demand up to k-1
#             total_demand_up_to_k_minus_1 = sum(
#                 R_vars[i, k - 2] * demand_row[: k - 1].sum() for k in range(2, T)
#             )
#             total_demand_up_to_k_minus_1_vars[i] = total_demand_up_to_k_minus_1

#             # Calculate the demand from k to T
#             total_demand_from_k_to_T = sum(
#                 R_vars[i, k - 2] * demand_row[k - 1 :].sum() for k in range(2, T)
#             )
#             total_demand_from_k_to_T_vars[i] = total_demand_from_k_to_T

#             Sold_0s[i] = min(total_demand_up_to_k_minus_1_vars[i], Q0_vars[i])
#             Left_0s[i] = max(Q0_vars[i] - Sold_0s[i], 0)
#             Lost_0s[i] = max(total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i], 0)
#             Q1_plus_lefts[i] = Q1_vars[i] + Left_0s[i]

#             Sold_1s[i] = min(total_demand_from_k_to_T_vars[i], Q1_plus_lefts[i])
#             Left_1s[i] = max(Q1_plus_lefts[i] - Sold_1s[i], 0)
#             Lost_1s[i] = max(total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i], 0)

#             profits_vars[i] = (
#                 (price - cost) * (Sold_0s[i] + Sold_1s[i])  # Revenue from sales
#                 - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # Lost sales cost
#                 - (cost - salvage_value) * Left_1s[i]
#             )

#         results_df = pd.DataFrame(
#             {
#                 "average_profits": [np.mean(profits_vars)],
#                 "average_loss_penalty": [
#                     np.mean((price - cost) * (Lost_0s[i] + Lost_1s[i]))
#                 ],
#                 "average_left_penalty": [
#                     np.mean((cost - salvage_value) * (Left_0s[i] + Left_1s[i]))
#                 ],
#                 "average_loss": [np.mean((Lost_0s[i] + Lost_1s[i]))],
#                 "average_left": [np.mean((Left_0s[i] + Left_1s[i]))],
#                 "alpha_values": [alphas],
#                 "R(T)": assigned_R + 2,
#             }
#         )

#         stimulation_result = pd.DataFrame(
#             {
#                 "F": F_vars,
#                 "R(T)": assigned_R + 2,
#                 "Sold_0": Sold_0s,
#                 "Left_0": Left_0s,
#                 "Lost_0": Lost_0s,
#                 "Sold_1": Sold_1s,
#                 "Left_1": Left_1s,
#                 "Lost_1": Lost_1s,
#                 "profits": profits_vars,
#                 "Q0": Q0_vars,
#                 "Q1": Q1_vars,
#                 "hc0": all_holding_costs_0,
#                 "hc1": all_holding_costs_1,
#             }
#         )

#         return results_df, stimulation_result

#     def __cal_flexible_F_fixed_R(
#         self,
#         assigned_R,
#         salvage_value,
#         cost,
#         price,
#         Q_star,
#         demand_df_train,
#         Qk_hat_df,
#         training_df,
#         lambda_alpha,
#     ):
#         print(
#             f"+++++++++++++++++++++++++++++++++++++++ THis is R={assigned_R} +++++++++++++++++++++++++++++++++++++++++++++++++"
#         )
#         with gp.Model("profit_maximization", env=env) as model:
#             model.setParam("OutputFlag", True)
#             model.setParam("Threads", THREADS)
#             model.setParam("MIPGap", MIPGAP)
#             model.setParam("TimeLimit", TIME_LIMIT)
#             model.setParam("NonConvex", 2)
#             model.setParam("IntFeasTol", 1e-9)
#             model.setParam("NumericFocus", 3)

#             # ======================= Decision Variables =======================
#             alphas = model.addVars(
#                 self.features_num + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="alphas"
#             )
#             abs_alphas = model.addVars(alphas.keys(), lb=0, name="abs_alpha")

#             # 進行 L1 正則化處理：alphas
#             for i in alphas.keys():
#                 model.addConstr(abs_alphas[i] >= alphas[i])
#                 model.addConstr(abs_alphas[i] >= -alphas[i])

#             Sold_0s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_0")
#             Left_0s = model.addVars(len(demand_df_train), lb=0.0, name="Left_0")
#             Lost_0s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_0")

#             Sold_1s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_1")
#             Left_1s = model.addVars(len(demand_df_train), lb=0.0, name="Left_1")
#             Lost_1s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_1")

#             f_vars = model.addVars(
#                 len(demand_df_train), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="f_var"
#             )
#             F_vars = model.addVars(len(demand_df_train), lb=0, ub=1, name="Fraction")

#             Q0_vars = model.addVars(
#                 len(demand_df_train), lb=0.0, ub=(Q_star + 1), name="Q0_var"
#             )
#             Q1_vars = model.addVars(len(Qk_hat_df), lb=0.0, name="Q1_var")

#             Q1_plus_lefts = model.addVars(
#                 len(demand_df_train),
#                 lb=0,
#                 name=f"Q1_plus_left",
#             )  # k 之前的剩餘 + 新進貨的 Q1 量

#             profits_vars = model.addVars(
#                 len(demand_df_train), lb=-GRB.INFINITY, name="profits_vars"
#             )

#             # ======================= Model Constraints =======================
#             for i, row in demand_df_train.iterrows():
#                 demand_row = demand_df_train.iloc[i]
#                 Qk_hat_df_row = Qk_hat_df.iloc[i].tolist()
#                 X_data = training_df.iloc[i].tolist()
#                 X_data.append(1)

#                 model.addConstr(F_vars[i] >= 0, name=f"Fraction_lower_bound_{i}")
#                 model.addConstr(F_vars[i] <= 1, name=f"Fraction_upper_bound_{i}")

#                 # Calculate F using logistic regression
#                 model.addConstr(
#                     f_vars[i]
#                     == gp.quicksum(X_data[j] * alphas[j] for j in range(self.features_num + 1))
#                 )
#                 model.addGenConstrLogistic(
#                     xvar=f_vars[i], yvar=F_vars[i], options="FuncNonlinear=1"
#                 )

#                 # Calculate initial order quantity
#                 model.addConstr(Q0_vars[i] == F_vars[i] * Q_star)

#                 # Define demand variables for before and after reorder point
#                 total_demand_before_R = demand_row[: assigned_R + 1].sum()
#                 total_demand_after_R = demand_row[assigned_R + 1 :].sum()

#                 # 定義輔助變數
#                 Left_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
#                 Lost_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")
#                 Left_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
#                 Lost_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")

#                 # 計算 Sold_0，為 total_demand_up_to_k_minus_1_vars 和 Q0_vars 的最小值
#                 model.addGenConstrMin(
#                     Sold_0s[i],
#                     [total_demand_before_R, Q0_vars[i]],
#                     name=f"Constr_Sold_0_min_{i}",
#                 )

#                 # 計算 Left_0，為 max(Q0_vars[i] - Sold_0s[i], 0)
#                 model.addConstr(
#                     Left_0_aux == Q0_vars[i] - Sold_0s[i],
#                     name=f"Constr_Left_0_diff_aux_{i}",
#                 )
#                 model.addGenConstrMax(
#                     Left_0s[i], [Left_0_aux, 0], name=f"Constr_Left_0_max_{i}"
#                 )

#                 # 計算 Lost_0，為 max(total_demand_before_R - Q0_vars[i], 0)
#                 model.addConstr(
#                     Lost_0_aux == total_demand_before_R - Q0_vars[i],
#                     name=f"Constr_Lost_0_diff_aux_{i}",
#                 )
#                 model.addGenConstrMax(
#                     Lost_0s[i], [Lost_0_aux, 0], name=f"Constr_Lost_0_max_{i}"
#                 )

#                 # Calculate Q1 based on reorder point estimate
#                 Q_hat = Qk_hat_df_row[assigned_R]
#                 Q_hat_adjusted = Q_hat - Q0_vars[i]
#                 Q_hat_adjusted_var = model.addVar(
#                     lb=-GRB.INFINITY, name=f"Q_hat_adjusted_{i}"
#                 )
#                 model.addConstr(Q_hat_adjusted_var == Q_hat_adjusted)

#                 model.addGenConstrMax(
#                     Q1_vars[i], [Q_hat_adjusted_var, 0], name=f"max_Q1_constr_{i}"
#                 )

#                 # 計算 Q1 + left_0
#                 model.addConstr(
#                     Q1_plus_lefts[i] == Q1_vars[i] + Left_0s[i],
#                     name=f"Constr_Q1_plus_left_{i}",
#                 )

#                 # 計算 Sold_1，為 total_demand_from_k_to_T_vars 和 Q1_plus_lefts 的最小值
#                 model.addGenConstrMin(
#                     Sold_1s[i],
#                     [total_demand_after_R, Q1_plus_lefts[i]],
#                     name=f"Constr_Sold_1_min_{i}",
#                 )

#                 # 計算 Left_1，為 max(Q1_plus_lefts[i] - Sold_1s[i], 0)
#                 model.addConstr(
#                     Left_1_aux == Q1_plus_lefts[i] - Sold_1s[i],
#                     name=f"Constr_Left_1_diff_aux_{i}",
#                 )
#                 model.addGenConstrMax(
#                     Left_1s[i], [Left_1_aux, 0], name=f"Constr_Left_1_max_{i}"
#                 )

#                 # 計算 Lost_1，為 max(total_demand_after_R - Q1_plus_lefts[i], 0)
#                 model.addConstr(
#                     Lost_1_aux == total_demand_after_R - Q1_plus_lefts[i],
#                     name=f"Constr_Lost_1_diff_aux_{i}",
#                 )
#                 model.addGenConstrMax(
#                     Lost_1s[i], [Lost_1_aux, 0], name=f"Constr_Lost_1_max_{i}"
#                 )

#                 model.addConstr(
#                     profits_vars[i]
#                     == (
#                         (price - cost) * (Sold_0s[i] + Sold_1s[i])  # sold
#                         - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # lost sales
#                         - (cost - salvage_value) * (Left_1s[i])  # left cost
#                     ),
#                     name=f"Profit_Constraint_{i}",
#                 )

#             # Set objective
#             model.setObjective(
#                 gp.quicksum(profits_vars[i] for i in range(len(demand_df_train)))
#                 - lambda_alpha * gp.quicksum(abs_alphas[i] for i in abs_alphas.keys()),
#                 GRB.MAXIMIZE,
#             )

#             model.write("s17_model_debug.lp")
#             model.write("s17_model.mps")

#             # Solve model
#             try:
#                 model.optimize()

#                 if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
#                     print(f"Model status: {model.status}")

#                     # Collect results
#                     alpha_values = np.array([alpha.X for alpha in alphas.values()])

#                     results = {
#                         "losses": [],
#                         "lefts": [],
#                         "profits": [],
#                         "operation_profits": [],
#                         "Q0s": [],
#                         "Q1s": [],
#                         "Fs": [],
#                         "fs": [],
#                     }

#                     for i in range(len(demand_df_train)):
#                         sold0, sold1 = Sold_0s[i].X, Sold_1s[i].X
#                         lost0, lost1 = Lost_0s[i].X, Lost_1s[i].X
#                         left1 = Left_1s[i].X
#                         left0 = Left_0s[i].X
#                         print(
#                             f"Lost0: {lost0}, Lost1: {lost1}, Left0: {left0}, Left1: {left1}"
#                         )

#                         # Record results
#                         results["losses"].append(lost0 + lost1)

#                         # results["lefts"].append(left1)
#                         results["lefts"].append(left0 + left1)

#                         results["operation_profits"].append(
#                             (price - cost) * (sold0 + sold1)
#                         )
#                         results["profits"].append(profits_vars[i].X)
#                         results["Q1s"].append(Q1_vars[i].X)

#                         # Check f
#                         x_data = training_df.iloc[i].tolist()
#                         x_data.append(1)
#                         f_train, F_train, Q0_train = self.__compute_f_F_Q(
#                             x_data, alpha_values, Q_star
#                         )
#                         print(
#                             f"f_train: {f_train}, F_train: {F_train}, Q0_train: {Q0_train}"
#                         )
#                         if (
#                             self.__truncate_to_2(f_train) == self.__truncate_to_2(f_vars[i].X)
#                             and self.__truncate_to_2(F_train) == self.__truncate_to_2(F_vars[i].X)
#                             and self.__truncate_to_2(Q0_train) == self.__truncate_to_2(Q0_vars[i].X)
#                         ):
#                             print("f_train, F_train, Q0_train 都相等")
#                             results["Q0s"].append(Q0_vars[i].X)
#                             results["Fs"].append(F_vars[i].X)
#                             results["fs"].append(f_vars[i].X)
#                         else:
#                             print(f"f_train, F_train, Q0_train 不相等")
#                             results["Q0s"].append(-1)
#                             results["Fs"].append(-1)
#                             results["fs"].append(-1)
#                     return (
#                         [assigned_R] * len(demand_df_train),
#                         results["losses"],
#                         results["lefts"],
#                         results["profits"],
#                         results["operation_profits"],
#                         alpha_values,
#                         results["Fs"],
#                         results["fs"],
#                         results["Q0s"],
#                         results["Q1s"],
#                     )

#                 else:
#                     print("===================== 找不到最佳解 ==================")
#                     print(f"Model is feasible. Status: {model.status}")
#                     model.computeIIS()
#                     model.write("model.ilp")

#                     for constr in model.getConstrs():
#                         if constr.IISConstr:
#                             print(f"導致不可行的約束： {constr.constrName}")

#                     for var in model.getVars():
#                         if var.IISLB > 0 or var.IISUB > 0:
#                             print(
#                                 f"導致不可行的變量： {var.VarName}, IIS下界： {var.IISLB}, IIS上界： {var.IISUB}"
#                             )

#                     return None

#             except gp.GurobiError as e:
#                 print(f"Error code {str(e.errno)}: {str(e)}")
#                 return None


#     # 線性模型預測公式
#     def __compute_f_F_Q(self, X_data, alphas, Q_star):
#         f = sum(X_data[j] * alphas[j] for j in range(len(alphas)))
#         big_f = 1 / (1 + np.exp(-f))
#         q0 = big_f * Q_star
#         print(f"f_vars[i]: {f:.4f}, F_vars[i]: {big_f:.4f}, Q0_vars[i]: {q0:.4f}")
#         return f, big_f, q0

#     def __truncate_to_2(self, x):
#         return math.floor(x * 100) / 100
