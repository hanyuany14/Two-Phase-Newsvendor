# %%
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


params = {
    "WLSACCESSID": "73a6e3bf-2a9d-41e8-85eb-dd9b9eda802b",
    "WLSSECRET": "c394298a-96ea-4c8c-9d5e-ef2bd5032427",
    "LICENSEID": 2563044,
}

env = gp.Env(params=params)
model = gp.Model(env=env)

##########################################

salvage_value = 0
cost = 300
price = 1000
holding_cost = 0

# CHUNK_SIZE = 100
# data_size = CHUNK_SIZE * 3
# LASSO_BETA = 100

model_prefix = f"med_with_holding_cost_{holding_cost}"

##########################################

train_size = 0.5
testing_size = 0.5

T = 10
M = 5000000

ASSIGNED_FS = np.arange(0.1, 1.0, 0.1)
ASSIGNED_TS = list(range(2, T))  # 2 到 T-1

np.random.seed(0)

# Gurobi Model Constants
THREADS = 12
TIME_LIMIT = 20000
MIPGAP = 0.01
CURRENT_TIMESTAMP = int(datetime.now().strftime("%Y%m%d%H%M"))

##############################################################################################################################


def perform_single(data_size):
    
    env = gp.Env(params=params)

    ##########################################

    salvage_value = 0
    cost = 300
    price = 1000
    holding_cost = 0
    model_prefix = f"med_with_holding_cost_{holding_cost}"

    ##########################################

    train_size = 0.5
    testing_size = 0.5

    T = 10
    M = 5000000

    ASSIGNED_FS = np.arange(0.1, 1.0, 0.1)
    ASSIGNED_TS = list(range(2, T))  # 2 到 T-1

    np.random.seed(0)

    # Gurobi Model Constants
    THREADS = 12
    TIME_LIMIT = 20000
    MIPGAP = 0.01
    OUTPUTFLAG = False
    CURRENT_TIMESTAMP = int(datetime.now().strftime("%Y%m%d%H%M"))

    
    def save_model_parameters(
        name: str,
        alpha_values=None,
        beta_values=None,
        f_values=None,
        tau_values=None,
        data_size=data_size,
        current_timestamp=CURRENT_TIMESTAMP,
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

        # 如果有參數才進行保存
        if params:
            with open(f"models/{name}_{data_size}_{current_timestamp}.pkl", "wb") as f:
                pickle.dump(params, f)
            print(
                f"Model parameters saved as models/{name}_{data_size}_{current_timestamp}.pkl"
            )
        else:
            print("No parameters provided to save.")

    # %%
    def delete_model_parameters(name: str, data_size: int):
        # 構建檔案的路徑
        file_path = f"models/{name}_{data_size}_{CURRENT_TIMESTAMP}.pkl"

        # 檢查檔案是否存在
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Model parameters file '{file_path}' has been deleted.")
        else:
            print(f"File '{file_path}' does not exist.")

    # %%
    def show_models(model_prefix):
        file_paths = sorted(glob.glob(f"models/{model_prefix}_*.pkl"))

        # 逐一讀取並打印每個檔案的內容
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                params = pickle.load(f)
                print(f"Contents of {file_path}:")
                print(params)
                print()  # 空行分隔每個檔案的內容

    # %%
    def plot_strategies_profits_scatter(save_type, dfs: dict):
        names = list(dfs.keys())
        df_list = [dfs[name] for name in names]

        if len(df_list) <= 1:
            print("No dataframes to plot.")
            return

        pairs = list(itertools.combinations(range(len(df_list)), 2))
        num_pairs = len(pairs)
        grid_size = math.ceil(math.sqrt(num_pairs))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle("Scatter Plots of Profits (Matrix View)")

        for idx, (i, j) in enumerate(pairs):
            row, col = divmod(idx, grid_size)
            df_i, df_j = df_list[i], df_list[j]

            if df_i is None or df_j is None or df_i.empty or df_j.empty:
                continue
            if len(df_i) != len(df_j):
                continue

            ax = axes[row, col]
            ax.scatter(df_i["profits"], df_j["profits"], alpha=0.6)
            ax.plot(
                [
                    min(df_i["profits"].min(), df_j["profits"].min()),
                    max(df_i["profits"].max(), df_j["profits"].max()),
                ],
                [
                    min(df_i["profits"].min(), df_j["profits"].min()),
                    max(df_i["profits"].max(), df_j["profits"].max()),
                ],
                "k--",
                linewidth=1,
            )
            ax.set_xlabel(names[i])
            ax.set_ylabel(names[j])
            ax.set_title(f"{names[i]} vs {names[j]}")

        # Remove empty subplots
        for idx in range(num_pairs, grid_size * grid_size):
            row, col = divmod(idx, grid_size)
            fig.delaxes(axes[row, col])

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        os.makedirs("plots", exist_ok=True)
        save_path = f"plots/plot_strategies_profits_scatter_{save_type}.png"
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved as {save_path}")

        plt.show()
        plt.close()

    # %%
    def plot_relative_profit_deviation(save_type, baseline_profit, max_profits):
        """
        繪製多個策略相對於基準的平均利潤偏差。

        :param baseline_profit: 基準利潤值
        :param max_profits: 各策略的最大利潤列表，包含 None 值或 -1 表示無效數據
        """
        print(f"Baseline is: {baseline_profit}")
        for i, profit in enumerate(max_profits):
            print(f"S{i+1}'s profit: {profit}")

        # 計算相對值
        ratios = {}
        for idx, max_profit in enumerate(max_profits, start=1):
            if max_profit is not None and max_profit != -1:
                if baseline_profit != 0:
                    ratio = (max_profit - baseline_profit) / abs(baseline_profit)
                    ratios[f"S{idx}"] = ratio
                else:
                    # 基準利潤為零時，直接記錄增量
                    ratio = max_profit
                    ratios[f"S{idx}"] = ratio

        # 設置 y 軸範圍
        if ratios:
            y_min = min(ratios.values()) - 0.1
            y_max = max(ratios.values()) + 0.1
        else:
            y_min, y_max = -0.1, 0.1

        # 創建圖表顯示結果
        plt.figure(figsize=(12, 8))

        if ratios:
            bars = plt.bar(
                ratios.keys(), ratios.values(), color=plt.cm.tab10(range(len(ratios)))
            )

            # 在每個柱狀圖上標出數值
            for bar in bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval,
                    f"{yval:.4f}",
                    ha="center",
                    va="bottom",
                )

        # 添加基準線，表示基準值（No Opt）
        plt.axhline(y=0, color="gray", linestyle="--", label="Baseline (No Opt)")

        # 設置圖表標題和軸標籤
        plt.title("Relative Avg Profit Deviation from Baseline (1)")
        plt.xlabel("Strategies")
        plt.ylabel("Deviation from Baseline (1)")
        plt.ylim(y_min, y_max)
        plt.legend()

        name = "plot_relative_profit_deviation"

        os.makedirs("plots", exist_ok=True)
        save_path = f"plots/{name}_{save_type}_{data_size}_{CURRENT_TIMESTAMP}.png"

        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"Plot saved as {save_path}")

        # Show plot
        plt.show()
        plt.close()

    # %%
    def plot_relative_profit_comparison(
        save_type,
        train_baseline_profit,
        test_baseline_profit,
        test_max_profits,
        train_max_profits,
    ):

        # Calculate relative deviations from baseline for test and train data
        test_ratios, train_ratios = {}, {}
        for idx, (test_profit, train_profit) in enumerate(
            zip(test_max_profits, train_max_profits), start=1
        ):
            if test_profit is not None and test_profit != -1:
                if test_baseline_profit != 0:
                    test_ratio = (test_profit - test_baseline_profit) / abs(
                        test_baseline_profit
                    )  # Relative deviation
                else:
                    test_ratio = test_profit  # Use profit directly if baseline is zero
                test_ratios[f"S{idx}"] = test_ratio

            if train_profit is not None and train_profit != -1:
                if train_baseline_profit != 0:
                    train_ratio = (train_profit - train_baseline_profit) / abs(
                        train_baseline_profit
                    )  # Relative deviation
                else:
                    train_ratio = train_profit  # Use profit directly if baseline is zero
                train_ratios[f"S{idx}"] = train_ratio

        # Define the fixed range of the y-axis
        max_value = max(
            max(test_ratios.values(), default=0), max(train_ratios.values(), default=0)
        )
        y_max = min(max_value + 0.1, 1.0)  # Limit max y to 1.0
        y_min = -y_max  # Keep symmetric scaling

        # Ensure y-axis tick marks are at intervals of 0.05
        y_ticks = np.arange(y_min, y_max + 0.05, 0.05)  # Generate ticks

        # Create bar plot for relative profit deviation comparison
        plt.figure(figsize=(14, 8))
        bar_width = 0.35
        indices = np.arange(len(train_ratios))

        # Plot bars for train and test ratios, with train on the left for each pair
        train_bars = plt.bar(
            indices - bar_width / 2,
            train_ratios.values(),
            bar_width,
            label="Train Data",
            color="salmon",
        )
        test_bars = plt.bar(
            indices + bar_width / 2,
            test_ratios.values(),
            bar_width,
            label="Test Data",
            color="skyblue",
        )

        # Add baseline line
        plt.axhline(y=0, color="gray", linestyle="--", label="Baseline (No Opt)")

        # Add labels for each bar
        for bar in train_bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.2f}",
                ha="center",
                va="bottom",
            )
        for bar in test_bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.2f}",
                ha="center",
                va="bottom",
            )

        # Set plot labels and title
        plt.xlabel("Strategies")
        plt.ylabel("Deviation from Baseline")
        plt.title("Relative Profit Deviation Comparison between Train and Test Data")
        plt.xticks(indices, train_ratios.keys())

        # Set fixed y-axis range and ticks
        plt.ylim(y_min, y_max)
        plt.yticks(y_ticks)  # Apply fixed 0.05 intervals

        plt.legend()

        name = "plot_relative_profit_comparison"

        os.makedirs("plots", exist_ok=True)
        save_path = f"plots/{name}_{save_type}.png"

        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"Plot saved as {save_path}")

        # Show plot
        plt.show()
        plt.close()

    # %%
    def plot_Q0_Q1_distribution(save_type, stimulation_results_dfs):

        for idx, df in enumerate(stimulation_results_dfs, start=1):
            if df is None or len(df) == 0:
                continue

            df["Q0"] = pd.to_numeric(df["Q0"], errors="coerce")
            df["Q1"] = pd.to_numeric(df["Q1"], errors="coerce")
            df.dropna(subset=["Q0", "Q1"], inplace=True)

            plt.figure(figsize=(10, 6))
            plt.hist(df["Q0"], bins=20, alpha=0.6, label="Q0", edgecolor="black")
            plt.hist(df["Q1"], bins=20, alpha=0.6, label="Q1", edgecolor="black")
            plt.title(f"Histogram of Q0 and Q1 for stimulation_results_df_{idx}")
            plt.xlabel("Value")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True)

            name = "plot_Q0_Q1_distribution"

            os.makedirs("plots", exist_ok=True)
            save_path = (
                f"plots/{name}_{save_type}_{data_size}_S{idx}_{CURRENT_TIMESTAMP}.png"
            )

            plt.savefig(save_path, format="png", bbox_inches="tight")
            print(f"Plot saved as {save_path}")

            plt.show()

    # %%


    def plot_profits_deviation_box_plot(
        save_type, stimulation_results_dfs, baseline_avg_profits
    ):

        for idx, df in enumerate(stimulation_results_dfs, start=1):
            if df is not None and "profits" in df.columns:
                df["profits"] = pd.to_numeric(df["profits"], errors="coerce")
                df.dropna(subset=["profits"], inplace=True)

                # Calculate deviation
                df["Deviation"] = df["profits"] - baseline_avg_profits

                # Plot deviation as a boxplot
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=df["Deviation"])
                plt.axhline(0, color="red", linestyle="--", label="Baseline")
                plt.title(
                    f"Boxplot of Deviation of Profits from Baseline for stimulation_results_df_{idx}"
                )
                plt.ylabel("Deviation")
                plt.legend()
                plt.grid(True, axis="y")

                name = "plot_profits_deviation_box_plot"

                os.makedirs("plots", exist_ok=True)
                save_path = (
                    f"plots/{name}_{save_type}_{data_size}_S{idx}_{CURRENT_TIMESTAMP}.png"
                )

                plt.savefig(save_path, format="png", bbox_inches="tight")
                print(f"Plot saved as {save_path}")

                plt.show()
            else:
                print(f"Skipping stimulation_results_df_{idx}: Missing 'profits' column.")

    # %% [markdown]
    # ## Others

    # %%
    # Function to replace negative values with 0
    def replace_negative_with_zero(df):
        return df.applymap(lambda x: max(x, 0))

    # %%
    def check_values(
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

    # %%
    # Calculate service level
    def calculate_service_level(*, salvage_value, cost, price):

        cu = price - cost
        co = cost - salvage_value
        service_lv = cu / (co + cu)

        return service_lv

    # %%
    def make_s3_related_strtegies_result(
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
        holding_costs_0s,
        holding_costs_1s,
        all_left0s,
        all_left1s,
        all_lost0s,
        all_lost1s,
        gamma_values=None
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
            "gamma_values": [gamma_values],
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
            "hc0": holding_costs_0s,
            "hc1": holding_costs_1s,
            "Left0s": all_left0s,
            "Left1s": all_left1s,
            "lost0s": all_lost0s,
            "lost1s": all_lost1s,
        }

        return pd.DataFrame(results_dict).sort_values(
            by="average_profits", ascending=False
        ), pd.DataFrame(stimulations_result)

    # %% [markdown]
    # # Generate Data
    # 

    # %% [markdown]
    # ## Data1: Training data for LR
    # 

    # %% [markdown]
    # ### Making full data
    # 

    # %%
    np.random.seed(0)

    full_df = pd.DataFrame(
        {
            "X1": np.random.uniform(10, 15, data_size),
            "X2": np.random.uniform(-30, -10, data_size),
            # "X3": np.random.uniform(1000, 2000, data_size),
            "X3": np.random.uniform(50, 300, data_size),
            # "X3": np.random.uniform(50, 150, data_size),
            # "X3": np.random.uniform(50, 60, data_size),
            "X4": np.random.uniform(5, 15, data_size),
        }
    )

    features_num = full_df.shape[1]


    # %% [markdown]
    # ### Split training and testing data
    # 

    # %%


    def train_data_split_and_normalized(data, train_size=0.5):
        folds = []
        scalers = []

        # 計算訓練集的大小
        train_len = int(len(data) * train_size)

        # 將資料切分為前半部分為訓練集，後半部分為測試集
        train_data = data.iloc[:train_len].reset_index(drop=True)
        test_data = data.iloc[train_len:].reset_index(drop=True)

        # 標準化處理
        scaler = StandardScaler()
        train_data_normalized = scaler.fit_transform(train_data)
        test_data_normalized = scaler.transform(test_data)

        # 將標準化資料轉回 DataFrame
        train_data_normalized = pd.DataFrame(train_data_normalized, columns=data.columns)
        test_data_normalized = pd.DataFrame(test_data_normalized, columns=data.columns)

        # 將資料加入 folds 與 scaler
        folds.append((train_data_normalized, test_data_normalized))
        scalers.append(scaler)

        return folds, scalers


    training_data_folds, scalers = train_data_split_and_normalized(full_df, train_size)


    # %%


    def train_data_split_and_normalized_k_fold(data, train_size=0.5, chunk_size=CHUNK_SIZE):

        folds = []
        scalers = []
        train_chunk = int(train_size * chunk_size)
        n = len(data)

        # 依序將資料切分成 chunk_size 大小的子集
        for start in range(0, n, chunk_size):
            if start + chunk_size > n:
                break  # 若剩餘資料不足一個完整的 chunk，則跳過
            chunk = data.iloc[start : start + chunk_size].reset_index(drop=True)
            train_data = chunk.iloc[:train_chunk].reset_index(drop=True)
            test_data = chunk.iloc[train_chunk:].reset_index(drop=True)

            # 建立並使用 StandardScaler 分別標準化當前的訓練與測試資料
            scaler = StandardScaler()
            train_data_normalized = scaler.fit_transform(train_data)
            test_data_normalized = scaler.transform(test_data)

            # 轉回 DataFrame 格式
            train_data_normalized = pd.DataFrame(
                train_data_normalized, columns=data.columns
            )
            test_data_normalized = pd.DataFrame(test_data_normalized, columns=data.columns)

            folds.append((train_data_normalized, test_data_normalized))
            scalers.append(scaler)

        return folds, scalers


    training_data_folds, scalers = train_data_split_and_normalized_k_fold(full_df)


    # %% [markdown]
    # ## Data2: demand_df
    # 

    # %% [markdown]
    # ### mu of each time(t)
    # 

    # %%
    # 設定 b0, b1, b2

    # b0 = 0
    # b1 = 1
    # b2 = 2
    # b3 = -1
    # b4 = 2
    # bt = 0

    b0 = 0
    b1 = 0
    b2 = 0
    b3 = 1
    b4 = 0
    bt = 0

    # b0 = 0
    # b1 = 1
    # b2 = 1


    def cal_mu_matrix_with_random_noise(data_size, T, training_df, sigma_t):
        np.random.seed(0)

        # 初始化 mu_matrix
        mu_matrix = np.zeros((data_size, T))

        # 生成每個 t 的隨機數
        random_noises = np.random.normal(0, sigma_t, T)

        # 計算 mu_matrix
        for t in range(1, T + 1):
            mu_matrix[:, t - 1] = (
                b0 * random_noises[t - 1]
                + b1 * training_df["X1"]
                + b2 * training_df["X2"]
                + b3 * training_df["X3"]
                + b4 * training_df["X4"]
                + bt * t
            )

        return mu_matrix

    # %%
    mu_matrix = cal_mu_matrix_with_random_noise(data_size, T, full_df, sigma_t=1)

    # %% [markdown]
    # ### sigma matrix
    # 

    # %%
    X = full_df.values
    feature_num = X.shape[1]

    np.random.seed(0)

    c = np.random.uniform(0, 1)
    coefficients = np.random.uniform(-1, 1, (feature_num, T))  # shape: (feature_num, T)


    # %%

    linear_combination = c + X @ coefficients
    sigma_matrix = 1 / (1 + np.exp(-linear_combination))  # shape: (data_size, T)

    min_value = np.min(sigma_matrix)
    max_value = np.max(sigma_matrix)

    ####################################################################################
    # shape: (data_size, T)
    # sigma_matrix = 0 + sigma_matrix * 300

    # sigma_matrix = 0 + sigma_matrix * 200
    # sigma_matrix = 100 + sigma_matrix * 100
    sigma_matrix = 0 + sigma_matrix * 100
    # sigma_matrix = 50 + sigma_matrix * 50
    # sigma_matrix = 0 + sigma_matrix * 10

    # sigma_matrix = 0 + sigma_matrix * 80
    # sigma_matrix = 40 + sigma_matrix * 40
    # sigma_matrix = 0 + sigma_matrix * 40
    # sigma_matrix = 20 + sigma_matrix * 20
    # sigma_matrix = 0 + sigma_matrix * 5

    # sigma_matrix = 0 + sigma_matrix * 8
    # sigma_matrix = 4 + sigma_matrix * 4
    # sigma_matrix = 0 + sigma_matrix * 4
    # sigma_matrix = 2 + sigma_matrix * 2
    # sigma_matrix = 0 + sigma_matrix * 0.3
    ####################################################################################


    # 計算每個元素的最小值和最大值
    min_value = np.min(sigma_matrix)
    max_value = np.max(sigma_matrix)

    # 輸出 sigma_matrix 的形狀和內容
    sigma_matrix_shape = sigma_matrix.shape
    sigma_matrix_content = sigma_matrix


    # %% [markdown]
    # ### corr matrix
    # 

    # %%
    # Generate correlation matrix
    np.random.seed(0)

    A = np.random.uniform(-1, 1, (T, T))
    corr_matrix = np.dot(A, A.T)

    D = np.diag(1 / np.sqrt(np.diag(corr_matrix)))
    corr_matrix = D @ corr_matrix @ D


    # %% [markdown]
    # ### cov matrix
    # 

    # %%
    # Generate covariance matrices
    cov_matrices = []
    for i in range(data_size):
        cov_matrix = np.zeros((T, T))  # 每一個模擬都會有 T*T 的共變異矩陣
        for j in range(T):
            for k in range(T):
                cov_matrix[j, k] = (
                    corr_matrix[j, k] * sigma_matrix[i, j] * sigma_matrix[i, k]
                )
        cov_matrices.append(cov_matrix)


    # %%
    def is_positive_definite(matrix):
        return np.all(np.linalg.eigvals(matrix) > 0)

    positive_definite_check = all(is_positive_definite(cov) for cov in cov_matrices)
    print("All covariance matrices are positive definite:", positive_definite_check)

    # %% [markdown]
    # ### MVN stimulation for demand_df
    # 

    # %%
    def simulate_demand_data(data_size, T, cov_matrices, mu_matrix):
        np.random.seed(0)

        simulated_data = np.array(
            [
                np.random.multivariate_normal(mu_matrix[i], cov_matrices[i])
                for i in range(data_size)
            ]
        )

        demand_df = pd.DataFrame(
            simulated_data, columns=[f"demand_t{t}" for t in range(1, T + 1)]
        )
        return demand_df


    demand_df = simulate_demand_data(data_size, T, cov_matrices, mu_matrix)

    # %% [markdown]
    # ### Replace negative values to 0
    # 

    # %%
    demand_df = replace_negative_with_zero(demand_df)

    # %% [markdown]
    # ### Split test and train demand_df
    # 

    # %%
    def demand_data_split_data_k_fold(data):
        folds = []
        chunk_size = CHUNK_SIZE  # 每組 60 筆資料
        train_chunk = int(train_size * chunk_size)

        n = len(data)
        # 依序切分每一個 chunk
        for start in range(0, n, chunk_size):
            # 若剩餘資料不足 60 筆，這裡直接跳過
            if start + chunk_size > n:
                break
            chunk = data.iloc[start : start + chunk_size].reset_index(drop=True)
            train_data = chunk.iloc[:train_chunk].reset_index(drop=True)
            test_data = chunk.iloc[train_chunk:].reset_index(drop=True)
            folds.append((train_data, test_data))

        return folds

    demand_folds = demand_data_split_data_k_fold(demand_df)


    # %% [markdown]
    # ### Define the Q star(Q optimal)
    # 

    # %%
    def calculate_Q_star(demand_df, service_level=0.95):

        # 計算每一行的總和
        demand_sum = demand_df.sum(axis=1)

        # 計算總和的均值和標準差
        mean_sum = demand_sum.mean()
        std_sum = demand_sum.std()

        # 計算總和的95%百分位數值
        Q_star = norm.ppf(service_level, loc=mean_sum, scale=std_sum)

        # 打印結果
        print(f"mean of sum: {mean_sum}")
        print(f"std of sum: {std_sum}")
        print(f"{service_level*100} percentile of sum: {Q_star}")

        return Q_star

    # %% [markdown]
    # ## Data3: Qk hat df
    # 

    # %% [markdown]
    # ### Functions
    # 

    # %%
    # 計算條件分佈的函數
    def calculate_conditional_distribution(mu, covariance_matrix, x_observed, k):
        mu_1 = mu[:k]
        mu_2 = mu[k:]
        Sigma_11 = covariance_matrix[:k, :k]
        Sigma_22 = covariance_matrix[k:, k:]
        Sigma_12 = covariance_matrix[k:, :k]
        Sigma_21 = covariance_matrix[:k, k:]

        # Compute conditional mean and covariance
        Sigma_11_inv = np.linalg.pinv(Sigma_11)
        mu_cond = mu_2 + np.dot(Sigma_12, np.dot(Sigma_11_inv, (x_observed - mu_1)))
        sigma_cond = Sigma_22 - np.dot(Sigma_12, np.dot(Sigma_11_inv, Sigma_21))

        return mu_cond, sigma_cond

    # %%
    def cal_Var_Y(sigma_cond):

        # Extract the variances (diagonal elements)
        variances = np.diag(sigma_cond)

        # Calculate the sum of covariances (off-diagonal elements)
        covariances_sum = np.sum(sigma_cond) - np.sum(variances)

        # Total variance for the sum of mu_cond
        total_variance = np.sum(variances) + covariances_sum

        return total_variance

    # %%
    def cal_Qk_hat(mu_cond, sigma_cond, service_level, x_observed):
        # predict_quantity = mu_cond + norm.ppf(service_level) * np.sqrt(np.diag(sigma_cond))
        # Qk_hat = x_observed.sum() + predict_quantity.sum()

        mean_Y = np.sum(mu_cond)
        var_Y = cal_Var_Y(sigma_cond)

        sd_Y = np.sqrt(var_Y)
        if sd_Y < 0 or np.isnan(sd_Y):  # scale must be positive
            sd_Y = 1e-6

        percentile_95_Y = norm.ppf(service_level, loc=mean_Y, scale=sd_Y)

        # print(f"        mean_Y: {mean_Y}")
        # print(f"        sd_Y: {sd_Y}")
        # print(f"    percentile_95_Y: {percentile_95_Y}")

        Qk_hat = x_observed.sum() + percentile_95_Y
        return Qk_hat

    # %%
    def cal_mu_and_cov_matrix(demand_df_train):

        mu_matrix = demand_df_train.mean().values
        covariance_matrix = demand_df_train.cov().values

        # print(f"mu_matrix: {mu_matrix}")
        # print(f"covariance_matrix: \n{covariance_matrix}\n")

        return mu_matrix, covariance_matrix

    # %%
    def make_Qk_hat_df(demand_df, T, service_level, mu_matrix, covariance_matrix):
        results_df = pd.DataFrame(index=demand_df.index)

        for index, row_data in demand_df.iterrows():
            for k in range(2, T):
                # print(f"Now processing index: {index}, t={k}")

                x_observed = row_data[
                    : k - 1
                ].values  # 取出前 k 個觀測值 -> Qk_hat_2(t=2): 則 observerd: T=1

                mu_cond, sigma_cond = calculate_conditional_distribution(
                    mu_matrix, covariance_matrix, x_observed, len(x_observed)
                )

                Qk_hat = cal_Qk_hat(mu_cond, sigma_cond, service_level, x_observed)

                results_df.loc[index, f"Qk_hat_k{k}"] = Qk_hat

                # print(f"    x_observed: {x_observed}")
                # print(f"    mu_cond: {mu_cond}")
                # print(f"    sigma_cond: \n{sigma_cond}")
                # print(f"    Qk_hat: {Qk_hat}")
                # print("\n")

        return results_df

    # %% [markdown]
    # # Strategies utils
    # 

    # %% [markdown]
    # ## S0 - One-time Procurement
    # 

    # %%
    def one_time_procurement(Q_star, demand_df, cost, price, salvage_value):

        all_losses = []
        all_lefts = []
        all_operation_profits = []
        all_profits = []

        for i, row in demand_df.iterrows():
            inventory = Q_star
            losses = []
            lefts = []
            daily_operation_profits = []
            daily_profits = []
            total_sold = 0  # 追蹤總售出量
            total_lost = 0  # 追蹤總丟失量

            # print("=" * 50)
            # print(
            #     f"Processing row {i+1}/{len(demand_df)} with initial inventory Q_star={Q_star}"
            # )
            # print("=" * 50)

            for day, demand in enumerate(row):
                sales = min(inventory, demand)
                loss = max(demand - inventory, 0)
                left = max(inventory - sales, 0)
                total_sold += sales
                total_lost += loss

                inventory -= sales

                # print("-" * 50)
                # print(f"Day {day+1}")
                # print(f"Demand      : {demand}")
                # print(f"Sales       : {sales}")
                # print(f"Loss        : {loss}")
                # print(f"Left        : {left}")
                # print(f"Inventory   : {inventory}")
                # print("-" * 50)

                if day == len(row) - 1:
                    left_penalty_cost = (cost - salvage_value) * left
                    lefts.append(left)
                    # print(f"End of period: Left Penalty Cost = {left_penalty_cost}")
                    # print("-" * 50)
                else:
                    left_penalty_cost = 0

            operation_profit = (price - cost) * total_sold
            profit = operation_profit - left_penalty_cost - (price - cost) * total_lost

            # print("=" * 50)
            # print(f"Row {i+1} Summary")
            # print(f"Total Sold         : {total_sold}")
            # print(f"Total Lost         : {total_lost}")
            # print(f"Operation Profit   : {operation_profit}")
            # print(f"Profit             : {profit}")
            # print("=" * 50)

            all_losses.append(total_lost)
            all_lefts.append(sum(lefts))
            all_operation_profits.append(operation_profit)
            all_profits.append(profit)

        avg_losses = np.mean(all_losses)
        avg_lefts = np.mean(all_lefts)
        avg_operation_profits = np.mean(all_operation_profits)
        avg_profits = np.mean(all_profits)

        # print("=" * 50)
        # print("Overall Summary")
        # print(f"Average Losses           : {avg_losses}")
        # print(f"Average Lefts            : {avg_lefts}")
        # print(f"Average Operation Profits: {avg_operation_profits}")
        # print(f"Average Profits          : {avg_profits}")
        # print("=" * 50)

        stimulation_df = pd.DataFrame(
            {
                "losses": all_losses,
                "lefts": all_lefts,
                "operation_profits": all_operation_profits,
                "profits": all_profits,
            }
        )

        return avg_losses, avg_lefts, avg_profits, avg_operation_profits, stimulation_df

    # %% [markdown]
    # ## S1 - Grid for Fixed F & Fixed Rk
    # 

    # %%
    def cal_fixed_F_fixed_R(
        Q_star, assigned_F, assigned_R, demand_df, cost, price, salvage_value, Qk_hat_df
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

        # print(f"\n")
        # print(f"====" * 10)
        # print(f"\n")

        for i, row in demand_df.iterrows():

            # 第一階段計算
            total_sold_0 = min(Q0, row[: assigned_R + 1].sum())  # 第一階段售出量
            left_0 = max(Q0 - total_sold_0, 0)  # 第一階段剩餘
            lost_0 = max(row[: assigned_R + 1].sum() - Q0, 0)

            # 第二階段開始補貨，根據指定的 R
            Qk_hat = Qk_hat_df.iloc[i, assigned_R]
            Q1 = max((Qk_hat - Q0), 0)  # 二次訂貨量
            total_sold_1 = min(Q1 + left_0, row[assigned_R + 1 :].sum())  # 第二階段售出量
            left_1 = max((Q1 + left_0) - total_sold_1, 0)  # 第二階段剩餘
            lost_1 = max(row[assigned_R + 1 :].sum() - (Q1 + left_0), 0)

            # 統計
            total_sold = total_sold_0 + total_sold_1
            total_lost = lost_0 + lost_1

            # 計算運營利潤和總利潤
            operation_profit = (price - cost) * total_sold
            left_penalty_cost = (cost - salvage_value) * left_1
            lost_penalty_cost = (price - cost) * total_lost
            profit = operation_profit - left_penalty_cost - lost_penalty_cost

            all_losses.append(total_lost)
            all_lefts.append(left_1)
            all_operation_profits.append(operation_profit)
            all_profits.append(profit)
            all_q0s.append(Q0)
            all_q1s.append(Q1)
            all_left0s.append(left_0)
            all_left1s.append(left_1)

            # print(f"這是第 {i+1} 筆模擬資料\n")
            # print(f"F: {assigned_F}, R: {assigned_R+2}")
            # print(f"Q_star 為 {Q_star}")
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
            "Left0s": all_left0s,
            "Left1s": all_left1s,
            "operation_profits": all_operation_profits,
            "Q0": all_q0s,
            "Q1": all_q1s,
        }

        return result_df, stimulation_result

    # %%
    def grid_fixed_F_fixed_R(
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
                mean_result, stimulation_result = cal_fixed_F_fixed_R(
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
                    # print(
                    #     f"max_profit is changed from {max_profit} to {mean_result['average_profits']}"
                    # )
                    max_profit = mean_result["average_profits"]
                    max_profit_stimulation_result = stimulation_result

        results_df_1 = pd.DataFrame(results_list).sort_values(
            by="average_profits", ascending=False
        )

        return results_df_1, pd.DataFrame(max_profit_stimulation_result)

    # %% [markdown]
    # ## S2 - Grid for Fixed Rk & Flexible F
    # 

    # %%
    def cal_flexible_F_fixed_R(
        assigned_R,
        salvage_value,
        cost,
        price,
        Q_star,
        demand_df_train,
        Qk_hat_df,
        training_df,
    ):
        # print(
        #     f"+++++++++++++++++++++++++++++++++++++++ THis is R={assigned_R} +++++++++++++++++++++++++++++++++++++++++++++++++"
        # )
        with gp.Model("profit_maximization", env=env) as model:
            model.setParam("OutputFlag", OUTPUTFLAG)
            model.setParam("Threads", THREADS)
            model.setParam("MIPGap", MIPGAP)
            model.setParam("TimeLimit", TIME_LIMIT)

            # ======================= Decision Variables =======================
            alphas = model.addVars(
                features_num + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="alphas"
            )
            Sold_0s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_0")
            Sold_1s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_1")
            Lost_0s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_0")
            Lost_1s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_1")
            Left_1s = model.addVars(len(demand_df_train), lb=0.0, name="Left_1")

            f_vars = model.addVars(
                len(demand_df_train), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="f_var"
            )
            F_vars = model.addVars(len(demand_df_train), lb=0, ub=1, name="Fraction")

            Q0_vars = model.addVars(
                len(demand_df_train), lb=0.0, ub=(Q_star + 1), name="Q0_var"
            )
            Q1_vars = model.addVars(len(Qk_hat_df), lb=0.0, name="Q1_var")

            profits_vars = model.addVars(
                len(demand_df_train), lb=-GRB.INFINITY, name="profits_vars"
            )

            # ======================= Model Constraints =======================
            for i, row in demand_df_train.iterrows():
                demand_row = demand_df_train.iloc[i]
                Qk_hat_df_row = Qk_hat_df.iloc[i].tolist()
                X_data = training_df.iloc[i].tolist()
                X_data.append(1)

                model.addConstr(F_vars[i] >= 0, name=f"Fraction_lower_bound_{i}")
                model.addConstr(F_vars[i] <= 1, name=f"Fraction_upper_bound_{i}")

                # Calculate F using logistic regression
                model.addConstr(
                    f_vars[i]
                    == gp.quicksum(X_data[j] * alphas[j] for j in range(features_num + 1))
                )
                model.addGenConstrLogistic(xvar=f_vars[i], yvar=F_vars[i])

                # Calculate initial order quantity
                model.addConstr(Q0_vars[i] == F_vars[i] * Q_star)

                # Define demand variables for before and after reorder point
                total_demand_before_R = demand_row[: assigned_R + 1].sum()
                total_demand_after_R = demand_row[assigned_R + 1 :].sum()

                # Calculate first period sales and lost sales
                model.addGenConstrMin(
                    Sold_0s[i],
                    [total_demand_before_R, Q0_vars[i]],
                    name=f"min_sales_constr_{i}",
                )

                # Calculate lost sales
                Lost_0_expr = total_demand_before_R - Q0_vars[i]
                Lost_0_var = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_expr_{i}")
                model.addConstr(Lost_0_var == Lost_0_expr)
                model.addGenConstrMax(
                    Lost_0s[i], [Lost_0_var, 0], name=f"max_lost_constr_{i}"
                )

                # Calculate inventory left after first period
                left_0 = Q0_vars[i] - Sold_0s[i]

                # Calculate Q1 based on reorder point estimate
                Q_hat = Qk_hat_df_row[assigned_R]
                Q_hat_adjusted = Q_hat - Q0_vars[i]
                Q_hat_adjusted_var = model.addVar(
                    lb=-GRB.INFINITY, name=f"Q_hat_adjusted_{i}"
                )
                model.addConstr(Q_hat_adjusted_var == Q_hat_adjusted)

                model.addGenConstrMax(
                    Q1_vars[i], [Q_hat_adjusted_var, 0], name=f"max_Q1_constr_{i}"
                )

                # Calculate second period sales and lost sales
                total_stock_second_period = Q1_vars[i] + left_0
                total_stock_second_period_var = model.addVar(
                    lb=0, name=f"total_stock_second_period_{i}"
                )
                model.addConstr(total_stock_second_period_var == total_stock_second_period)

                model.addGenConstrMin(
                    Sold_1s[i],
                    [total_demand_after_R, total_stock_second_period_var],
                    name=f"min_sales2_constr_{i}",
                )

                # Calculate second period lost sales
                Lost_1_expr = total_demand_after_R - total_stock_second_period_var
                Lost_1_var = model.addVar(lb=-GRB.INFINITY, name=f"Lost_1_expr_{i}")
                model.addConstr(Lost_1_var == Lost_1_expr)

                model.addGenConstrMax(
                    Lost_1s[i], [Lost_1_var, 0], name=f"max_lost2_constr_{i}"
                )

                model.addConstr(Left_1s[i] == total_stock_second_period_var - Sold_1s[i])

                # # Calculate holding costs directly in profit equation
                # holding_cost_1 = (
                #     (Q0_vars[i] + total_stock_second_period) * (assigned_R + 2 - 1) / 2
                # )
                # holding_cost_2 = (
                #     (total_stock_second_period + Left_1s[i]) * (T - (assigned_R + 2)) / 2
                # )

                # Calculate profit
                model.addConstr(
                    profits_vars[i]
                    == (
                        (price - cost) * (Sold_0s[i] + Sold_1s[i])  # Revenue
                        - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # Lost sales cost
                        - (cost - salvage_value) * Left_1s[i]  # Salvage cost
                        # - holding_cost * (holding_cost_1 + holding_cost_2)  # Holding cost
                    )
                )

            # Set objective
            model.setObjective(
                gp.quicksum(profits_vars[i] for i in range(len(demand_df_train))),
                GRB.MAXIMIZE,
            )

            model.write("s2_model_debug.lp")
            model.write("s2_model.mps")

            # Solve model
            try:
                model.optimize()

                if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                    # print(f"Model status: {model.status}")

                    # Collect results
                    alpha_values = np.array([alpha.X for alpha in alphas.values()])

                    results = {
                        "losses": [],
                        "lefts": [],
                        "profits": [],
                        "operation_profits": [],
                        "Q0s": [],
                        "Q1s": [],
                        "Fs": [],
                    }

                    for i in range(len(demand_df_train)):
                        sold0, sold1 = Sold_0s[i].X, Sold_1s[i].X
                        lost0, lost1 = Lost_0s[i].X, Lost_1s[i].X
                        left1 = Left_1s[i].X

                        # Record results
                        results["losses"].append(lost0 + lost1)
                        results["lefts"].append(left1)
                        results["operation_profits"].append(
                            (price - cost) * (sold0 + sold1)
                        )
                        results["profits"].append(profits_vars[i].X)
                        results["Q0s"].append(Q0_vars[i].X)
                        results["Q1s"].append(Q1_vars[i].X)
                        results["Fs"].append(F_vars[i].X)

                        # print(f"\nObservation {i+1}:")
                        # print(f"Reorder day: {assigned_R}")
                        # print(f"Profit: {profits_vars[i].X:.2f}")

                    return (
                        [assigned_R] * len(demand_df_train),  # Fixed R for all observations
                        results["losses"],
                        results["lefts"],
                        results["profits"],
                        results["operation_profits"],
                        alpha_values,
                        results["Fs"],
                        results["Q0s"],
                        results["Q1s"],
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

    # %%
    def grid_flexible_F_fixed_R(
        assigned_Ts,
        salvage_value,
        cost,
        price,
        Q_star,
        demand_df_train,
        Qk_hat_df_train,
        training_df,
    ):
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

        max_profit = None
        max_profit_stimulation_result = {}

        for assigned_T in assigned_Ts:
            assigned_R = assigned_T - 2
            result = cal_flexible_F_fixed_R(
                assigned_R=assigned_R,
                salvage_value=salvage_value,
                cost=cost,
                price=price,
                Q_star=Q_star,
                demand_df_train=demand_df_train,
                Qk_hat_df=Qk_hat_df_train,
                training_df=training_df,
            )

            if result is None:
                print(f"模型沒有最佳解")
                continue

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

            # 計算平均值
            average_losses = sum(losses) / len(losses) if losses else 0
            average_lefts = sum(lefts) / len(lefts) if lefts else 0
            average_profits = sum(profits) / len(profits) if profits else 0
            average_operation_profits = (
                sum(operation_profits) / len(operation_profits) if operation_profits else 0
            )

            # 將結果存儲到字典中
            results_dict["R(T)"].append(assigned_T)
            results_dict["R"].append(all_Rs)
            results_dict["average_losses"].append(average_losses)
            results_dict["average_lefts"].append(average_lefts)
            results_dict["average_profits"].append(average_profits)
            results_dict["average_operation_profits"].append(average_operation_profits)
            results_dict["alpha_values"].append(alpha_values)
            results_dict["F_vars"].append(F_vars)
            results_dict["Q0_vars"].append(Q0_vars)
            results_dict["Q1_vars"].append(Q1_vars)

            # print(f"The average profits is {average_profits}")

            if max_profit is None or max_profit < average_profits:
                # print(f"max_profit is changed from {max_profit} to {average_profits}")
                max_profit = average_profits
                max_profit_stimulation_result = {
                    "R": all_Rs,
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

    # %% [markdown]
    # ## Fully flexible F & Rk
    # 

    # %%
    def __fully_flexible_beta_with_softmax_4(
        salvage_value, cost, price, Q_star, demand_df_train, Qk_hat_df, training_df
    ):

        with gp.Model("profit_maximization", env=env) as model:

            model.setParam("OutputFlag", OUTPUTFLAG)
            model.setParam("Threads", THREADS)
            model.setParam("MIPGap", MIPGAP)
            model.setParam("TimeLimit", TIME_LIMIT)
            model.setParam("IntFeasTol", 1e-9)

            # ======================= Global Variables =======================

            # Category 1 - Some variables that is important to future work
            K = T - 2  # this is for k=2~T-1. => if T = 10(1~10), K will be 8. (0~7)

            alphas = model.addVars(features_num + 1, lb=-GRB.INFINITY, name="alphas")
            betas = model.addVars(K, features_num + 1, lb=-GRB.INFINITY, name="betas")

            # Category 2 - Variables about this stimulation
            ### 1. Variables for Model 1: Maximum Profit Model
            Sold_0s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_0")
            Left_0s = model.addVars(len(demand_df_train), lb=0.0, name="Left_0")
            Lost_0s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_0")

            Sold_1s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_1")
            Left_1s = model.addVars(len(demand_df_train), lb=0.0, name="Left_1")
            Lost_1s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_1")

            Holding_Cost_0s = model.addVars(
                len(demand_df_train), lb=0.0, name="Holding_Cost_0"
            )

            Holding_Cost_1s = model.addVars(
                len(demand_df_train), lb=0.0, name="Holding_Cost_1"
            )

            profits_vars = model.addVars(
                len(demand_df_train), lb=-GRB.INFINITY, name="profits_vars"
            )

            #### 1-2. 用於計算 k 時期之前與之後的需求量
            total_demand_up_to_k_minus_1_vars = model.addVars(
                len(demand_df_train),
                lb=0,
                name="Total_Demand_Up_to_K_minus_1",
            )
            total_demand_from_k_to_T_vars = model.addVars(
                len(demand_df_train), lb=0, name="Total_Demand_from_k_to_T"
            )
            Q1_plus_lefts = model.addVars(
                len(demand_df_train),
                lb=0,
                name=f"Q1_plus_left",
            )  # k 之前的剩餘 + 新進貨的 Q1 量

            ### 2. Variables for Model 2: Optimal Fraction Model
            f_vars = model.addVars(len(demand_df_train), lb=-GRB.INFINITY, name="f_var")
            F_vars = model.addVars(
                len(demand_df_train), lb=0, ub=1, name="Fraction_for_second_order_amount"
            )
            Q0_vars = model.addVars(
                len(demand_df_train), lb=0, ub=(Q_star + 1), name="Q0_var"
            )

            ### 3. Variables for Model 3: Optimal Order Time Model
            # tau_vars = model.addVars(len(demand_df_train), K, lb=-GRB.INFINITY, name="tau")
            tau_vars = model.addVars(len(demand_df_train), K, lb=-GRB.INFINITY, name="tau")
            r_vars = model.addVars(len(demand_df_train), K, lb=0.0, ub=1.0, name="r")
            R_vars = model.addVars(len(demand_df_train), K, vtype=GRB.BINARY, name="R")

            ### 4. Variables for Model 4: re-estimate order-up-to-level
            Q1_vars = model.addVars(len(Qk_hat_df), lb=0.0, name="Q1_var")
            Q_hats = model.addVars(
                len(Qk_hat_df),
                lb=0.0,
                name="Q_hat",
            )
            Q_hat_adjusteds = model.addVars(
                len(Qk_hat_df), lb=-GRB.INFINITY, name=f"Q_hat_adjusted"
            )

            # ======================= Start Stimulation! =======================

            for i, _ in demand_df_train.iterrows():

                ### Data for this stimulation
                demand_row = demand_df_train.iloc[i]
                Qk_hat_df_row = Qk_hat_df.iloc[i]
                X_data = training_df.iloc[i].tolist()
                X_data.append(1)

                # =================== Model 1: Optimal Fraction Model ===================

                ### 用線性回歸計算F_var
                model.addConstr(
                    f_vars[i]
                    == gp.quicksum(X_data[j] * alphas[j] for j in range(features_num + 1))
                )
                model.addGenConstrLogistic(
                    xvar=f_vars[i], yvar=F_vars[i], name=f"logistic_constraint_{i}"
                )
                model.addConstr(Q0_vars[i] == F_vars[i] * Q_star, f"Q0_upper_bound_{i}")

                # =================== Model 2: Optimal Order Time Model ===================

                # 用線性回歸計算確定最佳補貨時間
                exp_tau_vars = []
                for k in range(K):

                    # 計算 tau_vars 作為 beta 和特徵的線性組合
                    model.addConstr(
                        tau_vars[i, k]
                        == gp.quicksum(
                            X_data[j] * betas[k, j] for j in range(features_num + 1)
                        ),
                        name=f"tau_computation_{i}_{k}",
                    )
                    # model.addConstr(tau_vars[i, k] >= -5, name=f"tau_lb_{i}_{k}")

                    exp_tau_var = model.addVar(lb=1e-6, name=f"exp_tau_var_{i}_{k}")
                    exp_tau_var = model.addVar(
                        lb=-GRB.INFINITY, name=f"exp_tau_var_{i}_{k}"
                    )
                    model.addGenConstrExp(xvar=tau_vars[i, k], yvar=exp_tau_var)
                    exp_tau_vars.append(exp_tau_var)

                ### 找到最大 R 以及 r, R 的相關限制式 -> k: 0~7
                for k in range(K):
                    model.addConstr(
                        r_vars[i, k] * gp.quicksum(exp_tau_vars) == exp_tau_vars[k],
                        name=f"softmax_{i}_{k}",
                    )

                model.addConstr(
                    gp.quicksum(r_vars[i, k] for k in range(K)) == 1,
                    name=f"sum_r_{i}",
                )

                max_r_helpers = model.addVar(lb=0.0, ub=1.0, name="max_r_helper")
                model.addGenConstrMax(
                    max_r_helpers,
                    [r_vars[i, k] for k in range(K)],
                    name=f"MaxRConstraint_{i}",
                )

                # 確保 R_vars 的邏輯行為
                for k in range(K):
                    model.addGenConstrIndicator(
                        R_vars[i, k], True, r_vars[i, k] == max_r_helpers
                    )

                model.addConstr(
                    gp.quicksum(R_vars[i, k] for k in range(K)) == 1,
                    name=f"Ensure_only_one_R_true_{i}",
                )

                # ============ Model 3: re-estimate order-up-to-level =================

                ### 計算 Q_hat -> k: 2~9 -> k-2: 0~7
                model.addConstr(
                    Q_hats[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * Qk_hat_df_row[k - 2] for k in range(2, T)
                    ),
                    name=f"Define_Q_hat_{i}",
                )
                model.addConstr(
                    Q_hat_adjusteds[i] == Q_hats[i] - Q0_vars[i], name=f"Adjust_Q_hat_{i}"
                )
                model.addConstr(
                    Q1_vars[i] == max_(Q_hat_adjusteds[i], 0),
                    name=f"Max_Constraint_{i}",
                )

                # =================== Model 4: Maximum Profit Model ===================

                # ### 0~k-1 的需求量
                model.addConstr(
                    total_demand_up_to_k_minus_1_vars[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * demand_row[: k - 1].sum() for k in range(2, T)
                    ),
                    name=f"Constr_Total_Demand_Up_to_K_Minus_1_{i}",
                )

                # ### k~T 的需求量
                model.addConstr(
                    total_demand_from_k_to_T_vars[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * demand_row[k - 1 :].sum() for k in range(2, T)
                    ),
                    name=f"Constr_Total_Demand_from_K_to_T_{i}",
                )

                # 定義輔助變數
                Left_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
                Lost_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")
                Left_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
                Lost_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")

                # 計算 Sold_0，為 total_demand_up_to_k_minus_1_vars 和 Q0_vars 的最小值
                model.addGenConstrMin(
                    Sold_0s[i],
                    [total_demand_up_to_k_minus_1_vars[i], Q0_vars[i]],
                    name=f"Constr_Sold_0_min_{i}",
                )

                # 計算 Left_0，為 max(Q0_vars[i] - Sold_0s[i], 0)
                model.addConstr(
                    Left_0_aux == Q0_vars[i] - Sold_0s[i],
                    name=f"Constr_Left_0_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Left_0s[i], [Left_0_aux, 0], name=f"Constr_Left_0_max_{i}"
                )

                # 計算 Lost_0，為 max(total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i], 0)
                model.addConstr(
                    Lost_0_aux == total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i],
                    name=f"Constr_Lost_0_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Lost_0s[i], [Lost_0_aux, 0], name=f"Constr_Lost_0_max_{i}"
                )

                # 計算 Q1 + left_0
                model.addConstr(
                    Q1_plus_lefts[i] == Q1_vars[i] + Left_0s[i],
                    name=f"Constr_Q1_plus_left_{i}",
                )

                # 計算 Sold_1，為 total_demand_from_k_to_T_vars 和 Q1_plus_lefts 的最小值
                model.addGenConstrMin(
                    Sold_1s[i],
                    [total_demand_from_k_to_T_vars[i], Q1_plus_lefts[i]],
                    name=f"Constr_Sold_1_min_{i}",
                )

                # 計算 Left_1，為 max(Q1_plus_lefts[i] - Sold_1s[i], 0)
                model.addConstr(
                    Left_1_aux == Q1_plus_lefts[i] - Sold_1s[i],
                    name=f"Constr_Left_1_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Left_1s[i], [Left_1_aux, 0], name=f"Constr_Left_1_max_{i}"
                )

                # 計算 Lost_1，為 max(total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i], 0)
                model.addConstr(
                    Lost_1_aux == total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i],
                    name=f"Constr_Lost_1_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Lost_1s[i], [Lost_1_aux, 0], name=f"Constr_Lost_1_max_{i}"
                )

                model.addConstr(
                    profits_vars[i]
                    == (
                        (price - cost) * (Sold_0s[i] + Sold_1s[i])  # sold
                        - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # lost sales
                        - (cost - salvage_value) * Left_1s[i]  # left cost
                    ),
                    name=f"Profit_Constraint_{i}",
                )

            #  ======================================= Model optimize =======================================

            model.setObjective(
                gp.quicksum(profits_vars[i] for i in range(len(demand_df_train))),
                GRB.MAXIMIZE,
            )
            model.write("s4_model_debug.lp")
            model.write("s4_model.mps")
            try:
                model.optimize()

                if model.status == GRB.OPTIMAL:
                    # print(f"\nmodel.status is optimal: {model.status == GRB.OPTIMAL}")
                    # print(f"model.status is TIME_LIMIT: {model.status == GRB.TIME_LIMIT}\n")

                    # print("===================== 找到最佳解 ==================")
                    # print(f"Q0_optimal（最佳總庫存量）: {Q_star}")

                    # print("Alphas values:")
                    # for key, alpha in alphas.items():
                    #     print(f"alpha[{key}]: {alpha.X}")

                    alpha_values = np.array([alpha.X for _, alpha in alphas.items()])
                    beta_values = np.array(
                        [[betas[k, j].X for j in range(features_num + 1)] for k in range(K)]
                    )
                    # print(f"beta_values:\n{beta_values}")

                    f_values = np.array([f.X for _, f in f_vars.items()])
                    tau_values = np.array(
                        [
                            [tau_vars[i, j].X for j in range(K)]
                            for i in range(len(demand_df_train))
                        ]
                    )

                    # print(f"------------")
                    # print(f"f_values:\n{f_values}")
                    # print(f"tau_values:\n{tau_values}")

                    all_losses = []
                    all_lefts = []
                    all_operation_profits = []
                    all_profits = []
                    all_rs = []
                    all_Rs = []
                    all_Q0s = []
                    all_Q1s = []
                    all_Fs = []
                    all_holding_costs_0 = []
                    all_holding_costs_1 = []
                    all_left0s = []
                    all_left1s = []
                    all_lost0s = []
                    all_lost1s = []

                    for i in range(len(demand_df_train)):

                        # print("----------------------------------------------")
                        # print(f"第 {i+1} 筆觀察資料:")

                        sold0 = Sold_0s[i].X
                        sold1 = Sold_1s[i].X
                        left0 = Left_0s[i].X
                        left1 = Left_1s[i].X
                        lost0 = Lost_0s[i].X
                        lost1 = Lost_1s[i].X
                        Holding_Cost_0 = Holding_Cost_0s[i].X
                        Holding_Cost_1 = Holding_Cost_1s[i].X

                        operation_profit = (price - cost) * (sold0 + sold1)
                        daily_profit = profits_vars[i].X

                        all_losses.append(lost0 + lost1)
                        all_lefts.append(left0 + left1)
                        all_operation_profits.append(operation_profit)
                        all_profits.append(daily_profit)
                        all_Q0s.append(Q0_vars[i].X)
                        all_Q1s.append(Q1_vars[i].X)
                        all_Fs.append(F_vars[i].X)
                        all_holding_costs_0.append(Holding_Cost_0)
                        all_holding_costs_1.append(Holding_Cost_1)
                        all_left0s.append(left0)
                        all_left1s.append(left1)
                        all_lost0s.append(lost0)
                        all_lost1s.append(lost1)

                        reorder_day = None
                        rs = []
                        for k in range(K):
                            rs.append(r_vars[i, k].X)
                            R_value = R_vars[i, k].X
                            # print(
                            #     f"第 {k+2} 天補貨策略: R_vars = {R_value}, r_vars = {rs[k]}"
                            # )
                            # print(
                            #     f"第 {k+2} 天補貨策略: R_vars = {R_value}, tau_vars = {tau_vars[i, k].X}"
                            # )

                            if int(R_value) == 1:
                                reorder_day = k + 2
                        # print(f"*** 於第[{reorder_day}]天進貨 ***\n")

                        all_Rs.append(reorder_day)
                        all_rs.append(rs)

                        demand_row = demand_df_train.iloc[i]

                        total_demand_up = total_demand_up_to_k_minus_1_vars[i].X
                        total_demand_down = total_demand_from_k_to_T_vars[i].X

                        check_results_df = check_values(
                            Q1_vars=Q1_vars,
                            Q_hat_adjusteds=Q_hat_adjusteds,
                            Q0_vars=Q0_vars,
                            Sold_0s=Sold_0s,
                            total_demand_up_to_k_minus_1_vars=total_demand_up_to_k_minus_1_vars,
                            Sold_1s=Sold_1s,
                            total_demand_from_k_to_T_vars=total_demand_from_k_to_T_vars,
                            Q1_plus_lefts=Q1_plus_lefts,
                            Left_0s=Left_0s,
                            Lost_0s=Lost_0s,
                            Left_1s=Left_1s,
                            Lost_1s=Lost_1s,
                        )
                        # print(check_results_df)

                        # for t in range(2):
                        #     if t == 0:
                        #         print(
                        #             f"  第 {t+1} 階段: 本階段期初庫存 = {Q0_vars[i].X}, 第一階段總需求 = {total_demand_up}, 銷售量 = {Sold_0s[i].X}, 本階段期末剩餘庫存 = {Left_0s[i].X}, 本期損失 = {Lost_0s[i].X}, 本期 holding cost = {Holding_Cost_0}"
                        #         )
                        #     else:
                        #         print(
                        #             f"  第 {t+1} 階段: 本階段期初庫存 = {Q1_plus_lefts[i].X}, 重新預估需求 = {Q_hats[i].X}, 第二階段總需求 = {total_demand_down}, 銷售量 = {Sold_1s[i].X}, 本階段期末剩餘庫存 = {Left_1s[i].X}, 本期損失 = {Lost_1s[i].X}, 本期 holding cost = {Holding_Cost_1}"
                        #         )

                        # print(f"  本觀察資料總利潤 = {daily_profit}\n")

                    # print("==========================================")
                    # print(f"最佳化模型平均利潤 = {np.mean(all_profits)}")

                    return (
                        all_Rs,
                        all_losses,
                        all_lefts,
                        all_profits,
                        all_operation_profits,
                        alpha_values,
                        beta_values,
                        all_Fs,
                        all_Q0s,
                        all_Q1s,
                        f_values,
                        tau_values,
                        all_holding_costs_0,
                        all_holding_costs_1,
                        all_left0s,
                        all_left1s,
                        all_lost0s,
                        all_lost1s,
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

    # %% [markdown]
    # ### S12 - Beta without r

    # %%
    def __fully_flexible_beta_with_softmax_12(
        salvage_value, cost, price, Q_star, demand_df_train, Qk_hat_df, training_df
    ):

        with gp.Model("profit_maximization", env=env) as model:

            model.setParam("OutputFlag", OUTPUTFLAG)
            model.setParam("Threads", THREADS)
            model.setParam("MIPGap", MIPGAP)
            model.setParam("TimeLimit", TIME_LIMIT)
            model.setParam("IntFeasTol", 1e-9)

            # ======================= Global Variables =======================

            # Category 1 - Some variables that is important to future work
            K = T - 2  # this is for k=2~T-1. => if T = 10(1~10), K will be 8. (0~7)

            alphas = model.addVars(features_num + 1, lb=-GRB.INFINITY, name="alphas")
            betas = model.addVars(K, features_num + 1, lb=-GRB.INFINITY, name="betas")

            # Category 2 - Variables about this stimulation
            ### 1. Variables for Model 1: Maximum Profit Model
            Sold_0s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_0")
            Left_0s = model.addVars(len(demand_df_train), lb=0.0, name="Left_0")
            Lost_0s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_0")

            Sold_1s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_1")
            Left_1s = model.addVars(len(demand_df_train), lb=0.0, name="Left_1")
            Lost_1s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_1")

            Holding_Cost_0s = model.addVars(
                len(demand_df_train), lb=0.0, name="Holding_Cost_0"
            )

            Holding_Cost_1s = model.addVars(
                len(demand_df_train), lb=0.0, name="Holding_Cost_1"
            )

            profits_vars = model.addVars(
                len(demand_df_train), lb=-GRB.INFINITY, name="profits_vars"
            )

            #### 1-2. 用於計算 k 時期之前與之後的需求量
            total_demand_up_to_k_minus_1_vars = model.addVars(
                len(demand_df_train),
                lb=0,
                name="Total_Demand_Up_to_K_minus_1",
            )
            total_demand_from_k_to_T_vars = model.addVars(
                len(demand_df_train), lb=0, name="Total_Demand_from_k_to_T"
            )
            Q1_plus_lefts = model.addVars(
                len(demand_df_train),
                lb=0,
                name=f"Q1_plus_left",
            )  # k 之前的剩餘 + 新進貨的 Q1 量

            ### 2. Variables for Model 2: Optimal Fraction Model
            f_vars = model.addVars(len(demand_df_train), lb=-GRB.INFINITY, name="f_var")
            F_vars = model.addVars(
                len(demand_df_train), lb=0, ub=1, name="Fraction_for_second_order_amount"
            )
            Q0_vars = model.addVars(
                len(demand_df_train), lb=0, ub=(Q_star + 1), name="Q0_var"
            )

            ### 3. Variables for Model 3: Optimal Order Time Model
            # tau_vars = model.addVars(len(demand_df_train), K, lb=-GRB.INFINITY, name="tau")
            tau_vars = model.addVars(len(demand_df_train), K, lb=-GRB.INFINITY, name="tau")
            r_vars = model.addVars(len(demand_df_train), K, lb=0.0, ub=1.0, name="r")
            R_vars = model.addVars(len(demand_df_train), K, vtype=GRB.BINARY, name="R")

            ### 4. Variables for Model 4: re-estimate order-up-to-level
            Q1_vars = model.addVars(len(Qk_hat_df), lb=0.0, name="Q1_var")
            Q_hats = model.addVars(
                len(Qk_hat_df),
                lb=0.0,
                name="Q_hat",
            )
            Q_hat_adjusteds = model.addVars(
                len(Qk_hat_df), lb=-GRB.INFINITY, name=f"Q_hat_adjusted"
            )

            # ======================= Start Stimulation! =======================

            for i, _ in demand_df_train.iterrows():

                ### Data for this stimulation
                demand_row = demand_df_train.iloc[i]
                Qk_hat_df_row = Qk_hat_df.iloc[i]
                X_data = training_df.iloc[i].tolist()
                X_data.append(1)

                # =================== Model 1: Optimal Fraction Model ===================

                ### 用線性回歸計算F_var
                model.addConstr(
                    f_vars[i]
                    == gp.quicksum(X_data[j] * alphas[j] for j in range(features_num + 1))
                )
                model.addGenConstrLogistic(
                    xvar=f_vars[i], yvar=F_vars[i], name=f"logistic_constraint_{i}"
                )
                model.addConstr(Q0_vars[i] == F_vars[i] * Q_star, f"Q0_upper_bound_{i}")

                # =================== Model 2: Optimal Order Time Model(Alternative Model) ===================

                # Step 1: 利用線性回歸計算 tau
                for k in range(K):
                    model.addConstr(
                        tau_vars[i, k]
                        == gp.quicksum(
                            X_data[j] * betas[k, j] for j in range(features_num + 1)
                        ),
                        name=f"tau_computation_{i}_{k}",
                    )

                delta = 1e-3
                tau_star = model.addVar(lb=-GRB.INFINITY, name=f"tau_star_{i}")

                for k in range(K):
                    # 如果候選 k 被選中 (R_vars[i,k] == 1)，則強制 tau_vars[i,k] 等於 tau_star
                    model.addGenConstrIndicator(
                        R_vars[i, k],
                        True,
                        tau_vars[i, k] == tau_star,
                        name=f"tau_star_eq_{i}_{k}",
                    )

                    # 如果候選 k 未被選中 (R_vars[i,k] == 0)，則必須有 tau_vars[i,k] <= tau_star - delta
                    # 利用 Big-M 技巧：當 R_vars[i,k]==0 時，約束變為 tau_vars[i,k] <= tau_star - delta
                    # 當 R_vars[i,k]==1 時，由於前面的 indicator 約束已強制 tau_vars[i,k] == tau_star，
                    # 此約束則不會影響模型（因為 tau_star <= tau_star - delta + M 已經成立）
                    model.addConstr(
                        tau_vars[i, k] <= tau_star - delta + M * R_vars[i, k],
                        name=f"tau_gap_{i}_{k}",
                    )

                # Step 3: 保證只有一個候選被選中 (即 R_vars 為 1 的只有一個)
                model.addConstr(
                    gp.quicksum(R_vars[i, k] for k in range(K)) == 1,
                    name=f"one_R_{i}",
                )

                # ============ Model 3: re-estimate order-up-to-level =================

                ### 計算 Q_hat -> k: 2~9 -> k-2: 0~7
                model.addConstr(
                    Q_hats[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * Qk_hat_df_row[k - 2] for k in range(2, T)
                    ),
                    name=f"Define_Q_hat_{i}",
                )
                model.addConstr(
                    Q_hat_adjusteds[i] == Q_hats[i] - Q0_vars[i], name=f"Adjust_Q_hat_{i}"
                )
                model.addConstr(
                    Q1_vars[i] == max_(Q_hat_adjusteds[i], 0),
                    name=f"Max_Constraint_{i}",
                )

                # =================== Model 4: Maximum Profit Model ===================

                # ### 0~k-1 的需求量
                model.addConstr(
                    total_demand_up_to_k_minus_1_vars[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * demand_row[: k - 1].sum() for k in range(2, T)
                    ),
                    name=f"Constr_Total_Demand_Up_to_K_Minus_1_{i}",
                )

                # ### k~T 的需求量
                model.addConstr(
                    total_demand_from_k_to_T_vars[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * demand_row[k - 1 :].sum() for k in range(2, T)
                    ),
                    name=f"Constr_Total_Demand_from_K_to_T_{i}",
                )

                # 定義輔助變數
                Left_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
                Lost_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")
                Left_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
                Lost_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")

                # 計算 Sold_0，為 total_demand_up_to_k_minus_1_vars 和 Q0_vars 的最小值
                model.addGenConstrMin(
                    Sold_0s[i],
                    [total_demand_up_to_k_minus_1_vars[i], Q0_vars[i]],
                    name=f"Constr_Sold_0_min_{i}",
                )

                # 計算 Left_0，為 max(Q0_vars[i] - Sold_0s[i], 0)
                model.addConstr(
                    Left_0_aux == Q0_vars[i] - Sold_0s[i],
                    name=f"Constr_Left_0_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Left_0s[i], [Left_0_aux, 0], name=f"Constr_Left_0_max_{i}"
                )

                # 計算 Lost_0，為 max(total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i], 0)
                model.addConstr(
                    Lost_0_aux == total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i],
                    name=f"Constr_Lost_0_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Lost_0s[i], [Lost_0_aux, 0], name=f"Constr_Lost_0_max_{i}"
                )

                # 計算 Q1 + left_0
                model.addConstr(
                    Q1_plus_lefts[i] == Q1_vars[i] + Left_0s[i],
                    name=f"Constr_Q1_plus_left_{i}",
                )

                # 計算 Sold_1，為 total_demand_from_k_to_T_vars 和 Q1_plus_lefts 的最小值
                model.addGenConstrMin(
                    Sold_1s[i],
                    [total_demand_from_k_to_T_vars[i], Q1_plus_lefts[i]],
                    name=f"Constr_Sold_1_min_{i}",
                )

                # 計算 Left_1，為 max(Q1_plus_lefts[i] - Sold_1s[i], 0)
                model.addConstr(
                    Left_1_aux == Q1_plus_lefts[i] - Sold_1s[i],
                    name=f"Constr_Left_1_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Left_1s[i], [Left_1_aux, 0], name=f"Constr_Left_1_max_{i}"
                )

                # 計算 Lost_1，為 max(total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i], 0)
                model.addConstr(
                    Lost_1_aux == total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i],
                    name=f"Constr_Lost_1_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Lost_1s[i], [Lost_1_aux, 0], name=f"Constr_Lost_1_max_{i}"
                )

                model.addConstr(
                    profits_vars[i]
                    == (
                        (price - cost) * (Sold_0s[i] + Sold_1s[i])  # sold
                        - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # lost sales
                        - (cost - salvage_value) * Left_1s[i]  # left cost
                    ),
                    name=f"Profit_Constraint_{i}",
                )

            #  ======================================= Model optimize =======================================

            model.setObjective(
                gp.quicksum(profits_vars[i] for i in range(len(demand_df_train))),
                GRB.MAXIMIZE,
            )
            model.write("s4_model_debug.lp")
            model.write("s4_model.mps")
            try:
                model.optimize()

                if model.status == GRB.OPTIMAL:
                    # print(f"\nmodel.status is optimal: {model.status == GRB.OPTIMAL}")
                    # print(f"model.status is TIME_LIMIT: {model.status == GRB.TIME_LIMIT}\n")

                    # print("===================== 找到最佳解 ==================")
                    # print(f"Q0_optimal（最佳總庫存量）: {Q_star}")

                    # print("Alphas values:")
                    # for key, alpha in alphas.items():
                    #     print(f"alpha[{key}]: {alpha.X}")

                    alpha_values = np.array([alpha.X for _, alpha in alphas.items()])
                    beta_values = np.array(
                        [[betas[k, j].X for j in range(features_num + 1)] for k in range(K)]
                    )
                    # print(f"beta_values:\n{beta_values}")

                    f_values = np.array([f.X for _, f in f_vars.items()])
                    tau_values = np.array(
                        [
                            [tau_vars[i, j].X for j in range(K)]
                            for i in range(len(demand_df_train))
                        ]
                    )

                    # print(f"------------")
                    # print(f"f_values:\n{f_values}")
                    # print(f"tau_values:\n{tau_values}")

                    all_losses = []
                    all_lefts = []
                    all_operation_profits = []
                    all_profits = []
                    all_rs = []
                    all_Rs = []
                    all_Q0s = []
                    all_Q1s = []
                    all_Fs = []
                    all_holding_costs_0 = []
                    all_holding_costs_1 = []
                    all_left0s = []
                    all_left1s = []
                    all_lost0s = []
                    all_lost1s = []

                    for i in range(len(demand_df_train)):

                        # print("----------------------------------------------")
                        # print(f"第 {i+1} 筆觀察資料:")

                        sold0 = Sold_0s[i].X
                        sold1 = Sold_1s[i].X
                        left0 = Left_0s[i].X
                        left1 = Left_1s[i].X
                        lost0 = Lost_0s[i].X
                        lost1 = Lost_1s[i].X
                        Holding_Cost_0 = Holding_Cost_0s[i].X
                        Holding_Cost_1 = Holding_Cost_1s[i].X

                        operation_profit = (price - cost) * (sold0 + sold1)
                        daily_profit = profits_vars[i].X

                        all_losses.append(lost0 + lost1)
                        all_lefts.append(left0 + left1)
                        all_operation_profits.append(operation_profit)
                        all_profits.append(daily_profit)
                        all_Q0s.append(Q0_vars[i].X)
                        all_Q1s.append(Q1_vars[i].X)
                        all_Fs.append(F_vars[i].X)
                        all_holding_costs_0.append(Holding_Cost_0)
                        all_holding_costs_1.append(Holding_Cost_1)
                        all_left0s.append(left0)
                        all_left1s.append(left1)
                        all_lost0s.append(lost0)
                        all_lost1s.append(lost1)

                        reorder_day = None
                        rs = []
                        for k in range(K):
                            rs.append(r_vars[i, k].X)
                            R_value = R_vars[i, k].X
                            # print(
                            #     f"第 {k+2} 天補貨策略: R_vars = {R_value}, tau_vars = {tau_vars[i, k].X}"
                            # )

                            if int(R_value) == 1:
                                reorder_day = k + 2
                        # print(f"*** 於第[{reorder_day}]天進貨 ***\n")

                        all_Rs.append(reorder_day)
                        all_rs.append(rs)

                        demand_row = demand_df_train.iloc[i]

                        total_demand_up = total_demand_up_to_k_minus_1_vars[i].X
                        total_demand_down = total_demand_from_k_to_T_vars[i].X

                        check_results_df = check_values(
                            Q1_vars=Q1_vars,
                            Q_hat_adjusteds=Q_hat_adjusteds,
                            Q0_vars=Q0_vars,
                            Sold_0s=Sold_0s,
                            total_demand_up_to_k_minus_1_vars=total_demand_up_to_k_minus_1_vars,
                            Sold_1s=Sold_1s,
                            total_demand_from_k_to_T_vars=total_demand_from_k_to_T_vars,
                            Q1_plus_lefts=Q1_plus_lefts,
                            Left_0s=Left_0s,
                            Lost_0s=Lost_0s,
                            Left_1s=Left_1s,
                            Lost_1s=Lost_1s,
                        )
                        # print(check_results_df)

                        # for t in range(2):
                        #     if t == 0:
                        #         print(
                        #             f"  第 {t+1} 階段: 本階段期初庫存 = {Q0_vars[i].X}, 第一階段總需求 = {total_demand_up}, 銷售量 = {Sold_0s[i].X}, 本階段期末剩餘庫存 = {Left_0s[i].X}, 本期損失 = {Lost_0s[i].X}, 本期 holding cost = {Holding_Cost_0}"
                        #         )
                        #     else:
                        #         print(
                        #             f"  第 {t+1} 階段: 本階段期初庫存 = {Q1_plus_lefts[i].X}, 重新預估需求 = {Q_hats[i].X}, 第二階段總需求 = {total_demand_down}, 銷售量 = {Sold_1s[i].X}, 本階段期末剩餘庫存 = {Left_1s[i].X}, 本期損失 = {Lost_1s[i].X}, 本期 holding cost = {Holding_Cost_1}"
                        #         )

                        # print(f"  本觀察資料總利潤 = {daily_profit}\n")

                    # print("==========================================")
                    # print(f"最佳化模型平均利潤 = {np.mean(all_profits)}")

                    return (
                        all_Rs,
                        all_losses,
                        all_lefts,
                        all_profits,
                        all_operation_profits,
                        alpha_values,
                        beta_values,
                        all_Fs,
                        all_Q0s,
                        all_Q1s,
                        f_values,
                        tau_values,
                        all_holding_costs_0,
                        all_holding_costs_1,
                        all_left0s,
                        all_left1s,
                        all_lost0s,
                        all_lost1s,
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

    # %%
    def fully_flexible_beta_with_softmax_12(
        salvage_value, cost, price, Q_star, demand_df_train, Qk_hat_df, training_df
    ):

        result = __fully_flexible_beta_with_softmax_12(
            salvage_value=salvage_value,
            cost=cost,
            price=price,
            Q_star=Q_star,
            demand_df_train=demand_df_train,
            Qk_hat_df=Qk_hat_df,
            training_df=training_df,
        )
        if result is None:
            print(f"找不到最佳解")
            return None, None
        else:
            (
                all_Rs,
                all_losses,
                all_lefts,
                all_profits,
                all_operation_profits,
                alpha_values,
                beta_values,
                all_Fs,
                all_Q0s,
                all_Q1s,
                f_values,
                tau_values,
                holding_costs_0s,
                holding_costs_1s,
                all_left0s,
                all_left1s,
                all_lost0s,
                all_lost1s,
            ) = result

            # print(f"all_Rs: {all_Rs}")

            return make_s3_related_strtegies_result(
                all_Rs=all_Rs,
                losses=all_losses,
                lefts=all_lefts,
                profits=all_profits,
                operation_profits=all_operation_profits,
                alpha_values=alpha_values,
                beta_values=beta_values,
                F_vars=all_Fs,
                Q0_vars=all_Q0s,
                Q1_vars=all_Q1s,
                f_values=f_values,
                tau_values=tau_values,
                holding_costs_0s=holding_costs_0s,
                holding_costs_1s=holding_costs_1s,
                all_left0s=all_left0s,
                all_left1s=all_left1s,
                all_lost0s=all_lost0s,
                all_lost1s=all_lost1s,
            )

    # %% [markdown]
    # ### S14 - Optimized F & Rk

    # %%
    def __cal_optimized_F_R(
        salvage_value, cost, price, Q_star, demand_df_train, Qk_hat_df, training_df
    ):

        with gp.Model("profit_maximization", env=env) as model:

            model.setParam("OutputFlag", OUTPUTFLAG)
            model.setParam("Threads", THREADS)
            model.setParam("MIPGap", MIPGAP)
            model.setParam("TimeLimit", TIME_LIMIT)
            model.setParam("IntFeasTol", 1e-9)

            # ======================= Global Variables =======================

            # Category 1 - Some variables that is important to future work
            K = T - 2  # this is for k=2~T-1. => if T = 10(1~10), K will be 8. (0~7)

            # Category 2 - Variables about this stimulation
            ### 1. Variables for Model 1: Maximum Profit Model
            Sold_0s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_0")
            Left_0s = model.addVars(len(demand_df_train), lb=0.0, name="Left_0")
            Lost_0s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_0")

            Sold_1s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_1")
            Left_1s = model.addVars(len(demand_df_train), lb=0.0, name="Left_1")
            Lost_1s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_1")

            Holding_Cost_0s = model.addVars(
                len(demand_df_train), lb=0.0, name="Holding_Cost_0"
            )

            Holding_Cost_1s = model.addVars(
                len(demand_df_train), lb=0.0, name="Holding_Cost_1"
            )

            profits_vars = model.addVars(
                len(demand_df_train), lb=-GRB.INFINITY, name="profits_vars"
            )

            #### 1-2. 用於計算 k 時期之前與之後的需求量
            total_demand_up_to_k_minus_1_vars = model.addVars(
                len(demand_df_train),
                lb=0,
                name="Total_Demand_Up_to_K_minus_1",
            )
            total_demand_from_k_to_T_vars = model.addVars(
                len(demand_df_train), lb=0, name="Total_Demand_from_k_to_T"
            )
            Q1_plus_lefts = model.addVars(
                len(demand_df_train),
                lb=0,
                name=f"Q1_plus_left",
            )  # k 之前的剩餘 + 新進貨的 Q1 量

            ### 2. Variables for Model 2: Optimal Fraction Model
            F_vars = model.addVars(
                len(demand_df_train), lb=0, ub=1, name="Fraction_for_second_order_amount"
            )
            Q0_vars = model.addVars(
                len(demand_df_train), lb=0, ub=(Q_star + 1), name="Q0_var"
            )

            ### 3. Variables for Model 3: Optimal Order Time Model
            R_vars = model.addVars(len(demand_df_train), K, vtype=GRB.BINARY, name="R")

            ### 4. Variables for Model 4: re-estimate order-up-to-level
            Q1_vars = model.addVars(len(Qk_hat_df), lb=0.0, name="Q1_var")
            Q_hats = model.addVars(
                len(Qk_hat_df),
                lb=0.0,
                name="Q_hat",
            )
            Q_hat_adjusteds = model.addVars(
                len(Qk_hat_df), lb=-GRB.INFINITY, name=f"Q_hat_adjusted"
            )

            # ======================= Start Stimulation! =======================

            for i, _ in demand_df_train.iterrows():

                ### Data for this stimulation
                demand_row = demand_df_train.iloc[i]
                Qk_hat_df_row = Qk_hat_df.iloc[i]
                X_data = training_df.iloc[i].tolist()
                X_data.append(1)

                # =================== Model 1: Optimal Fraction Model ===================

                model.addConstr(F_vars[i] >= 0, name=f"Fraction_lower_bound_{i}")
                model.addConstr(F_vars[i] <= 1, name=f"Fraction_upper_bound_{i}")
                model.addConstr(Q0_vars[i] == F_vars[i] * Q_star, f"Q0_upper_bound_{i}")

                # =================== Model 2: Optimal Order Time Model ===================

                model.addConstr(
                    gp.quicksum(R_vars[i, k] for k in range(K)) == 1,
                    name=f"Ensure_only_one_R_true_{i}",
                )

                # ============ Model 3: re-estimate order-up-to-level =================

                ### 計算 Q_hat -> k: 2~9 -> k-2: 0~7
                model.addConstr(
                    Q_hats[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * Qk_hat_df_row[k - 2] for k in range(2, T)
                    ),
                    name=f"Define_Q_hat_{i}",
                )
                model.addConstr(
                    Q_hat_adjusteds[i] == Q_hats[i] - Q0_vars[i], name=f"Adjust_Q_hat_{i}"
                )
                model.addConstr(
                    Q1_vars[i] == max_(Q_hat_adjusteds[i], 0),
                    name=f"Max_Constraint_{i}",
                )

                # =================== Model 4: Maximum Profit Model ===================

                # ### 0~k-1 的需求量
                model.addConstr(
                    total_demand_up_to_k_minus_1_vars[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * demand_row[: k - 1].sum() for k in range(2, T)
                    ),
                    name=f"Constr_Total_Demand_Up_to_K_Minus_1_{i}",
                )

                # ### k~T 的需求量
                model.addConstr(
                    total_demand_from_k_to_T_vars[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * demand_row[k - 1 :].sum() for k in range(2, T)
                    ),
                    name=f"Constr_Total_Demand_from_K_to_T_{i}",
                )

                # 定義輔助變數
                Left_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
                Lost_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")
                Left_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
                Lost_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")

                # 計算 Sold_0，為 total_demand_up_to_k_minus_1_vars 和 Q0_vars 的最小值
                model.addGenConstrMin(
                    Sold_0s[i],
                    [total_demand_up_to_k_minus_1_vars[i], Q0_vars[i]],
                    name=f"Constr_Sold_0_min_{i}",
                )

                # 計算 Left_0，為 max(Q0_vars[i] - Sold_0s[i], 0)
                model.addConstr(
                    Left_0_aux == Q0_vars[i] - Sold_0s[i],
                    name=f"Constr_Left_0_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Left_0s[i], [Left_0_aux, 0], name=f"Constr_Left_0_max_{i}"
                )

                # 計算 Lost_0，為 max(total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i], 0)
                model.addConstr(
                    Lost_0_aux == total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i],
                    name=f"Constr_Lost_0_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Lost_0s[i], [Lost_0_aux, 0], name=f"Constr_Lost_0_max_{i}"
                )

                # 計算 Q1 + left_0
                model.addConstr(
                    Q1_plus_lefts[i] == Q1_vars[i] + Left_0s[i],
                    name=f"Constr_Q1_plus_left_{i}",
                )

                # 計算 Sold_1，為 total_demand_from_k_to_T_vars 和 Q1_plus_lefts 的最小值
                model.addGenConstrMin(
                    Sold_1s[i],
                    [total_demand_from_k_to_T_vars[i], Q1_plus_lefts[i]],
                    name=f"Constr_Sold_1_min_{i}",
                )

                # 計算 Left_1，為 max(Q1_plus_lefts[i] - Sold_1s[i], 0)
                model.addConstr(
                    Left_1_aux == Q1_plus_lefts[i] - Sold_1s[i],
                    name=f"Constr_Left_1_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Left_1s[i], [Left_1_aux, 0], name=f"Constr_Left_1_max_{i}"
                )

                # 計算 Lost_1，為 max(total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i], 0)
                model.addConstr(
                    Lost_1_aux == total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i],
                    name=f"Constr_Lost_1_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Lost_1s[i], [Lost_1_aux, 0], name=f"Constr_Lost_1_max_{i}"
                )

                model.addConstr(
                    profits_vars[i]
                    == (
                        (price - cost) * (Sold_0s[i] + Sold_1s[i])  # sold
                        - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # lost sales
                        - (cost - salvage_value) * Left_1s[i]  # left cost
                    ),
                    name=f"Profit_Constraint_{i}",
                )

            #  ======================================= Model optimize =======================================

            model.setObjective(
                gp.quicksum(profits_vars[i] for i in range(len(demand_df_train))),
                GRB.MAXIMIZE,
            )
            model.write("s4_model_debug.lp")
            model.write("s4_model.mps")
            try:
                model.optimize()

                if model.status == GRB.OPTIMAL:
                    # print(f"\nmodel.status is optimal: {model.status == GRB.OPTIMAL}")
                    # print(f"model.status is TIME_LIMIT: {model.status == GRB.TIME_LIMIT}\n")

                    # print("===================== 找到最佳解 ==================")
                    # print(f"Q0_optimal（最佳總庫存量）: {Q_star}")

                    all_losses = []
                    all_lefts = []
                    all_operation_profits = []
                    all_profits = []
                    # all_rs = []
                    all_Rs = []
                    all_Q0s = []
                    all_Q1s = []
                    all_Fs = []
                    all_holding_costs_0 = []
                    all_holding_costs_1 = []
                    all_left0s = []
                    all_left1s = []
                    all_lost0s = []
                    all_lost1s = []

                    for i in range(len(demand_df_train)):

                        # print("----------------------------------------------")
                        # print(f"第 {i+1} 筆觀察資料:")

                        sold0 = Sold_0s[i].X
                        sold1 = Sold_1s[i].X
                        left0 = Left_0s[i].X
                        left1 = Left_1s[i].X
                        lost0 = Lost_0s[i].X
                        lost1 = Lost_1s[i].X
                        Holding_Cost_0 = Holding_Cost_0s[i].X
                        Holding_Cost_1 = Holding_Cost_1s[i].X

                        operation_profit = (price - cost) * (sold0 + sold1)
                        daily_profit = profits_vars[i].X

                        all_losses.append(lost0 + lost1)
                        all_lefts.append(left0 + left1)
                        all_operation_profits.append(operation_profit)
                        all_profits.append(daily_profit)
                        all_Q0s.append(Q0_vars[i].X)
                        all_Q1s.append(Q1_vars[i].X)
                        all_Fs.append(F_vars[i].X)
                        all_holding_costs_0.append(Holding_Cost_0)
                        all_holding_costs_1.append(Holding_Cost_1)
                        all_left0s.append(left0)
                        all_left1s.append(left1)
                        all_lost0s.append(lost0)
                        all_lost1s.append(lost1)

                        reorder_day = None
                        rs = []
                        for k in range(K):
                            R_value = R_vars[i, k].X
                            # print(f"第 {k+2} 天補貨策略: R_vars = {R_value}")

                            if int(R_value) == 1:
                                reorder_day = k + 2
                        # print(f"*** 於第[{reorder_day}]天進貨 ***\n")

                        all_Rs.append(reorder_day)
                        demand_row = demand_df_train.iloc[i]

                        total_demand_up = total_demand_up_to_k_minus_1_vars[i].X
                        total_demand_down = total_demand_from_k_to_T_vars[i].X

                        check_results_df = check_values(
                            Q1_vars=Q1_vars,
                            Q_hat_adjusteds=Q_hat_adjusteds,
                            Q0_vars=Q0_vars,
                            Sold_0s=Sold_0s,
                            total_demand_up_to_k_minus_1_vars=total_demand_up_to_k_minus_1_vars,
                            Sold_1s=Sold_1s,
                            total_demand_from_k_to_T_vars=total_demand_from_k_to_T_vars,
                            Q1_plus_lefts=Q1_plus_lefts,
                            Left_0s=Left_0s,
                            Lost_0s=Lost_0s,
                            Left_1s=Left_1s,
                            Lost_1s=Lost_1s,
                        )
                        # print(check_results_df)

                    #     for t in range(2):
                    #         if t == 0:
                    #             print(
                    #                 f"  第 {t+1} 階段: 本階段期初庫存 = {Q0_vars[i].X}, 第一階段總需求 = {total_demand_up}, 銷售量 = {Sold_0s[i].X}, 本階段期末剩餘庫存 = {Left_0s[i].X}, 本期損失 = {Lost_0s[i].X}, 本期 holding cost = {Holding_Cost_0}"
                    #             )
                    #         else:
                    #             print(
                    #                 f"  第 {t+1} 階段: 本階段期初庫存 = {Q1_plus_lefts[i].X}, 重新預估需求 = {Q_hats[i].X}, 第二階段總需求 = {total_demand_down}, 銷售量 = {Sold_1s[i].X}, 本階段期末剩餘庫存 = {Left_1s[i].X}, 本期損失 = {Lost_1s[i].X}, 本期 holding cost = {Holding_Cost_1}"
                    #             )

                    #     print(f"  本觀察資料總利潤 = {daily_profit}\n")

                    # print("==========================================")
                    # print(f"最佳化模型平均利潤 = {np.mean(all_profits)}")

                    return (
                        all_Rs,
                        all_losses,
                        all_lefts,
                        all_profits,
                        all_operation_profits,
                        all_Fs,
                        all_Q0s,
                        all_Q1s,
                        all_holding_costs_0,
                        all_holding_costs_1,
                        all_left0s,
                        all_left1s,
                        all_lost0s,
                        all_lost1s,
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

    # %%
    def cal_optimized_F_R(
        salvage_value,
        cost,
        price,
        Q_star,
        demand_df,
        Qk_hat_df,
        training_df,
    ):
        results_dict = {
            "R(T)": [],
            "average_profits": [],
            "average_losses": [],
            "average_lefts": [],
            "average_operation_profits": [],
            "alpha_values": [],
            "F_vars": [],
            "Q0_vars": [],
            "Q1_vars": [],
        }

        max_profit = None
        max_profit_stimulation_result = {}

        (
            all_Rs,
            losses,
            lefts,
            profits,
            operation_profits,
            F_vars,
            Q0_vars,
            Q1_vars,
            all_holding_costs_0,
            all_holding_costs_1,
            all_left0s,
            all_left1s,
            all_lost0s,
            all_lost1s,
        ) = __cal_optimized_F_R(
            salvage_value=salvage_value,
            cost=cost,
            price=price,
            Q_star=Q_star,
            demand_df_train=demand_df,
            Qk_hat_df=Qk_hat_df,
            training_df=training_df,
        )

        # 計算平均值
        average_losses = sum(losses) / len(losses) if losses else 0
        average_lefts = sum(lefts) / len(lefts) if lefts else 0
        average_profits = sum(profits) / len(profits) if profits else 0
        average_operation_profits = (
            sum(operation_profits) / len(operation_profits) if operation_profits else 0
        )

        # 將結果存儲到字典中
        results_dict["R(T)"].append(all_Rs)
        results_dict["average_losses"].append(average_losses)
        results_dict["average_lefts"].append(average_lefts)
        results_dict["average_profits"].append(average_profits)
        results_dict["average_operation_profits"].append(average_operation_profits)
        results_dict["alpha_values"].append(None)
        results_dict["F_vars"].append(F_vars)
        results_dict["Q0_vars"].append(Q0_vars)
        results_dict["Q1_vars"].append(Q1_vars)

        # print(f"The average profits is {average_profits}")

        if max_profit is None or max_profit < average_profits:
            # print(f"max_profit is changed from {max_profit} to {average_profits}")
            max_profit = average_profits
            max_profit_stimulation_result = {
                "R": all_Rs,
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

    # %%


    # %% [markdown]
    # ### S15 - Beta with Lasso

    # %%
    def __fully_flexible_beta_with_lasso_15(
        salvage_value,
        cost,
        price,
        Q_star,
        demand_df_train,
        Qk_hat_df,
        training_df,
        lambda_beta,
    ):

        with gp.Model("profit_maximization", env=env) as model:

            model.setParam("OutputFlag", OUTPUTFLAG)
            model.setParam("Threads", THREADS)
            model.setParam("MIPGap", MIPGAP)
            model.setParam("TimeLimit", TIME_LIMIT)
            model.setParam("IntFeasTol", 1e-9)

            # ======================= Global Variables =======================

            # Category 1 - Some variables that is important to future work
            K = T - 2  # this is for k=2~T-1. => if T = 10(1~10), K will be 8. (0~7)

            alphas = model.addVars(features_num + 1, lb=-GRB.INFINITY, name="alphas")
            betas = model.addVars(K, features_num + 1, lb=-GRB.INFINITY, name="betas")
            abs_betas = model.addVars(betas.keys(), lb=0, name="abs_beta")

            # 進行 lasso 處理
            for k, j in betas.keys():
                model.addConstr(abs_betas[k, j] >= betas[k, j])
                model.addConstr(abs_betas[k, j] >= -betas[k, j])

            # Category 2 - Variables about this stimulation
            ### 1. Variables for Model 1: Maximum Profit Model
            Sold_0s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_0")
            Left_0s = model.addVars(len(demand_df_train), lb=0.0, name="Left_0")
            Lost_0s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_0")

            Sold_1s = model.addVars(len(demand_df_train), lb=0.0, name="Sold_1")
            Left_1s = model.addVars(len(demand_df_train), lb=0.0, name="Left_1")
            Lost_1s = model.addVars(len(demand_df_train), lb=0.0, name="Lost_1")

            Holding_Cost_0s = model.addVars(
                len(demand_df_train), lb=0.0, name="Holding_Cost_0"
            )

            Holding_Cost_1s = model.addVars(
                len(demand_df_train), lb=0.0, name="Holding_Cost_1"
            )

            profits_vars = model.addVars(
                len(demand_df_train), lb=-GRB.INFINITY, name="profits_vars"
            )

            #### 1-2. 用於計算 k 時期之前與之後的需求量
            total_demand_up_to_k_minus_1_vars = model.addVars(
                len(demand_df_train),
                lb=0,
                name="Total_Demand_Up_to_K_minus_1",
            )
            total_demand_from_k_to_T_vars = model.addVars(
                len(demand_df_train), lb=0, name="Total_Demand_from_k_to_T"
            )
            Q1_plus_lefts = model.addVars(
                len(demand_df_train),
                lb=0,
                name=f"Q1_plus_left",
            )  # k 之前的剩餘 + 新進貨的 Q1 量

            ### 2. Variables for Model 2: Optimal Fraction Model
            f_vars = model.addVars(len(demand_df_train), lb=-GRB.INFINITY, name="f_var")
            F_vars = model.addVars(
                len(demand_df_train), lb=0, ub=1, name="Fraction_for_second_order_amount"
            )
            Q0_vars = model.addVars(
                len(demand_df_train), lb=0, ub=(Q_star + 1), name="Q0_var"
            )

            ### 3. Variables for Model 3: Optimal Order Time Model
            # tau_vars = model.addVars(len(demand_df_train), K, lb=-GRB.INFINITY, name="tau")
            tau_vars = model.addVars(len(demand_df_train), K, lb=-GRB.INFINITY, name="tau")
            r_vars = model.addVars(len(demand_df_train), K, lb=0.0, ub=1.0, name="r")
            R_vars = model.addVars(len(demand_df_train), K, vtype=GRB.BINARY, name="R")

            ### 4. Variables for Model 4: re-estimate order-up-to-level
            Q1_vars = model.addVars(len(Qk_hat_df), lb=0.0, name="Q1_var")
            Q_hats = model.addVars(
                len(Qk_hat_df),
                lb=0.0,
                name="Q_hat",
            )
            Q_hat_adjusteds = model.addVars(
                len(Qk_hat_df), lb=-GRB.INFINITY, name=f"Q_hat_adjusted"
            )

            # ======================= Start Stimulation! =======================

            for i, _ in demand_df_train.iterrows():

                ### Data for this stimulation
                demand_row = demand_df_train.iloc[i]
                Qk_hat_df_row = Qk_hat_df.iloc[i]
                X_data = training_df.iloc[i].tolist()
                X_data.append(1)

                # =================== Model 1: Optimal Fraction Model ===================

                ### 用線性回歸計算F_var
                model.addConstr(
                    f_vars[i]
                    == gp.quicksum(X_data[j] * alphas[j] for j in range(features_num + 1))
                )
                model.addGenConstrLogistic(
                    xvar=f_vars[i], yvar=F_vars[i], name=f"logistic_constraint_{i}"
                )
                model.addConstr(Q0_vars[i] == F_vars[i] * Q_star, f"Q0_upper_bound_{i}")

                # =================== Model 2: Optimal Order Time Model(Alternative Model) ===================

                # Step 1: 利用線性回歸計算 tau
                for k in range(K):
                    model.addConstr(
                        tau_vars[i, k]
                        == gp.quicksum(
                            X_data[j] * betas[k, j] for j in range(features_num + 1)
                        ),
                        name=f"tau_computation_{i}_{k}",
                    )

                delta = 1e-3
                tau_star = model.addVar(lb=-GRB.INFINITY, name=f"tau_star_{i}")

                for k in range(K):
                    # 如果候選 k 被選中 (R_vars[i,k] == 1)，則強制 tau_vars[i,k] 等於 tau_star
                    model.addGenConstrIndicator(
                        R_vars[i, k],
                        True,
                        tau_vars[i, k] == tau_star,
                        name=f"tau_star_eq_{i}_{k}",
                    )

                    model.addConstr(
                        tau_vars[i, k] <= tau_star - delta + M * R_vars[i, k],
                        name=f"tau_gap_{i}_{k}",
                    )

                # Step 3: 保證只有一個候選被選中 (即 R_vars 為 1 的只有一個)
                model.addConstr(
                    gp.quicksum(R_vars[i, k] for k in range(K)) == 1,
                    name=f"one_R_{i}",
                )

                # ============ Model 3: re-estimate order-up-to-level =================

                ### 計算 Q_hat -> k: 2~9 -> k-2: 0~7
                model.addConstr(
                    Q_hats[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * Qk_hat_df_row[k - 2] for k in range(2, T)
                    ),
                    name=f"Define_Q_hat_{i}",
                )
                model.addConstr(
                    Q_hat_adjusteds[i] == Q_hats[i] - Q0_vars[i], name=f"Adjust_Q_hat_{i}"
                )
                model.addConstr(
                    Q1_vars[i] == max_(Q_hat_adjusteds[i], 0),
                    name=f"Max_Constraint_{i}",
                )

                # =================== Model 4: Maximum Profit Model ===================

                # ### 0~k-1 的需求量
                model.addConstr(
                    total_demand_up_to_k_minus_1_vars[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * demand_row[: k - 1].sum() for k in range(2, T)
                    ),
                    name=f"Constr_Total_Demand_Up_to_K_Minus_1_{i}",
                )

                # ### k~T 的需求量
                model.addConstr(
                    total_demand_from_k_to_T_vars[i]
                    == gp.quicksum(
                        R_vars[i, k - 2] * demand_row[k - 1 :].sum() for k in range(2, T)
                    ),
                    name=f"Constr_Total_Demand_from_K_to_T_{i}",
                )

                # 定義輔助變數
                Left_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
                Lost_0_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")
                Left_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Left_0_aux_{i}")
                Lost_1_aux = model.addVar(lb=-GRB.INFINITY, name=f"Lost_0_aux_{i}")

                # 計算 Sold_0，為 total_demand_up_to_k_minus_1_vars 和 Q0_vars 的最小值
                model.addGenConstrMin(
                    Sold_0s[i],
                    [total_demand_up_to_k_minus_1_vars[i], Q0_vars[i]],
                    name=f"Constr_Sold_0_min_{i}",
                )

                # 計算 Left_0，為 max(Q0_vars[i] - Sold_0s[i], 0)
                model.addConstr(
                    Left_0_aux == Q0_vars[i] - Sold_0s[i],
                    name=f"Constr_Left_0_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Left_0s[i], [Left_0_aux, 0], name=f"Constr_Left_0_max_{i}"
                )

                # 計算 Lost_0，為 max(total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i], 0)
                model.addConstr(
                    Lost_0_aux == total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i],
                    name=f"Constr_Lost_0_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Lost_0s[i], [Lost_0_aux, 0], name=f"Constr_Lost_0_max_{i}"
                )

                # 計算 Q1 + left_0
                model.addConstr(
                    Q1_plus_lefts[i] == Q1_vars[i] + Left_0s[i],
                    name=f"Constr_Q1_plus_left_{i}",
                )

                # 計算 Sold_1，為 total_demand_from_k_to_T_vars 和 Q1_plus_lefts 的最小值
                model.addGenConstrMin(
                    Sold_1s[i],
                    [total_demand_from_k_to_T_vars[i], Q1_plus_lefts[i]],
                    name=f"Constr_Sold_1_min_{i}",
                )

                # 計算 Left_1，為 max(Q1_plus_lefts[i] - Sold_1s[i], 0)
                model.addConstr(
                    Left_1_aux == Q1_plus_lefts[i] - Sold_1s[i],
                    name=f"Constr_Left_1_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Left_1s[i], [Left_1_aux, 0], name=f"Constr_Left_1_max_{i}"
                )

                # 計算 Lost_1，為 max(total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i], 0)
                model.addConstr(
                    Lost_1_aux == total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i],
                    name=f"Constr_Lost_1_diff_aux_{i}",
                )
                model.addGenConstrMax(
                    Lost_1s[i], [Lost_1_aux, 0], name=f"Constr_Lost_1_max_{i}"
                )

                model.addConstr(
                    profits_vars[i]
                    == (
                        (price - cost) * (Sold_0s[i] + Sold_1s[i])  # sold
                        - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # lost sales
                        - (cost - salvage_value) * Left_1s[i]  # left cost
                    ),
                    name=f"Profit_Constraint_{i}",
                )

            #  ======================================= Model optimize =======================================

            model.setObjective(
                gp.quicksum(profits_vars[i] for i in range(len(demand_df_train)))
                - lambda_beta * gp.quicksum(abs_betas[k, j] for k, j in abs_betas.keys()),
                GRB.MAXIMIZE,
            )
            model.write("s4_model_debug.lp")
            model.write("s4_model.mps")
            try:
                model.optimize()

                if model.status == GRB.OPTIMAL:
                    # print(f"\nmodel.status is optimal: {model.status == GRB.OPTIMAL}")
                    # print(f"model.status is TIME_LIMIT: {model.status == GRB.TIME_LIMIT}\n")

                    # print("===================== 找到最佳解 ==================")
                    # print(f"Q0_optimal（最佳總庫存量）: {Q_star}")

                    # print("Alphas values:")
                    # for key, alpha in alphas.items():
                    #     print(f"alpha[{key}]: {alpha.X}")

                    alpha_values = np.array([alpha.X for _, alpha in alphas.items()])
                    beta_values = np.array(
                        [[betas[k, j].X for j in range(features_num + 1)] for k in range(K)]
                    )
                    # print(f"beta_values:\n{beta_values}")

                    f_values = np.array([f.X for _, f in f_vars.items()])
                    tau_values = np.array(
                        [
                            [tau_vars[i, j].X for j in range(K)]
                            for i in range(len(demand_df_train))
                        ]
                    )

                    # print(f"------------")
                    # print(f"f_values:\n{f_values}")
                    # print(f"tau_values:\n{tau_values}")

                    all_losses = []
                    all_lefts = []
                    all_operation_profits = []
                    all_profits = []
                    all_rs = []
                    all_Rs = []
                    all_Q0s = []
                    all_Q1s = []
                    all_Fs = []
                    all_holding_costs_0 = []
                    all_holding_costs_1 = []
                    all_left0s = []
                    all_left1s = []
                    all_lost0s = []
                    all_lost1s = []

                    for i in range(len(demand_df_train)):

                        # print("----------------------------------------------")
                        # print(f"第 {i+1} 筆觀察資料:")

                        sold0 = Sold_0s[i].X
                        sold1 = Sold_1s[i].X
                        left0 = Left_0s[i].X
                        left1 = Left_1s[i].X
                        lost0 = Lost_0s[i].X
                        lost1 = Lost_1s[i].X
                        Holding_Cost_0 = Holding_Cost_0s[i].X
                        Holding_Cost_1 = Holding_Cost_1s[i].X

                        operation_profit = (price - cost) * (sold0 + sold1)
                        daily_profit = profits_vars[i].X

                        all_losses.append(lost0 + lost1)
                        all_lefts.append(left0 + left1)
                        all_operation_profits.append(operation_profit)
                        all_profits.append(daily_profit)
                        all_Q0s.append(Q0_vars[i].X)
                        all_Q1s.append(Q1_vars[i].X)
                        all_Fs.append(F_vars[i].X)
                        all_holding_costs_0.append(Holding_Cost_0)
                        all_holding_costs_1.append(Holding_Cost_1)
                        all_left0s.append(left0)
                        all_left1s.append(left1)
                        all_lost0s.append(lost0)
                        all_lost1s.append(lost1)

                        reorder_day = None
                        rs = []
                        for k in range(K):
                            rs.append(r_vars[i, k].X)
                            R_value = R_vars[i, k].X
                            # print(
                            #     f"第 {k+2} 天補貨策略: R_vars = {R_value}, tau_vars = {tau_vars[i, k].X}"
                            # )

                            if int(R_value) == 1:
                                reorder_day = k + 2
                        # print(f"*** 於第[{reorder_day}]天進貨 ***\n")

                        all_Rs.append(reorder_day)
                        all_rs.append(rs)

                        demand_row = demand_df_train.iloc[i]

                        total_demand_up = total_demand_up_to_k_minus_1_vars[i].X
                        total_demand_down = total_demand_from_k_to_T_vars[i].X

                        check_results_df = check_values(
                            Q1_vars=Q1_vars,
                            Q_hat_adjusteds=Q_hat_adjusteds,
                            Q0_vars=Q0_vars,
                            Sold_0s=Sold_0s,
                            total_demand_up_to_k_minus_1_vars=total_demand_up_to_k_minus_1_vars,
                            Sold_1s=Sold_1s,
                            total_demand_from_k_to_T_vars=total_demand_from_k_to_T_vars,
                            Q1_plus_lefts=Q1_plus_lefts,
                            Left_0s=Left_0s,
                            Lost_0s=Lost_0s,
                            Left_1s=Left_1s,
                            Lost_1s=Lost_1s,
                        )
                        # print(check_results_df)

                    #     for t in range(2):
                    #         if t == 0:
                    #             print(
                    #                 f"  第 {t+1} 階段: 本階段期初庫存 = {Q0_vars[i].X}, 第一階段總需求 = {total_demand_up}, 銷售量 = {Sold_0s[i].X}, 本階段期末剩餘庫存 = {Left_0s[i].X}, 本期損失 = {Lost_0s[i].X}, 本期 holding cost = {Holding_Cost_0}"
                    #             )
                    #         else:
                    #             print(
                    #                 f"  第 {t+1} 階段: 本階段期初庫存 = {Q1_plus_lefts[i].X}, 重新預估需求 = {Q_hats[i].X}, 第二階段總需求 = {total_demand_down}, 銷售量 = {Sold_1s[i].X}, 本階段期末剩餘庫存 = {Left_1s[i].X}, 本期損失 = {Lost_1s[i].X}, 本期 holding cost = {Holding_Cost_1}"
                    #             )

                    #     print(f"  本觀察資料總利潤 = {daily_profit}\n")

                    # print("==========================================")
                    # print(f"最佳化模型平均利潤 = {np.mean(all_profits)}")

                    return (
                        all_Rs,
                        all_losses,
                        all_lefts,
                        all_profits,
                        all_operation_profits,
                        alpha_values,
                        beta_values,
                        all_Fs,
                        all_Q0s,
                        all_Q1s,
                        f_values,
                        tau_values,
                        all_holding_costs_0,
                        all_holding_costs_1,
                        all_left0s,
                        all_left1s,
                        all_lost0s,
                        all_lost1s,
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

    # %%
    def fully_flexible_beta_with_lasso_15(
        salvage_value,
        cost,
        price,
        Q_star,
        demand_df_train,
        Qk_hat_df,
        training_df,
        lambda_beta,
    ):

        result = __fully_flexible_beta_with_lasso_15(
            salvage_value=salvage_value,
            cost=cost,
            price=price,
            Q_star=Q_star,
            demand_df_train=demand_df_train,
            Qk_hat_df=Qk_hat_df,
            training_df=training_df,
            lambda_beta=lambda_beta,
        )
        if result is None:
            print(f"找不到最佳解")
            return None, None
        else:
            (
                all_Rs,
                all_losses,
                all_lefts,
                all_profits,
                all_operation_profits,
                alpha_values,
                beta_values,
                all_Fs,
                all_Q0s,
                all_Q1s,
                f_values,
                tau_values,
                holding_costs_0s,
                holding_costs_1s,
                all_left0s,
                all_left1s,
                all_lost0s,
                all_lost1s,
            ) = result

            # print(f"all_Rs: {all_Rs}")

            return make_s3_related_strtegies_result(
                all_Rs=all_Rs,
                losses=all_losses,
                lefts=all_lefts,
                profits=all_profits,
                operation_profits=all_operation_profits,
                alpha_values=alpha_values,
                beta_values=beta_values,
                F_vars=all_Fs,
                Q0_vars=all_Q0s,
                Q1_vars=all_Q1s,
                f_values=f_values,
                tau_values=tau_values,
                holding_costs_0s=holding_costs_0s,
                holding_costs_1s=holding_costs_1s,
                all_left0s=all_left0s,
                all_left1s=all_left1s,
                all_lost0s=all_lost0s,
                all_lost1s=all_lost1s,
            )

    # %% [markdown]
    # # Testing Utils
    # 

    # %% [markdown]
    # ## S1 - Grid for Fixed F & Fixed Rk
    # 

    # %%
    def cal_test_fixed_F_fixed_R(
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
        result, stimulation_result = cal_fixed_F_fixed_R(
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

    # %% [markdown]
    # ## S2 - Grid for Fixed Rk & Flexible F
    # 

    # %%
    def cal_test_flexible_F_fixed_R(
        assigned_R,
        alphas,
        salvage_value,
        cost,
        price,
        Q_star,
        demand_df_test,
        Qk_hat_df_test,
        testing_df,
    ):

        # ======================= Global Variables =======================

        # Category 1 - Some variables that is important to future work
        K = T - 2  # this is for k=2~T-1. => if T = 10(1~10), K will be 8. (0~7)
        n = len(demand_df_test)

        # Initialize lists or numpy arrays to replace Gurobi variables
        Sold_0s = np.zeros(n)
        Left_0s = np.zeros(n)
        Lost_0s = np.zeros(n)
        Sold_1s = np.zeros(n)
        Left_1s = np.zeros(n)
        Lost_1s = np.zeros(n)
        all_holding_costs_0 = np.zeros(n)
        all_holding_costs_1 = np.zeros(n)
        profits_vars = np.zeros(n)

        # 1-2. Arrays for demand calculation up to certain periods
        total_demand_up_to_k_minus_1_vars = np.zeros(n)
        total_demand_from_k_to_T_vars = np.zeros(n)
        Q1_plus_lefts = np.zeros(n)

        # 2. Variables for Model 2: Optimal Fraction Model
        f_vars = np.zeros(n)
        F_vars = np.zeros(n)  # Assuming values will be between 0 and 1
        Q0_vars = np.zeros(n)  # Replace Q_star with a specific value as needed

        # 3. Variables for Model 3: Optimal Order Time Model (2D array for binary values)
        R_vars = np.zeros((n, K), dtype=int)  # Use dtype=int to represent binary 0/1 values

        # 4. Variables for Model 4: Re-estimate order-up-to-level
        Q1_vars = np.zeros(n)
        Q_hats = np.zeros(n)
        Q_hat_adjusteds = np.zeros(n)

        # ======================= Start Stimulation! =======================

        for i, row in demand_df_test.iterrows():

            ### Data for this stimulation
            demand_row = demand_df_test.iloc[i]
            Qk_hat_df_test_row = Qk_hat_df_test.iloc[i]
            X_data = testing_df.iloc[i].tolist()
            X_data.append(1)

            # =================== Model 1: Optimal Fraction Model ===================

            ### 用線性回歸計算F_var
            f_vars[i] = sum(X_data[j] * alphas[j] for j in range(features_num + 1))
            F_vars[i] = 1 / (1 + np.exp(-(f_vars[i])))
            Q0_vars[i] = F_vars[i] * Q_star

            # =================== Model 2: Optimal Order Time Model ===================

            # Ensure only one `R` is set to 1 in each row by setting `assigned_R` to 1 and all others to 0
            R_vars[i, assigned_R] = 1

            # ============ Model 3: re-estimate order-up-to-level =================

            Q_hats[i] = sum(
                R_vars[i, k - 2] * Qk_hat_df_test_row[k - 2] for k in range(2, T)
            )
            Q_hat_adjusteds[i] = Q_hats[i] - Q0_vars[i]
            Q1_vars[i] = max(Q_hat_adjusteds[i], 0)

            # =================== Model 4: Maximum Profit Model ===================

            # Calculate the demand up to k-1
            total_demand_up_to_k_minus_1 = sum(
                R_vars[i, k - 2] * demand_row[: k - 1].sum() for k in range(2, T)
            )
            total_demand_up_to_k_minus_1_vars[i] = total_demand_up_to_k_minus_1

            # Calculate the demand from k to T
            total_demand_from_k_to_T = sum(
                R_vars[i, k - 2] * demand_row[k - 1 :].sum() for k in range(2, T)
            )
            total_demand_from_k_to_T_vars[i] = total_demand_from_k_to_T

            Sold_0s[i] = min(total_demand_up_to_k_minus_1_vars[i], Q0_vars[i])
            Left_0s[i] = max(Q0_vars[i] - Sold_0s[i], 0)
            Lost_0s[i] = max(total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i], 0)
            Q1_plus_lefts[i] = Q1_vars[i] + Left_0s[i]

            Sold_1s[i] = min(total_demand_from_k_to_T_vars[i], Q1_plus_lefts[i])
            Left_1s[i] = max(Q1_plus_lefts[i] - Sold_1s[i], 0)
            Lost_1s[i] = max(total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i], 0)

            # all_holding_costs_0[i] = (
            #     (Q0_vars[i] + Left_0s[i] + Q1_vars[i]) * ((assigned_R + 2) - 1) / 2
            # )
            # all_holding_costs_1[i] = (
            #     (Q1_vars[i] + Left_0s[i] + Left_1s[i]) * (T - (assigned_R + 2)) / 2
            # )

            profits_vars[i] = (
                (price - cost) * (Sold_0s[i] + Sold_1s[i])  # Revenue from sales
                - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # Lost sales cost
                - (cost - salvage_value) * Left_1s[i]
                # - holding_cost * (all_holding_costs_0[i] + all_holding_costs_1[i])
            )

        # Calculate the average profit
        print(f"assigned_R: {assigned_R}")
        results_df = pd.DataFrame(
            {
                "average_profits": [np.mean(profits_vars)],
                "average_loss": [np.mean((price - cost) * (Lost_0s[i] + Lost_1s[i]))],
                "average_left": [np.mean((price - cost) * (Lost_0s[i] + Lost_1s[i]))],
                "alpha_values": [alphas],
                "R(T)": assigned_R + 2,
            }
        )

        stimulation_result = pd.DataFrame(
            {
                "F": F_vars,
                "R(T)": assigned_R + 2,
                "Sold_0": Sold_0s,
                "Left_0": Left_0s,
                "Lost_0": Lost_0s,
                "Sold_1": Sold_1s,
                "Left_1": Left_1s,
                "Lost_1": Lost_1s,
                "profits": profits_vars,
                "Q0": Q0_vars,
                "Q1": Q1_vars,
                "hc0": all_holding_costs_0,
                "hc1": all_holding_costs_1,
            }
        )

        return results_df, stimulation_result

    # %% [markdown]
    # ## Fully flexible F & Rk
    # 

    # %% [markdown]
    # ### S12 - Beta without r

    # %%
    def cal_test_fully_flexible_beta_with_softmax_12(
        alphas,
        betas,
        salvage_value,
        cost,
        price,
        Q_star,
        demand_df_test,
        Qk_hat_df_test,
        testing_df,
    ):

        # ======================= Global Variables =======================

        # Category 1 - Some variables that is important to future work
        K = T - 2  # this is for k=2~T-1. => if T = 10(1~10), K will be 8. (0~7)
        n = len(demand_df_test)

        # Initialize lists or numpy arrays to replace Gurobi variables
        Sold_0s = np.zeros(n)
        Left_0s = np.zeros(n)
        Lost_0s = np.zeros(n)
        Sold_1s = np.zeros(n)
        Left_1s = np.zeros(n)
        Lost_1s = np.zeros(n)
        all_holding_costs_0 = np.zeros(n)
        all_holding_costs_1 = np.zeros(n)
        profits_vars = np.zeros(n)

        # 1-2. Arrays for demand calculation up to certain periods
        total_demand_up_to_k_minus_1_vars = np.zeros(n)
        total_demand_from_k_to_T_vars = np.zeros(n)
        Q1_plus_lefts = np.zeros(n)

        # 2. Variables for Model 2: Optimal Fraction Model
        f_vars = np.zeros(n)
        F_vars = np.zeros(n)  # Assuming values will be between 0 and 1
        Q0_vars = np.zeros(n)  # Replace Q_star with a specific value as needed

        # 3. Variables for Model 3: Optimal Order Time Model (2D array for binary values)
        tau_vars = np.zeros((n, K))
        exp_tau_vars = np.zeros((n, K))
        r_vars = np.zeros((n, K))
        max_r_index = np.zeros(n, dtype=int)
        R_vars = np.zeros(
            (n, K), dtype=int
        )  # Binary array to select one optimal replenishment time

        # 4. Variables for Model 4: Re-estimate order-up-to-level
        Q1_vars = np.zeros(n)
        Q_hats = np.zeros(n)
        Q_hat_adjusteds = np.zeros(n)

        # ======================= Start Stimulation! =======================

        for i, row in demand_df_test.iterrows():

            ### Data for this stimulation
            demand_row = demand_df_test.iloc[i]
            Qk_hat_df_test_row = Qk_hat_df_test.iloc[i]
            X_data = testing_df.iloc[i].tolist()
            X_data.append(1)

            # =================== Model 1: Optimal Fraction Model ===================

            ### 用線性回歸計算F_var
            f_vars[i] = sum(X_data[j] * alphas[j] for j in range(features_num + 1))
            F_vars[i] = 1 / (1 + np.exp(-(f_vars[i])))
            Q0_vars[i] = F_vars[i] * Q_star

            # =================== Model 2: Optimal Order Time Model ===================

            # Step 1: Calculate tau_vars as a linear combination of X_data and betas
            for k in range(K):
                tau_vars[i, k] = sum(
                    X_data[j] * betas[k, j] for j in range(features_num + 1)
                )

            max_r_index[i] = np.argmax(tau_vars[i])
            R_vars[i, max_r_index[i]] = 1

            # print(f"tau: {tau_vars[i]}")
            # print(f"R: {R_vars[i]}")
            # print(f"max_r_index: {max_r_index[i]}")
            # print("\n\n")

            # ============ Model 3: re-estimate order-up-to-level =================

            Q_hats[i] = sum(
                R_vars[i, k - 2] * Qk_hat_df_test_row[k - 2] for k in range(2, T)
            )
            Q_hat_adjusteds[i] = Q_hats[i] - Q0_vars[i]
            Q1_vars[i] = max(Q_hat_adjusteds[i], 0)

            # =================== Model 4: Maximum Profit Model ===================

            # Calculate the demand up to k-1
            total_demand_up_to_k_minus_1 = sum(
                R_vars[i, k - 2] * demand_row[: k - 1].sum() for k in range(2, T)
            )
            total_demand_up_to_k_minus_1_vars[i] = total_demand_up_to_k_minus_1

            # Calculate the demand from k to T
            total_demand_from_k_to_T = sum(
                R_vars[i, k - 2] * demand_row[k - 1 :].sum() for k in range(2, T)
            )
            total_demand_from_k_to_T_vars[i] = total_demand_from_k_to_T

            Sold_0s[i] = min(total_demand_up_to_k_minus_1_vars[i], Q0_vars[i])
            Left_0s[i] = max(Q0_vars[i] - Sold_0s[i], 0)
            Lost_0s[i] = max(total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i], 0)
            Q1_plus_lefts[i] = Q1_vars[i] + Left_0s[i]

            Sold_1s[i] = min(total_demand_from_k_to_T_vars[i], Q1_plus_lefts[i])
            Left_1s[i] = max(Q1_plus_lefts[i] - Sold_1s[i], 0)
            Lost_1s[i] = max(total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i], 0)

            # assigned_R = max_r_index[i]

            # all_holding_costs_0[i] = (
            #     (Q0_vars[i] + Left_0s[i] + Q1_vars[i]) * ((assigned_R + 2) - 1) / 2
            # )
            # all_holding_costs_1[i] = (
            #     (Q1_vars[i] + Left_0s[i] + Left_1s[i]) * (T - (assigned_R + 2)) / 2
            # )

            profits_vars[i] = (
                (price - cost) * (Sold_0s[i] + Sold_1s[i])  # Revenue from sales
                - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # Lost sales cost
                - (cost - salvage_value) * Left_1s[i]
                # - holding_cost * (all_holding_costs_0[i] + all_holding_costs_1[i])
            )

        # Calculate the average profit
        results_df = pd.DataFrame(
            {
                "average_profits": [np.mean(profits_vars)],
                "average_loss": [np.mean((price - cost) * (Lost_0s[i] + Lost_1s[i]))],
                "average_left": [np.mean((price - cost) * (Lost_0s[i] + Lost_1s[i]))],
                "alpha_values": [alphas],
                "beta_balues": [betas],
            }
        )

        stimulation_result = pd.DataFrame(
            {
                "F": F_vars,
                "R(T)": [x + 2 for x in max_r_index],
                "Sold_0": Sold_0s,
                "Left_0": Left_0s,
                "Lost_0": Lost_0s,
                "Sold_1": Sold_1s,
                "Left_1": Left_1s,
                "Lost_1": Lost_1s,
                "profits": profits_vars,
                "Q0": Q0_vars,
                "Q1": Q1_vars,
                "hc0": all_holding_costs_0,
                "hc1": all_holding_costs_1,
            }
        )

        return results_df, stimulation_result

    # %%


    # %% [markdown]
    # ### S15 - Beta with Lasso

    # %%
    def cal_test_fully_flexible_beta_with_lasso_15(
        alphas,
        betas,
        salvage_value,
        cost,
        price,
        Q_star,
        demand_df_test,
        Qk_hat_df_test,
        testing_df,
    ):

        # ======================= Global Variables =======================

        # Category 1 - Some variables that is important to future work
        K = T - 2  # this is for k=2~T-1. => if T = 10(1~10), K will be 8. (0~7)
        n = len(demand_df_test)

        # Initialize lists or numpy arrays to replace Gurobi variables
        Sold_0s = np.zeros(n)
        Left_0s = np.zeros(n)
        Lost_0s = np.zeros(n)
        Sold_1s = np.zeros(n)
        Left_1s = np.zeros(n)
        Lost_1s = np.zeros(n)
        all_holding_costs_0 = np.zeros(n)
        all_holding_costs_1 = np.zeros(n)
        profits_vars = np.zeros(n)

        # 1-2. Arrays for demand calculation up to certain periods
        total_demand_up_to_k_minus_1_vars = np.zeros(n)
        total_demand_from_k_to_T_vars = np.zeros(n)
        Q1_plus_lefts = np.zeros(n)

        # 2. Variables for Model 2: Optimal Fraction Model
        f_vars = np.zeros(n)
        F_vars = np.zeros(n)  # Assuming values will be between 0 and 1
        Q0_vars = np.zeros(n)  # Replace Q_star with a specific value as needed

        # 3. Variables for Model 3: Optimal Order Time Model (2D array for binary values)
        tau_vars = np.zeros((n, K))
        exp_tau_vars = np.zeros((n, K))
        r_vars = np.zeros((n, K))
        max_r_index = np.zeros(n, dtype=int)
        R_vars = np.zeros(
            (n, K), dtype=int
        )  # Binary array to select one optimal replenishment time

        # 4. Variables for Model 4: Re-estimate order-up-to-level
        Q1_vars = np.zeros(n)
        Q_hats = np.zeros(n)
        Q_hat_adjusteds = np.zeros(n)

        # ======================= Start Stimulation! =======================

        for i, row in demand_df_test.iterrows():

            ### Data for this stimulation
            demand_row = demand_df_test.iloc[i]
            Qk_hat_df_test_row = Qk_hat_df_test.iloc[i]
            X_data = testing_df.iloc[i].tolist()
            X_data.append(1)

            # =================== Model 1: Optimal Fraction Model ===================

            ### 用線性回歸計算F_var
            f_vars[i] = sum(X_data[j] * alphas[j] for j in range(features_num + 1))
            F_vars[i] = 1 / (1 + np.exp(-(f_vars[i])))
            Q0_vars[i] = F_vars[i] * Q_star

            # =================== Model 2: Optimal Order Time Model ===================

            # Step 1: Calculate tau_vars as a linear combination of X_data and betas
            for k in range(K):
                tau_vars[i, k] = sum(
                    X_data[j] * betas[k, j] for j in range(features_num + 1)
                )

            max_r_index[i] = np.argmax(tau_vars[i])
            R_vars[i, max_r_index[i]] = 1

            # print(f"tau: {tau_vars[i]}")
            # print(f"R: {R_vars[i]}")
            # print(f"max_r_index: {max_r_index[i]}")
            # print("\n\n")

            # ============ Model 3: re-estimate order-up-to-level =================

            Q_hats[i] = sum(
                R_vars[i, k - 2] * Qk_hat_df_test_row[k - 2] for k in range(2, T)
            )
            Q_hat_adjusteds[i] = Q_hats[i] - Q0_vars[i]
            Q1_vars[i] = max(Q_hat_adjusteds[i], 0)

            # =================== Model 4: Maximum Profit Model ===================

            # Calculate the demand up to k-1
            total_demand_up_to_k_minus_1 = sum(
                R_vars[i, k - 2] * demand_row[: k - 1].sum() for k in range(2, T)
            )
            total_demand_up_to_k_minus_1_vars[i] = total_demand_up_to_k_minus_1

            # Calculate the demand from k to T
            total_demand_from_k_to_T = sum(
                R_vars[i, k - 2] * demand_row[k - 1 :].sum() for k in range(2, T)
            )
            total_demand_from_k_to_T_vars[i] = total_demand_from_k_to_T

            Sold_0s[i] = min(total_demand_up_to_k_minus_1_vars[i], Q0_vars[i])
            Left_0s[i] = max(Q0_vars[i] - Sold_0s[i], 0)
            Lost_0s[i] = max(total_demand_up_to_k_minus_1_vars[i] - Q0_vars[i], 0)
            Q1_plus_lefts[i] = Q1_vars[i] + Left_0s[i]

            Sold_1s[i] = min(total_demand_from_k_to_T_vars[i], Q1_plus_lefts[i])
            Left_1s[i] = max(Q1_plus_lefts[i] - Sold_1s[i], 0)
            Lost_1s[i] = max(total_demand_from_k_to_T_vars[i] - Q1_plus_lefts[i], 0)

            # assigned_R = max_r_index[i]

            # all_holding_costs_0[i] = (
            #     (Q0_vars[i] + Left_0s[i] + Q1_vars[i]) * ((assigned_R + 2) - 1) / 2
            # )
            # all_holding_costs_1[i] = (
            #     (Q1_vars[i] + Left_0s[i] + Left_1s[i]) * (T - (assigned_R + 2)) / 2
            # )

            profits_vars[i] = (
                (price - cost) * (Sold_0s[i] + Sold_1s[i])  # Revenue from sales
                - (price - cost) * (Lost_0s[i] + Lost_1s[i])  # Lost sales cost
                - (cost - salvage_value) * Left_1s[i]
                # - holding_cost * (all_holding_costs_0[i] + all_holding_costs_1[i])
            )

        # Calculate the average profit
        results_df = pd.DataFrame(
            {
                "average_profits": [np.mean(profits_vars)],
                "average_loss": [np.mean((price - cost) * (Lost_0s[i] + Lost_1s[i]))],
                "average_left": [np.mean((price - cost) * (Lost_0s[i] + Lost_1s[i]))],
                "alpha_values": [alphas],
                "beta_balues": [betas],
            }
        )

        stimulation_result = pd.DataFrame(
            {
                "F": F_vars,
                "R(T)": [x + 2 for x in max_r_index],
                "Sold_0": Sold_0s,
                "Left_0": Left_0s,
                "Lost_0": Lost_0s,
                "Sold_1": Sold_1s,
                "Left_1": Left_1s,
                "Lost_1": Lost_1s,
                "profits": profits_vars,
                "Q0": Q0_vars,
                "Q1": Q1_vars,
                "hc0": all_holding_costs_0,
                "hc1": all_holding_costs_1,
            }
        )

        return results_df, stimulation_result

    # %%


    # %% [markdown]
    # # MAIN
    # 

    # %% [markdown]
    # ## Setting reasonable parameters
    # 

    # %%
    status = "train"

    service_lv = calculate_service_level(
        salvage_value=salvage_value, cost=cost, price=price
    )
    print(f"service_lv: {service_lv}")

    # %% [markdown]
    # ## K-folds training

    # %% [markdown]
    # ### Training & Testing Function 

    # %%
    # This is for single fold training.
    def perform_fold_training(
        training_df, demand_df_train, Qk_hat_df_train, Q_star
    ) -> dict[str, float]:

        # 1. Baseline model
        (
            baseline_avg_losses,
            baseline_avg_lefts,
            baseline_avg_profits,
            baseline_avg_operation_profits,
            baseline_stimulation_df,
        ) = one_time_procurement(
            Q_star=Q_star,
            demand_df=demand_df_train,
            cost=cost,
            price=price,
            salvage_value=salvage_value,
        )

        # 2. S1 - Grid F & Grid R

        results_df_1, stimulation_results_df_1 = None, None

        results_df_1, stimulation_results_df_1 = grid_fixed_F_fixed_R(
            assigned_Ts=ASSIGNED_TS,
            assigned_Fs=ASSIGNED_FS,
            cost=cost,
            price=price,
            salvage_value=salvage_value,
            Qk_hat_df=Qk_hat_df_train,
            demand_df_train=demand_df_train,
            Q_star=Q_star,
        )

        S1_profit_training = results_df_1.iloc[0]["average_profits"]

        # 3. S2 - Grid R & Flexible F

        results_df_2, stimulation_results_df_2 = None, None
        results_df_2, stimulation_results_df_2 = grid_flexible_F_fixed_R(
            assigned_Ts=ASSIGNED_TS,
            salvage_value=salvage_value,
            cost=cost,
            price=price,
            Q_star=Q_star,
            demand_df_train=demand_df_train,
            Qk_hat_df_train=Qk_hat_df_train,
            training_df=training_df,
        )

        S2_profit_training = results_df_2.iloc[0]["average_profits"]

        # 4. S12 - Beta without r

        # results_df_12, stimulation_results_df_12 = None, None
        # results_df_12, stimulation_results_df_12 = fully_flexible_beta_with_softmax_12(
        #     salvage_value=salvage_value,
        #     cost=cost,
        #     price=price,
        #     Q_star=Q_star,
        #     demand_df_train=demand_df_train,
        #     Qk_hat_df=Qk_hat_df_train,
        #     training_df=training_df,
        # )
        # if results_df_12 is not None:
        #     S12_profit_training = results_df_12.iloc[0]["average_profits"]
        # else:
        #     S12_profit_training = None

        # 5. S14 - Optimized F & Rk
        results_df_14, stimulation_results_df_14 = None, None
        results_df_14, stimulation_results_df_14 = cal_optimized_F_R(
            salvage_value=salvage_value,
            cost=cost,
            price=price,
            Q_star=Q_star,
            demand_df=demand_df_train,
            Qk_hat_df=Qk_hat_df_train,
            training_df=training_df,
        )
        S14_profit_training = results_df_14.iloc[0]["average_profits"]

        # # 6. S15 - Beta with Lasso
        # results_df_15, stimulation_results_df_15 = None, None
        # results_df_15, stimulation_results_df_15 = fully_flexible_beta_with_lasso_15(
        #     salvage_value=salvage_value,
        #     cost=cost,
        #     price=price,
        #     Q_star=Q_star,
        #     demand_df_train=demand_df_train,
        #     Qk_hat_df=Qk_hat_df_train,
        #     training_df=training_df,
        #     lambda_beta=LASSO_BETA,
        # )
        # if results_df_15 is not None:
        #     S15_profit_training = results_df_15.iloc[0]["average_profits"]
        # else:
        #     S15_profit_training = None

        # print(f"baseline_profit: {baseline_avg_profits}")
        # print(f"S1_profit_training: {S1_profit_training}")
        # print(f"S2_profit_training: {S2_profit_training}")
        # print(f"S12_profit_training: {S12_profit_training}")
        # print(f"S14_profit_training: {S14_profit_training}")
        # print(f"S15_profit_training: {S15_profit_training}")

        # 整理利潤結果
        training_profits = {
            "baseline": baseline_avg_profits,
            "S1": S1_profit_training,
            "S2": S2_profit_training,
            # "S12": S12_profit_training,
            # "S15": S15_profit_training,
            "S12": None,
            "S15": None,
            "S14": S14_profit_training,
        }

        training_results = {
            "S1": results_df_1,
            "S2": results_df_2,
            # "S12": results_df_12,
            # "S15": results_df_15,
            "S12": None,
            "S15": None,
            "S14": results_df_14,
        }

        training_stimulation_results = {
            "baseline": baseline_stimulation_df,
            "S1": stimulation_results_df_1,
            "S2": stimulation_results_df_2,
            # "S12": stimulation_results_df_12,
            # "S15": stimulation_results_df_15,
            "S12": None,
            "S15": None,
            "S14": stimulation_results_df_14,
        }

        return training_profits, training_results, training_stimulation_results

    # %%
    # This is for single fold testing.


    def perform_fold_testing(
        results_df_1,
        results_df_2,
        results_df_12,
        results_df_15,
        demand_df_test,
        Qk_hat_df_test,
        Q_star,
        testing_df,
    ) -> dict[str, float]:

        # 1. Baseline model

        (
            test_baseline_avg_loss,
            test_baseline_avg_lefts,
            test_baseline_avg_profits,
            test_baseline_avg_operation_profits,
            test_stimulation_df_baseline,
        ) = one_time_procurement(
            Q_star=Q_star,
            demand_df=demand_df_test,
            cost=cost,
            price=price,
            salvage_value=salvage_value,
        )

        # print(f"baseline_profit: {test_baseline_avg_profits}")

        # 2. S1 - Grid F & Grid R
        if results_df_1 is not None:
            assigned_T = results_df_1.iloc[0]["R(T)"]
            assigned_F = results_df_1.iloc[0]["F"]

            test_results_df_1, test_stimulation_results_df_1 = cal_test_fixed_F_fixed_R(
                assigned_T=int(assigned_T),
                assigned_F=assigned_F,
                salvage_value=salvage_value,
                cost=cost,
                price=price,
                Q_star=Q_star,
                demand_df_test=demand_df_test,
                Qk_hat_df_test=Qk_hat_df_test,
            )

        S1_profit_testing = test_results_df_1.iloc[0]["average_profits"]

        # 3. S2 - Grid R & Flexible F

        if results_df_2 is not None and len(results_df_2) > 0:
            assigned_R = results_df_2.iloc[0]["R"]
            alphas = results_df_2.iloc[0]["alpha_values"]

            test_results_df_2, test_stimulation_results_df_2 = cal_test_flexible_F_fixed_R(
                assigned_R=assigned_R[0],
                alphas=alphas,
                salvage_value=salvage_value,
                cost=cost,
                price=price,
                Q_star=Q_star,
                demand_df_test=demand_df_test,
                Qk_hat_df_test=Qk_hat_df_test,
                testing_df=testing_df,
            )

        S2_profit_testing = test_results_df_2.iloc[0]["average_profits"]

        # # 4. S12 - Beta without r
        # test_results_df_12, test_stimulation_results_df_12 = None, None
        # if results_df_12 is not None:
        #     alphas = results_df_12.iloc[0]["alpha_values"]
        #     betas = results_df_12.iloc[0]["beta_values"]

        #     test_results_df_12, test_stimulation_results_df_12 = (
        #         cal_test_fully_flexible_beta_with_softmax_12(
        #             alphas=alphas,
        #             betas=betas,
        #             salvage_value=salvage_value,
        #             cost=cost,
        #             price=price,
        #             Q_star=Q_star,
        #             demand_df_test=demand_df_test,
        #             Qk_hat_df_test=Qk_hat_df_test,
        #             testing_df=testing_df,
        #         )
        #     )
        #     S12_profit_testing = test_results_df_12.iloc[0]["average_profits"]
        # else:
        #     S12_profit_testing = None

        # 5. S14 - Optimized F & Rk
        test_results_df_14, test_stimulation_results_df_14 = cal_optimized_F_R(
            salvage_value=salvage_value,
            cost=cost,
            price=price,
            Q_star=Q_star,
            demand_df=demand_df_test,
            Qk_hat_df=Qk_hat_df_test,
            training_df=testing_df,
        )

        S14_profit_testing = test_results_df_14.iloc[0]["average_profits"]

        # # 6. S15 - Beta with Lasso
        # test_results_df_15, test_stimulation_results_df_15 = None, None
        # if results_df_15 is not None:
        #     alphas = results_df_15.iloc[0]["alpha_values"]
        #     betas = results_df_15.iloc[0]["beta_values"]

        #     test_results_df_15, test_stimulation_results_df_15 = (
        #         cal_test_fully_flexible_beta_with_lasso_15(
        #             alphas=alphas,
        #             betas=betas,
        #             salvage_value=salvage_value,
        #             cost=cost,
        #             price=price,
        #             Q_star=Q_star,
        #             demand_df_test=demand_df_test,
        #             Qk_hat_df_test=Qk_hat_df_test,
        #             testing_df=testing_df,
        #         )
        #     )
        #     S15_profit_testing = test_results_df_15.iloc[0]["average_profits"]
        # else:
        #     S15_profit_testing = None

        # print(f"baseline_profit: {test_baseline_avg_profits}")
        # print(f"S1_profit_testing: {S1_profit_testing}")
        # print(f"S2_profit_testing: {S2_profit_testing}")
        # print(f"S12_profit_testing: {S12_profit_testing}")
        # print(f"S14_profit_testing: {S14_profit_testing}")
        # print(f"S15_profit_testing: {S15_profit_testing}")

        # 整理利潤結果
        testing_profits = {
            "baseline": test_baseline_avg_profits,
            "S1": S1_profit_testing,
            "S2": S2_profit_testing,
            # "S12": S12_profit_testing,
            # "S15": S15_profit_testing,
            # "S12": None,
            # "S15": None,
            "S14": S14_profit_testing,
        }

        testing_stimulation_results = {
            "baseline": test_stimulation_df_baseline,
            "S1": test_stimulation_results_df_1,
            "S2": test_stimulation_results_df_2,
            # "S12": test_stimulation_results_df_12,
            # "S15": test_stimulation_results_df_15,
            # "S12": None,
            # "S15": None,
            "S14": test_stimulation_results_df_14,
        }

        return testing_profits, testing_stimulation_results


    train_all_fold_profits = []
    train_all_fold_stimulation_results = []
    test_all_fold_profits = []
    test_all_fold_stimulation_results = []
    beta_records = {"S12": [], "S15": []}

    # 迴圈遍歷所有 fold
    for fold_idx in range(len(training_data_folds)):
        print(f"===== Processing Fold {fold_idx + 1} =====")
        # 取出該 fold 的訓練資料與需求資料
        training_df, testing_df = training_data_folds[fold_idx]
        demand_df_train, demand_df_test = demand_folds[fold_idx]

        Q_star = calculate_Q_star(demand_df_train, service_level=service_lv)
        print(f"Fold {fold_idx + 1} Q_star: {Q_star}")

        # ====訓練階段====

        mu_matrix, covariance_matrix = cal_mu_and_cov_matrix(demand_df_train)
        Qk_hat_df_train = make_Qk_hat_df(
            demand_df_train, T, service_lv, mu_matrix, covariance_matrix
        )
        training_profits, training_results, training_stimulation_results = (
            perform_fold_training(training_df, demand_df_train, Qk_hat_df_train, Q_star)
        )
        train_all_fold_profits.append(training_profits)
        train_all_fold_stimulation_results.append(training_stimulation_results)

        # if training_results["S12"] is not None:
        #     beta_records["S12"].append(training_results["S12"].iloc[0]["beta_values"])
        # else:
        #     beta_records["S12"].append(None)

        # if training_results["S15"] is not None:
        #     beta_records["S15"].append(training_results["S15"].iloc[0]["beta_values"])
        # else:
        #     beta_records["S15"].append(None)

        # ====測試階段====
        print(f"Fold {fold_idx + 1} Q_star: {Q_star}")

        mu_matrix, covariance_matrix = cal_mu_and_cov_matrix(demand_df_test)
        Qk_hat_df_test = make_Qk_hat_df(
            demand_df_test, T, service_lv, mu_matrix, covariance_matrix
        )
        testing_profits, testing_stimulation_results = perform_fold_testing(
            training_results["S1"],
            training_results["S2"],
            training_results["S12"],
            training_results["S15"],
            demand_df_test,
            Qk_hat_df_test,
            Q_star,
            testing_df,
        )

        test_all_fold_profits.append(testing_profits)
        test_all_fold_stimulation_results.append(testing_stimulation_results)


    # 將所有 fold 的結果轉換為 DataFrame 便於檢查與保存
    train_all_fold_profit_df = pd.DataFrame(train_all_fold_profits)
    # print("All train fold profits:")
    # print(train_all_fold_profit_df)

    test_all_fold_profit_df = pd.DataFrame(test_all_fold_profits)
    # print("All test fold profits:")
    # print(test_all_fold_profit_df)
    
    return train_all_fold_profit_df, test_all_fold_profit_df


# 參數空間
CHUNK_SIZES = [400, 500]
# LASSO_BETAS = [10, 100, 1000]
LASSO_BETAS = [1]
FOLDS = [10]
# FOLDS = [3, 5, 10]


# CHUNK_SIZES = [10, 20, 25]
# LASSO_BETAS = [10, 100, 1000]
# FOLDS = [1, 2, 3]

os.makedirs("results_0327", exist_ok=True)

summary = []
for chunk_size, lasso_beta, folds in itertools.product(CHUNK_SIZES, LASSO_BETAS, FOLDS):
    CHUNK_SIZE = chunk_size
    data_size = CHUNK_SIZE * folds
    LASSO_BETA = lasso_beta

    print(f"Running: chunk={chunk_size}, lasso={lasso_beta}, folds={folds}")
    train_df, test_df = perform_single(data_size)

    # 存檔路徑
    tag = f"chunk{chunk_size}_lasso{lasso_beta}_fold{folds}"
    train_df.to_csv(f"results_0327/train_{tag}.csv", index=False)
    test_df.to_csv(f"results_0327/test_{tag}.csv", index=False)

    # 收集 summary metrics
    summary.append({
        "chunk_size": chunk_size,
        "lasso_beta": lasso_beta,
        "folds": folds,
        "train_df": train_df.to_dict(orient="records"),
        "test_df": test_df.to_dict(orient="records"),
    })

# 輸出 summary table
summary_df = pd.DataFrame(summary)
summary_df.to_csv("results_0327/summary.csv", index=False)
print("All experiments finished. Summary saved to results_0327/summary.csv")
print(summary_df)
