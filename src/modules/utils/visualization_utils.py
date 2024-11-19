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


class VisualizationUtils(Base):
    def __init__(self, current_timestamp: datetime):
        super().__init__(current_timestamp)

    def plot_strategies_profits_scatter(self, save_type, dfs: list):

        if len(dfs) <= 1:
            print("No dataframes to plot.")
            return None

        # 生成所有兩兩配對
        pairs = list(itertools.combinations(range(len(dfs)), 2))
        num_pairs = len(pairs)

        # 計算網格大小
        grid_size = math.ceil(math.sqrt(num_pairs))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle("Scatter Plots of Profits (Matrix View)")

        # 繪製每個配對的散佈圖
        for idx, (i, j) in enumerate(pairs):
            row, col = divmod(idx, grid_size)

            if (dfs[i] is None or len(dfs[i]) == 0) or (
                dfs[j] is None or len(dfs[j]) == 0
            ):
                continue

            profits_i = dfs[i]["profits"]
            profits_j = dfs[j]["profits"]

            if len(profits_i) != len(profits_j):
                continue

            ax = axes[row, col]
            ax.scatter(
                profits_i, profits_j, alpha=0.6, label=f"Profits {i} vs Profits {j}"
            )
            ax.set_xlabel(f"Profits {i}")
            ax.set_ylabel(f"Profits {j}")

            # 繪製 45 度虛線
            max_val = max(profits_i.max(), profits_j.max())
            min_val = min(profits_i.min(), profits_j.min())
            ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)

            ax.legend()
            ax.set_title(f"Profits {i} vs Profits {j}")

        # 隱藏未使用的子圖軸
        for idx in range(num_pairs, grid_size * grid_size):
            row, col = divmod(idx, grid_size)
            fig.delaxes(axes[row, col])

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        name = "plot_strategies_profits_scatter"

        os.makedirs("plots", exist_ok=True)
        save_path = (
            f"plots/{name}_{save_type}_{self.data_size}_{self.current_timestamp}.png"
        )

        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"Plot saved as {save_path}")

        plt.show()
        plt.close()

    def plot_relative_profit_deviation(self, save_type, baseline_profit, max_profits):

        print(f"Baseline is: {baseline_profit}")
        for i, profit in enumerate(max_profits):
            print(f"S{i+1}'s profit: {profit}")

        # 計算相對值
        ratios = {}
        for idx, max_profit in enumerate(max_profits, start=1):
            if max_profit is not None and max_profit != -1:
                ratio = max_profit / baseline_profit
                ratios[f"S{idx}"] = ratio - 1  # 相對偏差

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
        save_path = (
            f"plots/{name}_{save_type}_{self.data_size}_{self.current_timestamp}.png"
        )

        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"Plot saved as {save_path}")

        plt.show()
        plt.close()

    def plot_relative_profit_comparison(
        self,
        save_type,
        train_baseline_profit,
        test_baseline_profit,
        test_max_profits,
        train_max_profits,
    ):

        test_ratios, train_ratios = {}, {}
        for idx, (test_profit, train_profit) in enumerate(
            zip(test_max_profits, train_max_profits), start=1
        ):
            if test_profit is not None and test_profit != -1:
                test_ratio = (
                    test_profit / test_baseline_profit - 1
                )  # Relative deviation
                test_ratios[f"S{idx}"] = test_ratio
            if train_profit is not None and train_profit != -1:
                train_ratio = (
                    train_profit / train_baseline_profit - 1
                )  # Relative deviation
                train_ratios[f"S{idx}"] = train_ratio

        # Define the range of the y-axis
        y_min = (
            min(
                min(test_ratios.values(), default=0),
                min(train_ratios.values(), default=0),
            )
            - 0.1
        )
        y_max = (
            max(
                max(test_ratios.values(), default=0),
                max(train_ratios.values(), default=0),
            )
            + 0.1
        )

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
        plt.ylim(y_min, y_max)
        plt.legend()

        name = "plot_relative_profit_comparison"

        os.makedirs("plots", exist_ok=True)
        save_path = (
            f"plots/{name}_{save_type}_{self.data_size}_{self.current_timestamp}.png"
        )

        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"Plot saved as {save_path}")

        plt.show()
        plt.close()

    def plot_Q0_Q1_distribution(self, save_type, stimulation_results_dfs):

        for idx, df in enumerate(stimulation_results_dfs, start=1):
            if df is None:
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
            save_path = f"plots/{name}_{save_type}_{self.data_size}_S{idx}_{self.current_timestamp}.png"

            plt.savefig(save_path, format="png", bbox_inches="tight")
            print(f"Plot saved as {save_path}")

            plt.show()

    def plot_profits_deviation_box_plot(
        self, save_type, stimulation_results_dfs, baseline_avg_profits
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
                save_path = f"plots/{name}_{save_type}_{self.data_size}_S{idx}_{self.current_timestamp}.png"

                plt.savefig(save_path, format="png", bbox_inches="tight")
                print(f"Plot saved as {save_path}")

                plt.show()
            else:
                print(
                    f"Skipping stimulation_results_df_{idx}: Missing 'profits' column."
                )
