
import pandas as pd
import numpy as np

T = 10
M = 5000000
ASSIGNED_FS = np.arange(0.1, 1.0, 0.1)
ASSIGNED_TS = list(range(2, T))  # 2 到 T-1

np.random.seed(0)

class BaselineModel():
    def __init__(self):
        pass

    def one_time_procurement(self, Q_star, demand_df, cost, price, salvage_value):

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

            for day, demand in enumerate(row):
                sales = min(inventory, demand)
                loss = max(demand - inventory, 0)
                left = max(inventory - sales, 0)
                total_sold += sales
                total_lost += loss

                inventory -= sales

                if day == len(row) - 1:
                    left_penalty_cost = (cost - salvage_value) * left
                    lefts.append(left)
                else:
                    left_penalty_cost = 0

            operation_profit = (price - cost) * total_sold
            profit = operation_profit - left_penalty_cost - (price - cost) * total_lost

            all_losses.append(total_lost)
            all_lefts.append(sum(lefts))
            all_operation_profits.append(operation_profit)
            all_profits.append(profit)

        avg_losses = np.mean(all_losses)
        avg_lefts = np.mean(all_lefts)
        avg_operation_profits = np.mean(all_operation_profits)
        avg_profits = np.mean(all_profits)

        stimulation_df = pd.DataFrame(
            {
                "losses": all_losses,
                "lefts": all_lefts,
                "operation_profits": all_operation_profits,
                "profits": all_profits,
            }
        )

        return avg_losses, avg_lefts, avg_profits, avg_operation_profits, stimulation_df