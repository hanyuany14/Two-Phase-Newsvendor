
import numpy as np
import pandas as pd

from src.modules.strategies.train_base import Train

class Train_S0(Train):
    def __init__(self):
        super().__init__()
        
    def one_time_procurement(self):

        all_losses = []
        all_lefts = []
        all_operation_profits = []
        all_profits = []

        for i, row in self.demand_df.iterrows():
            inventory = self.Q_star
            losses = []
            lefts = []
            daily_operation_profits = []
            daily_profits = []
            total_sold = 0  # 追蹤總售出量
            total_lost = 0  # 追蹤總丟失量

            # print("=" * 50)
            # print(
            #     f"Processing row {i+1}/{len(self.demand_df)} with initial inventory self.Q_star={self.Q_star}"
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
                    left_penalty_cost = (self.cost - self.salvage_value) * left
                    lefts.append(left)
                    print(f"End of period: Left Penalty self.Cost = {left_penalty_cost}")
                    print("-" * 50)
                else:
                    left_penalty_cost = 0

            operation_profit = (self.price - self.cost) * total_sold
            profit = operation_profit - left_penalty_cost - (self.price - self.cost) * total_lost

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