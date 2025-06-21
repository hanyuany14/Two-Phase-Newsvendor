import pandas as pd
from simulation import Simulation


class Main:
    def __init__(self):
        pass

    def main(
        self,
        cost: int,
        price: int,
        salvage_value: int,
        time_period: int,
        simulation_times: int,
        items_num_train: int,
        items_num_test: int,
        Qks_test: list[float],
        demand_df_train: pd.DataFrame,
        demand_df_test: pd.DataFrame,
    ):
        self.simulation = Simulation(
            cost=cost,
            price=price,
            salvage_value=salvage_value,
            simulation_times=simulation_times,
            time_period=time_period,
            items_num_train=items_num_train,
            items_num_test=items_num_test,
        )

        train_profit_df, training_stimulation_result_df, training_results = (
            self.simulation.experiment(
                Qks_test=Qks_test,
                demand_df_train=demand_df_train,
                demand_df_test=demand_df_test,
            )
        )

        return train_profit_df, training_stimulation_result_df, training_results


# if __name__ == "__main__":

#     """
#     1. 先到 src/0615_custom/k_folds/simulation/configs.py 修改自己的 params
#     2. 以下 salvage_value, cost, price 都可以自己修改
#     3. 輸入算好的 Qk: list[tuple] -> [(k, Qk)]
#     4. full_df -> 訓練數據
#     5. demand_df -> 模擬出的 demand 數據

#     """

#     # 輸入參數
#     salvage_value = 0
#     cost = 400
#     price = 1000
#     time_period = 10
#     simulation_times = 1000  # 代表每一個 Qk 包含多少筆
#     items_num = 5

#     # input data
#     demand_df_train = pd.read_csv("your_demand_data.csv")  # ← 請替換成實際路徑
#     demand_df_test = pd.read_csv("your_demand_data.csv")  # ← 請替換成實際路徑

#     Qks_test = [
#         (1, 100),
#         (5, 100),
#     ]  # 代表有兩筆資料，其中一筆是在 k=1 時 Qk =100, 第二筆是 k=5 時 Qk=100

#     # 執行主程式
#     app = Main()

#     (
#         train_profit_df,
#         training_stimulation_result_df,
#         training_results,
#     ) = app.main(
#         cost=cost,
#         price=price,
#         salvage_value=salvage_value,
#         time_period=time_period,
#         items_num=items_num,
#         simulation_times=simulation_times,
#         Qks_test=Qks_test,
#         demand_df_train=demand_df_train,
#         demand_df_test=demand_df_test,
#     )
