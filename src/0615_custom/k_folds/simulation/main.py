import pandas as pd
from simulation import Simulation


class Main:
    def __init__(self):
        pass

    def main(
        self,
        Qk,
        cost: int,
        price: int,
        salvage_value: int,
        full_df: pd.DataFrame,
        demand_df: pd.DataFrame,
    ):
        self.simulation = Simulation(
            cost=cost,
            price=price,
            salvage_value=salvage_value,
            full_df=full_df,
            demand_df=demand_df,
        )
        (
            train_profit_df,
            test_profit_df,
            training_stimulation_result_df,
            testing_stimulation_result_df,
        ) = self.simulation.experiment(Qk)
        return (
            train_profit_df,
            test_profit_df,
            training_stimulation_result_df,
            testing_stimulation_result_df,
        )


if __name__ == "__main__":

    """
    1. 先到 src/0615_custom/k_folds/simulation/configs.py 修改自己的 params
    2. 以下 salvage_value, cost, price 都可以自己修改
    3. 輸入算好的 Qk: list[tuple] -> [(k, Qk)]
    4. full_df -> 訓練數據
    5. demand_df -> 模擬出的 demand 數據

    """

    # 輸入參數
    salvage_value = 0
    cost = 400
    price = 1000

    Qk = [
        (1, 100),
        (5, 100),
    ]  # 代表有兩筆資料，其中一筆是在 k=1 時 Qk =100, 第二筆是 k=5 時 Qk=100
    full_df = pd.read_csv("your_full_data.csv")  # ← 請替換成實際路徑
    demand_df = pd.read_csv("your_demand_data.csv")  # ← 請替換成實際路徑

    # 執行主程式
    app = Main()
    (
        train_profit_df,
        test_profit_df,
        training_stimulation_result_df,
        testing_stimulation_result_df,
    ) = app.main(Qk, cost, price, salvage_value, full_df, demand_df)
