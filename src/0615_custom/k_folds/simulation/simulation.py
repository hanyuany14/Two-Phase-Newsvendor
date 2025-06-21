import gurobipy as gp
import pandas as pd
import numpy as np
from scipy.stats import norm

from qk_hat import Qk_hat
from baseline_model import BaselineModel
from s1_model import S1_Model
from s2_model import S2_Model

np.random.seed(0)

T = 10
M = 5000000

ASSIGNED_FS = np.arange(0.1, 1.0, 0.1)
ASSIGNED_TS = list(range(2, T))  # 2 到 T-1

THREADS = 12
TIME_LIMIT = 20000
MIPGAP = 0.01


class Simulation:
    def __init__(
        self,
        cost: int,
        price: int,
        salvage_value: int,
        full_df: pd.DataFrame,
        demand_df: pd.DataFrame,
    ):
        self.cost = cost
        self.price = price
        self.salvage_value = salvage_value

        self.service_lv = self.calculate_service_level(
            salvage_value=salvage_value, cost=cost, price=price
        )
        self.features_num = full_df.shape[1]

        self.qk_hat = Qk_hat()
        self.baseline_model = BaselineModel()
        self.s1_model = S1_Model()
        self.s2_model = S2_Model()

        self.prepare_data(full_df, demand_df, train_size=0.5)

    def experiment(self, Qks_train: list[int], Qks_test: list[int]):
        """做實驗，包含訓練以及測試

        Args:
            Qks_train (list[int]): 從訓練資料得到的 Qk
            Qks_test (list[int]): 從測試資料得到的 Qk

        Returns:
            _type_: _description_
        """

        training_df = self.training_df
        testing_df = self.testing_df
        demand_df_train = self.demand_df_train
        demand_df_test = self.demand_df_test

        Q_star = self.calculate_Q_star(demand_df_train, service_level=self.service_lv)
        print(f"Q_star: {Q_star}")

        # ====訓練階段====
        Qk_hats_train = self.qk_hat.make_Qk_hat_df_with_known_Qk(demand_df_train, Qk)
        training_profits, training_results, training_stimulation_results = (
            self.perform_fold_training(
                training_df, demand_df_train, Qk_hats_train, Q_star
            )
        )

        # ====測試階段====
        Qk_hat_df_test = self.qk_hat.make_Qk_hat_df_with_known_Qk(demand_df_test, Qk)
        testing_profits, testing_stimulation_results = self.perform_fold_testing(
            training_results["S1"],
            training_results["S2"],
            demand_df_test,
            Qk_hat_df_test,
            Q_star,
            testing_df,
        )

        train_profit_df = pd.DataFrame(training_profits)
        test_profit_df = pd.DataFrame(testing_profits)

        training_stimulation_result_df = pd.DataFrame(training_stimulation_results)
        testing_stimulation_result_df = pd.DataFrame(testing_stimulation_results)

        return (
            train_profit_df,
            test_profit_df,
            training_stimulation_result_df,
            testing_stimulation_result_df,
        )

    def perform_fold_training(
        self,
        training_df: pd.DataFrame,
        demand_df_train: pd.DataFrame,
        Qk_hats_train: list[int],
        Q_star: float | int,
    ) -> dict[str, float]:
        """This is for single fold training."""

        # 1. Baseline model
        (
            baseline_avg_losses,
            baseline_avg_lefts,
            baseline_avg_profits,
            baseline_avg_operation_profits,
            baseline_stimulation_df,
        ) = self.baseline_model.one_time_procurement(
            Q_star=Q_star,
            demand_df=demand_df_train,
            cost=self.cost,
            price=self.price,
            salvage_value=self.salvage_value,
        )

        # 2. S1 - Grid F & Grid R
        results_df_1, stimulation_results_df_1 = None, None
        results_df_1, stimulation_results_df_1 = self.s1_model.grid_fixed_F_fixed_R(
            assigned_Ts=ASSIGNED_TS,
            assigned_Fs=ASSIGNED_FS,
            cost=self.cost,
            price=self.price,
            salvage_value=self.salvage_value,
            Qk_hat_df=Qk_hats_train,
            demand_df_train=demand_df_train,
            Q_star=Q_star,
        )

        S1_profit_training = results_df_1.iloc[0]["average_profits"]

        # 3. S2 - Grid R & Flexible F
        results_df_2, stimulation_results_df_2 = None, None
        results_df_2, stimulation_results_df_2 = self.s2_model.grid_flexible_F_fixed_R(
            assigned_Ts=ASSIGNED_TS,
            salvage_value=self.salvage_value,
            cost=self.cost,
            price=self.price,
            Q_star=Q_star,
            demand_df_train=demand_df_train,
            Qk_hat_df_train=Qk_hats_train,
            training_df=training_df,
        )

        S2_profit_training = results_df_2.iloc[0]["average_profits"]

        training_profits = {
            "baseline": baseline_avg_profits,
            "S1": S1_profit_training,
            "S2": S2_profit_training,
        }

        training_results = {
            "S1": results_df_1,
            "S2": results_df_2,
        }

        training_stimulation_results = {
            "baseline": baseline_stimulation_df,
            "S1": stimulation_results_df_1,
            "S2": stimulation_results_df_2,
        }

        return training_profits, training_results, training_stimulation_results

    def perform_fold_testing(
        self,
        results_df_1,
        results_df_2,
        demand_df_test,
        Qk_hat_df_test,
        Q_star,
        testing_df,
    ) -> dict[str, float]:
        """This is for testing

        Args:
            results_df_1 (_type_): _description_
            results_df_2 (_type_): _description_
            demand_df_test (_type_): _description_
            Qk_hat_df_test (_type_): _description_
            Q_star (_type_): _description_
            testing_df (_type_): _description_

        Returns:
            dict[str, float]: _description_
        """

        # 1. Baseline model
        (
            test_baseline_avg_loss,
            test_baseline_avg_lefts,
            test_baseline_avg_profits,
            test_baseline_avg_operation_profits,
            test_stimulation_df_baseline,
        ) = self.baseline_model.one_time_procurement(
            Q_star=Q_star,
            demand_df=demand_df_test,
            cost=self.cost,
            price=self.price,
            salvage_value=self.salvage_value,
        )

        print(f"baseline_profit: {test_baseline_avg_profits}")

        # 2. S1 - Grid F & Grid R
        if results_df_1 is not None:
            assigned_T = results_df_1.iloc[0]["R(T)"]
            assigned_F = results_df_1.iloc[0]["F"]

            test_results_df_1, test_stimulation_results_df_1 = (
                self.s1_model.cal_test_fixed_F_fixed_R(
                    assigned_T=int(assigned_T),
                    assigned_F=assigned_F,
                    salvage_value=self.salvage_value,
                    cost=self.cost,
                    price=self.price,
                    Q_star=Q_star,
                    demand_df_test=demand_df_test,
                    Qk_hat_df_test=Qk_hat_df_test,
                )
            )

        S1_profit_testing = test_results_df_1.iloc[0]["average_profits"]

        # 3. S2 - Grid R & Flexible F
        if results_df_2 is not None and len(results_df_2) > 0:
            assigned_R = results_df_2.iloc[0]["R"]
            alphas = results_df_2.iloc[0]["alpha_values"]

            test_results_df_2, test_stimulation_results_df_2 = (
                self.s2_model.cal_test_flexible_F_fixed_R(
                    assigned_R=assigned_R[0],
                    alphas=alphas,
                    salvage_value=self.salvage_value,
                    cost=self.cost,
                    price=self.price,
                    Q_star=Q_star,
                    demand_df_test=demand_df_test,
                    Qk_hat_df_test=Qk_hat_df_test,
                    testing_df=testing_df,
                )
            )

        S2_profit_testing = test_results_df_2.iloc[0]["average_profits"]

        # 整理利潤結果
        testing_profits = {
            "baseline": test_baseline_avg_profits,
            "S1": S1_profit_testing,
            "S2": S2_profit_testing,
        }

        testing_stimulation_results = {
            "baseline": test_stimulation_df_baseline,
            "S1": test_stimulation_results_df_1,
            "S2": test_stimulation_results_df_2,
        }

        return testing_profits, testing_stimulation_results

    def calculate_Q_star(self, demand_df, service_level=0.95):

        demand_sum = demand_df.sum(axis=1)
        mean_sum = demand_sum.mean()
        std_sum = demand_sum.std()
        Q_star = norm.ppf(service_level, loc=mean_sum, scale=std_sum)

        print(f"mean of sum: {mean_sum}")
        print(f"std of sum: {std_sum}")
        print(f"{service_level*100} percentile of sum: {Q_star}")

        return Q_star

    def calculate_service_level(self, *, salvage_value, cost, price):

        cu = price - cost
        co = cost - salvage_value
        service_lv = cu / (co + cu)

        return service_lv

    def split_data_by_ratio(self, data, train_size=0.5):
        """依照 train_size 對 data 做一次性切分"""
        n = len(data)
        split_idx = int(n * train_size)
        train_data = data.iloc[:split_idx].reset_index(drop=True)
        test_data = data.iloc[split_idx:].reset_index(drop=True)
        return train_data, test_data

    def prepare_data(self, full_df, demand_df, train_size=0.5):
        """準備一次性的訓練資料與需求資料切分"""
        self.training_df, self.testing_df = self.split_data_by_ratio(
            full_df, train_size
        )
        self.demand_df_train, self.demand_df_test = self.split_data_by_ratio(
            demand_df, train_size
        )
