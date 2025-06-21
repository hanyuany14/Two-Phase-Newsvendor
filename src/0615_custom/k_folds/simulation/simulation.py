import gurobipy as gp
import pandas as pd
import numpy as np
from scipy.stats import norm

from qk_hat import Qk_hat
from baseline_model import BaselineModel
from s1_model import S1_Model

np.random.seed(0)


class Simulation:
    def __init__(
        self,
        cost: int,
        price: int,
        salvage_value: int,
        time_period: int,
        simulation_times: int,
        items_num_train: int,
        items_num_test: int,
    ):
        self._cost = cost
        self._price = price
        self._salvage_value = salvage_value
        self._time_period = time_period
        self._simulation_times = simulation_times
        self._items_num_train = items_num_train
        self._items_num_test = items_num_test

        self.assigned_T = list(range(2, time_period))  # 2 到 T-1
        self.assigned_F = np.arange(0.1, 1.0, 0.1)

        self.service_lv = self.calculate_service_level(
            salvage_value=salvage_value, cost=cost, price=price
        )
        print(f"self.service_lv: {self.service_lv}")

        self.qk_hat = Qk_hat()
        self.baseline_model = BaselineModel()
        self.s1_model = S1_Model()

    def experiment(
        self,
        Qks_test: np.ndarray,
        demand_df_train: pd.DataFrame,
        demand_df_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run Qk simulation experiments.

        Parameters
        ----------
        Qks_test : np.ndarray of shape (T-2, N, S)
            A 3D array representing simulated Qk values.
            - Axis 0 (size T-2): Time steps to be predicted (i.e., len(self.assigned_T))
            - Axis 1 (size N): Number of items (i.e., len(demand_df_test))
            - Axis 2 (size S): Number of simulations per item per time (i.e., self._simulation_times)

        demand_df_train : pd.DataFrame
            Training data for demand estimation.
            columns: time period
            row: items

        demand_df_test : pd.DataFrame
            Test data for which the Qks are predicted.
            columns: time period
            row: items

        Raises
        ------
        AssertionError
            If Qks_test is not a np.ndarray or its shape does not match the expected (T-2, N, S)
        """

        # 驗證 train DataFrame 的 shape
        assert (
            demand_df_train.shape[1] == self._time_period
        ), f"demand_df_train 欄位數量錯誤：預期 {self._time_period}，但為 {demand_df_train.shape[1]}"
        assert (
            demand_df_train.shape[0] == self._items_num_train
        ), f"demand_df_train 列數錯誤：預期 {self._items_num_train}，但為 {demand_df_train.shape[0]}"

        assert (
            demand_df_test.shape[1] == self._time_period
        ), f"demand_df_test 欄位數量錯誤：預期 {self._time_period}，但為 {demand_df_test.shape[1]}"
        assert (
            demand_df_test.shape[0] == self._items_num_test
        ), f"demand_df_test 列數錯誤：預期 {self._items_num_test}，但為 {demand_df_test.shape[0]}"

        # Check shape and type
        assert isinstance(Qks_test, np.ndarray), "Qks_test 應為 numpy.ndarray"
        expected_shape = (
            len(self.assigned_T),
            len(demand_df_test),
            self._simulation_times,
        )
        assert (
            Qks_test.shape == expected_shape
        ), f"Qks_test 應為 shape {expected_shape}, 但收到 {Qks_test.shape}"

        # cal the Q_star
        Q_star = self.calculate_Q_star(demand_df_train, service_level=self.service_lv)

        # ====訓練階段====
        Qk_hats_test = self.qk_hat.make_Qk_hat_df_with_known_Qk(
            demand_df_test,
            Qks_test,
            self.service_lv,
        )  # 我們使用 test 的資料來跑最佳化 in 學姊 case

        training_profits, training_results, training_stimulation_results = (
            self.perform_fold_training(
                demand_df_train=demand_df_test,
                Qk_hats_train=Qk_hats_test,
                Q_star=Q_star,
            )
        )

        return (training_profits, training_stimulation_results, training_results)

    def perform_fold_training(
        self,
        demand_df_train: pd.DataFrame,
        Qk_hats_train: list[int],
        Q_star: float | int,
    ) -> dict[str, float]:
        """This is for single fold training."""

        # 1. Baseline model
        baseline_avg_profits, baseline_stimulation_df = (
            self.baseline_model.one_time_procurement(
                Q_star=Q_star,
                demand_df=demand_df_train,
                cost=self._cost,
                price=self._price,
                salvage_value=self._salvage_value,
            )
        )

        # 2. S1 - Grid F & Grid R
        results_df_1, stimulation_results_df_1 = None, None
        results_df_1, stimulation_results_df_1 = self.s1_model.grid_fixed_F_fixed_R(
            assigned_Ts=self.assigned_T,
            assigned_Fs=self.assigned_F,
            cost=self._cost,
            price=self._price,
            salvage_value=self._salvage_value,
            Qk_hat_df=Qk_hats_train,
            demand_df_train=demand_df_train,
            Q_star=Q_star,
        )

        S1_profit_training = results_df_1.iloc[0]["average_profits"]

        training_profits = {
            "baseline": baseline_avg_profits,
            "S1": S1_profit_training,
        }

        training_results = {
            "S1": results_df_1,
        }

        training_stimulation_results = {
            "baseline": baseline_stimulation_df,
            "S1": stimulation_results_df_1,
        }

        # 整理資料成 df
        train_profit_df = pd.DataFrame(training_profits, index=[0])

        dfs = []
        for model_name, df in training_stimulation_results.items():
            df_copy = df.copy()
            df_copy["model"] = model_name
            dfs.append(df_copy)
        training_stimulation_result_df = pd.concat(dfs, ignore_index=True)

        df_list = [
            df.reset_index(drop=False)
            .rename(columns={"index": "original_index"})
            .assign(model=model_name)
            for model_name, df in training_results.items()
        ]

        training_results = pd.concat(df_list, ignore_index=True).sort_values(
            "average_profits", ascending=False
        )

        return train_profit_df, training_results, training_stimulation_result_df

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
