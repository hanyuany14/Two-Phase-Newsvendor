import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm


class Qk_hat:
    def __init__(self):
        pass

    def make_Qk_hat_df_with_known_Qk(
        self,
        demand_df: pd.DataFrame,
        Qks: np.ndarray,
        service_level: float,
    ) -> pd.DataFrame:
        """
        計算每期的補貨量預估 (Qk_hat)。

        參數:
        ----------
        demand_df : pd.DataFrame
            每列為時間點 (t=0..T-1)，每欄為樣本 (N)。shape=(T, N)
        Qks : np.ndarray
            已知的模擬補貨量分布樣本，shape=(T-2, N, simulation_times)
        service_level : float
            服務水準 (例如 0.95 表示 95%)。

        回傳:
        ----------
        pd.DataFrame
            每列為樣本 (sample_0..sample_{N-1})，每欄為 Qk_hat_{2}..Qk_hat_{T-1}。

        範例 (Example):（只取一個 item 舉例）
        ----------
        假設:
            T = 6  # t = 1 ~ 6（共6期），我們會建立 Qk_hat_2 ~ Qk_hat_5（共4期）
            N = 1  #  item
            simulation_times = 3

            demand_df (shape 1×6):
                0   1   2   3   4   5
            0  20  13  13  12  18   8

            代表 T=1 需求是 20
            T=2 需求是 13
            ...

            Qks (shape 4×1×3): -> 這是其中一筆 item 的資料，包含
            [[[28.724, 16.194, 21.595]],
            [[18.753, 27.311,  9.699]],
            [[18.388, 18.080, 25.669]],
            [[14.501, 19.138, 15.611]]]

        計算過程:
            - t=2 (計算 Qk_hat_2):
            x_observed = [20]  → sum=20 -> 只取 t=1
            Qk_percentile = 第95百分位 ≈ 28.0112
            Qk_hat_2 = 20 + 28.0112 = 48.0112

            - t=3 (計算 Qk_hat_3):
            x_observed = [20,13] → sum=33 -> 只取 t=1,2
            Qk_percentile ≈ 26.4548
            Qk_hat_3 = 33 + 26.4548 = 59.4548

            - t=4 (計算 Qk_hat_4):
            x_observed = [20,13,13] → sum=46 -> 只取 t=1,2,3
            Qk_percentile ≈ 24.9408
            Qk_hat_4 = 46 + 24.9408 = 70.9408

            - t=5 (計算 Qk_hat_5):
            x_observed = [20,13,13,12] → sum=58 -> 只取 t=1,2,3,4
            Qk_percentile ≈ 18.7851
            Qk_hat_5 = 58 + 18.7851 = 76.7851

        最終輸出:
            Qk_hat DataFrame (1×4):
                Qk_hat_2   Qk_hat_3   Qk_hat_4   Qk_hat_5
                48.0112     59.4548     70.9408     76.7851
        """

        N, T = demand_df.shape
        Qk_hat_matrix = np.zeros((N, T - 2))

        for t_index in range(T - 2):  # 0～3
            Qks_t = Qks[t_index, :, :]  # shape: (3, simulation_times)

            for item_idx in range(N):

                # 取出單一 item 的所有 Qk，並且計算 Qk 的 service_level 分位數
                qk_dist = Qks_t[item_idx]  # shape: (simulation_times,)
                Qk_percentile = np.percentile(qk_dist, service_level * 100)

                # 現在是計算 t=k 的 Qk hat, 因此取出已經觀測到的值，是 1 ~ k-1 的真實值
                x_observed = demand_df.iloc[item_idx, : t_index + 1].values
                Qk_hat_matrix[item_idx, t_index] = x_observed.sum() + Qk_percentile

        time_columns = [f"Qk_hat_{t+2}" for t in range(T - 2)]
        return pd.DataFrame(Qk_hat_matrix, columns=time_columns)

    # def make_Qk_hat_df_with_known_Qk(
    #     self, demand_df: pd.DataFrame, Qks: np.ndarray, simulation_times: int
    # ) -> list[float]:

    #     result = []
    #     for index, row_data in demand_df.iterrows():
    #         k, Qk = Qks[index]
    #         x_observed = row_data[: k - 1].values
    #         Qk_hats = self.__cal_Qk_hat_by_known_Qk(Qk, x_observed)
    #         result.append(Qk_hats)
    #     return Qk_hats

    # def __cal_Qk_hat_by_known_Qk(self, Qk, x_observed):
    #     Qk_hat = x_observed.sum() + Qk
    #     return Qk_hat

    # def __cal_Qk_hat(self, mu_cond, sigma_cond, service_level, x_observed):

    #     mean_Y = np.sum(mu_cond)
    #     var_Y = self.__cal_Var_Y(sigma_cond)

    #     sd_Y = np.sqrt(var_Y)
    #     if sd_Y < 0 or np.isnan(sd_Y):  # scale must be positive
    #         sd_Y = 1e-6

    #     percentile_95_Y = norm.ppf(service_level, loc=mean_Y, scale=sd_Y)
    #     Qk_hat = x_observed.sum() + percentile_95_Y
    #     return Qk_hat

    # def make_Qk_hat_df(self, demand_df, T, service_level):

    #     results_df = pd.DataFrame(index=demand_df.index)
    #     mu_matrix, covariance_matrix = self.__cal_mu_and_cov_matrix(demand_df)

    #     for index, row_data in demand_df.iterrows():
    #         for k in range(2, T):

    #             x_observed = row_data[
    #                 : k - 1
    #             ].values  # 取出前 k 個觀測值 -> Qk_hat_2(t=2): 則 observerd: T=1

    #             mu_cond, sigma_cond = self.__calculate_conditional_distribution(
    #                 mu_matrix, covariance_matrix, x_observed, len(x_observed)
    #             )

    #             Qk_hat = self.__cal_Qk_hat(mu_cond, sigma_cond, service_level, x_observed)

    #             results_df.loc[index, f"Qk_hat_k{k}"] = Qk_hat

    #     return results_df

    # def __calculate_conditional_distribution(self, mu, covariance_matrix, x_observed, k):
    #     mu_1 = mu[:k]
    #     mu_2 = mu[k:]
    #     Sigma_11 = covariance_matrix[:k, :k]
    #     Sigma_22 = covariance_matrix[k:, k:]
    #     Sigma_12 = covariance_matrix[k:, :k]
    #     Sigma_21 = covariance_matrix[:k, k:]

    #     # Compute conditional mean and covariance
    #     Sigma_11_inv = np.linalg.pinv(Sigma_11)
    #     mu_cond = mu_2 + np.dot(Sigma_12, np.dot(Sigma_11_inv, (x_observed - mu_1)))
    #     sigma_cond = Sigma_22 - np.dot(Sigma_12, np.dot(Sigma_11_inv, Sigma_21))

    #     return mu_cond, sigma_cond

    # def __cal_Var_Y(self, sigma_cond):

    #     # Extract the variances (diagonal elements)
    #     variances = np.diag(sigma_cond)

    #     # Calculate the sum of covariances (off-diagonal elements)
    #     covariances_sum = np.sum(sigma_cond) - np.sum(variances)

    #     # Total variance for the sum of mu_cond
    #     total_variance = np.sum(variances) + covariances_sum

    #     return total_variance

    # def __cal_mu_and_cov_matrix(self, demand_df):

    #     mu_matrix = demand_df.mean().values
    #     covariance_matrix = demand_df.cov().values
    #     return mu_matrix, covariance_matrix
