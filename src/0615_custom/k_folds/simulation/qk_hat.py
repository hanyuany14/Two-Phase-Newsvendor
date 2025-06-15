
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

class Qk_hat():
    def __init__(self):
        pass

    def make_Qk_hat_df_with_known_Qk(self, demand_df, T, Qk: list[tuple]):
        """將算好的 Qk 放進來得到完整 Qk_hat_df

        Args:
            Qk (list[tuple]): [(k, Qk)] tuple 裡面第一個是這是對第 k 期算好的 Qk

        Returns:
            _type_: _description_
        """
        
        results_df = pd.DataFrame(index=demand_df.index)
        for index, row_data in demand_df.iterrows():
            for k in range(2, T):

                x_observed = row_data[
                    : k - 1
                ].values  # 取出前 k 個觀測值 -> Qk_hat_2(t=2): 則 observerd: T=1

                Qk_hat = self.__cal_Qk_hat_by_known_Qk(Qk, x_observed)

                results_df.loc[index, f"Qk_hat_k{k}"] = Qk_hat

        return results_df
    

    def make_Qk_hat_df(self, demand_df, T, service_level):
        
        results_df = pd.DataFrame(index=demand_df.index)
        mu_matrix, covariance_matrix = self.__cal_mu_and_cov_matrix(demand_df)

        for index, row_data in demand_df.iterrows():
            for k in range(2, T):

                x_observed = row_data[
                    : k - 1
                ].values  # 取出前 k 個觀測值 -> Qk_hat_2(t=2): 則 observerd: T=1

                mu_cond, sigma_cond = self.__calculate_conditional_distribution(
                    mu_matrix, covariance_matrix, x_observed, len(x_observed)
                )

                Qk_hat = self.__cal_Qk_hat(mu_cond, sigma_cond, service_level, x_observed)

                results_df.loc[index, f"Qk_hat_k{k}"] = Qk_hat

        return results_df

    def __calculate_conditional_distribution(self, mu, covariance_matrix, x_observed, k):
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
        

    def __cal_Var_Y(self, sigma_cond):

        # Extract the variances (diagonal elements)
        variances = np.diag(sigma_cond)

        # Calculate the sum of covariances (off-diagonal elements)
        covariances_sum = np.sum(sigma_cond) - np.sum(variances)

        # Total variance for the sum of mu_cond
        total_variance = np.sum(variances) + covariances_sum

        return total_variance

    def __cal_Qk_hat(self, mu_cond, sigma_cond, service_level, x_observed):

        mean_Y = np.sum(mu_cond)
        var_Y = self.__cal_Var_Y(sigma_cond)

        sd_Y = np.sqrt(var_Y)
        if sd_Y < 0 or np.isnan(sd_Y):  # scale must be positive
            sd_Y = 1e-6

        percentile_95_Y = norm.ppf(service_level, loc=mean_Y, scale=sd_Y)
        Qk_hat = x_observed.sum() + percentile_95_Y
        return Qk_hat

    def __cal_mu_and_cov_matrix(self, demand_df):

        mu_matrix = demand_df.mean().values
        covariance_matrix = demand_df.cov().values
        return mu_matrix, covariance_matrix
    
    def __cal_Qk_hat_by_known_Qk(self, Qk, x_observed):
        Qk_hat = x_observed.sum() + Qk
        return Qk_hat


