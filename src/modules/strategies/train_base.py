from datetime import datetime
from src.modules.base import Base
import gurobipy as gp
from gurobipy import GRB

THREADS = 12
TIME_LIMIT = 400
MIPGAP = 0.3
# CURRENT_TIMESTAMP = int(datetime.now().strftime("%Y%m%d%H%M"))


class Train(Base):
    def __init__(
        self,
        *,
        current_timestamp,
        Q_star,
        cost,
        price,
        salvage_value,
        holding_cost,
        demand_df,
        Qk_hat_df,
        training_df,
    ):
        super().__init__(current_timestamp)
        self.Q_star = Q_star
        self.cost = cost
        self.price = price
        self.salvage_value = salvage_value
        self.holding_cost = holding_cost
        self.Qk_hat_df = Qk_hat_df
        self.training_df = training_df
        self.demand_df = demand_df

        self.features_num = training_df.shape[1]

        self.env = self.create_env()
        self.THREADS = THREADS
        self.TIME_LIMIT = TIME_LIMIT
        self.MIPGAP = MIPGAP

        self.model = self.create_base_model()

    def create_env(self):
        return gp.Env(
            params={
                "WLSACCESSID": "73a6e3bf-2a9d-41e8-85eb-dd9b9eda802b",
                "WLSSECRET": "c394298a-96ea-4c8c-9d5e-ef2bd5032427",
                "LICENSEID": 2563044,
            }
        )

    def create_base_model(self):
        model = gp.Model("base_model", env=self.env)

        model.setParam("OutputFlag", True)
        model.setParam("Threads", THREADS)
        model.setParam("MIPGap", TIME_LIMIT)
        model.setParam("TimeLimit", MIPGAP)

        # ======================= Global Variables =======================

        # Category 1 - Some variables that is important to future work
        self.K = self.T - 2  # this is for k=2~T-1. => if T = 10(1~10), K will be 8. (0~7)

        # Category 2 - Variables about this stimulation
        ### 1. Variables for Model 1: Maximum Profit Model
        self.Sold_0s = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="Sold_0",
        )
        self.Left_0s = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="Left_0",
        )
        self.Lost_0s = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="Lost_0",
        )

        self.Sold_1s = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="Sold_1",
        )
        self.Left_1s = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="Left_1",
        )
        self.Lost_1s = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="Lost_1",
        )

        self.profits_vars = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name="profits_vars",
        )

        #### 1-2. 用於計算 k 時期之前與之後的需求量
        self.total_demand_up_to_k_minus_1_vars = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=GRB.INFINITY,
            name="Total_Demand_Up_to_K_minus_1",
        )
        self.total_demand_from_k_to_T_vars = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=GRB.INFINITY,
            name="Total_Demand_from_k_to_T",
        )
        self.Q1_plus_lefts = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=GRB.INFINITY,
            name=f"Q1_plus_left",
        )  # k 之前的剩餘 + 新進貨的 Q1 量

        ### 2. Variables for Model 2: Optimal Fraction Model
        self.f_vars = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name="f_var",
        )
        self.F_vars = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=1,
            name="Fraction_for_second_order_amount",
        )
        self.Q0_vars = model.addVars(
            len(self.demand_df),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=self.Q_star,
            name="Q0_var",
        )

        ### 3. Variables for Model 3: Optimal Order Time Model
        self.R_vars = model.addVars(len(self.demand_df), K, vtype=GRB.BINARY, name="R")

        ### 4. Variables for Model 4: re-estimate order-up-to-level
        self.Q1_vars = model.addVars(
            len(self.Qk_hat_df),
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=self.demand_df.max().max(),
            name="Q1_var",
        )  # Every stimulation will have there own Q1 vars.
        self.Q_hats = model.addVars(
            len(self.Qk_hat_df),
            vtype=GRB.CONTINUOUS,
            lb=(-self.demand_df.max().max() * 100),
            ub=self.demand_df.max().max() * 100,
            name="Q_hat",
        )
        self.Q_hat_adjusteds = model.addVars(
            len(self.Qk_hat_df),
            vtype=GRB.CONTINUOUS,
            lb=(-self.demand_df.max().max() * 100),
            ub=self.demand_df.max().max() * 100,
            name=f"Q_hat_adjusted",
        )

        return model
