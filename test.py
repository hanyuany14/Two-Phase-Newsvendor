import pandas as pd

# 建立範例需求資料框
data = {
    '需求1': [10, 5, 20],
    '需求2': [15, 10, 25],
    '需求3': [20, 15, 30]
}
demand_df = pd.DataFrame(data, index=['t=1', 't=2', 't=3'])

# 計算每一行的總和
demand_sum = demand_df.sum(axis=0)

# 輸出結果
print("需求資料框:")
print(demand_df)
print("\n每一行的總和:")
print(demand_sum)
