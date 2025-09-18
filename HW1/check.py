import pandas as pd

input_path = "X:/a_Study_Files/NEU_Graduate/INFO6105/HW1/InsuranceCharges.csv"
output_path = "X:/a_Study_Files/NEU_Graduate/INFO6105/HW1/HW1_CleanedDataset.csv"

df_in = pd.read_csv(input_path)
df_out = pd.read_csv(output_path)

print("原始数据 (前5行):")
print(df_in.head(20), "\n")

print("清洗后数据 (前5行):")
print(df_out.head(20), "\n")

print("原始数据形状:", df_in.shape)
print("清洗后数据形状:", df_out.shape)
