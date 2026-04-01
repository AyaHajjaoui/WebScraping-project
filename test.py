import pandas as pd

df_wu = pd.read_csv("data/wunderground_raw.csv")
print("COLUMNS:")
print(df_wu.columns)

print("\nFIRST ROWS:")
print(df_wu.head())