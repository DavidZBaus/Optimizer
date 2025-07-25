import pandas as pd
import glob


files = sorted(glob.glob("/home/ubuntu/Desktop/Optimizer Project/LiquidityData/*.csv"))

df = pd.DataFrame()
for file in files:
    print(file)
    temp_df = pd.read_csv(file).set_index("date")

    temp_df.index = pd.to_datetime(temp_df.index, errors='coerce')
    temp_df = temp_df[temp_df.index.second == 0]

    df = pd.concat([df, temp_df])
   
df = df.sort_index()
print(df)

df.to_csv("/home/ubuntu/Desktop/Optimizer Project/LiquidityDataCombined.csv")