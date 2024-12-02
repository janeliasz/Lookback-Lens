import pandas as pd

df = pd.read_parquet("data/hallu-ds.parquet")

print(df.iloc[0]["formatted_context"])
