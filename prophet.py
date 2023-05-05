#python -m pip install prophet
import prophet
import pandas as pd
df=pd.read_excel('D:/数据/特征/othery.xlsx')
m = prophet()
m.fit(df)  # df is a pandas.DataFrame with 'y' and 'ds' columns
future = m.make_future_dataframe(periods=365)
m.predict(future)