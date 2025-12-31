# filling empty data by replacing with the median
import pandas as pd 
data_train =pd.read_csv("AICompetetion\\dataset\\train.csv")
data_test=pd.read_csv("AICompetetion\\dataset\\test.csv")
df_train=pd.DataFrame(data_train)
df_test =pd.DataFrame(data_test)
#median=df.median()
#df["replament"].fillna(median,inplace=True)
#print(df)
isnull=df_train.isna().sum()
print(isnull)