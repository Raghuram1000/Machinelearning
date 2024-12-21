import pandas as pd


df=pd.read_csv('Testdata.csv')
df = df.replace(' ?', 'NaN')
column_names=['Age','Work Class','fnlwgt','Education','Education-Num','Marital-Status','Occupation','Relationship','Race','Sex','Capital-Gain','Capital-Loss','Hours-per-week','Native-Country','class']



df.columns=column_names
df.to_csv('modified_file.csv', index=False)
df=pd.read_csv('modified_file.csv')
for column in df.columns:
    
    if df[column].isnull().values.any():
        
        mode_value = df[column].mode().values[0]
        df[column].fillna(mode_value, inplace=True)


df.to_csv('modified_file2.csv', index=False)
import pandas as pd


df=pd.read_csv('Testdata.csv',header=None)
df = df.replace(' ?', 'NaN')
df=df.set_axis(['Age','Work Class','fnlwgt','Education','Education-Num','Marital-Status','Occupation','Relationship','Race','Sex','Capital-Gain','Capital-Loss','Hours-per-week','Native-Country','class'],axis=1,copy=False)

df.head()


df.to_csv('modified_file.csv', index=False)
df=pd.read_csv('modified_file.csv')
for column in df.columns:
    
    if df[column].isnull().values.any():
        
        mode_value = df[column].mode().values[0]
        df[column].fillna(mode_value, inplace=True)


df.to_csv('modified_file2.csv', index=False)