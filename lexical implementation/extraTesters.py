import pandas as pd

test_df1 = pd.DataFrame()
test_df2 = pd.DataFrame()

def setdf1(df):
    global test_df1
    test_df1 = df

def setdf2(df):
    global test_df2
    test_df2 = df
