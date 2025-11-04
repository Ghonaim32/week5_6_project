import pandas as pd
def convert_time_spent(df):
    df['time_spent'] = df['time_spent'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    return df

