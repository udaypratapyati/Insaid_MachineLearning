import pandas as pd

def show_missing_values(df):
    print('Data Shape:', df.shape)
    null_frame = pd.DataFrame()
    null_frame['Features'] = df.columns.values
    null_frame['Null Frequency'] = df.isnull().sum().values
    null_frame['Missing %age'] = np.round(null_frame['Null Frequency']/df.shape[0], decimals = 4) * 100
    null_frame.set_index('Features', inplace = True)
    return null_frame.transpose()