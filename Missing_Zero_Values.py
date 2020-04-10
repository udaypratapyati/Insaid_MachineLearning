import pandas as pd
import numpy as np

def show_missing_values(df):
    print('Data Shape:', df.shape)
    null_frame = pd.DataFrame()
    null_frame['Features'] = df.columns.values
    null_frame['Null Frequency'] = df.isnull().sum().values
    null_frame['Missing %age'] = np.round(null_frame['Null Frequency']/df.shape[0], decimals = 4) * 100
    null_frame.set_index('Features', inplace = True)
    return null_frame.transpose()

def show_zero_values(df):
    print('Data Shape:', df.shape)
    null_frame = pd.DataFrame()
    null_frame['Features'] = df.columns.values
    null_frame['Zeros Frequency'] = df[df == 0].count().values
    null_frame['Missing %age'] = np.round(null_frame['Zeros Frequency']/df.shape[0], decimals = 5) * 100
    null_frame.set_index('Features', inplace = True)
    null_frame.transpose()

def show_duplicate_columns(dataframe):
    '''Returns a list of labels of duplicate valued columns'''

    names = set()
    for i in range(dataframe.shape[1]):
        col1 = dataframe.iloc[:, i]
        for j in range(i+1, dataframe.shape[1]):
            col2 = dataframe.iloc[:, j]
            if col1.equals(col2):
                names.add(dataframe.columns.values[j])

    if (len(names) == 0):
        return None
    else:
        return list(names)
 