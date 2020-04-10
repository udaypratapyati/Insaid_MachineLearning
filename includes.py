
import pandas as pd     # For Panel Data Analysis
# from pandas_profiling import ProfileReport
import numpy as np      # For Numerical Python

from random import randint      # For Random seed values
from scipy import stats         # For Scientific Computation

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.classifier import PrecisionRecallCurve

# Machine Learning Model for Evaluation
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier
from xgboost import to_graphviz, plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import classification_report

# For Preprocessing & Scaling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

# For Feature Selection
from sklearn.feature_selection import SelectFromModel

# To handle class imbalance problem
from imblearn.over_sampling import SMOTE

# initial settings
import warnings         # To Disable Warnings
warnings.filterwarnings(action = "ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('mode.chained_assignment', None)
# %matplotlib inline


# add in include path for api inclusion
# import sys
# my_api_path='C:\\Users\\prata\\OneDrive\\Documents\\GitHub\\UD_API'
# if my_api_path not in sys.path:
#     sys.path.append(my_api_path)
#     
# import importlib
# importlib.reload(ud_visualiztion)