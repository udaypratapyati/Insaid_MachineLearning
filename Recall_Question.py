#%%
# Bank Marketing
# Abstract: The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

# Data Set Information: The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

# Source:
# Dataset from : http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#
# FALSE POSITIVE & FALSE NEGATIVE AND IT'S IMPACT ON BUSINESS
# FALSE POSITIVE - The events which are predicted to happen but they don't occur in real life. Here in the context of dataset and business involved, the false positive results in predicted term deposit by customers ont he basis of phone marketing, but they actually do not turn up to make the term deposit.
# FALSE NEGATIVE - The events which are not predicted to happen but they do occur in real life. In this case the prediction is made as the customer will not be making term deposit but they do actually make the term deposit.

# So, concluding from the above two case, we need to cater/ lower down the False negative becasue in any business we cannot assume customer will have prior knowledge of our products and services. Even though they know, we need to do the marketing in order to not loose a single customer .

# %%
import pandas as pd
bank = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-2/master/Data/bank.csv')
bank.head() 


# %%
print(bank.columns)
print("*"*50)
print(bank.info)
print("*"*50)
print(bank.shape)
print(bank.describe())

# %%
bank.y.value_counts()

# %%
# add in include path for api inclusion
import sys
my_api_path='C:\\Users\\prata\\OneDrive\\Documents\\GitHub\\Insaid_MachineLearning'
if my_api_path not in sys.path:
    sys.path.append(my_api_path)

from Missing_Zero_Values import show_missing_values

#%%
### Q 1. Write a code to check the missing values present in the dataset.
show_missing_values(bank)

#%%
### Q 2. Write a user defined function to calculate the Inter quartile 
### range for quantile values outside 25 to 75 range. And do the 
### outlier capping for lower level with min value and for upper level 
### with 'q3=1.5*iqr' value.
import numpy as np 
Q1 = np.percentile(bank.age, 25, interpolation = 'midpoint')
Q3 = np.percentile(bank.age, 75, interpolation = 'midpoint')
IQR = Q3 - Q1
print('IQR :', IQR)
print('Q1 : {} \nQ3: {}'.format(Q1, Q3))
print('Lower limit : {} \nUpper limit : {}'.format( Q1 - IQR*1.5, Q3 + IQR*1.5))

# bank[(bank.campaign < (Q1 - IQR*1.5)) & (bank.campaign > (Q3 + IQR*1.5))]

def remove_outliers(df, col):
    import numpy as np 
    Q1 = np.percentile(df[col], 25, interpolation = 'midpoint')
    Q3 = np.percentile(df[col], 75, interpolation = 'midpoint')
    IQR = Q3 - Q1
    print('IQR :', IQR)
    print('Q1 : {} \nQ3: {}'.format(Q1, Q3))
    print('Lower limit : {} \nUpper limit : {}'.format( Q1 - IQR*1.5, Q3 + IQR*1.5))

    return df[(df[col] < (Q1 - IQR*1.5)) & (df[col] > (Q3 + IQR*1.5))]

#%%
### Q 2.1 Using the above created function , remove the outlier from 'age' variables:
remove_outliers(bank, 'age')

#%%
### Q 2.2 Using the above created function , remove the outlier from 'campaign' variables:
remove_outliers(bank, 'campaign')

#%%
### Q 2.3 Using the above created function , remove the outlier from 'duration' variables:
remove_outliers(bank, 'duration')

#%%
# Dividing dataset into two, on the basis of categorical and numerical.

def get_numerical_categorical_cols(df):
    lis_cat = list(df.select_dtypes(include='object').columns)
    lis_num = list(df.select_dtypes(exclude='object').columns)
    return lis_num, lis_cat
# bank.select_dtypes(include='object').columns
n_col, c_col = get_numerical_categorical_cols(bank)
print("Numerical Col :\n{}".format(n_col))
print("Categorical Col :\n{}".format(c_col))

bank_num = bank[n_col]
bank_cat = bank[c_col]

#%%
### Q 3. Label encode the below mentioned categorical variable to numerical values.
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for col in c_col:
    print(col)
    bank_cat[col] = enc.fit_transform(bank_cat[col])


bank_final= pd.concat([bank_cat, bank_num], axis = 1)
bank_final.head()


#%%
### Q 4. Extract the independent columns to prepare X
### Q 5. Extract dependent column into a dataframe 'y' for model prediction
X = bank_final.iloc[:,:-1]
y = bank_final.y

#%%%
### Q 6. Splitting X and y into train and test dataset
### Check the shape of X an y of train dataset.
### Check the shape of X and y of test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, random_state=42)
print("Train shape : X_train={}".format(X_train.shape))
print("Test shape : X_test={}".format(X_test.shape))

#%%
### Q 7. Instantiate RandomForestClassifier using scikit-learn with (n_estimators=600)
### Q 8. Fit the model
### Q 9. Use the model for predictions
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("Done")

#%%
### Q 10. Model evaluation using accuracy score
### Q 11. Model evaluation using confusion matrix
### Q 12. Model evaluation using Precision score
### Q 13. Model evaluation using Recall score
### Q 14. Model evaluation using F1-score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from MetricEvaluation import show_confusion_matrix
print("Accuracy Score : {}".format(accuracy_score(y_pred,  y_test)))
print("Precision Score : {}".format(precision_score(y_pred,  y_test)))
print("F1 Score : {}".format(f1_score(y_pred,  y_test)))
print("Recall Score : {}".format(recall_score(y_pred,  y_test)))
show_confusion_matrix(y_test, y_pred, ['Predicted No_Deposit','Predicted Deposit'],['Actual No_Deposit','Actual Deposit'])

# %%
### Q 15. Model evaluation using ROC_AUC score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def show_roc_curve(model, xtest, ytest):
    probs = model.predict_proba(xtest)
    print(probs)
    preds = probs[:,1]
    fpr, tpr, _threshold = roc_curve(ytest, preds)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# from MetricEvaluation import show_roc_curve
show_roc_curve(rfc, X_test, y_test)

# %%
# Q 16.1. Instantiate RandomForestClassifier using scikit-learn with 
# (criterion='entropy',n_estimators = 100, random_state = 0, max_depth = 2, min_samples_split=4, min_samples_leaf=3, max_leaf_nodes=5, class_weight='balanced')
# Q 16.2. Fit the model
# Q 16.3. Use the model for predictions
# Q 16.4. Model evaluation using accuracy score
# Q 16.5. Model evaluation using confusion matrix
# Q 16.6. Model evaluation using Precision score
# Q 16.7. Model evaluation using Recall score
# Q 16.8. Model evaluation using F1-score
rfc_2 = RandomForestClassifier(criterion='entropy',n_estimators = 100, random_state = 0, max_depth = 2, min_samples_split=4, min_samples_leaf=3, max_leaf_nodes=5, class_weight='balanced')
rfc_2.fit(X_train, y_train)
y_pred2 = rfc_2.predict(X_test)
print('Done')

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from MetricEvaluation import show_confusion_matrix
print("Accuracy Score : {}".format(accuracy_score(y_pred2,  y_test)))
print("Precision Score : {}".format(precision_score(y_pred2,  y_test)))
print("F1 Score : {}".format(f1_score(y_pred2,  y_test)))
print("Recall Score : {}".format(recall_score(y_pred2,  y_test)))
show_confusion_matrix(y_test, y_pred2, ['Predicted No_Deposit','Predicted Deposit'],['Actual No_Deposit','Actual Deposit'])


#%%
# Q 16.9. Model evaluation using ROC_AUC curve
show_roc_curve(rfc_2, X_test, y_test)

#%%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred2)

#%%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred2))