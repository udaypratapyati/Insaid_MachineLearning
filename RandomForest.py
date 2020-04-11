#%%
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False) 
clf = RandomForestClassifier(max_depth=2, random_state=0) 
clf.fit(X, y) 

print(clf.feature_importances_)     # [0.14205973 0.76664038 0.0282433 0.06305659] 
print(clf.predict([[0, 0, 0, 0]]))  # [1]                            

# %%
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets import make_regression

X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False) 
regr = RandomForestRegressor(max_depth=2, random_state=0) 
regr.fit(X, y) 
print(regr.feature_importances_)    # [0.18146984 0.81473937 0.00145312 0.00233767] 
print(regr.predict([[0, 0, 0, 0]])) # [-8.32987858]

# %%
