#%%
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 
from mlxtend.preprocessing import TransactionEncoder
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%
storedata = pd.read_csv('store_data.csv', header=None)
storedata.head()

#%%
# Create a dataframe to list down all the records
df = pd.DataFrame({'Transaction':[], 'Items':[]})
print(storedata.shape[0])

#%%
for row in range(storedata.shape[0]):
    items = storedata.loc[row].dropna()
    length = len(items)
    tmpdf = pd.DataFrame({'Transaction':[row]*length, 'Items':items})
    df = df.append(tmpdf, ignore_index=True)
    # break

df.head()   

#%%
# Set the transactions as integer type
df.Transaction = df.Transaction.astype(int)

#%%
hot_encoded = df.groupby(['Transaction','Items'])['Items'].count()
hot_encoded = hot_encoded.unstack().reset_index()
hot_encoded = hot_encoded.fillna(0)
hot_encoded = hot_encoded.set_index('Transaction')
hot_encoded.head()

#%%
def encode_units(x):
    if x <=  0:
        return 0
    if x >=  1:
        return 1

#%%
hot_encoded = hot_encoded.applymap(encode_units)
hot_encoded.head(2)

#%%
# Apply apriori
freqItems = apriori(hot_encoded, min_support=0.01, use_colnames=True)
freqItems.head(10)

#%%
# Get association rules
rules = association_rules(freqItems, metric='lift', min_threshold=1)
rules.sort_values('lift', ascending = False, inplace = True)
print(rules.head(5))
rules = rules[rules['confidence']>= 0.50]
rules

#%%


#%%
# Filter with confidence greater than 50%   
rules = rules[rules['confidence']>= 0.50]
rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))
rules

#%%
import networkx as nx
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (10,6))
GA = nx.from_pandas_edgelist(rules, target = 'antecedents', source= 'consequents')
nx.draw(GA,with_labels = True)


#%%
#  Case 2 :
bakery = pd.read_csv('https://raw.githubusercontent.com/02asmita/Term-4/patch-1/Data/Assignment/BreadBasket_DMS.csv')
bakery.head(2)

#%%
bakery[bakery.Item  ==  'NONE'].index

#%%
# Safety check, drop rows with entry as none
bakery = bakery.drop(bakery[bakery.Item  ==  'NONE'].index)
bakery.head()

#%%
bakery = bakery.groupby(['Transaction','Item'])['Item'].count()
bakery = bakery.unstack().reset_index().fillna(0).set_index('Transaction')
bakery.head()

#%%
def adjust(x):
    if x <= 0:
        return 0
    else:
        return 1

bakery = bakery.applymap(adjust)
freqItems = apriori(bakery, min_support=0.01, use_colnames=True)
rules = association_rules(freqItems, metric="lift", min_threshold=0.8)
rules = rules.sort_values('lift', ascending = False)

rules = rules[rules['confidence'] >= 0.55]
rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))
rules

#%%
fig, ax = plt.subplots(figsize = (10,6))
GA = nx.from_pandas_edgelist(rules, source = 'antecedents', target= 'consequents')
nx.draw(GA,with_labels = True)