import includes
import numpy as np
import pandas as pd
from random import randint 
import matplotlib.pyplot as plt
import seaborn as sns

def show_numberical_dist(df, columns):

    length = len(columns)
    col_count = 4
    row_count = length // col_count

    if( (length/col_count) > (length//col_count) ):
        row_count += 1

    _fig, axes = plt.subplots(nrows = row_count, ncols = col_count, sharex = False, figsize=(15, 10))
    colors = []
    for _i in range(length):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    # columns = ['annual_pay', 'date_final', 'dti', 'emp_duration', 'installment', 'interest_rate', 'is_default',
    #         'loan_amount', 'recoveries', 'total_pymnt', 'total_rec_prncp', 'year']
    for ax, col, color in zip(axes.flat, columns, colors):
        sns.distplot(a = df[col], bins = 50, ax = ax, color = color)
        ax.set_title(col)
        plt.setp(axes, yticks=[])
        ax.grid(False)

    plt.tight_layout()
    plt.show()


def show_categorical_dist(df, cat_list):
    
    col_count = 2
    length = len(cat_list)
    row_count = length // col_count

    _fig, axes = plt.subplots(nrows = row_count, ncols = col_count, sharex = False, figsize=(12, 12))

    colors = []
    for _i in range(length):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
  
    for ax, col, color in zip(axes.flat, cat_list, colors):
        ax.bar(x = df[col].value_counts().index, height = df[col].value_counts(), color = color)
        ax.set_title(col)
        ax.set_xlabel(' ')
        ax.set_xticklabels(labels = ' ')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def show_donut_chart(df, num_of_classes):
    space = np.ones(num_of_classes)/10
    df['is_default'].value_counts().plot(kind = 'pie', explode = space, fontsize = 14, autopct = '%3.1f%%', wedgeprops = dict(width=0.15), 
                                        shadow = True, startangle = 160, figsize = [13.66, 7.68], legend = True)
    plt.legend(['Not Default', 'Default'])
    plt.ylabel('Category')
    plt.title('Proportion of default customers', size = 14)
    plt.show()

def show_count_plot(x_col_name, hue_col_name, df, max_ytick, ytick_step):
    _figure = plt.figure(figsize = [15, 8])

    ax = sns.countplot(x = x_col_name,  data = df, hue = hue_col_name, palette = ['darkcyan', 'crimson'])

    total = df.shape[0]
    for p in ax.patches:
        percentage = '{:.2f}%'.format(100*p.get_height()/total)
        x = p.get_x() + p.get_width() / 10
        y = p.get_y() + p.get_height() + 2
        ax.annotate(percentage, (x, y))

    plt.yticks(range(0, max_ytick, ytick_step))
    plt.xlabel('Ownership Type', size = 14)
    plt.ylabel('Frequency', size = 14)
    plt.legend(labels = ['Not Default', 'Default'], loc = 'upper right')
    plt.title('Frequency occurence of Ownership Type', y=1.05, size = 16)
    plt.show()       

def show_corr_matrix(dataframe, feature_cols, target_col) :
    '''
        Plots the co-relation matrix with of feature_cols with target_col
        Returns pyplot object incase one needs to save the plot as images...
    
    dataframe       : dataframe for which corr matrix needs to be plotted.
    feature_cols    : numerical features which need to be included in corr matrix
    target_col      : target feature variable.
    
    Return  :    
        plt     : matplotlib.pyplot object 
        
    Usage : 
        plt = show_corr_matrix(dataframe, feature_cols, target_col)
        plt.savefig('Co-relation Matrix.png') 
    '''

    target = target_col

    corr = dataframe.corr()
    corr_abs = corr.abs()

    nr_num_cols = len(feature_cols)

    cols = corr_abs.nlargest(nr_num_cols, target)[target].index
    cm = np.corrcoef(dataframe[cols].T)

    # Generate a mask for the upper triangle (taken from seaborn example gallery)
    mask = np.zeros_like(cm, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(15,8))
    _hm = sns.heatmap(cm, annot=True, cmap = 'coolwarm', vmax=.9, linecolor='white', linewidths=.1, mask=mask,
                     fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.title('Co-relation Matrix')
    
    return plt