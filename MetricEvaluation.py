import includes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import PrecisionRecallCurve
from xgboost import to_graphviz, plot_importance

def show_confusion_matrix(actual_val, pred_val, col_header_list, index_list):
    matrix = pd.DataFrame(confusion_matrix(actual_val, pred_val))
    matrix.columns = col_header_list
    matrix.index = index_list
    return matrix

def show_roc_curve(model, xtest, ytest):
    probs = model.predict_proba(xtest)
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

    return roc_auc    

def PRCurve(model, X_train, X_test, y_train, y_test):
    '''
    A function to visualize Precision Recall Curve.
    Returns average precision score of the model.
    Data to fit must be training i.e. X_train, y_train.
    Data score will be estimated on X_test, y_test.
    '''
    viz = PrecisionRecallCurve(model)
    viz.fit(X_train, y_train)
    avg_prec = viz.score(X_test, y_test)
    plt.legend(labels = ['Binary PR Curve',"AP=%.3f"%avg_prec], loc = 'lower right', prop={'size': 14})
    plt.xlabel(xlabel = 'Recall', size = 14)
    plt.ylabel(ylabel = 'Precision', size = 14)
    plt.title(label = 'Precision Recall Curve', size = 16)    


def show_feature_importance(model, df):
    fig = plt.figure(figsize = [12, 8])
    ax = fig.add_subplot(1, 1, 1)

    colours = ['#CAF270', '#95E681', '#65D794', '#3CC5A3', '#26B1AC', '#359CAC', '#4D86A4', '#607093', '#695B7C']

    ax = plot_importance(model, height = 1, color = colours, grid = False, show_values = False, importance_type = 'cover', ax = ax)

    total = df.shape[0]
    for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.2
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y))

    plt.xlabel(xlabel = 'F-Score', size = 14)
    plt.ylabel(ylabel = 'Features', size = 14)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    plt.title('Ordering feature importance learned by model', size = 16)
    plt.show()    