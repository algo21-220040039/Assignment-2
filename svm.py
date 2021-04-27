import numpy as np
import pandas as pd

class Para:
    method = 'SVM'
    month_in_sample = range(7, 68)
    month_test = range(68, 130+1)
    percent_select = [0.3, 0.3]
    percent_cv = 0.1
    path_data = 'D:/svmdata/final'
    path_result = 'D:/svmdata/result'
    svm_kernal = 'rbf'
    svm_c = 0.01

para = Para()

def lael_data(data):
    data['return_bin'] = np.nan
    data = data.sort_values(by='avg30dreturn', ascending=False)
    n_stock_select = np.multiply(para.percent_select, data.shape[0])
    n_stock_select = np.around(n_stock_select).astype(int)
    data.iloc[:n_stock_select[0],-1] = 1
    data.iloc[-n_stock_select[1]:, -1] = 0
    data.dropna(axis=0, inplace=True)
    return data

cols = ['avg30dreturn','turn','roeAvg','npMargin','gpMargin','YOYAsset','YOYNI','currentRatio','cashRatio','liabilityToAsset','CFOToNP','logprice','marketvalue']
# generate sample data
for month in para.month_in_sample:
    current_data = pd.read_csv(para.path_data+r'/%i.csv'%month, header=0, usecols=cols)[cols]
    current_data = lael_data(current_data)
    if month == para.month_in_sample[0]:
        data_in_sample = current_data
    else:
        data_in_sample = data_in_sample.append(current_data)

data_in_sample = data_in_sample.reset_index(drop=True)
X_in_sample = data_in_sample.loc[:, 'turn':'marketvalue']
y_in_sample = data_in_sample.loc[:,'return_bin']

#from sklearn.model_selection import train_test_split
#X_train, X_cv, y_train, y_cv = train_test_split(X_in_sample, y_in_sample, test_size=para.percent_cv)

#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_cv = scaler.transform(X_cv)

from sklearn.preprocessing import StandardScaler
X_in_sample =  StandardScaler().fit_transform(X_in_sample)

# grid search for parameters of svm using gaussian kernel
from sklearn.model_selection import GridSearchCV
from sklearn import svm
grid = GridSearchCV(svm.SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
grid.fit(X_in_sample, y_in_sample)
print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))

# best parameter is C=10 and gamma=0.01, score 0.52
model = svm.SVC(kernel='rbf', C=10, gamma=0.01)
model.fit(X_in_sample, y_in_sample)

cols.insert(0, 'code')
cols.insert(0, 'pctChg')
# test on testing data
for month in para.month_test:
    current_data = pd.read_csv(para.path_data + r'/%i.csv' % month, header=0, usecols=cols)[cols]
    X_current_month = current_data.loc[:, 'turn':'marketvalue']
    X_current_month = StandardScaler().fit_transform(X_current_month)
    y_pred = model.predict(X_current_month)
    y_score = model.decision_function(X_current_month)
    current_data['y_pred'] = y_pred
    current_data['y_score'] = y_score
    current_data.to_csv(r'D:/svmdata/pred/%i.csv'%month, encoding='gbk', index=False)

# prediction metrics

from sklearn import metrics
haccuracy = list()
hAUC = list()
for month in para.month_test:
    current_data = pd.read_csv(r'D:/svmdata/pred/%i.csv'%month, encoding='gbk')
    current_data = lael_data(current_data)
    accuracy = metrics.accuracy_score(current_data['return_bin'], current_data['y_pred'])
    AUC = metrics.roc_auc_score(current_data['return_bin'], current_data['y_score'])
    haccuracy.append(accuracy)
    hAUC.append(AUC)
    print('month : {}, accuracy: {:.2f}, AUC: {:.2f}'.format(month, accuracy, AUC))

# strategy demo: choose 50 stocks most likely to grow in each period
para.n_stock = 100
test_length = para.month_test[-1]
strategy = pd.DataFrame({'return': [0]*test_length, 'value': [1]*test_length})
for month in para.month_test:
    current_data = pd.read_csv(r'D:/svmdata/pred/%i.csv' % month, encoding='gbk')
    current_data = current_data.sort_values(by='y_score', ascending=False)
    strategy.iloc[month-1,0] = current_data['avg30dreturn'].iloc[0:para.n_stock].mean()
strategy['value'] = (strategy['return']+1).cumprod()

