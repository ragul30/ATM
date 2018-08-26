
# coding: utf-8

# In[1]:


from collections import Counter
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


__author__ = 'Yazan Obeidi, yazan.obeidi@uwaterloo.ca'
__license__ = 'Apache v3'


# In[ ]:


__author__ = 'Yazan Obeidi, yazan.obeidi@uwaterloo.ca'
__license__ = 'Apache v3'


# In[ ]:


# plot the features (not amount or class)
plt.figure(figsize=(10,8))
for a in range(1, 29):
    plt.plot(data['V'+str(a)])
plt.legend(loc='best', fontsize=10, ncol=8)
plt.title("Plot of the first 28 columns (after PCA)")
plt.xlabel("Sample")


# In[ ]:


data.Class.value_counts()


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
ax1.hist(data.Time[data.Class == 1], bins = 50)
ax1.set_title('Fraud')
ax2.hist(data.Time[data.Class == 0], bins = 50)
ax2.set_title('Normal')
plt.xlabel('Time (s)')
plt.ylabel('Transactions')
plt.show()


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
ax1.hist(data.Amount[data.Class == 1], bins = 30)
ax1.set_title('Fraud')
ax2.hist(data.Amount[data.Class == 0], bins = 30)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()


# In[ ]:


plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i in range(1, 29):
    ax = plt.subplot(gs[i-1])
    sns.distplot(data['V'+str(i)][data.Class == 1], bins=50)
    sns.distplot(data['V'+str(i)][data.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + 'V'+str(i))
plt.show()
plt.tight_layout()


# In[ ]:


# Based on observation of data overlap above, try out a second dataset with redunancies removed
clean_data = data.drop(['V28','V27','V23','V8'], axis =1)
# Later - can re run everything after running the following line
#data = clean_data


# In[ ]:


#Create dataframes of only Fraud and Normal transactions. Also Shuffle them.
fraud = shuffle(data[data.Class == 1])
normal = shuffle(data[data.Class == 0])
# Produce a training set of 80% of fraudulent and 80% normal transactions
X_train = fraud.sample(frac=0.8)
X_train = pd.concat([X_train, normal.sample(frac = 0.8)], axis = 0)
# Split remainder into testing and validation
remainder = data.loc[~data.index.isin(X_train.index)]
X_test = remainder.sample(frac=0.7)
X_validation = remainder.loc[~remainder.index.isin(X_test.index)]


# In[ ]:


with open('pickle/train_data_resampled.pkl', 'rb') as f:
    X_train_resampled = pickle.load(f)
with open('pickle/train_data_labels_resampled.pkl', 'rb') as f:
    X_train_labels_resampled = pickle.load(f)
    
print(Counter(X_train_labels_resampled))
    
X_train_resampled = pd.DataFrame(X_train_resampled)
X_train_labels_resampled = pd.DataFrame(X_train_labels_resampled)
X_train_resampled = pd.concat([X_train_resampled, X_train_labels_resampled], axis=1)
X_train_resampled.columns = X_train.columns
X_train_resampled.head()


# In[ ]:


X_train = shuffle(X_train)
X_test = shuffle(X_test)
X_validation = shuffle(X_validation)
X_train_ = shuffle(X_train_resampled)
X_test_ = shuffle(X_test)
X_validation_ = shuffle(X_validation)
data_resampled = pd.concat([X_train_, X_test_, X_validation_])


# In[ ]:


for feature in X_train.columns.values[:-1]:
    mean, std = data[feature].mean(), data[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std
    X_validation.loc[:, feature] = (X_validation[feature] - mean) / std
for feature in X_train_.columns.values[:-1]:
    mean, std = data_resampled[feature].mean(), data_resampled[feature].std()
    X_train_.loc[:, feature] = (X_train_[feature] - mean) / std
    X_test_.loc[:, feature] = (X_test_[feature] - mean) / std
    X_validation_.loc[:, feature] = (X_validation_[feature] - mean) / std


# In[ ]:


y_train = X_train.Class
y_test = X_test.Class
y_validation = X_validation.Class
y_train_ = X_train_.Class
y_test_ = X_test_.Class
y_validation_ = X_validation_.Class
# Remove labels from X's
X_train = X_train.drop(['Class'], axis=1)
X_train_ = X_train_.drop(['Class'], axis=1)
X_test = X_test.drop(['Class'], axis=1)
X_test_ = X_test_.drop(['Class'], axis=1)
X_validation = X_validation.drop(['Class'], axis=1)
X_validation_ = X_validation_.drop(['Class'], axis=1)


# In[ ]:


dataset = {'X_train' : X_train,
           'X_train_': X_train_,
           'X_test': X_test,
           'X_test_': X_test,
           'X_validation': X_validation,
           'X_validation_': X_validation_,
           'y_train': y_train,
           'y_train_': y_train_,
           'y_test': y_test,
           'y_test_': y_test_,
           'y_validation': y_validation,
           'y_validation_': y_validation_}
with open('pickle/data_with_resample_apr19.pkl', 'wb+') as f:
    pickle.dump(dataset, f)


# In[ ]:


with open('pickle/data_with_resample_apr19.pkl', 'rb+') as f:
    dataset = pickle.load(f)


# In[ ]:


for k, v in dataset.iteritems():
    if 'y' in k:
        print(k, Counter(v))


# In[ ]:


def plot_confusion_matrix(y_test, pred):
    
    y_test_legit = y_test.value_counts()[0]
    y_test_fraud = y_test.value_counts()[1]
    
    cfn_matrix = confusion_matrix(y_test, pred)
    cfn_norm_matrix = np.array([[1.0 / y_test_legit,1.0/y_test_legit],[1.0/y_test_fraud,1.0/y_test_fraud]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')

    ax = fig.add_subplot(1,2,2)
    sns.heatmap(norm_cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)

    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()
    
    print('---Classification Report---')
    print(classification_report(y_test,pred))


# In[ ]:


lsvm = svm.LinearSVC(C=1.0, dual=False)
lsvm.fit(dataset['X_train'], dataset['y_train'])
y_pred = lsvm.predict(dataset['X_test'])
plot_confusion_matrix(dataset['y_test'], y_pred)


# In[ ]:



# Linear SVM on ADASYN training data #svm_adasyn_unweighted_c0_01# Linear 
lsvm = svm.LinearSVC(C=1, dual=False)
lsvm.fit(dataset['X_train_'], dataset['y_train_'])
y_pred = lsvm.predict(dataset['X_test_'])
plot_confusion_matrix(dataset['y_test_'], y_pred)


# In[ ]:


lsvm = svm.LinearSVC(C=1.0, dual=False, class_weight={1:1000,0:1})
lsvm.fit(dataset['X_train'], dataset['y_train'])
y_pred = lsvm.predict(dataset['X_test'])
plot_confusion_matrix(dataset['y_test'], y_pred)


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

for c,k in zip([0.0001, 0.001, 0.1, 1, 10, 25, 50, 100],'bgrcmywk'):
    lsvm_ = svm.LinearSVC(C=c, dual=False, class_weight={1:1,0:1})
    lsvm_.fit(dataset['X_train_'], dataset['y_train_'])
    y_pred = lsvm_.predict(dataset['X_test_'])

    p,r,_ = precision_recall_curve(dataset['y_test_'], y_pred)
    tpr,fpr,_ = roc_curve(dataset['y_test_'], y_pred)
    
    ax1.plot(r,p,c=k,label=c)
    ax2.plot(tpr,fpr,c=k,label=c)

ax1.legend(loc='lower left')    
ax2.legend(loc='lower left')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

for c,k in zip([0.0001, 0.001, 0.1, 1, 10, 25, 50, 100],'bgrcmywk'):
    print(c)
    lsvm_ = svm.LinearSVC(C=c, dual=False, class_weight={1:1,0:1})
    lsvm_.fit(dataset['X_train'], dataset['y_train'])
    y_pred = lsvm_.predict(dataset['X_test'])

    p,r,_ = precision_recall_curve(dataset['y_test'], y_pred)
    tpr,fpr,_ = roc_curve(dataset['y_test'], y_pred)
    
    ax1.plot(r,p,c=k,label=c)
    ax2.plot(tpr,fpr,c=k,label=c)

ax1.legend(loc='lower left')    
ax2.legend(loc='lower left')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

for c,k in zip([0.0001, 0.001, 0.1, 1, 10, 25, 50, 10000],'bgrcmywk'):
    lsvm_ = svm.LinearSVC(C=c, dual=False, class_weight={1:10,0:1})
    lsvm_.fit(dataset['X_train'], dataset['y_train'])
    y_pred = lsvm_.predict(dataset['X_test'])

    p,r,_ = precision_recall_curve(dataset['y_test'], y_pred)
    tpr,fpr,_ = roc_curve(dataset['y_test'], y_pred)
    
    ax1.plot(r,p,c=k,label=c)
    ax2.plot(tpr,fpr,c=k,label=c)
    
ax1.legend(loc='lower left')    
ax2.legend(loc='lower left')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

for w,k in zip([1, 5, 10, 100, 1000, 10000, 100000],'bgrcmyk'):
    lsvm_ = svm.LinearSVC(C=1, dual=False, class_weight={1:w,0:1})
    lsvm_.fit(dataset['X_train'], dataset['y_train'])
    y_pred = lsvm_.predict(dataset['X_test'])

    p,r,_ = precision_recall_curve(dataset['y_test'], y_pred, pos_label=1)
    tpr,fpr,_ = roc_curve(dataset['y_test'], y_pred)
    
    ax1.plot(r,p,c=k,label=w)
    ax2.plot(tpr,fpr,c=k,label=w)
    
ax1.legend(loc='lower left')    
ax2.legend(loc='lower left')
plt.show()


# In[ ]:


lsvm = svm.LinearSVC(C=1, dual=False, class_weight={1:10,0:1})
lsvm.fit(dataset['X_train_'], dataset['y_train_'])
y_pred = lsvm.predict(dataset['X_test_'])
y_pred_validation = lsvm.predict(dataset['X_validation_'])
plot_confusion_matrix(dataset['y_test_'], y_pred)
plot_confusion_matrix(dataset['y_validation_'], y_pred_validation)
# Conclusion: no bueno.


# In[ ]:


lsvm = svm.LinearSVC(C=1, dual=False, class_weight={1:100000,0:1})
print('fitting')
lsvm.fit(dataset['X_train'], dataset['y_train'])
print('predicting test')
y_pred = lsvm.predict(dataset['X_test'])
print('predicting validation')
y_pred_validation = lsvm.predict(dataset['X_validation'])
plot_confusion_matrix(dataset['y_test'], y_pred)
plot_confusion_matrix(dataset['y_validation'], y_pred_validation


# In[ ]:


rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4)
rf.fit(dataset['X_train'], dataset['y_train'])
y_pred = rf.predict(dataset['X_test'])
plot_confusion_matrix(dataset['y_test'], y_pred)


# In[ ]:


rf_adasyn = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4)
rf_adasyn.fit(dataset['X_train_'], dataset['y_train_'])
y_pred = rf_adasyn.predict(dataset['X_test_'])
plot_confusion_matrix(dataset['y_test_'], y_pred)


# In[ ]:


# Unsampled training data # rf_prec_recall_n_est_unsampled_unweighted
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

for n_est,k in zip([10, 50, 100, 250, 500, 1000],'bgrcmy'):
    print(n_est)
    rf = RandomForestClassifier(n_estimators=n_est, bootstrap=False, max_features=0.33, n_jobs=4)
    rf.fit(dataset['X_train'], dataset['y_train'])
    y_pred = rf.predict(dataset['X_test'])

    p,r,_ = precision_recall_curve(dataset['y_test'], y_pred)
    tpr,fpr,_ = roc_curve(dataset['y_test'], y_pred)
    
    ax1.plot(r,p,c=k,label=n_est)
    ax2.plot(tpr,fpr,c=k,label=n_est)

ax1.legend(loc='lower left')
ax2.legend(loc='lower left')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

for n_est,k in zip([1, 10, 50, 100, 500, 1000],'bgrcmy'):
    rf_adasyn_ = RandomForestClassifier(n_estimators=n_est, oob_score=True, n_jobs=4)
    rf_adasyn_.fit(dataset['X_train_'], dataset['y_train_'])
    y_pred = rf_adasyn_.predict(dataset['X_test_'])

    p,r,_ = precision_recall_curve(dataset['y_test_'], y_pred)
    tpr,fpr,_ = roc_curve(dataset['y_test_'], y_pred)
    
    ax1.plot(r,p,c=k,label=n_est)
    ax2.plot(tpr,fpr,c=k,label=n_est)

ax1.legend(loc='lower left')    
ax2.legend(loc='lower left')
plt.show()


# In[ ]:


mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=50,
                    solver='sgd', verbose=10, tol=1e-4, momentum=0.9,
                    learning_rate='adaptive', learning_rate_init=0.1)
mlp.fit(dataset['X_train'], dataset['y_train'])
y_pred = mlp.predict(dataset['X_test'])
plot_confusion_matrix(dataset['y_test'], y_pred)


# In[ ]:


mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=50,
                    solver='adam', verbose=10, tol=1e-6, momentum=0.9,
                    learning_rate='adaptive', learning_rate_init=0.001)
mlp.fit(dataset['X_train'], dataset['y_train'])
y_pred = mlp.predict(dataset['X_test'])
plot_confusion_matrix(dataset['y_test'], y_pred)


# In[ ]:


mlp_adasyn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=50,
                    solver='adam', verbose=10, tol=1e-6, momentum=0.9,
                    learning_rate='adaptive', learning_rate_init=0.001)
mlp_adasyn.fit(dataset['X_train_'], dataset['y_train_'])
y_pred = mlp_adasyn.predict(dataset['X_test_'])
plot_confusion_matrix(dataset['y_test_'], y_pred)


# In[ ]:


mlp_adasyn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=50,
                    solver='sgd', verbose=10, tol=1e-6, momentum=0.9,
                    learning_rate='adaptive', learning_rate_init=0.1)
mlp_adasyn.fit(dataset['X_train_'], dataset['y_train_'])
y_pred = mlp_adasyn.predict(dataset['X_test_'])
plot_confusion_matrix(dataset['y_test_'], y_pred)


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

for n_l,k in zip([10, 50, 100, 250, 500, 1000],'bgrcmy'):
    print(n_l)
    mlp = MLPClassifier(hidden_layer_sizes=(n_l,), max_iter=50,
                    solver='sgd', verbose=10, tol=1e-4, momentum=0.9,
                    learning_rate='adaptive', learning_rate_init=0.01)
    mlp.fit(dataset['X_train'], dataset['y_train'])
    y_pred = mlp.predict(dataset['X_test'])

    p,r,_ = precision_recall_curve(dataset['y_test'], y_pred)
    tpr,fpr,_ = roc_curve(dataset['y_test'], y_pred)
    
    ax1.plot(r,p,c=k,label=n_l)
    ax2.plot(tpr,fpr,c=k,label=n_l)

ax1.legend(loc='lower left')
ax2.legend(loc='lower left')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

for n_l,k in zip([10, 50, 100, 250, 500, 1000],'bgrcmy'):
    print(n_l)
    mlp = MLPClassifier(hidden_layer_sizes=(n_l,), max_iter=50,
                    solver='sgd', verbose=10, tol=1e-4, momentum=0.9,
                    learning_rate='adaptive', learning_rate_init=0.01)
    mlp.fit(dataset['X_train_'], dataset['y_train_'])
    y_pred = mlp.predict(dataset['X_test_'])

    p,r,_ = precision_recall_curve(dataset['y_test_'], y_pred)
    tpr,fpr,_ = roc_curve(dataset['y_test_'], y_pred)
    
    ax1.plot(r,p,c=k,label=n_l)
    ax2.plot(tpr,fpr,c=k,label=n_l)

ax1.legend(loc='lower left')
ax2.legend(loc='lower left')
plt.show()

