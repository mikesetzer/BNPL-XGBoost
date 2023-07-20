!pip install xgboost
!pip install graphviz

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import cv
from xgboost import plot_importance
import warnings
import numpy as np
warnings.filterwarnings('ignore')

df = pd.read_csv('/work/loan_data_clean.csv')

# turn all columns to int type
loan_data = df.astype(int)

# drop loan_status from loan_data and assign to X
X = loan_data.drop('loan_status', axis=1)
y = loan_data['loan_status']

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
print('X_val shape:', X_val.shape)
print('y_val shape:', y_val.shape)

# import XGBClassifier
xgb_model = XGBClassifier()

# Fit the model on training data 
xgb_model.fit(X_train, y_train)

# check on the training set and visualize the performance
y_pred = xgb_model.predict(X_train)
print('Training Accuracy: ', round(accuracy_score(y_train, y_pred)*100, 2), '%')
print('Training Precision: ', round(precision_score(y_train, y_pred)*100, 2), '%')
print('Training Recall: ', round(recall_score(y_train, y_pred)*100, 2), '%')
print('Training F1: ', round(f1_score(y_train, y_pred)*100, 2), '%')
print('Training ROC AUC: ', round(roc_auc_score(y_train, y_pred)*100, 2), '%')
print('Training Confusion Matrix: \n', confusion_matrix(y_train, y_pred))

# visualize the performance
cm = metrics.confusion_matrix(y_train, y_pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_train, y_pred))
plt.title(all_sample_title, size = 15);

# make predictions on the testing set
y_pred_class = xgb_model.predict(X_test)

# print true vs predicted
print('True:', y_test.values[0:25])
print('Pred:', y_pred_class[0:25])

# test and validate the model
print('Testing Accuracy score: ', round(accuracy_score(y_test, y_pred_class)*100, 2), '%')
print('Testing Precision score: ', round(precision_score(y_test, y_pred_class)*100, 2), '%')
print('Testing Recall score: ', round(recall_score(y_test, y_pred_class)*100, 2), '%')
print('Testing F1 score: ', round(f1_score(y_test, y_pred_class)*100, 2), '%')
print('Testing ROC AUC score: ', round(roc_auc_score(y_test, y_pred_class)*100, 2), '%')
print('Testing Confusion matrix: \n', confusion_matrix(y_test, y_pred_class))

# k-fold cross validation using xgboost (see Banerjee, 2020)

# define data dmatrix 
dmatrix = xgb.DMatrix(data=X,label=y)

params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 7, 'alpha': 10}

xgb_cv = cv(dtrain=dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)

# full results of the cross validation
xgb_cv

# cross validation summary
print('Cross-validation ROC AUC score: ', round(xgb_cv['test-auc-mean'].iloc[-1]*100, 2), '%')

# visualize the performance
cm = metrics.confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test, y_pred_class))
plt.title(all_sample_title, size = 15);

# plot ROC curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_class)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for loan classifier')
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

# evaluate the model on the validation set
y_val_pred = xgb_model.predict(X_val)
print('Validation accuracy score: ', round(accuracy_score(y_val, y_val_pred)*100, 2), '%')
print('Validation precision score: ', round(precision_score(y_val, y_val_pred)*100, 2), '%')
print('Validation recall score: ', round(recall_score(y_val, y_val_pred)*100, 2), '%')
print('Validation F1 score: ', round(f1_score(y_val, y_val_pred)*100, 2), '%')
print('Validation ROC AUC score: ', round(roc_auc_score(y_val, y_val_pred)*100, 2), '%')

# visualize the validation confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# plot ROC curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_val_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for loan classifier')
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

# visualize the target variable in the original dataset to establish balanced dataset
sns.countplot(x='loan_status', data=df)
plt.show()

# plot correlation matrix
plt.figure(figsize=(12, 12))
sns.heatmap(loan_data.corr(), annot=True, fmt='.2f')
plt.show()

# plot feature importance
plot_importance(xgb_model)
plt.show()

# Kernel Density Estimation Plots for loan_status and last_fico_range_high and show legend
plt.figure(figsize=(12, 6))
sns.kdeplot(loan_data[loan_data['loan_status'] == 1]['last_fico_range_high'], label='Charged Off', shade=True)
sns.kdeplot(loan_data[loan_data['loan_status'] == 0]['last_fico_range_high'], label='Fully Paid', shade=True)
plt.xlabel('last_fico_range_high')
plt.ylabel('Density')
plt.legend()
plt.show()

# violin plot for loan_status and annual_inc
plt.figure(figsize=(12, 6))
sns.violinplot(x='loan_status', y='annual_inc', data=loan_data)
plt.show()

# violin plot for loan_status and dti
plt.figure(figsize=(12, 6))
sns.violinplot(x='loan_status', y='dti', data=loan_data)
plt.show()
