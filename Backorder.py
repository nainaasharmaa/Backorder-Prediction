import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


df_train=pd.read_csv("Kaggle_Training_Dataset_v2.csv")
df_test=pd.read_csv("Kaggle_Test_Dataset_v2.csv")


# Removing the last rows of Train as well as Test data which contains Nan For all columns.
df_train.drop(df_train.tail(1).index,inplace=True)
df_test.drop(df_test.tail(1).index,inplace=True)

df = pd.concat([df_train, df_test], ignore_index=True)


copy_df = df.copy()
target=df["went_on_backorder"]

#convert Categories Yes and No to 0s and 1s
copy_df['went_on_backorder'].replace({'Yes':1,'No':0},inplace=True)
copy_df['went_on_backorder']=copy_df['went_on_backorder'].astype(int)


numerical_features = ['national_inv', 'lead_time', 'in_transit_qty',
       'forecast_3_month', 'forecast_6_month', 'forecast_9_month',
       'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
       'min_bank', 'pieces_past_due', 'perf_6_month_avg',
       'perf_12_month_avg', 'local_bo_qty']


#filling the Nan values using median imputation to find correlation
medianValue = copy_df['lead_time'].median()
copy_df['lead_time'] = copy_df['lead_time'].fillna(medianValue)


#Converting Target Variable to 0 and 1
df['went_on_backorder'].replace({'Yes':1,'No':0},inplace=True)
df['went_on_backorder'].astype(int)


# replacing -99 by Nan in performance column
df.perf_6_month_avg.replace({-99.0 : np.nan},inplace=True)
df.perf_12_month_avg.replace({-99.0 : np.nan},inplace=True)


# ### Converting Categorical Features

categorical_columns = ['rev_stop','stop_auto_buy','ppap_risk','oe_constraint','deck_risk','potential_issue']
for col in categorical_columns:
    df[col].replace({'Yes':1,'No':0},inplace=True)
    df[col]=df[col].astype(int)

columns_to_drop = ['lead_time','pieces_past_due','local_bo_qty', 'potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'perf_6_month_avg','perf_12_month_avg']
df.drop(columns_to_drop, axis=1, inplace=True)


# Assigning the Target Variable Column to y_true variable and droping it from the DF
y_true = target

df = df.drop(['sku','went_on_backorder'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df,y_true,stratify=y_true,test_size=0.2)


# fillna using median
median_values = X_train.median()

X_train_median = X_train.fillna(median_values)

X_test_median = X_test.fillna(median_values)

print(X_train_median.shape,X_test_median.shape)


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train_median,y_train)


y_predict=lr.predict(X_test_median)


from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_predict)
print("F1 Score:", f1*100)



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
print(cm)

accuracy_score(y_test, y_predict)


import pickle

# Save the trained model
with open('backorder_model.pkl', 'wb') as f:
    pickle.dump(lr, f)