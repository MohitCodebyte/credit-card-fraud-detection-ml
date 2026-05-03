# libabries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# data gathering
df=pd.read_csv(r"D:\Python\1.project\credit-card-fraud-project\data\credit_card_fraud_custom.csv")
print(df.dtypes)

# encoders
from sklearn.preprocessing import LabelEncoder

# algorithm
from sklearn.ensemble import RandomForestClassifier

# sampling
from imblearn.over_sampling import SMOTE

# train test
from sklearn.model_selection import train_test_split

# evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df=df.drop('transaction_id', axis=1)
sns.countplot(data=df,x='is_fraud')
plt.title("target distribution")
plt.show()

### EDA
# 1. univariate analysis

sns.histplot(data=df,x='user_age',bins=50)
plt.show()
num_col=['transaction_amount',
       'user_age','transaction_frequency',
       'account_balance', 'fraud_risk_score']

for i in num_col:
    sns.histplot(data=df,x=i,bins=40)
    plt.title(i)
    plt.show()
cat_col=['merchant','category','user_gender','user_city','device_type']

for i in cat_col:
    plt.figure(figsize=(5,3))
    sns.histplot(data=df,x=i)
    plt.xticks(rotation=45)
    plt.title(i)
    plt.show()
sns.boxplot(data=df,x='is_fraud',y='transaction_amount')
plt.show()
df.info()
df['transaction_time']=pd.to_datetime(df['transaction_time'])

df['hour']=df['transaction_time'].dt.hour
df['transaction_ratio']=df['transaction_amount']/df['account_balance']
df.head(1)
new=[]
for i in df['transaction_amount']:
    if i>50000:
        new.append(1)
    else:
        new.append(0)
df['high_amount_flag']=new
# df['transaction_amount'] = df['transaction_amount'].apply(Lambda i : 1 if i>500 else 0)
df.drop(['transaction_amount','transaction_time'],axis=1,inplace=True)
df['user_gender'].value_counts()

# LabelEncoder
le=LabelEncoder()
df['user_gender']=le.fit_transform(df['user_gender'])
df['device_type']=le.fit_transform(df['device_type'])

df=pd.get_dummies(df,columns=['merchant','category','user_city'],dtype=int)
df.dtypes

### drop unused coloumns

df.drop(columns=['transaction_ratio','fraud_risk_score'],inplace=True)

# train_test_split

x=df.drop('is_fraud',axis=1)
y=df['is_fraud']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

# scaling
scaler=StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x=df.drop('is_fraud',axis=1)
y=df['is_fraud']

# sampling
from imblearn.over_sampling import SMOTE
sm = SMOTE()
x,y=sm.fit_resample(x,y)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# model training
model = RandomForestClassifier(n_estimators=300,
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42)
model.fit(x,y)

## RandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
model = RandomForestClassifier(
    n_estimators=200,          # zyada trees = better generalization
    max_depth=6,               # sabse important (overfitting rokta hai)
    min_samples_split=10,      
    min_samples_leaf=5,        
    max_features='sqrt',       
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
rscv.fit(x_train,y_train)

#best estimator

best_model=rscv.best_estimator_
best_model.fit(x_train,y_train)

#predict accuracy
pred_train=best_model.predict(x_train)
pred_test=best_model.predict(x_test)

#accuracy score
print(accuracy_score(y_train,pred_train))
print(accuracy_score(y_test,pred_test))

## pickel file
import pickle
#model
with open('model.pkl','wb') as f:
    pickle.dump(model,f)
#scaler
with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)
