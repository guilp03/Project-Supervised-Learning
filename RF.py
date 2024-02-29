import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from math import sqrt

df = pd.read_csv("apple_quality.csv")

std_scaler = StandardScaler()
#df_scaler = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)

y = df['Quality']

X_train, X_test= train_test_split(df, test_size=0.3, random_state=42, stratify=y)

y_test = X_test['Quality']
y_train = X_train['Quality']
X_train = X_train.drop('Quality', axis=1)
X_test = X_test.drop('Quality', axis=1)

y_train = y_train.apply(lambda c: 0 if c == 'good' else 1)
y_test = y_test.apply(lambda c: 0 if c == 'good' else 1)

X_train = pd.DataFrame(std_scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns)



def RandomForest(n_features, conjunto, labels, min_samples_split,min_samples_leaf ,n_estimators = 100):
    rf_model = RandomForestClassifier(n_estimators = n_estimators, max_features = int(sqrt(n_features)), random_state = 44, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    rf_model.fit(conjunto, labels)
    return rf_model

def get_metrics(model, validation, labels):
    preds = model.predict(validation)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels,preds)
    recall = recall_score(labels, preds)
    return accuracy, f1, precision, recall
best_score = [0.0,0.0,0.0,0.0]
best_n = 0
k = [10,100,200,300,400,500,600,700,800,900]
n = [2, 3 ,4, 5, 6, 7, 8, 9, 10]
m = [1, 2, 3, 4, 5, 6, 7,8,9,10]
for l in k:
    for i in n:
        for j in m:
            model_rf = RandomForest(8, X_train, y_train, min_samples_split=i, min_samples_leaf=j, n_estimators=l)
            accuracy, f1, precision, recall = get_metrics(model_rf, X_test, y_test)
            if f1 > best_score[1]:
                best_score[0] = accuracy
                best_score[1] = f1
                best_score[2] = precision
                best_score[3] = recall
                best_k = l
                best_n = i
                best_m = j
            print(i, j, "accuracy:", accuracy, "f1:", f1, "precision:", precision, "recall:", recall)
print("best_score:",best_k ,best_n, best_m, "accuracy:", best_score[0], "f1:", best_score[1], "precision:", best_score[2], "recall:", best_score[3])