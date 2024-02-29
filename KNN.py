import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import seaborn as sns

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

def K_nearest_neighbors(X_train, y_train, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def get_metrics(model, validation, labels):
    preds = model.predict(validation)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels,preds)
    recall = recall_score(labels, preds)
    return accuracy, f1, precision, recall

k_values = [i for i in range (1,31)]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5)
    scores.append(np.mean(score))

sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()

best_index = np.argmax(scores)
best_k = k_values[best_index]

knn = K_nearest_neighbors(X_train, y_train, n_neighbors=best_k)
accuracy, f1, precision, recall = get_metrics(knn, X_test, y_test)

print("best_score:",best_k, "accuracy:", accuracy, "f1:", f1, "precision:", precision, "recall:", recall)