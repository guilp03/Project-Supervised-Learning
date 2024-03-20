import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
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

y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

def MultilayerPerceptron(X_train, y_train, hidden_layer_sizes, solver, alpha, learning_rate_init, activation):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, solver=solver, alpha=alpha,activation=activation, max_iter=1000)
    mlp.fit(X_train, y_train.ravel())
    return mlp

def get_metrics(model, validation, labels):
    preds = model.predict(validation)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels,preds)
    recall = recall_score(labels, preds)
    return accuracy, f1, precision, recall

hidden_layer_sizes = [(10,7,5)]
activation = ("identity", "logistic", "tahn", "relu")
solver = ("lbfgs", "sgd", "adam")
alpha = 0.0001
learning_rate_init = 0.01
f1_list = []
accuracy_list = []
precision_list = []
hidden_layer_sizes_list = []
for i in range (2,100):
    print("itter", i)
    hidden_layer_sizes = (i,)
    model = MultilayerPerceptron(X_train=X_train, 
                                 y_train=y_train, 
                                 hidden_layer_sizes=hidden_layer_sizes, 
                                 solver=solver[2], 
                                 alpha=alpha,
                                 learning_rate_init=learning_rate_init, 
                                 activation=activation[2])
    
    accuracy, f1, precision, recall = get_metrics(model, X_test, y_test)
    f1_list.append(f1)
    precision_list.append(precision)
    accuracy_list.append(accuracy)
    hidden_layer_sizes_list.append(i)
    
sns.lineplot(x = hidden_layer_sizes_list, y = f1_list, marker = 'o', label = "f1_score")
sns.lineplot(x = hidden_layer_sizes_list, y = accuracy_list, marker = 'o', label = "accuracy score")
sns.lineplot(x = hidden_layer_sizes_list, y = precision_list, marker = 'o', label = "precision score")

print("accuracy:", accuracy, "f1:", f1, "precision:", precision, "recall:", recall)

plt.xlabel("hidden layer values")
plt.ylabel("f1 score")
plt.legend()
plt.show()
