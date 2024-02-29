import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf

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

learning_rate = 0.01
training_epochs = 5000
display_steps = 100

n_input = 8 
n_hidden = 10
n_output = 1

X = tf.keras.Input(shape=(n_input,))
Y = tf.keras.Input(shape=(n_output,))

weights = {
    "hidden": tf.Variable(tf.random.normal([n_input, n_hidden])),
    "output": tf.Variable(tf.random.normal([n_hidden, n_output])),
}

bias = {
	"hidden": tf.Variable(tf.random.normal([n_hidden])),
	"output": tf.Variable(tf.random.normal([n_output])),
}

def model(X, weights, bias):
	layer1 = tf.add(tf.matmul(X, weights["hidden"]),bias["hidden"])
	layer1 = tf.nn.relu(layer1)

	output_layer = tf.matmul(layer1,weights["output"]) + bias["output"]
	return output_layer

pred = model(X, weights, bias)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y))

optimizer = tf.keras.optimizers.Adam(learning_rate)
with tf.GradientTape() as tape:
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y))

gradients = tape.gradient(loss, list(weights.values()) + list(bias.values()))

optimizer.apply_gradients(zip(gradients, list(weights.values()) + list(bias.values())))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epochs in range(training_epochs):
		_, c= sess.run([optimizador,cost],feed_dict = {X: X_train, Y: y_train})
		if(epochs + 1) % display_steps == 0:
			print("Epoch:",epochs+1,"Cost:", c)
	print("Optimization Finished")

	test_result = sess.run(pred, feed_dict={X: X_test})
	test_result = tf.round(tf.sigmoid(test_result)).eval()  # Arredonde as previsões para 0 ou 1
	accuracy = accuracy_score(y_test, test_result)

	print("accuracy:", accuracy.eval({X: X_test, Y: y_test}))