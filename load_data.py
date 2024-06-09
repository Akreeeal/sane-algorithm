import tensorflow as tf
import numpy as np


mnist = tf.keras.datasets.mnist


(X_train, y_train), (X_val, y_val) = mnist.load_data()


X_train = X_train / 255.0
X_val = X_val / 255.0


# print(f'Размер обучающего набора: {X_train.shape}')
# print(f'Размер валидационного набора: {X_val.shape}')


y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)


X_train = X_train.reshape(-1, 28*28)
X_val = X_val.reshape(-1, 28*28)

X_train = X_train[:1000]
y_train = y_train[:1000]
X_val = X_val[:200]
y_val = y_val[:200]



print(f'Размеры обучающего набора: {X_train.shape}')
print(f'Размеры валидационного набора: {X_val.shape}')