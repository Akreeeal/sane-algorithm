import tensorflow as tf
import numpy as np

# Загрузка данных MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_val, y_val) = mnist.load_data()

# Нормализация данных
X_train = X_train / 255.0
X_val = X_val / 255.0

# Печать размеров массивов данных
print(f'Размер обучающего набора: {X_train.shape}')
print(f'Размер валидационного набора: {X_val.shape}')

# Преобразование меток в формат one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)

# Преобразование данных в нужный формат
X_train = X_train.reshape(-1, 28*28)
X_val = X_val.reshape(-1, 28*28)

print(f'Новые размеры обучающего набора: {X_train.shape}')
print(f'Новые размеры валидационного набора: {X_val.shape}')