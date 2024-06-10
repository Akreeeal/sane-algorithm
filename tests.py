import numpy as np
import pickle
from model import Model
from load_data import X_val, y_val
from sklearn.metrics import accuracy_score
from run_sane import one_hot_from_softmax
from generate_population import num_input_neurons, num_hidden_neurons, num_output_neurons
from topology import draw_neural_network
import matplotlib.pyplot as plt


with open("outputs/best_model.pkl", "rb") as f:
    best_model = pickle.load(f)


preds_val = best_model.forward(X_val, f='relu')
accuracy_val = accuracy_score(y_val, one_hot_from_softmax(preds_val))

print(f'Точность на валидационном наборе данных: {accuracy_val}')



# Функция для отображения изображения и его предсказания
def display_predictions(X_val, preds_val, y_val, num_images=10):
    for i in range(num_images):
        plt.imshow(X_val[i].reshape(28, 28), cmap='gray')
        plt.title(f'Предсказание: {np.argmax(preds_val[i])}, Реальное число: {np.argmax(y_val[i])}')
        plt.axis('off')
        plt.show()

# Отображение первых нескольких изображений с предсказаниями и истинными метками
display_predictions(X_val, preds_val, y_val)


# draw_neural_network(num_input_neurons, num_hidden_neurons, num_output_neurons)