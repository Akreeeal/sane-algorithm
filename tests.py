import numpy as np
import pickle
from model import Model
from load_data import X_val, y_val
from sklearn.metrics import accuracy_score
from run_sane import one_hot_from_softmax
from generate_population import num_input_neurons, num_hidden_neurons, num_output_neurons
from topology2 import draw_neural_network


with open("outputs/best_model.pkl", "rb") as f:
    best_model = pickle.load(f)


preds_val = best_model.forward(X_val, f='relu')
accuracy_val = accuracy_score(y_val, one_hot_from_softmax(preds_val))

print(f'Точность на валидационном наборе данных: {accuracy_val}')

draw_neural_network(num_input_neurons, num_hidden_neurons, num_output_neurons)