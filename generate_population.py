import numpy as np
from load_data import X_train, X_val, y_train, y_val
# from load_iris_dataset import *
from pprint import pprint



num_neuron_pop = 10
num_input_neurons = X_train.shape[1]
num_hidden_neurons = 3
num_output_neurons = len(y_train[0])



def generate_population():
    print(num_output_neurons)
    ''' генерируем матрицу размером (num_neuron_pop, num_neuron_connect * 2), каждая строка этой матрицы это нейрон,
    а каждый столбец - связь этого нейрона. Заполняется случаеными числами.'''

    individual_len = num_input_neurons + num_output_neurons
    population = np.zeros((num_neuron_pop, individual_len * 2))

    for i in range(num_neuron_pop):
        for j in range(0, individual_len * 2, 2):
            # neuron_id = np.random.randint(0, num_input_neurons + num_output_neurons) # случайный идентификатор нейрона, который может быть либо входным либо выходным

            population[i][j] = j / 2
            population[i][j + 1] = np.random.random() - 0.5

    pprint(population[0][-20:])


    return population



if __name__=='__main__':
    generate_population()