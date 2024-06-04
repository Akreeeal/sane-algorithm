import numpy as np
from load_data import X_train, X_val, y_train, y_val
# from load_iris_dataset import *



num_neuron_pop = 50
num_neuron_connect = 7
num_input_neurons = X_train.shape[1]
num_output_neurons = np.unique(y_train).shape[0]
num_blueprints = 10
num_hidden_neurons = 5


def generate_population():
    ''' генерируем матрицу размером (num_neuron_pop, num_neuron_connect * 2), каждая строка этой матрицы это нейрон,
    а каждый столбец - связь этого нейрона. Заполняется случаеными числами.'''
    population = np.zeros((num_neuron_pop, num_neuron_connect * 2))
    ''' '''
    for i in range(num_neuron_pop):
        for j in range(0, num_neuron_connect * 2, 2):
            neuron_id = np.random.randint(0, num_input_neurons + num_output_neurons) # случайный идентификатор нейрона, который может быть либо входным либо выходным
            population[i][j] = neuron_id
            population[i][j + 1] = np.random.random() - 0.5
    '''генерируем матрицу размером (num_blueprints, num_hidden_neurons), каждая строка - набор нейронов для создания сети. 
    Все заполняется случайными числами от 0 до кол-ва всей популяции'''
    blueprints = np.zeros((num_blueprints, num_hidden_neurons), dtype=int)
    for i in range(num_blueprints):
        blueprints[i] = np.random.randint(0, num_neuron_pop, size=(num_hidden_neurons))

    print(population)
    print(blueprints)

    return population, blueprints

if __name__=='__main__':
    generate_population()