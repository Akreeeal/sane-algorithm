import numpy as np
from generate_population import generate_population, n_epoches, num_blueprints, num_neuron_pop
from load_data import X_train, y_train, X_val, y_val

def run_algorithm():

    epoch = 0

    population, blueprints = generate_population()

    while epoch < n_epoches:
        ''' Первый этап алгоритма: Удалить все значения пригодности для каждого нейрона'''
        blueprints_fitness = np.zeros(num_blueprints) # пригодности нейронной сети
        neuron_fitness = np.zeros(num_neuron_pop) # пригодности нейронов
        num_neuron_include = np.ones(num_neuron_pop) # пригодности вхождений нейронов в нейронную сеть








