import numpy as np
from generate_population import generate_population, n_epoches, num_blueprints, num_neuron_pop, num_input_neurons, num_hidden_neurons,num_output_neurons
from load_data import X_train, y_train, X_val, y_val
from model import Model
from sklearn.metrics import log_loss

def run_algorithm():

    epoch = 0
    fitness_func = log_loss
    lost_arr = []

    population, blueprints = generate_population()

    while epoch < n_epoches:
        ''' Первый этап алгоритма: Удалить все значения пригодности для каждого нейрона'''
        blueprints_fitness = np.zeros(num_blueprints) # пригодности нейронной сети
        neuron_fitness = np.zeros(num_neuron_pop) # пригодности нейронов
        num_neuron_include = np.ones(num_neuron_pop) # пригодности вхождений нейронов в нейронную сеть
        '''Пересчитать приспогодные комбинации'''
        for bp_id in range(blueprints.shape(0)):
            net = population[blueprints[bp_id]]
            model = Model(net, num_input_neurons, num_output_neurons, num_hidden_neurons)
            preds = model.forward(X_train, f='relu')
            loss = fitness_func(y_train, preds)
            blueprints_fitness[bp_id] = loss
        '''сортировка модели от лучшей к худшей'''
        sort_id = np.argsort(blueprints_fitness)
        blueprints = blueprints[sort_id]
        blueprints_fitness = blueprints_fitness[sort_id]
        '''сохранение лучшей модели'''
        best_loss_train = blueprints_fitness[0]
        net = population[blueprints[0]]
        model = Model(net, num_input_neurons, num_output_neurons, num_hidden_neurons)
        preds = model.forward(X_val, f='relu')
        best_loss_val = fitness_func(y_val,preds)
        lost_arr.append(best_loss_val)










