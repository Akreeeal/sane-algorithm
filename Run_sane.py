import numpy as np
from generate_population import (generate_population,
                                 num_neuron_pop, num_input_neurons,
                                 num_hidden_neurons,num_output_neurons)
from load_data import X_train, y_train, X_val, y_val
from crossover import crossover
from model import Model
from sklearn.metrics import log_loss
from tqdm import tqdm



def update_neuron_fitness(n_fit, n_ids_in, loss):
    """
    Функция обновления значений приспособленности нейронов
    Аргументы:
        - n_fit - Массив приспособленностей нейронов (ndarray)
        - n_ids_in - Массив индексов нейронов вошедших в нейронную сеть (ndarray)
        - loss - Значение приспособленности нейронной сети
    """
    for i in np.unique(n_ids_in):
        n_fit[i] += loss
    return n_fit

def crossover(population):
    n_cross_neurons = int(0.25*population.shape[0])
    for i in range(1, n_cross_neurons, 2):
        # индексы скрещивания нейронов
        n1_id = np.random.randint(0, n_cross_neurons)
        n2_id = np.random.randint(0, n_cross_neurons)
        #точка разрыва
        p = np.random.randint(1, population.shape[1] - 1)
        if(n1_id != n2_id):
            population[-i] = np.concatenate((population[n1_id, 0:p], population[n2_id, p:]),
                                            axis=0)
        population[-(i+1)] = population[n1_id]

    return population

def mutaion(population):
    p = 0.001
    # p_id = 1 - (1 - p) ** 8
    p_weight = 0.08
    for i in range(num_neuron_pop):
        for j in range(population[i].shape[0]):
            if (np.random.random() < p_weight) and j % 2 != 0:
                population[i,j] - np.random.random() - 0.5

    return population


def run_algorithm():

    n_epoches = 10
    epoch = 0
    fitness_func = log_loss
    lost_arr = []


    population = generate_population()
    best_loss = np.inf

    pbar = tqdm(total=n_epoches)
    while epoch < n_epoches:
        ''' Первый этап алгоритма: Удалить все значения пригодности для каждого нейрона'''
        neuron_fitness = np.zeros(num_neuron_pop) # пригодности нейронов
        num_neuron_include = np.ones(num_neuron_pop) # пригодности вхождений нейронов в нейронную сеть

        for _ in range(10):

            '''cлучайным образом выбирается ~ нейронов из популяции'''

            random_NN = np.random.randint(0, num_neuron_pop,
                                          size=(num_hidden_neurons))

            for neuron_id in np.unique(random_NN):
                num_neuron_include[neuron_id] += 1

            net = population[random_NN]
            model = Model(
                net, num_input_neurons,
                num_output_neurons,
                num_hidden_neurons
            )

            preds = model.forward(X_train, f='relu')
            loss = fitness_func(y_train, preds)

            if loss < best_loss:
                best_loss = loss


            neuron_fitness = update_neuron_fitness(neuron_fitness,
                                                       random_NN, loss)

        neuron_fitness /= num_neuron_include

        sort_ids = np.argsort(neuron_fitness)
        population = population[sort_ids]

        population = crossover(population) # кросинговер нейронов
        population = mutaion(population) #мутация нейронов


        pbar.update(1)
        epoch += 1

        print(best_loss)

    pbar.close()


if __name__=='__main__':
    run_algorithm()



