from generate_population import generate_population, num_neuron_connect
import numpy as np


def crossover():
    population, _ = generate_population()
    n_cross_neurons = int(0.25*population.shape[0])
    for i in range(1, n_cross_neurons, 2):
        # индексы скрещивания нейронов
        n1_id = np.random.randint(0, n_cross_neurons)
        n2_id = np.random.randint(0, n_cross_neurons)
        #точка разрыва
        p = np.random.randint(1, num_neuron_connect * 2 - 1)
        if(n1_id != n2_id):
            population[-i] = np.concatenate((population[n1_id, 0:p], population[n2_id, p:num_neuron_connect * 2]),
                                            axis=0)
        population[-(i+1)] = population[n1_id]
    print(population)
    return population

if __name__=='__main__':
    crossover()