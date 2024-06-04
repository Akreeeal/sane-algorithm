import numpy as np

from crossover import crossover
from generate_population import num_neuron_pop,num_input_neurons,num_output_neurons

def mutaion():
    population = crossover()
    p = 0.001
    p_id = 1 - (1 - p) ** 8
    p_w = 1 - (1 - p) ** 16
    for i in range(num_neuron_pop):
        for j in range(population[i].shape[0]):
            if (j % 2 == 0):
                if (np.random.random() < p_id):
                    n_id = np.random.randint(0, num_input_neurons + num_output_neurons)
                    population[i,j] = n_id
            else:
                if (np.random.random() < p_w):
                    population[i,j] - np.random.random() - 0.5
    print(population)
    return population

if __name__=='__main__':
    mutaion()