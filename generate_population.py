import numpy as np
from pprint import pprint
from parameters import *




num_neuron_pop = TOTAL_NEURONS
num_input_neurons = INPUT_NEURONS
num_hidden_neurons = HIDDEN_NEURONS
num_output_neurons = OUTPUT_NEURONS




def generate_population():
    ''' генерируем матрицу размером (num_neuron_pop, num_neuron_connect * 2), каждая строка этой матрицы это нейрон,
    а каждый столбец - связь этого нейрона. Заполняется случаеными числами.'''

    individual_len = num_input_neurons + num_output_neurons
    population = np.zeros((num_neuron_pop, individual_len * 2))

    for i in range(num_neuron_pop):
        for j in range(0, individual_len * 2, 2):
            population[i][j] = j / 2
            population[i][j + 1] = np.random.random() - 0.5

    # pprint(population[0][-20:])

    return population



if __name__=='__main__':
    generate_population()