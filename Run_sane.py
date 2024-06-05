import numpy as np
from matplotlib.figure import Figure

from generate_population import (generate_population,
                                 num_neuron_pop, num_input_neurons,
                                 num_hidden_neurons,num_output_neurons)
from load_data import X_train, y_train, X_val, y_val
from crossover import crossover
from model import Model
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt


def make_charts(graph_data, save_img=False):
    figure = Figure(figsize=(12, 5), dpi=100)

    ax_loss = figure.add_subplot(1, 2, 1)
    ax_loss.plot(graph_data["loss_array_train"], color="blue", label="train")
    ax_loss.plot(graph_data["loss_array_test"], color="orange", label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()

    ax_accuracy = figure.add_subplot(1, 2, 2)
    ax_accuracy.plot(graph_data["acc_array_train"], color="blue", label="train")
    ax_accuracy.plot(graph_data["acc_array_test"], color="orange", label="val")
    ax_accuracy.set_xlabel("epoch")
    ax_accuracy.set_ylabel("accuracy")
    ax_accuracy.legend()

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


def run_algorithm(n_epoches):
    loss_array_train = []  # массив для хранения значений потерь на тренировочном наборе данных
    loss_array_test = []  # массив для хранения значений потерь на валидационном наборе данных
    acc_array_train = []  # массив для хранения значений точности на тренировочном наборе данных
    acc_array_test = []  # массив для хранения значений точности на валидационном наборе данных

    epoch = 0
    fitness_func = log_loss
    loss_history = []


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

            # preds = model.forward(X_train, f='relu')
            # loss = fitness_func(y_train, preds)

            preds_train = model.forward(X_train, f='relu')
            loss_train = fitness_func(y_train, preds_train)
            loss_array_train.append(loss_train)

            preds_val = model.forward(X_val, f='relu')
            loss_val = fitness_func(y_val, preds_val)
            loss_array_test.append(loss_val)

            acc_train = accuracy_score(y_train, one_hot_from_softmax(preds_train))
            acc_array_train.append(acc_train)

            acc_val = accuracy_score(y_val, one_hot_from_softmax(preds_val))
            acc_array_test.append(acc_val)

            if loss_train  < best_loss:
                best_loss = loss_train

            loss_history.append(best_loss)
            best_model = model

            neuron_fitness = update_neuron_fitness(neuron_fitness,
                                                       random_NN, loss_train)

        neuron_fitness /= num_neuron_include

        sort_ids = np.argsort(neuron_fitness)
        population = population[sort_ids]

        population = crossover(population)  # кросинговер нейронов
        population = mutaion(population)  # мутация нейронов

        pbar.update(1)
        epoch += 1

        preds_val = best_model.forward(X_val)
        print(accuracy_score(y_val, one_hot_from_softmax(preds_val)))

        print(best_loss)

    pbar.close()

    return loss_history, loss_array_train, loss_array_test, acc_array_train, acc_array_test

def one_hot_from_softmax(softmax_array):
    one_hot_array = np.zeros_like(softmax_array)
    max_indices = np.argmax(softmax_array, axis=1)
    for i, max_index in enumerate(max_indices):
        one_hot_array[i, max_index] = 1
    return one_hot_array




if __name__=='__main__':

    n_epoches = 1
    loss_history,loss_array_train, loss_array_test, acc_array_train, acc_array_test = run_algorithm(n_epoches)

    graph_data = {
        "loss_array_train": loss_array_train,
        "loss_array_test": loss_array_test,
        "acc_array_train": acc_array_train,
        "acc_array_test": acc_array_test
    }

    make_charts(graph_data)




