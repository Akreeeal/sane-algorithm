import numpy as np
from generate_population import (generate_population,
                                 num_neuron_pop, num_input_neurons,
                                 num_hidden_neurons,num_output_neurons)
from load_data import X_train, y_train, X_val, y_val
from model import Model
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from parameters import TOTAL_EPOCHES, PATIENCE, TOTAL_NEURONS, HIDDEN_NEURONS
import time
import pickle





def one_hot_from_softmax(softmax_array):
    one_hot_array = np.zeros_like(softmax_array)
    max_indices = np.argmax(softmax_array, axis=1)
    for i, max_index in enumerate(max_indices):
        one_hot_array[i, max_index] = 1
    return one_hot_array

def make_charts(graph_data, n_epoches, save_img=False):
    fig, (ax_loss, ax_accuracy) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

    ax_loss.plot(graph_data["loss_array_train"], color="orange", label="train")
    ax_loss.plot(graph_data["loss_array_test"], color="blue", label="val")
    ax_loss.set_xlabel("Эпоха")
    ax_loss.set_ylabel("Ошибка")
    ax_loss.legend()
    ax_loss.set_title(f"Потери на протяжении {n_epoches} эпох")

    ax_accuracy.plot(graph_data["acc_array_train"], color="orange", label="train")
    ax_accuracy.plot(graph_data["acc_array_test"], color="blue", label="val")
    ax_accuracy.set_xlabel("Эпоха")
    ax_accuracy.set_ylabel("Точность")
    ax_accuracy.legend()
    ax_accuracy.set_title(f"Точность на протяжении {n_epoches} эпох")

    plt.tight_layout()
    if save_img:
        plt.savefig(f"График обучения на {n_epoches} эпохах.png")
    plt.show()

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

def mutation(population):
    p_weight = 0.001
    for i in range(num_neuron_pop):
        for j in range(population[i].shape[0]):
            if (np.random.random() < p_weight) and j % 2 != 0:
                population[i,j] - np.random.random() - 0.5

    return population


def run_algorithm(n_epoches, patience):
    loss_array_train = []  # массив для хранения значений потерь на тренировочном наборе данных
    loss_array_test = []  # массив для хранения значений потерь на валидационном наборе данных
    acc_array_train = []  # массив для хранения значений точности на тренировочном наборе данных
    acc_array_test = []  # массив для хранения значений точности на валидационном наборе данных

    epochs_without_improvement = 0
    epoch = 0
    fitness_func = log_loss
    loss_history = []
    accuracy_history = []


    population = generate_population()
    best_loss = np.inf

    pbar = tqdm(total=n_epoches)
    start_time = time.time()
    while epoch < n_epoches:
        ''' Первый этап алгоритма: Удалить все значения пригодности для каждого нейрона'''
        neuron_fitness = np.zeros(num_neuron_pop) # пригодности нейронов
        num_neuron_include = np.ones(num_neuron_pop) # пригодности вхождений нейронов в нейронную сеть

        for _ in range(250):

            '''рандомно выбирается ~ нейронов из популяции'''

            random_NN = np.random.randint(0, num_neuron_pop,
                                          size=(num_hidden_neurons))

            ''''''
            for neuron_id in np.unique(random_NN):
                num_neuron_include[neuron_id] += 1

            net = population[random_NN]
            model = Model(
                net, num_input_neurons,
                num_output_neurons,
                num_hidden_neurons
            )

            preds_train = model.forward(X_train, f='relu')
            loss_train = fitness_func(y_train, preds_train)

            preds_val = model.forward(X_val, f='relu')
            loss_val = fitness_func(y_val, preds_val)

            acc_train = accuracy_score(y_train, one_hot_from_softmax(preds_train))
            acc_val = accuracy_score(y_val, one_hot_from_softmax(preds_val))

            if loss_train < best_loss:
                best_loss = loss_train
                epochs_without_improvement = epoch
                best_model = model

                with open("outputs/best_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)



            loss_history.append(best_loss)
            best_model = model

            neuron_fitness = update_neuron_fitness(neuron_fitness, random_NN, loss_train)

        if epoch - epochs_without_improvement >= patience:
            print(f"Остановка обучения на эпохе: {epoch}")
            break

        loss_array_train.append(loss_train)
        loss_array_test.append(loss_val)
        acc_array_train.append(acc_train)
        acc_array_test.append(acc_val)

        neuron_fitness /= num_neuron_include

        sort_ids = np.argsort(neuron_fitness)
        population = population[sort_ids]

        population = crossover(population)  # скрещивание нейронов
        population = mutation(population)  # мутация нейронов

        pbar.update(1)
        epoch += 1

        preds_val = best_model.forward(X_val)

        print(f'Точность{accuracy_score(y_val, one_hot_from_softmax(preds_val))}')
        accuracy_history.append(accuracy_score(y_val, one_hot_from_softmax(preds_val)))
        print(f'Ошибка на этапе обучения: {best_loss}')
    pbar.close()



    print('=======================================')
    print(f'Минимальная ошибка: {min(loss_history)}')
    print(f'Максимальная точность на этапе валидации: {max(accuracy_history)}')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Общее время обучения: {round(total_time, 2)}')
    print('=======================================')

    with open("outputs/training_summary.txt", "w", encoding="utf-8") as file:
        file.write('==========================================\n')
        file.write(f'Размеры обучающего набора MNIST: {X_train.shape}\n')
        file.write(f'Размеры валидационного набора MNIST: {X_val.shape}\n')
        file.write('==========================================\n')
        file.write(f'Количество нейронов в популяции: {TOTAL_NEURONS}\n')
        file.write(f'Количество скрытых нейронов: {HIDDEN_NEURONS}\n')
        file.write(f'Заданное количесво эпох: {TOTAL_EPOCHES}\n')
        file.write('==========================================\n')
        file.write(f'Обучение остановилось на {epoch} эпохе\n')
        file.write(f'Минимальная ошибка: {min(loss_history)}\n')
        file.write(f'Максимальная точность на этапе валидации: {max(accuracy_history)}\n')
        file.write(f'Общее время обучения: {round(total_time, 2)} сек.\n')
        file.write('==========================================\n')


    return loss_history, loss_array_train, loss_array_test, acc_array_train, acc_array_test

if __name__=='__main__':
    patience = PATIENCE
    n_epoches = TOTAL_EPOCHES
    loss_history,loss_array_train, loss_array_test, acc_array_train, acc_array_test = run_algorithm(n_epoches, patience)

    graph_data = {
        "loss_array_train": loss_array_train,
        "loss_array_test": loss_array_test,
        "acc_array_train": acc_array_train,
        "acc_array_test": acc_array_test
    }

    make_charts(graph_data,n_epoches, save_img=True)




