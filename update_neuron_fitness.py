import numpy as np
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