import sys
import csv
import numpy as np
from load_data import *
from model import Model

# Глобальные переменные
n_epoches = 100 # кол-во эпох
fitness_func = None
num_neuron_pop = 50 # Кол - во нейронов в популяции
num_input_neurons = 10
num_output_neurons = 1
num_hidden_neurons = 5
num_neuron_connect = 10 #кол-во связей нейрона (кол-во генов * 2)
num_blueprints = 10
act_func = 'relu'
patience = 7
early_stop = False

# Инициализация популяции
def generate_population():
    # создаем
    population = np.zeros((num_neuron_pop, num_neuron_connect * 2))
    for i in range(num_neuron_pop):
        for j in range(0, num_neuron_connect * 2, 2):
            neuron_id = np.random.randint(0, num_input_neurons + num_output_neurons)
            population[i][j] = neuron_id
            population[i][j + 1] = np.random.random() - 0.5
    blueprints = np.zeros((num_blueprints, num_hidden_neurons), dtype=int)
    for i in range(num_blueprints):
        blueprints[i] = np.random.randint(0, num_neuron_pop, size=(num_hidden_neurons))
    return population, blueprints

# Оценка приспособленности
def evaluate_fitness(population, blueprints, X, y):
    blueprint_fitness = np.zeros(num_blueprints)
    for bp_id in range(num_blueprints):
        net = population[blueprints[bp_id]]
        model = Model(net, num_input_neurons, num_output_neurons, num_hidden_neurons)
        preds = model.forward(X, f=act_func)
        loss = fitness_func(y, preds)
        blueprint_fitness[bp_id] = loss
    return blueprint_fitness

# Обновление приспособленности нейронов
def update_neuron_fitness(n_fit, n_ids_in, loss):
    for i in np.unique(n_ids_in):
        n_fit[i] += loss
    return n_fit

# Скрещивание нейронов
def neuron_crossover(population):
    n_cross_neurons = int(0.25 * population.shape[0])
    for i in range(1, n_cross_neurons, 2):
        n1_id = np.random.randint(0, n_cross_neurons)
        n2_id = np.random.randint(0, n_cross_neurons)
        p = np.random.randint(1, num_neuron_connect * 2 - 1)
        if n1_id != n2_id:
            population[-i] = np.concatenate((population[n1_id, 0:p], population[n2_id, p:num_neuron_connect * 2]), axis=0)
            population[-(i + 1)] = population[n1_id]
    return population

# Мутация нейронов
def neuron_mutation(population, p):
    pid = 1 - (1 - p) ** 8
    pw = 1 - (1 - p) ** 16
    for i in range(num_neuron_pop):
        for j in range(population[i].shape[0]):
            if j % 2 == 0:
                if np.random.random() < pid:
                    n_id = np.random.randint(0, num_input_neurons + num_output_neurons)
                    population[i, j] = n_id
            else:
                if np.random.random() < pw:
                    population[i, j] = np.random.random() - 0.5
    return population

# Скрещивание blueprints
def blueprint_crossover(blueprints):
    n_cross_bp = int(0.25 * num_blueprints)
    for i in range(1, n_cross_bp, 2):
        n1_id = np.random.randint(0, n_cross_bp)
        n2_id = np.random.randint(0, n_cross_bp)
        p = np.random.randint(1, num_hidden_neurons)
        if n1_id != n2_id:
            blueprints[-i] = np.concatenate((blueprints[n1_id, 0:p], blueprints[n2_id, p:num_hidden_neurons]), axis=0)
            blueprints[-(i + 1)] = blueprints[n1_id]
    return blueprints

# Мутация blueprints
def blueprint_mutation(blueprints, p1, p2):
    n_mut_bp = int(0.75 * num_blueprints)
    n_mut_neurons = int(0.25 * num_neuron_pop)
    p2 = p2 + p1
    for i in range(n_mut_bp, num_blueprints):
        for j in range(num_hidden_neurons):
            pm = np.random.random()
            if pm < p1:
                n_id = np.random.randint(0, num_neuron_pop - n_mut_neurons)
                blueprints[i, j] = n_id
            if p1 < pm < p2:
                n_id = np.random.randint(num_neuron_pop - n_mut_neurons, num_neuron_pop)
                blueprints[i, j] = n_id
    return blueprints

# Основной цикл алгоритма
def run_algorithm():
    global early_stop
    loss_arr = []
    population, blueprints = generate_population()
    best_score = None
    counter = 0

    for epoch in range(n_epoches):
        blueprint_fitness = evaluate_fitness(population, blueprints, X_train, y_train)
        sort_id = np.argsort(blueprint_fitness)
        blueprints = blueprints[sort_id]
        blueprint_fitness = blueprint_fitness[sort_id]

        best_loss_train = blueprint_fitness[0]
        net = population[blueprints[0]]
        model = Model(net, num_input_neurons, num_output_neurons, num_hidden_neurons)
        preds = model.forward(X_val, f=act_func)
        best_loss_val = fitness_func(y_val, preds)
        loss_arr.append(best_loss_val)

        # Early stopping logic
        best_score, counter, early_stop = early_stopping(best_loss_val, net, epoch, patience, best_score, counter, early_stop, "model")
        if early_stop:
            break

        neuron_fitness = np.zeros(num_neuron_pop)
        num_neuron_include = np.ones(num_neuron_pop)

        for _ in range(1000):
            random_NN = np.random.randint(0, num_neuron_pop, size=(num_hidden_neurons))
            for n_id in np.unique(random_NN):
                num_neuron_include[n_id] += 1
            net = population[random_NN]
            model = Model(net, num_input_neurons, num_output_neurons, num_hidden_neurons)
            preds = model.forward(X_train, f=act_func)
            loss = fitness_func(y_train, preds)
            neuron_fitness = update_neuron_fitness(neuron_fitness, random_NN, loss)

        neuron_fitness /= num_neuron_include
        sort_id = np.argsort(neuron_fitness)
        population = population[sort_id]

        for i in range(num_blueprints):
            for j in range(num_hidden_neurons):
                blueprints[i, j] = np.where(sort_id == blueprints[i, j])[0][0]

        population = neuron_crossover(population)
        population = neuron_mutation(population, 0.001)
        blueprints = blueprint_crossover(blueprints)
        blueprints = blueprint_mutation(blueprints, 0.01, 0.5)

    return population[blueprints[0]], loss_arr

def log_to_csv(model_name, epoch, counter, accuracy):
    try:
        if epoch == 0:
            mode = 'w'
            with open(f'./outputs/logs/{model_name}_{counter}.csv', mode, newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Train loss", "Val Loss"])
        else:
            mode = 'a'

        with open(f'./outputs/logs/{model_name}_{counter}.csv', mode, newline='') as file:
            writer = csv.writer(file)
            writer.writerow(accuracy)
        file.close()
    except FileNotFoundError:
        print("Wrong path to the log file.")

def save_best_model(model_name, current_valid_loss, best_valid_loss, epoch, model, counter):
    if current_valid_loss < best_valid_loss:
        best_valid_loss = current_valid_loss
        print(f"\nBest validation loss: {best_valid_loss}")
        print(f"\nSaving best model for epoch: {epoch + 1}\n")
        np.save(f'./outputs/models/{model_name}_{counter}', model)
    return best_valid_loss

def early_stopping(val_loss, model, epoch, patience, best_score, counter, early_stop, model_name):
    score = -val_loss
    if best_score is None:
        best_score = score
        best_score = save_best_model(model_name, val_loss, float('inf'), epoch, model, counter)
        counter = 0
    elif score < best_score:
        counter += 1
        print(f'EarlyStopping counter: {counter} out of {patience}')
        if counter >= patience:
            early_stop = True
    else:
        best_score = score
        best_score = save_best_model(model_name, val_loss, float('inf'), epoch, model, counter)
        counter = 0
    return best_score, counter, early_stop

# Пример вызова функции
if __name__ == "__main__":
    # Определите fitness_func, X_train, y_train, X_val, y_val, Model, early_stopping здесь
    fitness_func = fitness_func  # Замените на вашу функцию потерь
    X_train, y_train = X_train, y_train  # Замените на ваши данные для обучения
    X_val, y_val = X_val, y_val  # Замените на ваши данные для валидации
    best_model, loss_history = run_algorithm()