import networkx as nx
import matplotlib.pyplot as plt

def draw_neural_network(num_input_neurons, num_hidden_neurons, num_output_neurons):
    G = nx.DiGraph()

    # Добавим входные нейроны
    input_neurons = [f'Input {i+1}' for i in range(num_input_neurons)]
    for neuron in input_neurons:
        G.add_node(neuron, layer=0)

    # Добавим скрытые нейроны
    hidden_neurons = [f'Hidden {i+1}' for i in range(num_hidden_neurons)]
    for neuron in hidden_neurons:
        G.add_node(neuron, layer=1)

    # Добавим выходные нейроны
    output_neurons = [f'Output {i+1}' for i in range(num_output_neurons)]
    for neuron in output_neurons:
        G.add_node(neuron, layer=2)

    # Связи между входными и скрытыми нейронами
    for i in input_neurons:
        for j in hidden_neurons:
            G.add_edge(i, j)

    # Связи между скрытыми и выходными нейронами
    for i in hidden_neurons:
        for j in output_neurons:
            G.add_edge(i, j)

    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()
