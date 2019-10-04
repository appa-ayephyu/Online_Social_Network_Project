import numpy as np
import networkx as nx
import random
from xml.dom import minidom


def read_xml_file(path):
    G = nx.DiGraph()
    mydoc= minidom.parse(path)
    node_list = mydoc.getElementsByTagName('node')
    edge_list = mydoc.getElementsByTagName('edge')
    for node in node_list:
        G.add_node(node.attributes['ID'].value)

    for edge in edge_list:
        G.add_edge(edge.attributes['nodeID_1'].value, edge.attributes['nodeID_2'].value)

    return G


def DIC(G, S, pp, mc):
    """
    :param G: Graph
    :param S: Seed-set
    :param pp: Propagation Probability
    :param mc: Number of Monte-Carlo simulations
    :return: Expected nodes influenced by seed-set
    """
    average_spread = 0
    for _ in range(mc):
        A = S[:] # activated nodes
        new_a = S[:] # new activated nodes
        while new_a:
            newly_influenced_nodes = []
            for node in new_a:
                inactive_neighbors = list(set(G.neighbors(node)) - set(A))
                if not inactive_neighbors:
                    break

                prob_list = np.random.uniform(0, 1, len(inactive_neighbors))
                influenced_neighbors = prob_list < pp
                newly_influenced_nodes.extend(np.extract(influenced_neighbors, inactive_neighbors).tolist())
                A.extend(newly_influenced_nodes)

            new_a = newly_influenced_nodes

        average_spread += (len(A) * 1.0 / mc)

    return average_spread


def greedy(G, B, ap, pp, mc):
    """
    :param G: Graph
    :param B: Budget
    :param ap: Activation Probability for seed node
    :param pp: Propagation Probability
    :param mc: Number of Monte-Carlo simulations
    :return: max spread
    """

    S = [] # initial seed-set
    all_nodes = list(G.nodes)
    max_expected_spread = 0
    while B-len(S):
        current_best_node = []
        for s in list(set(all_nodes)-set(S)):
            if random.uniform(0, 1) < ap: # if new seed node is activated
                # print('new s is: {} and S is {}'.format(s, S))
                expected_spread = DIC(G, S + [s], pp, mc)
                if expected_spread > max_expected_spread:
                    max_expected_spread = expected_spread
                    current_best_node = [s]

        # greedy pick the best seed node and add to the current seed-set
        S.extend(current_best_node)

    return S, max_expected_spread


G = read_xml_file('N_2500_beta_1.2_01.xml')
print(nx.info(G))
print(greedy(G, 2, 0.5, 0.1, 10))

# GOT BUG!
# n = 5  # 50 nodes
# m = 20  # 200 edges
#
# G = nx.gnm_random_graph(n, m)
#
# print(greedy(G, 1, 0.5, 0.1, 10000))