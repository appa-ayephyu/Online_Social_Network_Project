import numpy as np
import networkx as nx
import random
import copy
import time
from xml.dom import minidom
import heapq
import sys, os, inspect

def read_xml_file(path):
    G = nx.DiGraph()
    mydoc= minidom.parse(path)
    node_list = mydoc.getElementsByTagName('node')
    edge_list = mydoc.getElementsByTagName('edge')
    for node in node_list:
        G.add_node(node.attributes['ID'].value, activated=False)

    for edge in edge_list:
        G.add_edge(edge.attributes['nodeID_1'].value, edge.attributes['nodeID_2'].value)

    return G


def DIC(G, S, new_s, ap, F, mc):
    """
    :param G: Graph
    :param S: Seed-set
    :param new_s: new potential seed node
    :param pp: Propagation Probability
    :param mc: Number of Monte-Carlo simulations
    :return: Expected nodes influenced by seed-set
    """
    average_spread = 0
    for _ in range(mc):
        A = S[:]  # activated nodes
        new_a = S[:]  # new activated nodes
        if random.uniform(0, 1) < ap:  # if new seed node is activated
            A.extend(new_s)
            new_a.extend(new_s)

        while new_a:
            newly_influenced_nodes = []
            for node in new_a:
                neighbors = list(G.neighbors(node))
                prob_list = np.random.uniform(0, 1, len(neighbors))
                pp = np.array([probability_distribution(F) for _ in range(len(neighbors))])
                influenced_neighbors = prob_list < pp
                newly_influenced_nodes.extend(np.extract(influenced_neighbors, neighbors).tolist())

            new_a = list(set(newly_influenced_nodes) - set(A))
            A.extend(new_a)

        average_spread += (len(A) * 1.0 / mc)
        # print(average_spread)

    return average_spread


def adaptive_DIC(G, activated_n, new_s, ap, F, mc):
    """
    :param G: Graph
    :param S: Seed-set
    :param new_s: new potential seed node
    :param pp: Propagation Probability
    :param mc: Number of Monte-Carlo simulations
    :return: Expected nodes influenced by seed-set
    """
    average_spread = 0
    for _ in range(mc):
        A = activated_n[:]  # activated nodes
        new_a = [] # new activated nodes
        if random.uniform(0, 1) < ap:  # if new seed node is activated
            A.extend(new_s)
            new_a.extend(new_s)

        while new_a:
            newly_influenced_nodes = []
            for node in new_a:
                neighbors = list(G.neighbors(node))
                prob_list = np.random.uniform(0, 1, len(neighbors))
                pp = np.array([probability_distribution(F) for _ in range(len(neighbors))])
                influenced_neighbors = prob_list < pp
                newly_influenced_nodes.extend(np.extract(influenced_neighbors, neighbors).tolist())

            new_a = list(set(newly_influenced_nodes) - set(A))
            A.extend(new_a)

        if len(list(set(A))) != len(A):
            print("gg wrong! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        average_spread += len(A)
        # print(average_spread)
    return average_spread * 1.0 / mc


def activate_nodes(G, activated_n, new_s, F):
    A = activated_n[:]
    new_a = new_s[:]
    while new_a:
        newly_influenced_nodes = []
        A.extend(new_a)
        for node in new_a:
            neighbors = list(G.neighbors(node))
            prob_list = np.random.uniform(0, 1, len(neighbors))
            pp = np.array([probability_distribution(F) for _ in range(len(neighbors))])

            influenced_neighbors = prob_list < pp
            newly_influenced_nodes.extend(np.extract(influenced_neighbors, neighbors).tolist())

        new_a = list(set(newly_influenced_nodes) - set(A))

    return A


def lazy_greedy(G, B, ap, pp, mc):
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

    spread_list = []

    for s in list(set(all_nodes) - set(S)):
        # print('new s is: {} and S is {}'.format(s, S))
        expected_spread = DIC(G, S, [s], ap, pp, mc)
        spread_list.append([s, expected_spread])

    spread_list.sort(key=lambda x: x[1], reverse=True)
    S.append(spread_list[0][0])
    spread = spread_list[0][1]
    spread_list.pop(0)

    # spread_list_copy = spread_list[:]
    while B-len(S):
        # current_best_node = []
        sorted_spread = spread_list[1:]
        for i in range(len(spread_list)):
            s = spread_list[i]
            position_check = 0
            expected_spread = DIC(G, S, [s[0]], ap, pp, mc)
            marginal_gain = expected_spread - spread
            if s[0] == sorted_spread[0][0]:
                position_check = 1

            if marginal_gain >= sorted_spread[position_check][1]:
                S.append(s[0])
                spread_list.pop(i)
                spread = expected_spread
                break
            else:
                spread_list[i][1] = marginal_gain
                sorted_spread = spread_list[:]
                sorted_spread.sort(key=lambda x:x[1], reverse=True)

        spread_list.sort(key=lambda x: x[1], reverse=True)

    return S, spread


def lazy_greedy_2(G, B, ap, F, mc):
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

    spread_list = []

    counter_1 = 0
    for s in list(set(all_nodes) - set(S)):
        counter_1 += 1
        # print("counter: {}".format(counter_1))
        # print('new s is: {} and S is {}'.format(s, S))
        expected_spread = DIC(G, S, [s], ap, F, mc)
        heapq.heappush(spread_list, (-expected_spread, s))

    spread, seed = heapq.heappop(spread_list)
    delta_spread = -spread

    S.append(seed)
    spread_list.pop(0)

    # spread_list_copy = spread_list[:]
    while B-len(S):
        # current_best_node = []
        found_best_seed = False
        node_lookup = 0
        while not found_best_seed:
            node_lookup += 1
            _, potential_seed = heapq.heappop(spread_list)
            expected_spread = DIC(G, S, [potential_seed], ap, F, mc)
            marginal_gain = expected_spread - delta_spread
            heapq.heappush(spread_list, (-marginal_gain, potential_seed))

            if spread_list[0][1] == potential_seed:
                found_best_seed = True

        delta_gain, seed = heapq.heappop(spread_list)
        delta_spread += -delta_gain
        S.append(seed)
        # print(node_lookup)

    return S, delta_spread


def probability_distribution(F):
    if F == 'F_1':
        return 0.01
    elif F == 'F_2':
        return np.random.exponential(scale=0.01)
    elif F == 'F_3':
        return random.choice([0.1, 0.01, 0.001])


def a_greedy(G, B, ap, F, mc):
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

    spread_list = []

    start_time = time.time()
    counter_1 = 0
    for s in list(set(all_nodes) - set(S)):
        counter_1+=1
        if (counter_1 - 1) % 100 == 0:
            print("counter: {}".format(counter_1))
        # print('new s is: {} and S is {}'.format(s, S))
        expected_spread = adaptive_DIC(G, S, [s], ap, F, mc)
        heapq.heappush(spread_list, (-expected_spread, s))

    spread, seed = heapq.heappop(spread_list)
    delta_spread = -spread

    print("took {} seconds to run first round".format(time.time() - start_time))

    S.append(seed)
    spread_list.pop(0)

    activated_nodes = []
    activated_nodes = activate_nodes(G, activated_nodes, S, F)

    # spread_list_copy = spread_list[:]
    while B-len(S):
        # current_best_node = []
        found_best_seed = False
        node_lookup = 0
        spread = len(activated_nodes)
        while not found_best_seed:
            _, potential_seed = heapq.heappop(spread_list)
            if potential_seed in activated_nodes:
                continue
            node_lookup += 1
            expected_spread = adaptive_DIC(G, activated_nodes, [potential_seed], ap, F, mc)
            marginal_gain = expected_spread - spread
            if marginal_gain < 0:
                print("gg something is wrong!")

            heapq.heappush(spread_list, (-marginal_gain, potential_seed))

            if spread_list[0][1] == potential_seed:
                found_best_seed = True

        delta_gain, new_seed = heapq.heappop(spread_list)
        delta_spread += -delta_gain
        S.append(new_seed)
        activated_nodes = activate_nodes(G, activated_nodes, [new_seed], F)
        print(node_lookup, -delta_gain)

    return S, delta_spread


G = read_xml_file('N_2500_beta_1.2_01.xml')
# G = nx.read_edgelist("C:/distributed-project/Online_Social_Network_Project/data/Wiki-Vote.txt", create_using = nx.DiGraph(), nodetype = int)
print(nx.info(G))
# print(list(G.nodes)[:10])

for num_seed in [2, 5, 10, 15, 20, 25, 30]:
    start_time = time.time()
    _, delta_spread = a_greedy(G, num_seed, 1.0, 'F_3', 1000)
    print("\n================ Number of seeds: {} ====================".format(num_seed))
    print("delta_spread from a_greedy: {}".format(delta_spread))
    print("took {} seconds to run".format(time.time()-start_time))

# start_time = time.time()
# _, spread = lazy_greedy_2(G, 30, 1.0, 0.01, 1000)
# print("spread from lazy greedy: {}".format(spread))
# print("took {} seconds to run".format(time.time()-start_time))
