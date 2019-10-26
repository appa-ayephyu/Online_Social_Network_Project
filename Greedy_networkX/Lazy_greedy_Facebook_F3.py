import numpy as np
import networkx as nx
import random
import copy
import time
from xml.dom import minidom
import heapq
import sys, os, inspect

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
        S.append(seed)
        # print(node_lookup)
        with open('output_Facebook_F3.txt', 'a') as f:
            if len(S) == 2 :
                print("Spread for seed node 2", -delta_gain, file = f)
            if len(S) == 5 :
                print("Spread for seed node 5", -delta_gain, file = f)
            if len(S) == 10 :
                print("Spread for seed node 10", -delta_gain, file = f)
            if len(S) == 15 :
                print("Spread for seed node 15", -delta_gain, file = f)
            if len(S) == 20 :
                print("Spread for seed node 20", -delta_gain, file = f)
            if len(S) == 25 :
                print("Spread for seed node 25", -delta_gain, file = f)
            if len(S) == 30 :
                print("Spread for seed node 30", -delta_gain, file = f)
            f.close()
    return S, -delta_gain


def probability_distribution(F):
    if F == 'F_1':
        return 0.01
    elif F == 'F_2':
        return np.random.exponential(scale=0.01)
    elif F == 'F_3':
        return random.choice([0.1, 0.01, 0.001])



G = nx.read_edgelist("facebook_combined.txt", create_using = nx.Graph(), nodetype = int)
print(nx.info(G))


start_time = time.time()
_, delta_spread = lazy_greedy_2(G, 30, 0.5, 'F_3', 100)
print("took {} seconds to run".format(time.time()-start_time))

