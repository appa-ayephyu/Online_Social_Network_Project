import numpy as np
import networkx as nx
import random
import copy
import time
from xml.dom import minidom
import heapq


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


def DIC(G, S, new_s, ap, pp, mc):
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
                # inactive_neighbors = list(set(G.neighbors(node)) - set(A))
                neighbors = list(G.neighbors(node))
                # if not inactive_neighbors:
                #     break

                # prob_list = np.random.uniform(0, 1, len(inactive_neighbors))
                prob_list = np.random.uniform(0, 1, len(neighbors))
                influenced_neighbors = prob_list < pp
                # newly_influenced_nodes.extend(np.extract(influenced_neighbors, inactive_neighbors).tolist())
                newly_influenced_nodes.extend(np.extract(influenced_neighbors, neighbors).tolist())

                # A.extend(newly_influenced_nodes)

            # new_a = list(set(newly_influenced_nodes))
            new_a = list(set(newly_influenced_nodes) - set(A))
            # A = list(set(A))
            A.extend(new_a)

        average_spread += (len(A) * 1.0 / mc)
        # print(average_spread)

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
    S = []  # initial seed-set
    all_nodes = list(G.nodes)
    max_expected_spread = 0

    while B - len(S):
        current_best_seed = []
        for s in list(set(all_nodes) - set(S)):
            expected_spread = DIC(G, S, [s], ap, pp, mc)
            if expected_spread > max_expected_spread:
                max_expected_spread = expected_spread
                current_best_seed = [s]

        if current_best_seed:
            S.extend(current_best_seed)

    return S, max_expected_spread


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

def lazy_greedy_2(G, B, ap, pp, mc):
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
        heapq.heappush(spread_list, (-expected_spread, s))

    spread, seed = heapq.heappop(spread_list)
    spread = -spread

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
            expected_spread = DIC(G, S, [potential_seed], ap, pp, mc)
            marginal_gain = expected_spread - spread
            heapq.heappush(spread_list, (-marginal_gain, potential_seed))

            if spread_list[0][1] == potential_seed:
                found_best_seed = True

        spread, seed = heapq.heappop(spread_list)
        spread = -spread
        S.append(seed)
        print(node_lookup)

    return S, spread


G = read_xml_file('N_2500_beta_1.2_01.xml')
print(nx.info(G))
start_time = time.time()
print(lazy_greedy_2(G, 5, 1.0, 0.01, 1500))
print("took {} seconds to run".format(time.time()-start_time))
# print(greedy(G, 1, 1.0, 0.01, 10000))
