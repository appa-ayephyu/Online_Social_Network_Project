import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import networkx as nx
import time

n = 10  # 10 nodes
m = 20  # 20 edges

G = nx.gnm_random_graph(n, m)


# some properties
print("node degree clustering")
for v in nx.nodes(G):
    print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

# print the adjacency list
for line in nx.generate_adjlist(G):
    print(line)

nx.draw(G)
plt.show()

def IC(g,S,p,mc):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """

    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):

        # Simulate propagation process
        new_active, A = S[:], S[:]
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:

                # Determine those neighbors that become infected
                np.random.seed(i)
                neighbor = list(g.neighbors(node))
                success = np.random.uniform(0,1,len(neighbor)) < p
                new_ones += list(np.extract(success, list(g.neighbors(node))))
            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active
        spread.append(len(A))

    return(np.mean(spread))
S = [2,4]
mean_spread = IC(G,S,0.5,10)

def greedy(g,k,p=0.1,mc=1000):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    S, spread, timelapse, start_time = [], [], [], time.time()

    # Find k nodes with largest marginal gain
    for _ in range(k):

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in set(range(g.number_of_nodes()))-set(S):

            # Get the spread
            s = IC(g,S + [j],p,mc)

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j

        # Add the selected node to the seed set
        S.append(node)

        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)

    return(S,spread,timelapse)

greedy_output = greedy(G,2,p=0.2,mc=1000)
print("greedy output: " + str(greedy_output[0]))
