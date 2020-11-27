import networkx as nx
import numpy as np
from timeit import default_timer as timer

def get_ppr(G, Q, c=0.15, t=0.001):
    '''
    Run ppr using particle filtering on graph `G` and using query nodes `Q`

    Arguments
    -------
        G (networkx graph): Path to the input csv file. The CSV file will be converted into a pandas dataframe

        Q (list of str or int): A list of the query nodes

        c (float): Restart probability

        t (float): Parameter that controls the number of particles used. There are 1/t particles from each query node
        TODO: If we want to use a non-uniform distribution of particles for each node, we can feed a vector of `t` values
        for each node in the query nodes set Q. 

    Returns
    -------
    Returns the personalized page rank for each node in the graph (python dictionary keyed by node).
    '''

    start = timer()
    print('Calculating PPR using particle filtering...')

    # TODO: This step could be done as a priori for optimization.
    # So then we only deal with graphs specified by ids and not names of nodes which can be very lengthy (i.e. string values)
    #  
    # Create a mapping of every node to a unique integer index starting from 0
    indices = range(G.number_of_nodes())
    nodes = list(G.nodes())
    node_to_index_dict = {nodes[i]: indices[i] for i in range(G.number_of_nodes())}

    v = np.zeros(G.number_of_nodes())
    p = np.zeros(G.number_of_nodes())
    for q in Q:
        p[node_to_index_dict[q]] = 1/t
    while np.any(p):
        temp = np.zeros(G.number_of_nodes())
        # Loop all non-zero values in the p array
        for i in range(len(p)):
            if p[i] > 0:
                particles = p[i]*(1-c)
                # Get list of outgoing edges from node n sorted by descending order (highest weight edge first)
                edges=sorted(G.out_edges(nodes[i], data=True), key=lambda t: t[2]['weight'], reverse=True)
                
                for edge in edges:
                    if particles <= t:
                        break
                    passing = max(particles * edge[2]['weight'], t)
                    temp[node_to_index_dict[edge[1]]] = temp[node_to_index_dict[edge[1]]] + passing
                    particles -= passing
        p = temp
        for i in range(len(v)):
            v[i] = v[i] + p[i] * c

    # Construct the dictionary of the personalized page rank
    ppr_dict = {}
    for i in range(len(v)):
        ppr_dict[nodes[i]] = v[i]


    print('Finished calculating PPR using particle filtering. Elapsed time is:', timer()-start, 'seconds.\n')

    return ppr_dict
