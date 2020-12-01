import networkx as nx
import numpy as np

import os

from timeit import default_timer as timer

def get_ppr(G, Q, c=0.15, t=0.001, return_type='dict', quiet=True):
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

        return_type (str): This string is either 'dict' or 'array'. If 'dict' return a dictionary keyed by node id
        to its corresponding ppr score. If 'array' return a numpy array where ppr scores are indexed by the node id.

        quiet (bool): If set to true this function will not print() debugging information to standard output.

    Returns
    -------
    Returns two objects:

        1. The personalized page rank for each node in the graph (python dictionary keyed by node or just an numpy array
        indexed by the node id depending on the specified `return_type` argument).

        2. The number of iterations it took to the algorithm to converge
    '''

    allowed_return_types = ['dict', 'array']
    if return_type not in allowed_return_types:
        raise ValueError("Invalid return type argument. Expected one of: %s" % allowed_return_types)

    if not quiet:
        start = timer()
        print('Calculating PPR using particle filtering...')

    v = np.zeros(G.number_of_nodes())
    p = np.zeros(G.number_of_nodes())
    num_iterations = 0
    for q in Q:
        p[q] = 1/t
    while np.any(p):
        num_iterations += 1
        temp = np.zeros(G.number_of_nodes())
        # Loop all non-zero values in the p array
        for i in range(len(p)):
            if p[i] > 0:
                particles = p[i]*(1-c)
                # Get list of outgoing edges from node n sorted by descending order (highest weight edge first)
                edges=sorted(G.out_edges(i, data=True), key=lambda t: t[2]['weight'], reverse=True)
                
                for edge in edges:
                    if particles <= t:
                        break
                    passing = max(particles * edge[2]['weight'], t)
                    temp[edge[1]] = temp[edge[1]] + passing
                    particles -= passing
        p = temp
        for i in range(len(v)):
            v[i] = v[i] + p[i] * c

    if return_type == 'dict':
        # Construct the dictionary of the personalized page rank
        ppr_dict = {}
        for i in range(len(v)):
            ppr_dict[i] = v[i]

    if not quiet:
        print('Finished calculating PPR using particle filtering. Took', num_iterations, 'iterations for convergence.',
        'Elapsed time is:', timer()-start, 'seconds.\n')

    if return_type == 'dict':
        return ppr_dict, num_iterations
    elif return_type == 'array':
        return v, num_iterations

def get_ppr_from_single_source_nodes(dir):
    '''
    Given a directory of the PPR scores from each source node, combine them to
    get an estimate of the PPR score from all the source nodes.

    The combined vector is simply the average of all the ppr vectors from each source

    Arguments
    -------
        dir (str): Directory where all the ppr scores are stored for each source.
        All the files in the directory must be .npy binary files.

    Returns
    -------
    A numpy array of the combined ppr scores. The array is indexed by the node id 
    '''
    combined_ppr_scores = np.array([])
    numpy_files = os.listdir(dir)
    for numpy_file in numpy_files:
        with open(dir + numpy_file, 'rb') as f:
            ppr_scores = np.load(f)
            if combined_ppr_scores.size == 0:
                # Initialize to zero
                combined_ppr_scores = np.zeros(len(ppr_scores))
            combined_ppr_scores += ppr_scores

    # Average the array by the number source nodes
    combined_ppr_scores /= len(numpy_files)
    return combined_ppr_scores
