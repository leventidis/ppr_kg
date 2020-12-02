import networkx as nx
import numpy as np

import random
import pickle
import argparse
import sys
import json

import utils

from pathlib import Path
from timeit import default_timer as timer

import matplotlib.pyplot as plt

def main(args):
    print('Constructing random graph...')
    start = timer()

    if args.graph_generator == 'scale-free':
        # Ideally we want a scale-free graph generator but it runs very slowly, so not recommended for large graphs
        G = nx.scale_free_graph(n=args.num_nodes, seed=args.seed)
    elif args.graph_generator == 'erdos-renyi':
        # Using a fast Erdos Renyi graph generator.
        # p is set to ln(p)/n that that the generated graph is almost always guaranteed to be connected 
        p = np.log(args.num_nodes) / args.num_nodes
        G = nx.fast_gnp_random_graph(n=args.num_nodes, p=p, seed=args.seed, directed=True)

    # Add weights to the edges
    for node in G:
        edges = list(G.out_edges(node, data=True))
        if len(edges) >= 1:
            # Assign a uniform weight to each outgoing edge such that all outgoing edges have weights that sum to 1
            weight = 1 / len(edges)
            for edge in edges:
                G[edge[0]][edge[1]]['weight'] = weight

    elapsed_time = timer()-start
    print('Finished constructing random graph. Elapsed time:', elapsed_time, 'seconds.')
    print('Constructed graph has:', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges.\n\n')

    # Save the graph 
    nx.write_gpickle(G, path=args.output_dir + 'graph.gpickle')

    utils.auxiliary_functions.set_json_attr_val(
        'time_to_generate_in_seconds', elapsed_time, file_path=args.output_dir+'info.json'
    )

if __name__ == "__main__":
    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Construct random directed graphs')

    # Output directory for the randomly constructed graph that is saved as a pickle file
    parser.add_argument('-od', '--output_dir', metavar='output_dir', required=True,
    help='Output directory for the randomly constructed graph that is saved as a pickle file')

    # Specifies the number of nodes for the random graph
    parser.add_argument('-n', '--num_nodes', metavar='num_nodes', type=int, default=100,
    help='Specifies the number of nodes for the random graph')

    # Graph generator model used for the generated random graph. scale-free is very slow for large graphs
    parser.add_argument('--graph_generator', choices=['erdos-renyi', 'scale-free'], default='erdos-renyi',
    help='Graph generator model used for the generated random graph. scale-free is very slow for large graphs')

    # Seed used for randomization
    parser.add_argument('-s', '--seed', metavar='seed', type=int,
    help='Seed used for randomization')

    # Parse the arguments
    args = parser.parse_args() 

    print('##### ----- Running random_graph_creator.py with the following parameters ----- #####\n')

    print('Output directory:', args.output_dir)
    print('Number of nodes:', args.num_nodes)
    print('Graph generator:', args.graph_generator)
    if args.seed:
        print('User specified seed:', args.seed)
        random.seed(args.seed) # Set the seed
    else:
        # Generate a random seed if not specified
        args.seed = random.randrange(sys.maxsize)
        random.seed(args.seed)
        print('No seed specified, picking one at random. Seed chosen is:', args.seed)
    print('\n\n')

    # Create the output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save the input arguments in the output_dir
    with open(args.output_dir + 'info.json', 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    main(args)