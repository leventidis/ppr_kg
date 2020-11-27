import networkx as nx
import pandas as pd

import argparse
import pickle
import json
import random
import sys

from heapq import nlargest 

import utils
from pathlib import Path
from timeit import default_timer as timer


def main(args):
    # Get graph from csv
    G = utils.graph.get_graph_from_csv(file='Data/Movies/triples.csv', source='head_uri', target='tail_uri', edge_attr=['relation'])

    # Save the graph
    nx.write_gpickle(G, path=args.graph_output_dir + 'graph.gpickle')

    print('Input graph has', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges')

    # Select 'k' query nodes randomly. The nodes selected must have an out-degree of at least 1.
    valid_nodes = []
    for tup in G.out_degree():
        if (tup[1] >= 1):
            valid_nodes.append(tup[0])
    Q = random.choices(valid_nodes, k=args.num_q_nodes)

    # Save the chosen nodes Q into the json file
    utils.auxiliary_functions.set_json_attr_val('query_nodes', Q, args.output_dir)

    # Get the PPR scores for every node in G given a set of query nodes Q
    ppr_dict = utils.ppr.get_ppr(G, Q)
    with open(args.output_dir + 'particle_filtering_ppr_scores.pickle', 'wb') as handle:
        pickle.dump(ppr_dict, handle)


    # Quick Evaluation of the results 

    # Top-10 nodes using particle filtering
    top_10_nodes_ppr = nlargest(10, ppr_dict, key=ppr_dict.get)
    print('TOP-10 nodes using particle filtering')
    for key in top_10_nodes_ppr:
        print(str(key) + ': ' + str(ppr_dict[key]))

    # Top-10 nodes using networkx implementation of PPR
    personalization_dict = {}
    for q in Q:
        personalization_dict[q] = 1
    start = timer()
    print('\n\nCalculating PPR using NetworkX implementation of PPR')
    ppr_dict_nx = nx.pagerank(G, alpha=0.85, personalization=personalization_dict)
    print('Finished calculating PPR using NetworkX implementation of PPR. Elapsed time is:', timer()-start, 'seconds.')
    with open(args.output_dir + 'networkx_ppr_scores.pickle', 'wb') as handle:
        pickle.dump(ppr_dict_nx, handle)

    top_10_nodes_ppr = nlargest(10, ppr_dict_nx, key=ppr_dict_nx.get)
    print('\n\nTOP-10 nodes using Networkx implementation of PPR')
    for key in top_10_nodes_ppr:
        print(str(key) + ': ' + str(ppr_dict_nx[key]))

if __name__ == "__main__":

    # -------------------------- Argparse Configuration -------------------------- #
    parser = argparse.ArgumentParser(description='Run Personalized Page Rank (PPR) on a graph')

    # Path to the input csv file used to build the graph
    parser.add_argument('-f', '--file_path', metavar='file_path', required=True,
    help='Path to the input csv file used to build the graph')

    # Column header for the source nodes in the csv file
    parser.add_argument('--source', metavar='source', required=True,
    help='Column header for the source nodes in the csv file')

    # Column header for the target nodes in the csv file
    parser.add_argument('--target', metavar='target', required=True,
    help='Column header for the target nodes in the csv file')

    # Column headers for edge attributes. Can be a list of column headers.
    parser.add_argument('--edge_attr', metavar='edge_attr', nargs='+',
    help='Column headers for edge attributes. Can be a list of column headers.')

    # Directory where the constructed graph is saved as a pickle file
    parser.add_argument('-go', '--graph_output_dir', metavar='graph_output', required=True,
    help='Directory where the constructed graph is saved as a pickle file')

    # Directory where the ppr scores for every node are saved
    parser.add_argument('-od', '--output_dir', metavar='output_dir', required=True,
    help=' Directory where the ppr scores for every node are saved')

    # Number of query nodes that are randomly chosen
    parser.add_argument('--num_q_nodes', metavar='num_q_nodes', type=int, default=5,
    help='Number of query nodes that are randomly chosen')

    # Seed used for randomization
    parser.add_argument('-s', '--seed', metavar='seed', type=int,
    help='Seed used for randomization')

    # Parse the arguments
    args = parser.parse_args() 

    print('##### ----- Running main.py with the following parameters ----- #####\n')

    print('CSV file path:', args.file_path)
    print('Source column header:', args.source)
    print('Target column header:', args.target)
    print('Edge attributes column headers:', args.edge_attr)
    print('Graph output directory:', args.graph_output_dir)
    print('Output directory', args.output_dir)
    print('Number of query nodes randomly chosen', args.num_q_nodes)
    if args.seed:
        print('User specified seed:', args.seed)
        # Set the seed
        random.seed(args.seed)
    else:
        # Generate a random seed if not specified
        args.seed = random.randrange(sys.maxsize)
        random.seed(args.seed)
        print('No seed specified, picking one at random. Seed chosen is:', args.seed)
    print('\n\n')

    # Create the output directories if they doesn't exist
    Path(args.graph_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    # Save the input arguments in the output_dir
    with open(args.output_dir + 'args.json', 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    main(args)