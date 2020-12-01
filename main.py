import networkx as nx
import pandas as pd
import numpy as np

import argparse
import pickle
import json
import random
import sys

from heapq import nlargest
from tqdm import tqdm

import utils
from pathlib import Path
from timeit import default_timer as timer

from sklearn.metrics import ndcg_score, dcg_score


def main(args):
    if args.graph_path:
        G = nx.read_gpickle(args.graph_path)
    else:
        # We need to create the graph from a csv
        G = utils.graph.get_graph_from_csv(file=args.file_path, source=args.source, target=args.target, edge_attr=args.edge_attr)

        # Convert all node names to integer IDs (starting with ID=0)
        ids = range(G.number_of_nodes())
        nodes = list(G.nodes())
        id_to_node_dict = {ids[i]: nodes[i] for i in range(len(ids))}
        node_to_id_dict = {nodes[i]: ids[i] for i in range(G.number_of_nodes())}
        G = nx.relabel_nodes(G, node_to_id_dict)

        # Save the graph and the dictionaries
        nx.write_gpickle(G, path=args.graph_output_dir + 'graph.gpickle')
        with open(args.output_dir + 'id_to_node_dict.pickle', 'wb') as handle:
            pickle.dump(id_to_node_dict, handle)
        with open(args.output_dir + 'node_to_id_dict.pickle', 'wb') as handle:
            pickle.dump(node_to_id_dict, handle)

    print('Input graph has', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges')
    utils.auxiliary_functions.set_json_attr_val('graph_info', {'num_nodes': G.number_of_nodes(), 'num_edges': G.number_of_edges()}, file_path=args.output_dir+'args.json')

    # Select 'k' query nodes randomly. The nodes selected must have an out-degree of at least 1.
    Q = utils.auxiliary_functions.get_query_nodes(G, k=args.num_q_nodes)

    # Save the chosen nodes Q into the json file
    utils.auxiliary_functions.set_json_attr_val('query_nodes', Q, file_path=args.output_dir+'args.json')

    # Get the PPR scores for every node in G given a set of query nodes Q using particle filtering
    start = timer()
    print('Calculating PPR using particle filtering...')
    ppr_np_array, num_iterations = utils.ppr.get_ppr(G, Q, return_type='array')
    elapsed_time = timer()-start
    print('Finished calculating PPR using particle filtering. Took', num_iterations, 'iterations for convergence. Elapsed time is:', elapsed_time, 'seconds.\n')
    with open(args.output_dir + 'particle_filtering_ppr_scores.npy', 'wb') as f:
        np.save(f, ppr_np_array)
    utils.auxiliary_functions.set_json_attr_val('ppr_using_pf', {'runtime': elapsed_time, 'num_iterations': num_iterations }, file_path=args.output_dir+'info.json')

    # Check if we want to also run PPR from each query node seperately
    if args.run_ppr_from_each_query_node:
        single_source_output_dir = args.output_dir + 'single_source_ppr_scores/'
        Path(args.output_dir + 'single_source_ppr_scores/').mkdir(parents=True, exist_ok=True)
        print('Calculating PPR from each source in the query set...')
        stats_dict = {}
        # Aggregate vector of the single source nodes
        aggregate_ppr_single_source_node_np_array = np.zeros(G.number_of_nodes())
        #TODO: Parallelize this operation
        for query_node in tqdm(Q):
            start = timer()
            ppr_single_source_node_np_array, num_iterations = utils.ppr.get_ppr(G, [query_node], return_type='array')
            elapsed_time = timer()-start
            stats_dict[query_node] = {'runtime': elapsed_time, 'num_iterations': num_iterations}
            aggregate_ppr_single_source_node_np_array += ppr_single_source_node_np_array

            # Instead of saving the ppr vectors to disk, store in-memory and only save the combined ppr vector
            # with open(single_source_output_dir + str(query_node) + '.npy', 'wb') as f:
            #     np.save(f, ppr_single_source_node_np_array)

        # Calculate a combined ppr vector for all sources in the query 
        ppr_single_sources = aggregate_ppr_single_source_node_np_array / len(Q) 

        utils.auxiliary_functions.set_json_attr_val('ppr_single_source_using_pf', stats_dict, file_path=args.output_dir+'info.json')
        print('Finished calculating PPR from each source in the query set.\n')

        # ppr_single_sources = utils.ppr.get_ppr_from_single_source_nodes(single_source_output_dir)

        with open(args.output_dir + 'ppr_single_source_scores.npy', 'wb') as f:
            np.save(f, ppr_single_sources)

    # Evaluation of the results 
    # Top-10 nodes using particle filtering
    top_k_ppr = utils.auxiliary_functions.get_top_k_vals_numpy(ppr_np_array, k=10)
    print('TOP-10 nodes using particle filtering')
    for tup in top_k_ppr:
        print(str(tup[0]) + ': ' + str(tup[1]))

    if args.run_ppr_from_each_query_node:
        # Get top-k values from numpy array
        top_k_ppr_single_sources = utils.auxiliary_functions.get_top_k_vals_numpy(ppr_single_sources, 10)
        print('\nTOP-10 nodes using multiple sources particle filtering')
        for tup in top_k_ppr_single_sources:
            print(str(tup[0]) + ': ' + str(tup[1]))

        # Calculate the normalized discounted cumulative gain (NDCG) between the ppr vs the ppr_single_source rankings
        k_vals = [1, 5, 10, 50, 100, 200, 500, 1000]
        ndcg_dict = {}
        print('\n\nNormalized discounted cumulative gain (NDCG) scores at various k values')
        for k in k_vals:
            ndcg_dict[str(k)] = ndcg_score(np.array([ppr_np_array]), np.array([ppr_single_sources]), k=k)
            print('NDCG score at k=' + str(k) + ':', ndcg_dict[str(k)])
        # Calculate NDCG scores for all rankings (k=total_number_of_nodes)
        ndcg_dict['full'] = ndcg_score(np.array([ppr_np_array]), np.array([ppr_single_sources]))

        utils.auxiliary_functions.set_json_attr_val('ndcg_scores', ndcg_dict, file_path=args.output_dir+'info.json')

    if args.run_networkx_ppr:
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
    parser.add_argument('-f', '--file_path', metavar='file_path',
    help='Path to the input csv file used to build the graph')

    # Column header for the source nodes in the csv file
    parser.add_argument('--source', metavar='source',
    help='Column header for the source nodes in the csv file')

    # Column header for the target nodes in the csv file
    parser.add_argument('--target', metavar='target',
    help='Column header for the target nodes in the csv file')

    # Column headers for edge attributes. Can be a list of column headers.
    parser.add_argument('--edge_attr', metavar='edge_attr', nargs='+',
    help='Column headers for edge attributes. Can be a list of column headers.')

    # Directory where the constructed graph is saved as a pickle file
    parser.add_argument('-go', '--graph_output_dir', metavar='graph_output',
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

    # Denotes if we want to run PPR from each each query node seperately.
    parser.add_argument('--run_ppr_from_each_query_node', action='store_true', 
    help='Denotes if we want to run PPR from each each query node seperately.')

    # Denotes if we want to run the PPR implementation provided by NetworkX.
    parser.add_argument('--run_networkx_ppr', action='store_true', 
    help='Denotes if we want to run the PPR implementation provided by NetworkX.')

    # If specified graph already exists so we do not need to build it and can directly load it
    # The path to the graph must be a pickle object of a networkx graph.
    parser.add_argument('--graph_path', metavar='graph_path',
    help='If specified graph already exists so we do not need to build it and can directly load it.\
    The path to the graph must be a pickle object of a networkx graph.')

    # Parse the arguments
    args = parser.parse_args() 

    # Ensure we have all necessary arguments
    if args.graph_path is None and (args.file_path is None or args.source is None or args.target is None or
        args.edge_attr is None or args.graph_output_dir is None):
        parser.error('''
        if --graph_path is not specified must specify all of the following arguments:
        --file_path, --source, --target, --edge_attr, --graph_output_dir''')
    
    print('##### ----- Running main.py with the following parameters ----- #####\n')

    if args.graph_path:
        print('Graph path:', args.graph_path)
    else:
        print('CSV file path:', args.file_path)
        print('Source column header:', args.source)
        print('Target column header:', args.target)
        print('Edge attributes column headers:', args.edge_attr)
        print('Graph output directory:', args.graph_output_dir)
        Path(args.graph_output_dir).mkdir(parents=True, exist_ok=True)

    print('Output directory', args.output_dir)
    print('Number of query nodes randomly chosen', args.num_q_nodes)
    if args.run_ppr_from_each_query_node:
        print('In addition: Run PPR using PF from each query node seperately')
    if args.run_networkx_ppr:
        print('In addition: Run PPR using the implementation by NetworkX')
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
    with open(args.output_dir + 'args.json', 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    # Create an empty json file (info.json) that can store useful run information
    with open(args.output_dir + 'info.json', 'w') as fp:
        json.dump({}, fp, sort_keys=True, indent=4)

    main(args)