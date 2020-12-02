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

        dict_path = args.graph_path[0:args.graph_path.rfind('/') + 1]

        if (not args.no_dictionary):
            node_to_id_dict = pickle.load(open(dict_path+'node_to_id_dict.pickle', 'rb'))

        id_to_node_dict = None
        if args.print_node_names_in_top_k:
            id_to_node_dict = pickle.load(open(dict_path+'id_to_node_dict.pickle', 'rb'))
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
        with open(args.graph_output_dir + 'id_to_node_dict.pickle', 'wb') as handle:
            pickle.dump(id_to_node_dict, handle)
        with open(args.graph_output_dir + 'node_to_id_dict.pickle', 'wb') as handle:
            pickle.dump(node_to_id_dict, handle)

    print('Input graph has', G.number_of_nodes(), 'nodes and', G.number_of_edges(), 'edges')
    utils.auxiliary_functions.set_json_attr_val('graph_info', {'num_nodes': G.number_of_nodes(), 'num_edges': G.number_of_edges()}, file_path=args.output_dir+'args.json')

    if args.user_specified_query_nodes:
        # Use the user specified query nodes
        Q = []
        for q_name in args.user_specified_query_nodes:
            Q.append(node_to_id_dict[q_name])
    else:
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
        print('Calculating PPR from each source in the query set...')

        if args.distributed_single_source_ppr:
            # Single source ppr multi-core implementation
            start_timer = timer()
            aggregate_ppr_single_source_node_np_array, stats_dict = utils.ppr.get_ppr_from_single_source_nodes_parallel(G, Q)
            print('Total Elapsed time distribute implementation:', timer()-start_timer)
        else:
            # Single source ppr single-core implementation
            aggregate_ppr_single_source_node_np_array = np.zeros(G.number_of_nodes())
            stats_dict = {}
            start_timer = timer()
            for query_node in tqdm(Q):
                start = timer()
                ppr_single_source_node_np_array, num_iterations = utils.ppr.get_ppr(G, [query_node], return_type='array')
                elapsed_time = timer()-start
                stats_dict[query_node] = {'runtime': elapsed_time, 'num_iterations': num_iterations}
                aggregate_ppr_single_source_node_np_array += ppr_single_source_node_np_array
            print('Total Elapsed time with single cpu:', timer()-start_timer)

        # Calculate a combined ppr vector for all sources in the query 
        ppr_single_sources = aggregate_ppr_single_source_node_np_array / len(Q) 

        utils.auxiliary_functions.set_json_attr_val('ppr_single_source_using_pf', stats_dict, file_path=args.output_dir+'info.json')
        print('Finished calculating PPR from each source in the query set.\n')

        with open(args.output_dir + 'ppr_single_source_scores.npy', 'wb') as f:
            np.save(f, ppr_single_sources)

    # Evaluation of the results 
    # Top-10 nodes using particle filtering
    top_k_ppr = utils.auxiliary_functions.get_top_k_vals_numpy(ppr_np_array, k=10)
    print('TOP-10 nodes using particle filtering')
    utils.auxiliary_functions.print_top_k_nodes(top_k_ppr, id_to_node_dict, args.print_node_names_in_top_k)

    if args.run_ppr_from_each_query_node:
        # Get top-k values from numpy array
        top_k_ppr_single_sources = utils.auxiliary_functions.get_top_k_vals_numpy(ppr_single_sources, 10)
        print('\nTOP-10 nodes using multiple sources particle filtering')
        utils.auxiliary_functions.print_top_k_nodes(top_k_ppr_single_sources, id_to_node_dict, args.print_node_names_in_top_k)

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

    if args.distributed_pf:
        #Evaluation of Distributed Particle Filtering
        ppr_dist, num_iterations_dist = utils.ppr.get_ppr_distributed(G, Q, return_type='array')
        top_k_ppr_dist = utils.auxiliary_functions.get_top_k_vals_numpy(ppr_dist, 10)
        utils.auxiliary_functions.print_top_k_nodes(top_k_ppr_dist, id_to_node_dict, args.print_node_names_in_top_k)

        k_vals = [1, 5, 10, 50, 100, 200, 500, 1000]
        ndcg_dist_dict = {}
        print('The number of iterations for distributed PPR', num_iterations_dist)
        print('\n\nNormalized discounted cumulative gain (NDCG) scores at various k values for Dist PPR')
        for k in k_vals:
            ndcg_dict[k] = ndcg_score(np.array([ppr_np_array]), np.array([ppr_dist]), k=k)
            print('NDCG score at k=' + str(k) + ':', ndcg_dict[k])
    

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

        # Convert 'ppr_dict_nx' into an array for easy NDCG scores comparison
        ppr_array_nx = []
        for id in ppr_dict_nx:
            ppr_array_nx.append(ppr_dict_nx[id])

        print('\n\nTOP-10 nodes using Networkx implementation of PPR')
        top_k_ppr_nx = utils.auxiliary_functions.get_top_k_vals_numpy(np.array(ppr_array_nx), 10)
        utils.auxiliary_functions.print_top_k_nodes(top_k_ppr_nx, id_to_node_dict, args.print_node_names_in_top_k)
        
        # Calculate the NDCG scores using networkx vs ppr_np_array. The networkx scores are used as the ground truth
        k_vals = [1, 5, 10, 50, 100, 200, 500, 1000]
        ndcg_dict = {}
        print('\n\nNormalized discounted cumulative gain (NDCG) scores at various k values for networkx PPR vs PPR using PF')
        for k in k_vals:
            ndcg_dict[str(k)] = ndcg_score(np.array([ppr_array_nx]), np.array([ppr_np_array]), k=k)
            print('NDCG score at k=' + str(k) + ':', ndcg_dict[str(k)])
        ndcg_dict['full'] = ndcg_score(np.array([ppr_array_nx]), np.array([ppr_np_array]))
        utils.auxiliary_functions.set_json_attr_val('ndcg_scores_nx', ndcg_dict, file_path=args.output_dir+'info.json')
        

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

    # Sets the query nodes to the ones specified in the command line.
    # The query nodes must be given as node names that are seperated by space
    # This assumes that a node_to_id_dict.pickle file exists in the --graph_path directory
    parser.add_argument('--user_specified_query_nodes', nargs='+', 
    help='Sets the query nodes to the ones specified in the command line. \
    The query nodes must be given as node names that are seperated by space. \
    This assumes that a node_to_id_dict.pickle file exists in the --graph_path directory')

    # Seed used for randomization
    parser.add_argument('-s', '--seed', metavar='seed', type=int,
    help='Seed used for randomization')

    # Denotes if we want to run PPR from each each query node seperately.
    parser.add_argument('--run_ppr_from_each_query_node', action='store_true', 
    help='Denotes if we want to run PPR from each each query node seperately.')

    # Denotes if we want to run the PPR implementation provided by NetworkX.
    parser.add_argument('--run_networkx_ppr', action='store_true', 
    help='Denotes if we want to run the PPR implementation provided by NetworkX.')

    # Denotes if we want to run PPR using particle filtering in a distributed manner.
    parser.add_argument('--distributed_pf', action='store_true', 
    help='Denotes if we want to run PPR using particle filtering in a distributed manner.')

    # Denotes if we want run single source ppr in a distributed manner. If not specified a single-cpu implementation is used
    parser.add_argument('--distributed_single_source_ppr', action='store_true', 
    help='Denotes if we want run single source ppr in a distributed manner. If not specified a single-cpu implementation is used')

    # If specified graph already exists so we do not need to build it and can directly load it
    # The path to the graph must be a pickle object of a networkx graph.
    parser.add_argument('--graph_path', metavar='graph_path',
    help='If specified graph already exists so we do not need to build it and can directly load it.\
    The path to the graph must be a pickle object of a networkx graph.')

    # If specified top-k ppr scores are printed in the terminal using the node names and not the ids
    # The dictionaries between id to node must be stored in the same directory as the --graph_path
    parser.add_argument('--print_node_names_in_top_k', action='store_true',
    help='If specified top-k ppr scores are printed in the terminal using the node names and not the ids. \
    The dictionaries between id to node must be stored in the same directory as the --graph_path.')

    # If specified it is assumed that no mapping dictionaries exist.
    # This is to be used only for random graph testing
    parser.add_argument('--no_dictionary', action='store_true',
    help='If specified it is assumed that no mapping dictionaries exist. This is to be used only for random graph testing')

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

    print('Output directory:', args.output_dir)
    if args.user_specified_query_nodes:
        print('Using the following query nodes as specified by the user:', args.user_specified_query_nodes)
    else:
        print('Number of query nodes randomly chosen:', args.num_q_nodes)
    if args.run_ppr_from_each_query_node:
        print('In addition: Run PPR using PF from each query node seperately')
        if args.distributed_single_source_ppr:
            print('Single source PPR is run in a distributed manner')
    if args.run_networkx_ppr:
        print('In addition: Run PPR using the implementation by NetworkX')
    if args.distributed_pf:
        print('In addition: Run PPR with PF in a distributed manner')
    if args.print_node_names_in_top_k:
        print('Top-k scores are printed using the node\'s name instead of its ID.')
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