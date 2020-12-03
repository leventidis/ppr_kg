import json
import random
import networkx as nx
import numpy as np

from wikidata.client import Client
import json
from urllib.request import urlopen

def get_query_nodes(G, k):
    '''
    Given a graph it randomly selects `k` nodes to be used as the query nodes.
    The query nodes must have an out degree of at least 1. 

    Arguments
    -------
        G (networkx graph): Input graph

        k (int): number of query nodes to randomly select. `k` must be less than the number of valid nodes in the graph

    Returns
    -------
    A list of `k` randomly selected query nodes.
    '''

    # Select 'k' query nodes randomly. The nodes selected must have an out-degree of at least 1.
    valid_nodes = []
    for tup in G.out_degree():
        if (tup[1] >= 1):
            valid_nodes.append(tup[0])

    assert (len(valid_nodes) >= k), 'The number of query nodes cannot be larger that the number of valid nodes in the graph'
    Q = random.choices(valid_nodes, k=k)
    return Q

def get_top_k_vals_numpy(arr, k):
    '''
    Given a numpy array, return the top k values of the array.

    It uses argpartition which is more efficient than sorting the whole array.
    More details in this post: https://stackoverflow.com/a/23734295/7407311

    Arguments
    -------
        arr (numpy array): Input numpy array

        k (int): Specifies the number of top k values to return

    Returns
    -------
    A list of 2D tuples. The first element in the tuple is the index and the second element the value of array at that index.
    In the context of ppr scores it will be the node_id and its respective ppr score. 
    '''
    # We are using -k because we want to sort from high->low
    arr_indices = np.argpartition(arr, -k)[-k:]
    # argpartition doesn't sort the values within the partition so we perform a full sort over k-values 
    arr_indices = arr_indices[np.argsort(arr[arr_indices])][::-1]

    return_list = []
    for idx in arr_indices:
        return_list.append((idx, arr[idx]))
    return return_list

def set_json_attr_val(attr, data, file_path):
    '''
    Sets attribute `attr` to `data` for the json file at `file_path`
    '''
    with open(file_path, "r") as jsonFile:
        json_data = json.load(jsonFile)

    json_data[attr] = data

    with open(file_path, "w") as jsonFile:
        json.dump(json_data, jsonFile, sort_keys=True, indent=4)

def get_json_attr_val(attr, file_path):
    '''
    Returns the data value of the object specified by attribute `attr` from the json specified in `file_path` 
    If attribute `attr` doesn't exist then return None 
    '''
    with open(file_path, "r") as jsonFile:
        data = json.load(jsonFile)
    if attr in data:
        return data[attr]
    else:
        return None

def print_top_k_nodes(top_k_ppr, id_to_node_dict, print_node_names):
    '''
    Prints out the node and its respective ppr score

    Arguments
    -------
        top_k_ppr (list of 2D tuples): The first element of the tuple is the node id.
        The second element is the score. The tuples are sorted high->low ppr scores.

        id_to_node_dict (dict): Dictionary mapping each node id to each name

        print_node_names (bool): If true then prints node names instead of IDs 

    Returns
    -------
    Nothing
    '''
    for tup in top_k_ppr:
        if print_node_names:
            node = str(id_to_node_dict[tup[0]]) 
            print(node + '\t'+ getWikiDataNameFromNode(node) +'\t: ' + str(tup[1]))
        else:
            print(str(tup[0]) + ': ' + str(tup[1]))


def getWikiDataNameFromNode (url):
    try: 
        data = json.load(urlopen(url))
        id = url.split("/")[-1]
        label = data["entities"][id]["labels"]["en"]["value"]
        return label
    except:
        return ""